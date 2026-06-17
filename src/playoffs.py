"""
Playoff bracket simulator.

A regular-season game model predicts a SINGLE game. The playoffs are best-of-7
series organised into a 16-team bracket. To predict the championship we:

  1. Predict each potential matchup's single-game win probability (home/away)
  2. Translate that into a SERIES win probability (best-of-7 with 2-2-1-1-1 format)
  3. Monte-Carlo simulate the whole bracket N times, counting how often each
     team advances and wins it all

NBA playoff format
──────────────────
  • 16 teams: 8 East + 8 West, seeded 1-8 within each conference
  • Round 1 (Conference Quarter-finals):  1v8, 4v5, 3v6, 2v7
  • Round 2 (Conference Semi-finals):     R1 winners pair up
  • Round 3 (Conference Finals):          Survivors play
  • Round 4 (NBA Finals):                 East champion vs West champion
  • Each series is best-of-7
  • Higher seed gets home court (games 1, 2, 5*, 7*)

This module also supports MID-BRACKET simulation via simulate_from_round_2:
pass the current R2 matchups + series scores and we'll project the rest of
the bracket from there. Useful once R1 is over and you want sharper odds.
"""

import random
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

import config
from src.predict import (
    load_artifacts, team_id_from_abbr, build_prediction_features,
)


# Standard NBA playoff schedule: which team is HOME in each game (True = higher seed)
SERIES_HOME_PATTERN = [True, True, False, False, True, False, True]


# ── Model wrapper: predict ONE matchup, return P(home wins) ──────────────────

class MatchupPredictor:
    """
    Loads model + data ONCE, then answers many predict-matchup queries fast.
    Cached so repeated (home, away, date) queries don't recompute.
    """

    def __init__(self, game_date: str | None = None):
        self.model, self.scaler, self.feature_cols = load_artifacts()

        from src.data_fetch import fetch_all_seasons, build_game_table
        raw = fetch_all_seasons()
        self.games = build_game_table(raw)

        self.game_date = pd.Timestamp(game_date) if game_date else pd.Timestamp.now().normalize()
        self._cache: dict[tuple[str, str], float] = {}

    def predict(self, home_abbr: str, away_abbr: str) -> float:
        """Return calibrated P(home team wins) for a single game."""
        key = (home_abbr.upper(), away_abbr.upper())
        if key in self._cache:
            return self._cache[key]

        home_id, _ = team_id_from_abbr(home_abbr)
        away_id, _ = team_id_from_abbr(away_abbr)

        feats = build_prediction_features(home_id, away_id, self.game_date, self.games)
        X = feats[self.feature_cols].values.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        # v2: applies temperature scaling + XGBoost ensemble if available
        from src.predict import predict_calibrated, load_artifacts as _la
        prob = predict_calibrated(
            self.model, self.scaler, X_scaled,
            xgb_model=getattr(_la, "last_xgb", None),
            temperature=getattr(_la, "last_temperature", 1.0),
        )
        self._cache[key] = prob
        return prob


# ── Series simulation (best-of-7) ─────────────────────────────────────────────

def series_win_prob(
    higher_seed: str,
    lower_seed: str,
    predictor: MatchupPredictor,
    n_sims: int = 5000,
    rng: random.Random = None,
) -> dict:
    """
    Simulate a best-of-7 series with the 2-2-1-1-1 home/away pattern.

    Returns:
        win_prob:   probability higher seed wins the series
        avg_games:  expected number of games (4 to 7)
        game_dist:  histogram of series length (4,5,6,7)
    """
    rng = rng or random.Random(config.RANDOM_SEED)

    # Single-game probabilities (cached after first call)
    p_higher_at_home = predictor.predict(higher_seed, lower_seed)
    p_higher_away    = 1.0 - predictor.predict(lower_seed, higher_seed)

    wins = 0
    game_counts = []
    for _ in range(n_sims):
        higher_score, lower_score = 0, 0
        for game_idx, higher_is_home in enumerate(SERIES_HOME_PATTERN):
            p = p_higher_at_home if higher_is_home else p_higher_away
            if rng.random() < p:
                higher_score += 1
            else:
                lower_score += 1
            if higher_score == 4 or lower_score == 4:
                break
        if higher_score == 4:
            wins += 1
        game_counts.append(higher_score + lower_score)

    return {
        "higher_seed":      higher_seed,
        "lower_seed":       lower_seed,
        "p_higher_seed":    wins / n_sims,
        "p_lower_seed":     1 - wins / n_sims,
        "p_higher_at_home": p_higher_at_home,
        "p_higher_away":    p_higher_away,
        "avg_games":        sum(game_counts) / len(game_counts),
    }


# ── Mid-series simulation (given current state) ──────────────────────────────

@dataclass
class SeriesState:
    """A best-of-7 series in progress (or completed)."""
    higher_seed: str
    lower_seed:  str
    higher_wins: int = 0
    lower_wins:  int = 0

    @property
    def is_over(self) -> bool:
        return self.higher_wins >= 4 or self.lower_wins >= 4

    @property
    def winner(self) -> str | None:
        if not self.is_over:
            return None
        return self.higher_seed if self.higher_wins >= 4 else self.lower_seed

    @property
    def games_played(self) -> int:
        return self.higher_wins + self.lower_wins

    def describe(self) -> str:
        if self.is_over:
            return f"{self.higher_seed} vs {self.lower_seed}: {self.winner} won {self.higher_wins}-{self.lower_wins}"
        return f"{self.higher_seed} vs {self.lower_seed}: {self.higher_wins}-{self.lower_wins} (in progress)"


def simulate_remaining(
    state: SeriesState,
    predictor: "MatchupPredictor",
    n_sims: int = 5000,
    rng: random.Random = None,
) -> dict:
    """
    Given a series IN PROGRESS, simulate the remaining games. If the series
    is already over, returns probability 1.0 / 0.0.
    """
    rng = rng or random.Random(config.RANDOM_SEED)

    if state.is_over:
        return {
            "p_higher_seed": 1.0 if state.winner == state.higher_seed else 0.0,
            "p_lower_seed":  1.0 if state.winner == state.lower_seed  else 0.0,
            "avg_games_remaining": 0.0,
            "winner_known":       True,
        }

    p_higher_at_home = predictor.predict(state.higher_seed, state.lower_seed)
    p_higher_away    = 1.0 - predictor.predict(state.lower_seed, state.higher_seed)

    higher_series_wins = 0
    remaining_lengths  = []
    for _ in range(n_sims):
        h, l = state.higher_wins, state.lower_wins
        games_remaining = 0
        # Step through game numbers starting from the next un-played game
        for game_idx in range(state.games_played, 7):
            higher_is_home = SERIES_HOME_PATTERN[game_idx]
            p = p_higher_at_home if higher_is_home else p_higher_away
            games_remaining += 1
            if rng.random() < p:
                h += 1
            else:
                l += 1
            if h >= 4 or l >= 4:
                break
        if h >= 4:
            higher_series_wins += 1
        remaining_lengths.append(games_remaining)

    return {
        "p_higher_seed":       higher_series_wins / n_sims,
        "p_lower_seed":        1 - higher_series_wins / n_sims,
        "avg_games_remaining": sum(remaining_lengths) / len(remaining_lengths),
        "winner_known":        False,
    }


# ── Mid-bracket simulation (from Round 2 with current state) ─────────────────

def simulate_from_round_2(
    r2_east: list[SeriesState],   # 2 series: top-quadrant and bottom-quadrant
    r2_west: list[SeriesState],
    predictor: "MatchupPredictor" = None,
    n_sims: int = 5000,
    series_sims: int = 5000,
) -> dict:
    """
    Simulate the rest of the playoffs given the current Round 2 state.

    Args:
        r2_east: [top-quadrant series, bottom-quadrant series]
                 e.g. [SeriesState(DET, CLE, 2, 1), SeriesState(NYK, PHI, 3, 0)]
        r2_west: same shape for West
        predictor: MatchupPredictor (reused if provided)
        n_sims: number of full simulations from this point
        series_sims: Monte Carlo runs per series

    Returns dict with championship / round-by-round probabilities.
    """
    if predictor is None:
        predictor = MatchupPredictor()

    rng = random.Random(config.RANDOM_SEED)

    # Pre-compute "rest of series" win probabilities for the 4 R2 series
    print("Simulating remaining games in current R2 series …")
    r2_outcomes: dict[tuple[str, str], dict] = {}  # series key → result
    for state in r2_east + r2_west:
        key = (state.higher_seed, state.lower_seed)
        result = simulate_remaining(state, predictor, n_sims=series_sims, rng=rng)
        r2_outcomes[key] = {
            "state": state,
            "p_higher": result["p_higher_seed"],
        }
        if result["winner_known"]:
            print(f"  · {state.describe()}  [already over]")
        else:
            print(f"  · {state.describe()}  → higher seed wins remaining "
                  f"{result['p_higher_seed']:.1%}")

    # Counters
    counts_conf_final = defaultdict(int)
    counts_finals     = defaultdict(int)
    counts_champ      = defaultdict(int)

    # Cache for hypothetical future series probabilities
    future_series_cache: dict[tuple[str, str], float] = {}

    def series_prob(higher: str, lower: str) -> float:
        """P(higher beats lower) in a FRESH best-of-7 series."""
        key = (higher, lower)
        if key in future_series_cache:
            return future_series_cache[key]
        # Fresh series: full Monte Carlo from 0-0
        fresh = SeriesState(higher, lower, 0, 0)
        result = simulate_remaining(fresh, predictor, n_sims=series_sims, rng=rng)
        future_series_cache[key] = result["p_higher_seed"]
        return future_series_cache[key]

    print(f"\nRunning {n_sims} bracket continuations …")
    for _ in range(n_sims):
        # Step 1: resolve the 4 R2 series
        def draw_r2(states_pair):
            winners = []
            for state in states_pair:
                p = r2_outcomes[(state.higher_seed, state.lower_seed)]["p_higher"]
                winner = state.higher_seed if rng.random() < p else state.lower_seed
                winners.append(winner)
            return winners

        east_top, east_bot = draw_r2(r2_east)
        west_top, west_bot = draw_r2(r2_west)

        # Step 2: Conference Finals (top quadrant vs bottom quadrant)
        # Pick higher seed by looking at recent ELO via predictor (simple proxy:
        # whoever has higher predicted home prob if they were home)
        def conf_final(a: str, b: str) -> str:
            # Use a-at-home prob > 0.5 as "a is favored" heuristic
            p_a_home = predictor.predict(a, b)
            higher, lower = (a, b) if p_a_home >= 0.5 else (b, a)
            p = series_prob(higher, lower)
            return higher if rng.random() < p else lower

        east_champ = conf_final(east_top, east_bot)
        west_champ = conf_final(west_top, west_bot)
        counts_conf_final[east_top] += 1
        counts_conf_final[east_bot] += 1
        counts_conf_final[west_top] += 1
        counts_conf_final[west_bot] += 1
        counts_finals[east_champ] += 1
        counts_finals[west_champ] += 1

        # Step 3: NBA Finals (neutral home-court — average both directions)
        p_east_home = series_prob(east_champ, west_champ)
        p_west_home = series_prob(west_champ, east_champ)
        p_east_wins = (p_east_home + (1 - p_west_home)) / 2
        champion = east_champ if rng.random() < p_east_wins else west_champ
        counts_champ[champion] += 1

    def normalise(counter: dict) -> dict:
        return {t: c / n_sims for t, c in sorted(counter.items(), key=lambda x: -x[1])}

    return {
        "r2_outcomes":       r2_outcomes,
        "conf_finals_probs": normalise(counts_conf_final),
        "finals_probs":      normalise(counts_finals),
        "champion_probs":    normalise(counts_champ),
        "n_sims":            n_sims,
    }


# ── Full bracket Monte Carlo ─────────────────────────────────────────────────

def simulate_bracket(
    east_seeds: list[str],
    west_seeds: list[str],
    n_sims: int = 2000,
    series_sims: int = 1000,
    predictor: MatchupPredictor = None,
) -> dict:
    """
    Run a full playoff Monte-Carlo simulation.

    Args:
        east_seeds:  8 team abbreviations in seed order (seed 1 first)
        west_seeds:  same for West
        n_sims:      number of full-bracket simulations
        series_sims: Monte Carlo runs PER series within each bracket sim
        predictor:   re-use a MatchupPredictor if you have one (faster)

    Returns:
        Dict with:
          - 'champion_probs': {team: probability}
          - 'finals_probs':   {team: probability}
          - 'conf_finals_probs': {team: probability}
          - 'round2_probs':   {team: probability}
          - 'series_results': pre-computed first-round series outcomes
    """
    if predictor is None:
        predictor = MatchupPredictor()

    rng = random.Random(config.RANDOM_SEED)

    # Pre-compute Round 1 series probabilities (only 8 fixed matchups)
    east_r1 = [
        (east_seeds[0], east_seeds[7]),   # 1 vs 8
        (east_seeds[3], east_seeds[4]),   # 4 vs 5
        (east_seeds[2], east_seeds[5]),   # 3 vs 6
        (east_seeds[1], east_seeds[6]),   # 2 vs 7
    ]
    west_r1 = [
        (west_seeds[0], west_seeds[7]),
        (west_seeds[3], west_seeds[4]),
        (west_seeds[2], west_seeds[5]),
        (west_seeds[1], west_seeds[6]),
    ]

    # Build per-series win probabilities (cached single-game predictions reused)
    print("Computing single-game probabilities for all potential matchups …")
    series_cache: dict[tuple[str, str], float] = {}

    def get_series_prob(higher: str, lower: str) -> float:
        """Cached probability that `higher` beats `lower` in a series."""
        key = (higher, lower)
        if key not in series_cache:
            result = series_win_prob(higher, lower, predictor, n_sims=series_sims, rng=rng)
            series_cache[key] = result["p_higher_seed"]
        return series_cache[key]

    # Pre-fill round-1 series probabilities (deterministic — only 8 calls)
    for higher, lower in east_r1 + west_r1:
        get_series_prob(higher, lower)
    print(f"  ✓ Computed {len(series_cache)} round-1 series")

    # Counters across full-bracket simulations
    counts_round2     = defaultdict(int)
    counts_conf_final = defaultdict(int)
    counts_finals     = defaultdict(int)
    counts_champ      = defaultdict(int)

    # Run N full-bracket simulations
    for sim_idx in range(n_sims):
        # Round 1
        east_r2_teams = []
        for higher, lower in east_r1:
            p = get_series_prob(higher, lower)
            winner = higher if rng.random() < p else lower
            east_r2_teams.append((higher, lower, winner))
            counts_round2[winner] += 1

        west_r2_teams = []
        for higher, lower in west_r1:
            p = get_series_prob(higher, lower)
            winner = higher if rng.random() < p else lower
            west_r2_teams.append((higher, lower, winner))
            counts_round2[winner] += 1

        # Round 2 — bracket pairs: (1v8 winner) plays (4v5 winner); (3v6) plays (2v7)
        def seed_of(team, seeds):
            return seeds.index(team) + 1

        def pair_and_play(r2_teams, seeds):
            # r2_teams[0] = 1v8 winner, [1] = 4v5, [2] = 3v6, [3] = 2v7
            pairs = [(r2_teams[0][2], r2_teams[1][2]),  # 1/8 vs 4/5
                     (r2_teams[2][2], r2_teams[3][2])]  # 3/6 vs 2/7
            winners = []
            for a, b in pairs:
                higher, lower = (a, b) if seed_of(a, seeds) < seed_of(b, seeds) else (b, a)
                p = get_series_prob(higher, lower)
                winner = higher if rng.random() < p else lower
                winners.append(winner)
                counts_conf_final[winner] += 1
            return winners

        east_finalists = pair_and_play(east_r2_teams, east_seeds)
        west_finalists = pair_and_play(west_r2_teams, west_seeds)

        # Conference finals
        def conf_final(finalists, seeds):
            a, b = finalists
            higher, lower = (a, b) if seed_of(a, seeds) < seed_of(b, seeds) else (b, a)
            p = get_series_prob(higher, lower)
            winner = higher if rng.random() < p else lower
            counts_finals[winner] += 1
            return winner

        east_champ = conf_final(east_finalists, east_seeds)
        west_champ = conf_final(west_finalists, west_seeds)

        # NBA Finals — higher seed (overall) gets home court
        # We approximate by assigning home-court to the team with the better
        # regular-season ELO (which we read from the model's input features)
        # Simpler: just call series_win_prob both directions and pick the higher
        p_east_at_home  = get_series_prob(east_champ, west_champ)
        p_west_at_home  = get_series_prob(west_champ, east_champ)
        # Use average of the two as a neutral estimate
        p_east_wins_finals = (p_east_at_home + (1 - p_west_at_home)) / 2
        champion = east_champ if rng.random() < p_east_wins_finals else west_champ
        counts_champ[champion] += 1

    print(f"  ✓ Completed {n_sims} bracket simulations")

    def normalise(counter: dict) -> dict:
        return {t: c / n_sims for t, c in sorted(counter.items(), key=lambda x: -x[1])}

    return {
        "round2_probs":      normalise(counts_round2),
        "conf_finals_probs": normalise(counts_conf_final),
        "finals_probs":      normalise(counts_finals),
        "champion_probs":    normalise(counts_champ),
        "series_cache":      series_cache,
        "n_sims":            n_sims,
    }
