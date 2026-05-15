"""
ELO rating system for NBA teams.

ELO origins
───────────
ELO was invented by Arpad Elo for chess in the 1960s. It assigns each player
(here: each team) a numeric rating. After every match, ratings shift based on
the gap between EXPECTED and ACTUAL outcomes:

    new_rating = old_rating + K · (actual_result − expected_result)

  - Beating a stronger team → big rating gain
  - Beating a weaker team   → small gain (or even small loss if you barely won)

This implementation follows the FiveThirtyEight NBA ELO conventions:
  - Home court advantage worth +100 ELO points
  - Margin-of-victory multiplier so blowouts count more than nail-biters
  - Between seasons, ratings regress 25% toward the mean (1500)

References:
  - https://fivethirtyeight.com/methodology/how-our-nba-predictions-work/
"""

from collections import defaultdict
import numpy as np
import pandas as pd

import config


# ── Core ELO math ─────────────────────────────────────────────────────────────

def expected_score(rating_a: float, rating_b: float) -> float:
    """
    Probability that team A beats team B given their ELO ratings.
    Standard ELO formula — every 400 points = 10× more likely to win.
    """
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def margin_multiplier(score_diff: int, elo_diff_winner: float) -> float:
    """
    FiveThirtyEight's MOV (margin of victory) adjustment.

    Larger margin → bigger ELO change. The denominator dampens the effect
    when a heavy favorite wins big (already expected, so smaller update).

      mov = ((|diff| + 3)^0.8) / (7.5 + 0.006 · elo_diff_winner)
    """
    return (abs(score_diff) + 3) ** 0.8 / (7.5 + 0.006 * elo_diff_winner)


# ── Season detection ──────────────────────────────────────────────────────────

def _season_label(date: pd.Timestamp) -> str:
    """
    NBA seasons span two calendar years (Oct → June).
    A game on 2022-12-25 belongs to the '2022-23' season.
    """
    year = date.year
    if date.month >= 8:           # August onwards = new season starting
        return f"{year}-{str(year + 1)[-2:]}"
    else:
        return f"{year - 1}-{str(year)[-2:]}"


# ── Main feature builder ──────────────────────────────────────────────────────

def compute_elo_features(games: pd.DataFrame, return_final_ratings: bool = False):
    """
    Walk through games chronologically and compute pre-game ELO ratings
    for both teams. Adds three new columns to the games DataFrame:
        home_elo_pre  – home team's ELO rating BEFORE the game
        away_elo_pre  – away team's ELO rating BEFORE the game
        elo_diff      – home_elo_pre − away_elo_pre (positive = home favored)

    'pre' is critical: we never use post-game ratings as features (would leak
    the game outcome).

    If `return_final_ratings=True`, also returns a {team_id: post-game ELO} dict
    representing each team's current rating after all processed games.
    Used by src.predict to look up "ELO going into tomorrow's game".
    """
    games = games.sort_values("GAME_DATE").reset_index(drop=True)

    ratings: dict[int, float] = defaultdict(lambda: config.ELO_INITIAL)
    last_season: str | None = None

    home_pre, away_pre = [], []

    for _, game in games.iterrows():
        season = _season_label(game["GAME_DATE"])

        # Between-season regression toward the mean
        if last_season is not None and season != last_season:
            for tid in list(ratings.keys()):
                ratings[tid] = (
                    config.ELO_SEASON_CARRYOVER * ratings[tid]
                    + (1 - config.ELO_SEASON_CARRYOVER) * config.ELO_INITIAL
                )
        last_season = season

        home_id = game["home_team_id"]
        away_id = game["away_team_id"]
        home_r  = ratings[home_id]
        away_r  = ratings[away_id]

        # Save PRE-game ratings (the features)
        home_pre.append(home_r)
        away_pre.append(away_r)

        # Update ratings with the actual result
        # Effective home rating includes home court advantage
        eff_home = home_r + config.ELO_HOME_ADVANTAGE
        expected_home = expected_score(eff_home, away_r)
        actual_home   = float(game["home_win"])
        score_diff    = int(game["home_pts"] - game["away_pts"])

        # ELO advantage from winner's perspective (for MOV multiplier)
        winner_advantage = eff_home - away_r if actual_home == 1 else away_r - eff_home
        mov = margin_multiplier(score_diff, winner_advantage)

        delta = config.ELO_K_FACTOR * mov * (actual_home - expected_home)
        ratings[home_id] = home_r + delta
        ratings[away_id] = away_r - delta

    games = games.copy()
    games["home_elo_pre"] = home_pre
    games["away_elo_pre"] = away_pre
    games["elo_diff"]     = games["home_elo_pre"] - games["away_elo_pre"]

    print(f"ELO features added. Final rating range: "
          f"{min(ratings.values()):.0f} – {max(ratings.values()):.0f}")

    if return_final_ratings:
        return games, dict(ratings)
    return games
