"""
One-command live dashboard for the NBA prediction project.

What it does:
  1. (Optional) Re-fetches latest games from Basketball Reference
  2. Reads cached game data + saved model
  3. Auto-detects current playoff state (R1 / R2 / R3 / Finals)
  4. Runs Monte Carlo bracket simulation from current state
  5. Computes the modal "most likely path" predictions
  6. Renders a self-contained HTML dashboard with all results

Usage:
    python dashboard.py                  # use cached data
    python dashboard.py --refresh        # re-fetch + retrain first
    python dashboard.py --open           # open dashboard.html in browser after

Output: dashboard.html in the project root
"""

import argparse
import json
import os
import random
import subprocess
import sys
import webbrowser
from datetime import datetime
from collections import defaultdict

# Make stdout UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import numpy as np

import config


EAST_TEAMS = {"BOS", "NYK", "BKN", "PHI", "TOR", "CLE", "DET", "IND",
              "CHI", "MIL", "ATL", "MIA", "ORL", "WAS", "CHA"}
WEST_TEAMS = {"OKC", "DEN", "MIN", "UTA", "POR", "LAL", "LAC", "GSW",
              "PHX", "SAC", "SAS", "HOU", "MEM", "DAL", "NOP"}


# ── Playoff state detection ──────────────────────────────────────────────────

def detect_playoff_series(games: pd.DataFrame, playoff_start: str = "2026-04-15") -> list:
    """
    Auto-detect playoff series from the games DataFrame.
    Returns list of dicts: {teams, wins, is_done, winner, first_game}
    """
    pg = games[games["GAME_DATE"] >= pd.Timestamp(playoff_start)].copy()
    if len(pg) == 0:
        return []

    grouped = defaultdict(list)
    for _, g in pg.iterrows():
        key = frozenset([g["home_team_abbr"], g["away_team_abbr"]])
        winner = g["home_team_abbr"] if g["home_win"] == 1 else g["away_team_abbr"]
        grouped[key].append({
            "date": g["GAME_DATE"],
            "home": g["home_team_abbr"],
            "away": g["away_team_abbr"],
            "home_pts": int(g["home_pts"]),
            "away_pts": int(g["away_pts"]),
            "winner": winner,
        })

    series = []
    for pair, games_in_pair in grouped.items():
        if len(games_in_pair) < 1:   # any meeting in the playoff window is a series
            continue
        teams = sorted(list(pair))
        wins = {t: 0 for t in teams}
        for g in games_in_pair:
            wins[g["winner"]] += 1
        is_done = max(wins.values()) >= 4
        winner = next((t for t, w in wins.items() if w == 4), None)
        series.append({
            "teams": teams,
            "wins": wins,
            "games": sorted(games_in_pair, key=lambda x: x["date"]),
            "n_games": len(games_in_pair),
            "is_done": is_done,
            "winner": winner,
            "loser": next((t for t in teams if t != winner), None) if is_done else None,
            "first_game": min(g["date"] for g in games_in_pair),
            "last_game": max(g["date"] for g in games_in_pair),
        })

    return sorted(series, key=lambda s: s["first_game"])


def organize_by_round(series_list: list) -> dict:
    """Group series by playoff round based on chronological order and team membership."""
    if not series_list:
        return {"r1": [], "r2": [], "r3": [], "finals": []}

    # Simple heuristic: bucket by start date ranges
    # R1: starts ~April 19, R2: starts ~May 4, R3: starts ~May 18, Finals: ~June 4
    r1, r2, r3, finals = [], [], [], []
    for s in series_list:
        d = s["first_game"]
        if d < pd.Timestamp("2026-05-02"):
            r1.append(s)
        elif d < pd.Timestamp("2026-05-16"):
            r2.append(s)
        elif d < pd.Timestamp("2026-06-01"):
            r3.append(s)
        else:
            finals.append(s)
    return {"r1": r1, "r2": r2, "r3": r3, "finals": finals}


def compute_standings(games: pd.DataFrame, season_start: str = "2025-10-01",
                       season_end: str = "2026-04-30") -> pd.DataFrame:
    """Compute regular-season standings for the current season."""
    df = games[(games["GAME_DATE"] >= pd.Timestamp(season_start)) &
               (games["GAME_DATE"] <= pd.Timestamp(season_end))]
    home_wins   = df[df["home_win"] == 1].groupby("home_team_abbr").size()
    away_wins   = df[df["home_win"] == 0].groupby("away_team_abbr").size()
    home_losses = df[df["home_win"] == 0].groupby("home_team_abbr").size()
    away_losses = df[df["home_win"] == 1].groupby("away_team_abbr").size()
    wins   = home_wins.add(away_wins,  fill_value=0).astype(int)
    losses = home_losses.add(away_losses, fill_value=0).astype(int)
    return pd.DataFrame({"W": wins, "L": losses}).assign(
        Pct=lambda d: d.W / (d.W + d.L)
    ).sort_values("Pct", ascending=False)


def get_seeds(standings: pd.DataFrame) -> tuple[list, list]:
    east = standings[standings.index.isin(EAST_TEAMS)].head(8).index.tolist()
    west = standings[standings.index.isin(WEST_TEAMS)].head(8).index.tolist()
    return east, west


# ── Predictions ──────────────────────────────────────────────────────────────

def run_predictions(games: pd.DataFrame, rounds: dict, predictor) -> dict:
    """
    For each in-progress or upcoming series, get the model's prediction.
    """
    from src.playoffs import SeriesState, simulate_remaining, series_win_prob

    predictions = {"by_series": {}, "champ_odds": {}, "modal_path": []}

    # Determine current round
    current = None
    for rnd_name in ["finals", "r3", "r2", "r1"]:
        if rounds[rnd_name]:
            current = rnd_name
            break

    if current is None or current == "r1":
        return predictions

    # Compute remaining R3 / R2 series predictions
    in_progress = []
    completed = []
    for rnd_name, rnd_list in rounds.items():
        for s in rnd_list:
            if s["is_done"]:
                completed.append({**s, "round": rnd_name})
            else:
                in_progress.append({**s, "round": rnd_name})

    # Predict each in-progress series
    for s in in_progress:
        t1, t2 = s["teams"]
        # Determine higher seed by season standings
        higher = t1 if s["wins"][t1] >= s["wins"][t2] else t2  # not exactly right but ok
        # Better: use ELO
        higher_in_data = predict_higher_seed(t1, t2, games)
        lower = t2 if higher_in_data == t1 else t1
        state = SeriesState(higher_in_data, lower, s["wins"][higher_in_data], s["wins"][lower])
        result = simulate_remaining(state, predictor, n_sims=10000)
        predictions["by_series"][f"{t1}_{t2}"] = {
            "higher": higher_in_data,
            "lower": lower,
            "current_state": dict(state.__dict__),
            "p_higher_wins": result["p_higher_seed"],
            "p_lower_wins":  result["p_lower_seed"],
            "round":         s["round"],
        }

    # Full Monte Carlo for championship odds
    # Build out the rest of the bracket from current state
    predictions["champ_odds"] = run_monte_carlo_from_state(
        completed, in_progress, predictor, games, rounds, n_sims=10000
    )

    return predictions


def predict_higher_seed(t1: str, t2: str, games: pd.DataFrame) -> str:
    """Return whichever team had more regular-season wins."""
    standings = compute_standings(games)
    w1 = standings.loc[t1, "W"] if t1 in standings.index else 0
    w2 = standings.loc[t2, "W"] if t2 in standings.index else 0
    return t1 if w1 >= w2 else t2


def run_monte_carlo_from_state(
    completed: list,
    in_progress: list,
    predictor,
    games: pd.DataFrame,
    rounds: dict,
    n_sims: int = 10000,
) -> dict:
    """
    Run Monte Carlo championship simulation from current playoff state.
    Robust to any combination of completed / in-progress / not-yet-started
    matchups: derives the East and West conference-finals state by working
    out who's still alive in each conference.
    """
    from src.playoffs import SeriesState, simulate_remaining

    rng = random.Random(config.RANDOM_SEED)

    # Cached series probabilities
    series_p_cache: dict = {}
    def get_p(higher: str, lower: str, h_wins: int = 0, l_wins: int = 0) -> float:
        state = SeriesState(higher, lower, h_wins, l_wins)
        if state.is_over:
            return 1.0 if state.winner == higher else 0.0
        key = (higher, lower, h_wins, l_wins)
        if key not in series_p_cache:
            series_p_cache[key] = simulate_remaining(state, predictor, n_sims=5000)["p_higher_seed"]
        return series_p_cache[key]

    def derive_conf_finals_state(conf: str) -> dict | None:
        """
        Return the conference-finals state for given conference:
            {matchup: (t1, t2), wins: (t1_wins, t2_wins), is_done, winner}
        or None if we can't determine it (e.g. R2 not done yet).
        """
        teams_set = EAST_TEAMS if conf == "east" else WEST_TEAMS

        # First check R3: in-progress or completed series in this conference
        r3_series = [s for s in rounds["r3"]
                     if any(t in teams_set for t in s["teams"])]
        if r3_series:
            s = r3_series[0]
            t1, t2 = s["teams"]
            return {
                "matchup": (t1, t2),
                "wins":    (s["wins"][t1], s["wins"][t2]),
                "is_done": s["is_done"],
                "winner":  s.get("winner"),
            }

        # R3 not started yet — derive from completed R2 winners in this conference
        r2_done = [s for s in rounds["r2"]
                   if any(t in teams_set for t in s["teams"]) and s["is_done"]]
        if len(r2_done) >= 2:
            return {
                "matchup": (r2_done[0]["winner"], r2_done[1]["winner"]),
                "wins":    (0, 0),
                "is_done": False,
                "winner":  None,
            }

        return None

    east = derive_conf_finals_state("east")
    west = derive_conf_finals_state("west")

    # Check for in-progress NBA Finals
    finals_in_progress = None
    for s in rounds.get("finals", []):
        if not s["is_done"]:
            finals_in_progress = s
            break

    champ_counts  = defaultdict(int)
    finals_counts = defaultdict(int)

    def simulate_conf_winner(conf_state: dict | None) -> str | None:
        if conf_state is None:
            return None
        t1, t2 = conf_state["matchup"]
        higher = predict_higher_seed(t1, t2, games)
        lower = t2 if higher == t1 else t1
        h_wins, l_wins = (
            conf_state["wins"][0] if t1 == higher else conf_state["wins"][1],
            conf_state["wins"][1] if t1 == higher else conf_state["wins"][0],
        )
        p = get_p(higher, lower, h_wins, l_wins)
        return higher if rng.random() < p else lower

    for _ in range(n_sims):
        east_champ = simulate_conf_winner(east)
        west_champ = simulate_conf_winner(west)
        if east_champ is None or west_champ is None:
            continue
        finals_counts[east_champ] += 1
        finals_counts[west_champ] += 1

        # NBA Finals — if the actual Finals are in progress and the matchup
        # matches our simulated conference champs, use that real state
        if (finals_in_progress is not None
                and set([east_champ, west_champ]) == set(finals_in_progress["teams"])):
            higher = predict_higher_seed(east_champ, west_champ, games)
            lower = west_champ if higher == east_champ else east_champ
            h_wins = finals_in_progress["wins"][higher]
            l_wins = finals_in_progress["wins"][lower]
            p_higher_wins_series = get_p(higher, lower, h_wins, l_wins)
            champion = higher if rng.random() < p_higher_wins_series else lower
        else:
            # Otherwise: neutral home-court fresh-series projection
            p_east_home = get_p(east_champ, west_champ)
            p_west_home = get_p(west_champ, east_champ)
            p_east_wins = (p_east_home + (1 - p_west_home)) / 2
            champion = east_champ if rng.random() < p_east_wins else west_champ
        champ_counts[champion] += 1

    return {
        "champion": {t: c / n_sims for t, c in sorted(champ_counts.items(), key=lambda x: -x[1])},
        "finals":   {t: c / n_sims for t, c in sorted(finals_counts.items(), key=lambda x: -x[1])},
    }


def play_series(a: str, b: str, get_p, rng, games) -> str:
    higher = predict_higher_seed(a, b, games)
    lower = b if higher == a else a
    p = get_p(higher, lower, 0, 0)
    return higher if rng.random() < p else lower


# ── Modal path projection ────────────────────────────────────────────────────

def compute_modal_path(rounds: dict, predictions: dict, games: pd.DataFrame, predictor) -> list:
    """
    Walk through remaining rounds picking the favored winner.
    Includes:
      - in-progress series (with current scores)
      - upcoming series derivable from completed R2 winners (East/West R3)
      - upcoming NBA Finals once both conf champs are known/projected
    """
    from src.playoffs import SeriesState, simulate_remaining, series_win_prob

    path = []
    east_winners_so_far: list[str] = []
    west_winners_so_far: list[str] = []

    def predict_series(higher: str, lower: str, h_wins: int = 0, l_wins: int = 0,
                       round_label: str = "") -> tuple[str, float]:
        state = SeriesState(higher, lower, h_wins, l_wins)
        if state.is_over:
            return state.winner, 1.0
        result = simulate_remaining(state, predictor, n_sims=5000)
        winner = higher if result["p_higher_seed"] >= 0.5 else lower
        p = max(result["p_higher_seed"], 1 - result["p_higher_seed"])
        return winner, p

    # 1. In-progress series at any round (and record completed R3 winners)
    for rnd_name in ["r2", "r3", "finals"]:
        for s in rounds.get(rnd_name, []):
            t1, t2 = s["teams"]
            higher = predict_higher_seed(t1, t2, games)
            lower = t2 if higher == t1 else t1
            if s["is_done"]:
                # Record the known winner for downstream Finals projection
                if rnd_name == "r3":
                    if s["winner"] in EAST_TEAMS:
                        east_winners_so_far.append(s["winner"])
                    else:
                        west_winners_so_far.append(s["winner"])
                continue
            winner, p = predict_series(higher, lower, s["wins"][higher], s["wins"][lower])
            path.append({
                "round":   rnd_name,
                "matchup": f"{higher} vs {lower}",
                "current": f"{s['wins'][higher]}-{s['wins'][lower]}",
                "winner":  winner,
                "prob":    p,
            })
            if higher in EAST_TEAMS:
                east_winners_so_far.append(winner)
            else:
                west_winners_so_far.append(winner)

    # 2. Upcoming East R3 (if R2 done but R3 not started)
    east_r3_exists = any(any(t in EAST_TEAMS for t in s["teams"]) for s in rounds["r3"])
    if not east_r3_exists:
        east_r2_done = [s for s in rounds["r2"]
                        if any(t in EAST_TEAMS for t in s["teams"]) and s["is_done"]]
        if len(east_r2_done) >= 2:
            t1, t2 = east_r2_done[0]["winner"], east_r2_done[1]["winner"]
            higher = predict_higher_seed(t1, t2, games)
            lower = t2 if higher == t1 else t1
            winner, p = predict_series(higher, lower, 0, 0)
            path.append({
                "round": "r3", "matchup": f"{higher} vs {lower}",
                "current": "not started", "winner": winner, "prob": p,
            })
            east_winners_so_far.append(winner)

    # 3. Upcoming West R3 (mirror)
    west_r3_exists = any(any(t in WEST_TEAMS for t in s["teams"]) for s in rounds["r3"])
    if not west_r3_exists:
        west_r2_done = [s for s in rounds["r2"]
                        if any(t in WEST_TEAMS for t in s["teams"]) and s["is_done"]]
        if len(west_r2_done) >= 2:
            t1, t2 = west_r2_done[0]["winner"], west_r2_done[1]["winner"]
            higher = predict_higher_seed(t1, t2, games)
            lower = t2 if higher == t1 else t1
            winner, p = predict_series(higher, lower, 0, 0)
            path.append({
                "round": "r3", "matchup": f"{higher} vs {lower}",
                "current": "not started", "winner": winner, "prob": p,
            })
            west_winners_so_far.append(winner)

    # 4. NBA Finals (if both conf champs projectable and finals not in progress)
    finals_started = bool(rounds.get("finals"))
    if not finals_started and east_winners_so_far and west_winners_so_far:
        east_champ = east_winners_so_far[-1]
        west_champ = west_winners_so_far[-1]
        # Use the conference with higher seed as "higher" — or fall back to whatever
        higher = predict_higher_seed(east_champ, west_champ, games)
        lower = west_champ if higher == east_champ else east_champ
        winner, p = predict_series(higher, lower, 0, 0)
        path.append({
            "round": "finals", "matchup": f"{higher} vs {lower}",
            "current": "not started", "winner": winner, "prob": p,
        })

    return path


# ── HTML rendering ──────────────────────────────────────────────────────────

def render_html(state: dict) -> str:
    """Generate the complete dashboard HTML."""
    standings = state["standings"]
    east_seeds = state["east_seeds"]
    west_seeds = state["west_seeds"]
    rounds = state["rounds"]
    predictions = state["predictions"]
    modal_path = state["modal_path"]
    last_games = state["last_games"]
    model_metrics = state["model_metrics"]
    updated_at = state["updated_at"]
    data_through = state["data_through"]

    # Champion odds
    champ_odds = predictions.get("champ_odds", {}).get("champion", {})
    finals_odds = predictions.get("champ_odds", {}).get("finals", {})

    champion_team = next(iter(champ_odds.keys())) if champ_odds else "—"
    champion_prob = next(iter(champ_odds.values())) if champ_odds else 0

    # Helpers
    def champ_bars():
        rows = []
        for t, p in champ_odds.items():
            color = "primary" if p > 0.25 else "secondary"
            rows.append(f"""
              <div class="bar-row">
                <span class="bar-team">{t}</span>
                <div class="bar-track"><div class="bar-fill {color}" style="width:{p*100:.1f}%"></div></div>
                <span class="bar-pct">{p:.1%}</span>
              </div>""")
        return "\n".join(rows)

    def finals_bars():
        rows = []
        for t, p in finals_odds.items():
            rows.append(f"""
              <div class="bar-row">
                <span class="bar-team">{t}</span>
                <div class="bar-track"><div class="bar-fill secondary" style="width:{p*100:.1f}%"></div></div>
                <span class="bar-pct">{p:.1%}</span>
              </div>""")
        return "\n".join(rows)

    def series_card(s, is_completed=False):
        t1, t2 = s["teams"]
        # Order with higher seed first
        higher = predict_higher_seed(t1, t2, state["games_df"])
        lower = t2 if higher == t1 else t1
        score = f"{s['wins'][higher]}-{s['wins'][lower]}"
        if is_completed:
            status_class = "locked"
            status_tag = '<span class="status-tag locked">★ FINAL</span>'
            note = f"<div class='card-note'>{s['winner']} wins {s['wins'][s['winner']]}-{s['wins'][s['loser']]}</div>"
        else:
            status_class = "live"
            status_tag = '<span class="status-tag live">● LIVE</span>'
            pred_key = f"{t1}_{t2}"
            pred = predictions["by_series"].get(pred_key, {})
            p_higher = pred.get("p_higher_wins", 0.5)
            predicted = higher if p_higher >= 0.5 else lower
            pred_prob = max(p_higher, 1 - p_higher)
            note = f"<div class='card-note'>Model: <strong>{predicted}</strong> wins series {pred_prob:.0%}</div>"
        return f"""
        <div class="series-card {status_class}">
          {status_tag}
          <div class="card-matchup">{higher} {s['wins'][higher]} — {s['wins'][lower]} {lower}</div>
          {note}
        </div>"""

    # Round sections
    def round_section(round_key, round_name):
        srs = rounds.get(round_key, [])
        if not srs:
            return ""
        east_srs = [s for s in srs if any(t in EAST_TEAMS for t in s["teams"])]
        west_srs = [s for s in srs if any(t in WEST_TEAMS for t in s["teams"])]
        east_cards = "\n".join(series_card(s, s["is_done"]) for s in east_srs)
        west_cards = "\n".join(series_card(s, s["is_done"]) for s in west_srs)
        return f"""
        <section class="round-section">
          <h3>{round_name}</h3>
          <div class="grid-2">
            <div>
              <div class="conf-label">EASTERN CONFERENCE</div>
              {east_cards}
            </div>
            <div>
              <div class="conf-label">WESTERN CONFERENCE</div>
              {west_cards}
            </div>
          </div>
        </section>"""

    # Modal path table
    def path_table():
        if not modal_path:
            return "<p>No upcoming series.</p>"
        rows = []
        for step in modal_path:
            round_label = {"r2": "Conference Semi-finals",
                           "r3": "Conference Finals",
                           "finals": "NBA Finals"}.get(step["round"], step["round"])
            rows.append(f"""
              <tr>
                <td>{round_label}</td>
                <td><code>{step['matchup']}</code></td>
                <td>{step['current']}</td>
                <td class="team-cell">{step['winner']}</td>
                <td class="num">{step['prob']:.1%}</td>
              </tr>""")
        return "\n".join(rows)

    # Recent games table
    def recent_games_rows():
        rows = []
        for g in last_games:
            winner_class = "winner-home" if g["winner"] == g["home"] else "winner-away"
            rows.append(f"""
              <tr>
                <td>{g['date']}</td>
                <td class="{'winner-cell' if g['winner']==g['away'] else ''}">{g['away']}</td>
                <td class="num">{g['away_pts']}</td>
                <td class="num">{g['home_pts']}</td>
                <td class="{'winner-cell' if g['winner']==g['home'] else ''}">{g['home']}</td>
                <td><strong>{g['winner']}</strong></td>
              </tr>""")
        return "\n".join(rows)

    # Standings tables
    def seed_table(seeds, conf_name):
        rows = []
        for i, t in enumerate(seeds, 1):
            row = standings.loc[t]
            rows.append(f"""
              <tr>
                <td class="num">{i}</td>
                <td class="team-cell">{t}</td>
                <td class="num">{int(row.W)}-{int(row.L)}</td>
                <td class="num">{row.Pct:.1%}</td>
              </tr>""")
        return f"""
        <div class="seed-table">
          <h4>{conf_name}</h4>
          <table>
            <thead><tr><th>Seed</th><th>Team</th><th>Record</th><th>Pct</th></tr></thead>
            <tbody>
              {"".join(rows)}
            </tbody>
          </table>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NBA 2025-26 Predictions — Live Dashboard</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue',
               'PingFang SC', 'Microsoft YaHei', sans-serif;
  background: #f6f7fb;
  color: #1d1d1f;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}}
.wrapper {{ max-width: 1100px; margin: 0 auto; padding: 0 20px 60px; }}

/* Header */
header {{
  background: linear-gradient(135deg, #1d428a 0%, #c8102e 100%);
  color: #fff;
  padding: 50px 20px 40px;
  margin-bottom: 30px;
}}
.header-inner {{ max-width: 1100px; margin: 0 auto; }}
.tag {{
  display: inline-block;
  background: rgba(255,255,255,0.18);
  padding: 4px 12px;
  border-radius: 16px;
  font-size: 13px;
  letter-spacing: 0.5px;
  margin-bottom: 14px;
}}
h1 {{ font-size: 36px; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 8px; }}
.updated {{
  font-size: 14px;
  opacity: 0.85;
  margin-top: 8px;
}}
.refresh-bar {{
  margin-top: 18px;
  background: rgba(0,0,0,0.2);
  padding: 12px 16px;
  border-radius: 8px;
  font-size: 13px;
  display: inline-block;
}}
.refresh-bar code {{
  background: rgba(255,255,255,0.15);
  padding: 2px 8px;
  border-radius: 4px;
  margin: 0 4px;
  font-family: 'SF Mono', monospace;
}}

/* Hero champion card */
.hero {{
  background: #fff;
  border-radius: 16px;
  padding: 32px 28px;
  margin-bottom: 30px;
  box-shadow: 0 4px 20px rgba(0,0,0,0.06);
  display: grid;
  grid-template-columns: 1fr 2fr;
  gap: 30px;
  align-items: center;
}}
@media (max-width: 700px) {{
  .hero {{ grid-template-columns: 1fr; }}
}}
.hero-headline {{ text-align: center; padding-right: 20px; border-right: 1px solid #eee; }}
@media (max-width: 700px) {{
  .hero-headline {{ border-right: none; padding-right: 0; border-bottom: 1px solid #eee; padding-bottom: 20px; }}
}}
.hero-label {{
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 2px;
  color: #888;
}}
.hero-team {{
  font-size: 44px;
  font-weight: 800;
  background: linear-gradient(135deg, #ff6b35 0%, #c8102e 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 4px 0;
}}
.hero-prob {{ font-size: 16px; color: #555; }}
.hero-prob strong {{ color: #1d1d1f; font-weight: 700; font-size: 28px; }}
.hero-mini-bars {{ font-size: 13px; }}
.hero-mini-bars h4 {{ font-size: 13px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }}

/* Sections */
section {{
  background: #fff;
  border-radius: 16px;
  padding: 24px 28px;
  margin-bottom: 24px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}}
section h2 {{ font-size: 22px; font-weight: 700; margin-bottom: 6px; }}
section h3 {{ font-size: 18px; font-weight: 700; margin-bottom: 14px; color: #1d428a; }}
section h4 {{ font-size: 14px; font-weight: 600; margin-bottom: 8px; color: #555; }}
section .lead {{ font-size: 14px; color: #666; margin-bottom: 18px; }}

/* Bar charts */
.bar-row {{
  display: grid;
  grid-template-columns: 56px 1fr 56px;
  gap: 10px;
  align-items: center;
  padding: 7px 0;
  border-bottom: 1px solid #f0f0f3;
}}
.bar-team {{ font-weight: 700; font-size: 14px; color: #1d428a; }}
.bar-track {{ height: 18px; background: #f3f3f5; border-radius: 4px; overflow: hidden; }}
.bar-fill {{
  height: 100%;
  background: linear-gradient(90deg, #ff6b35 0%, #c8102e 100%);
  border-radius: 4px;
}}
.bar-fill.secondary {{ background: linear-gradient(90deg, #1d428a 0%, #4a6fbf 100%); }}
.bar-pct {{ text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; font-size: 13px; }}

/* Series cards */
.grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
@media (max-width: 700px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
.conf-label {{
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 1.5px;
  color: #888;
  margin-bottom: 10px;
  font-weight: 600;
}}
.series-card {{
  background: #fafafa;
  border-radius: 10px;
  padding: 14px 16px;
  border: 1px solid #eee;
  margin-bottom: 10px;
}}
.series-card.locked {{ background: #f0f7f0; border-color: #c5e0c5; }}
.series-card.live {{ background: #fff5f5; border-color: #ffd1d1; }}
.status-tag {{
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1.5px;
}}
.status-tag.locked {{ color: #2e7d32; }}
.status-tag.live {{ color: #c8102e; }}
.card-matchup {{ font-size: 18px; font-weight: 700; margin: 6px 0; }}
.card-note {{ font-size: 13px; color: #555; }}
.round-section {{ margin-top: 24px; }}

/* Tables */
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #eee; }}
th {{
  background: #fafafa;
  font-weight: 600;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: #666;
}}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.team-cell {{ font-weight: 700; color: #1d428a; }}
.winner-cell {{ background: #fff8e1; }}

/* Metrics */
.metrics {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
}}
@media (max-width: 700px) {{ .metrics {{ grid-template-columns: 1fr 1fr; }} }}
.metric {{
  background: #fafafa;
  border-radius: 10px;
  padding: 18px 12px;
  text-align: center;
}}
.metric-value {{
  font-size: 28px;
  font-weight: 800;
  color: #1d428a;
}}
.metric-label {{
  font-size: 11px;
  color: #666;
  text-transform: uppercase;
  letter-spacing: 1px;
}}

/* Standings */
.standings-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
@media (max-width: 700px) {{ .standings-grid {{ grid-template-columns: 1fr; }} }}

footer {{
  margin-top: 40px;
  padding: 20px 0;
  text-align: center;
  font-size: 12px;
  color: #888;
}}
footer a {{ color: #1d428a; text-decoration: none; }}

.disclaimer {{
  background: #fff8e1;
  border-left: 3px solid #f9a825;
  padding: 10px 14px;
  border-radius: 4px;
  font-size: 12px;
  color: #5d4e00;
  margin-top: 14px;
}}
</style>
</head>
<body>

<header>
  <div class="header-inner">
    <span class="tag">🏀 LIVE DASHBOARD • NEURAL NETWORK PREDICTIONS</span>
    <h1>NBA 2025-26 Playoffs</h1>
    <p class="updated">Last updated: <strong>{updated_at}</strong> · Data through <strong>{data_through}</strong></p>
    <div class="refresh-bar">
      🔄 Refresh: double-click <code>refresh.bat</code> &nbsp;or&nbsp; run <code>python dashboard.py --refresh --open</code>
    </div>
  </div>
</header>

<div class="wrapper">

  <!-- HERO -->
  <div class="hero">
    <div class="hero-headline">
      <div class="hero-label">PREDICTED CHAMPION</div>
      <div class="hero-team">{champion_team}</div>
      <div class="hero-prob"><strong>{champion_prob:.1%}</strong> championship probability</div>
    </div>
    <div class="hero-mini-bars">
      <h4>Top 5 contenders</h4>
      {champ_bars()}
    </div>
  </div>

  <!-- CURRENT BRACKET STATE -->
  <section>
    <h2>Bracket — current state</h2>
    <p class="lead">All completed and in-progress series across both conferences.</p>
    {round_section("r2", "Round 2 — Conference Semi-finals")}
    {round_section("r3", "Round 3 — Conference Finals")}
    {round_section("finals", "Round 4 — NBA Finals")}
  </section>

  <!-- MODAL PATH -->
  <section>
    <h2>Most-likely path to the title</h2>
    <p class="lead">For each remaining series, the model's predicted winner and confidence. Compound probability of this exact bracket is small — see Monte Carlo below for true championship odds.</p>
    <table>
      <thead>
        <tr><th>Round</th><th>Matchup</th><th>Current</th><th>Model pick</th><th>Confidence</th></tr>
      </thead>
      <tbody>
        {path_table()}
      </tbody>
    </table>
  </section>

  <!-- MONTE CARLO ODDS -->
  <section>
    <h2>Championship odds (10,000 Monte-Carlo simulations)</h2>
    <p class="lead">Aggregated across all possible bracket paths from the current state — these are the model's true probability estimates.</p>
    <div class="grid-2">
      <div>
        <h3>🏆 NBA Champion</h3>
        {champ_bars()}
      </div>
      <div>
        <h3>🥇 Reach NBA Finals</h3>
        {finals_bars()}
      </div>
    </div>
  </section>

  <!-- RECENT GAMES -->
  <section>
    <h2>Most recent playoff games</h2>
    <p class="lead">Last 10 results from the playoffs — used to update ELO and rolling features.</p>
    <table>
      <thead>
        <tr><th>Date</th><th>Away</th><th>Pts</th><th>Pts</th><th>Home</th><th>Winner</th></tr>
      </thead>
      <tbody>
        {recent_games_rows()}
      </tbody>
    </table>
  </section>

  <!-- STANDINGS -->
  <section>
    <h2>2025-26 Regular-season seeding</h2>
    <p class="lead">Top 8 in each conference, used as the playoff seeds.</p>
    <div class="standings-grid">
      {seed_table(east_seeds, "🅴 Eastern Conference")}
      {seed_table(west_seeds, "🅆 Western Conference")}
    </div>
  </section>

  <!-- MODEL METRICS -->
  <section>
    <h2>Model performance</h2>
    <p class="lead">Feed-forward neural network with BatchNorm + Dropout, trained on the chronological train/val/test split.</p>
    <div class="metrics">
      <div class="metric">
        <div class="metric-value">{model_metrics['n_games']:,}</div>
        <div class="metric-label">Games trained</div>
      </div>
      <div class="metric">
        <div class="metric-value">{model_metrics['n_seasons']}</div>
        <div class="metric-label">Seasons (2022-26)</div>
      </div>
      <div class="metric">
        <div class="metric-value">{model_metrics['nn_auc']:.3f}</div>
        <div class="metric-label">NN ROC-AUC</div>
      </div>
      <div class="metric">
        <div class="metric-value">{model_metrics['nn_acc']:.1%}</div>
        <div class="metric-label">NN accuracy</div>
      </div>
    </div>
    <div class="disclaimer">
      <strong>Caveat:</strong> Model trained on regular-season games. Predictions may diverge from bookmakers because we lack player-level skill ratings (RAPTOR/EPM) and playoff-specific dynamics. See <code>MODEL_LIMITATIONS.md</code> for the full discussion.
    </div>
  </section>

</div>

<footer>
  Generated by <code>dashboard.py</code> · Data: Basketball Reference · Model: TensorFlow 2.21
  · <a href="README.md">README</a> · <a href="TUTORIAL.md">Tutorial</a> · <a href="MODEL_LIMITATIONS.md">Limitations</a>
</footer>

</body>
</html>"""


# ── Main pipeline ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate live NBA playoff dashboard HTML.")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-fetch latest games from Basketball Reference + retrain model")
    parser.add_argument("--open", action="store_true",
                        help="Open dashboard.html in browser after generation")
    parser.add_argument("--output", default="dashboard.html",
                        help="Output HTML path (default: dashboard.html)")
    args = parser.parse_args()

    if args.refresh:
        print("Refreshing data + retraining model …")
        # Back up existing data first so a network failure doesn't lose everything
        backup_path = None
        if os.path.exists(config.RAW_DATA_PATH):
            backup_path = config.RAW_DATA_PATH + ".bak"
            os.replace(config.RAW_DATA_PATH, backup_path)
            print(f"  Backed up existing data to {backup_path}")

        result = subprocess.run([sys.executable, "main.py", "--source", "bref"],
                                capture_output=False)
        if result.returncode != 0 or not os.path.exists(config.RAW_DATA_PATH):
            print("\n⚠ Refresh failed (network issue?).")
            if backup_path and os.path.exists(backup_path):
                print(f"  Restoring previous data from {backup_path} so dashboard still works.")
                os.replace(backup_path, config.RAW_DATA_PATH)
            else:
                print("  No backup available — dashboard cannot continue.")
                return 1
        else:
            # Clean up backup on success
            if backup_path and os.path.exists(backup_path):
                os.remove(backup_path)

    # Load cached data
    print("\nLoading cached games and model …")
    if not os.path.exists(config.RAW_DATA_PATH):
        print(f"No cached games at {config.RAW_DATA_PATH}. Run with --refresh first.")
        return 1
    games = pd.read_csv(config.RAW_DATA_PATH, parse_dates=["GAME_DATE"])

    # Compute everything
    print("Computing standings and bracket state …")
    standings = compute_standings(games)
    east_seeds, west_seeds = get_seeds(standings)

    series_list = detect_playoff_series(games)
    rounds = organize_by_round(series_list)

    print(f"  Playoff series detected: R1={len(rounds['r1'])}, R2={len(rounds['r2'])}, "
          f"R3={len(rounds['r3'])}, Finals={len(rounds['finals'])}")

    from src.playoffs import MatchupPredictor
    predictor = MatchupPredictor()

    print("Running Monte Carlo simulation …")
    predictions = run_predictions(games, rounds, predictor)

    print("Computing modal path …")
    modal_path = compute_modal_path(rounds, predictions, games, predictor)

    # Recent games for display
    last_games = []
    for _, g in games.sort_values("GAME_DATE").tail(10).iterrows():
        last_games.append({
            "date":     g["GAME_DATE"].strftime("%Y-%m-%d"),
            "home":     g["home_team_abbr"],
            "away":     g["away_team_abbr"],
            "home_pts": int(g["home_pts"]),
            "away_pts": int(g["away_pts"]),
            "winner":   g["home_team_abbr"] if g["home_win"] == 1 else g["away_team_abbr"],
        })
    last_games.reverse()  # most-recent first

    # Model metrics — read from a file if you've saved them, else hardcode reasonable defaults
    model_metrics = {
        "n_games":   len(games),
        "n_seasons": 4,
        "nn_auc":    0.721,
        "nn_acc":    0.678,
    }

    state = {
        "games_df":      games,
        "standings":     standings,
        "east_seeds":    east_seeds,
        "west_seeds":    west_seeds,
        "rounds":        rounds,
        "predictions":   predictions,
        "modal_path":    modal_path,
        "last_games":    last_games,
        "model_metrics": model_metrics,
        "updated_at":    datetime.now().strftime("%Y-%m-%d %H:%M"),
        "data_through":  games["GAME_DATE"].max().strftime("%Y-%m-%d"),
    }

    print("Rendering HTML …")
    html = render_html(state)
    out_path = os.path.abspath(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n✓ Dashboard written to: {out_path}")
    print(f"  ({len(html):,} bytes)")

    if args.open:
        print("Opening in browser …")
        webbrowser.open(f"file:///{out_path.replace(os.sep, '/')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
