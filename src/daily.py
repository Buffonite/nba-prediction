"""
Batch prediction: fetch all NBA games scheduled on a date and predict each.

Used by:
  - daily.py (command-line tool)
  - app.py (Streamlit UI, "Daily Slate" tab)

The schedule comes from `nba_api.scoreboardv2`, which returns games for any
date — past or future. So this works both for live predictions and historical
backtesting.
"""

import time
import pandas as pd
from nba_api.stats.endpoints import scoreboardv2

from src.predict import predict_game


def get_games_on_date(game_date: str) -> pd.DataFrame:
    """
    Return all NBA games scheduled on a given date.

    Args:
        game_date: 'YYYY-MM-DD' string

    Returns:
        DataFrame with columns: GAME_ID, home_abbr, away_abbr, status
        (empty if no games that day, e.g. off-season or All-Star break)
    """
    sb = scoreboardv2.ScoreboardV2(game_date=game_date)
    frames = sb.get_data_frames()
    header     = frames[0]   # GameHeader: GAME_ID, HOME_TEAM_ID, VISITOR_TEAM_ID, status
    line_score = frames[1]   # LineScore: TEAM_ID ↔ TEAM_ABBREVIATION lookup

    if len(header) == 0:
        return pd.DataFrame(columns=["GAME_ID", "home_abbr", "away_abbr", "status"])

    abbr_lookup = dict(zip(line_score["TEAM_ID"], line_score["TEAM_ABBREVIATION"]))

    rows = []
    for _, g in header.iterrows():
        rows.append({
            "GAME_ID":   g["GAME_ID"],
            "home_abbr": abbr_lookup.get(g["HOME_TEAM_ID"], "?"),
            "away_abbr": abbr_lookup.get(g["VISITOR_TEAM_ID"], "?"),
            "status":    g["GAME_STATUS_TEXT"],
        })
    return pd.DataFrame(rows)


def predict_daily_slate(game_date: str, throttle: float = 0.0) -> pd.DataFrame:
    """
    Predict every game scheduled on `game_date`.

    Returns a DataFrame with one row per game and columns ready for display:
        Matchup, Predicted Winner, Win Prob, Confidence,
        Home ELO, Away ELO, Recent Form (H/A)
    """
    matchups = get_games_on_date(game_date)
    if len(matchups) == 0:
        return pd.DataFrame()

    rows = []
    for i, m in matchups.iterrows():
        try:
            r = predict_game(m["home_abbr"], m["away_abbr"], game_date=game_date)

            # Use the WINNING side's probability as "Win Prob"
            winner_prob = max(r["home_win_probability"], r["away_win_probability"])

            rows.append({
                "Matchup":          f"{r['away_team']} @ {r['home_team']}",
                "Predicted Winner": r["predicted_winner"],
                "Win Prob":         winner_prob,
                "Confidence":       r["confidence"],
                "Home ELO":         round(r["home_elo"]),
                "Away ELO":         round(r["away_elo"]),
                "Home L10":         f"{r['home_recent_winpct']:.0%}",
                "Away L10":         f"{r['away_recent_winpct']:.0%}",
                "Status":           m["status"],
            })
        except Exception as e:
            rows.append({
                "Matchup":          f"{m['away_abbr']} @ {m['home_abbr']}",
                "Predicted Winner": "—",
                "Win Prob":         None,
                "Confidence":       "error",
                "Home ELO":         None,
                "Away ELO":         None,
                "Home L10":         "—",
                "Away L10":         "—",
                "Status":           f"Error: {str(e)[:40]}",
            })

        # Optional polite delay so we don't hammer stats.nba.com
        if throttle > 0 and i < len(matchups) - 1:
            time.sleep(throttle)

    return pd.DataFrame(rows)
