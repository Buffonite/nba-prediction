"""
Injury / star-availability features.

Real injury reports are messy and not in a clean public API.
We use a tractable proxy: 'star availability' = how many of a team's top
players actually played in this game.

Method
──────
1. For each (season, team), identify the TOP_N_STARS players by total
   minutes played in that season  → "stars".
2. For each game, count how many of each team's stars actually appeared
   in the box score (MIN > 0)  → star availability.
3. Feature: home_stars_avail, away_stars_avail (integers 0..N).

Mild leakage caveat
───────────────────
Identifying "stars" using full-season minutes is technically using future
information (we're at game G, but we know who'll lead in minutes by season-end).
A stricter version would use the prior season's top-N or rolling minutes
leadership. For a portfolio project, the simpler version is fine — but worth
mentioning in interviews that you're aware of the trade-off.
"""

import pandas as pd

import config


def identify_stars(player_logs: pd.DataFrame, top_n: int = config.TOP_N_STARS) -> pd.DataFrame:
    """
    For each (SEASON, TEAM_ID), return the top-N players by total minutes.
    Returns DataFrame with columns: SEASON, TEAM_ID, PLAYER_ID, total_min.
    """
    # SEASON_ID looks like '22022' for 2022-23. We convert to a readable label.
    logs = player_logs.copy()
    logs["MIN"] = pd.to_numeric(logs["MIN"], errors="coerce").fillna(0)

    totals = (
        logs.groupby(["SEASON_ID", "TEAM_ID", "PLAYER_ID"], as_index=False)["MIN"]
            .sum()
            .rename(columns={"MIN": "total_min"})
    )

    # Sort and take top-N within each (season, team)
    totals = totals.sort_values(["SEASON_ID", "TEAM_ID", "total_min"],
                                ascending=[True, True, False])
    stars = totals.groupby(["SEASON_ID", "TEAM_ID"], as_index=False).head(top_n)

    print(f"Identified {len(stars):,} star-team-season combos "
          f"(top {top_n} per team per season)")
    return stars


def compute_star_availability(
    player_logs: pd.DataFrame,
    stars: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    """
    For every game, compute how many of each team's stars actually played.

    Returns DataFrame with columns:
        GAME_ID, home_stars_avail, away_stars_avail
    """
    logs = player_logs.copy()
    logs["MIN"] = pd.to_numeric(logs["MIN"], errors="coerce").fillna(0)
    logs["played"] = (logs["MIN"] > 0).astype(int)

    # Keep only star appearances (filter player logs by star list)
    star_keys = stars[["SEASON_ID", "TEAM_ID", "PLAYER_ID"]]
    star_apps = logs.merge(star_keys, on=["SEASON_ID", "TEAM_ID", "PLAYER_ID"], how="inner")

    # Count stars-played per (game, team)
    avail = (
        star_apps.groupby(["GAME_ID", "TEAM_ID"], as_index=False)["played"]
                 .sum()
                 .rename(columns={"played": "stars_avail"})
    )

    # Merge twice — once for home team, once for away team
    home_avail = avail.rename(columns={
        "TEAM_ID": "home_team_id", "stars_avail": "home_stars_avail"
    })
    away_avail = avail.rename(columns={
        "TEAM_ID": "away_team_id", "stars_avail": "away_stars_avail"
    })

    out = (
        games[["GAME_ID", "home_team_id", "away_team_id"]]
            .merge(home_avail, on=["GAME_ID", "home_team_id"], how="left")
            .merge(away_avail, on=["GAME_ID", "away_team_id"], how="left")
    )

    # Fill missing (rare — would mean we have no player log entries for that game)
    out["home_stars_avail"] = out["home_stars_avail"].fillna(config.TOP_N_STARS).astype(int)
    out["away_stars_avail"] = out["away_stars_avail"].fillna(config.TOP_N_STARS).astype(int)

    # Difference feature — relative star advantage
    out["stars_avail_diff"] = out["home_stars_avail"] - out["away_stars_avail"]

    print(f"Star availability: avg home = {out['home_stars_avail'].mean():.2f}, "
          f"avg away = {out['away_stars_avail'].mean():.2f} (out of {config.TOP_N_STARS})")

    return out[["GAME_ID", "home_stars_avail", "away_stars_avail", "stars_avail_diff"]]


# ── Synthetic version for --demo mode ────────────────────────────────────────

def synthetic_star_availability(games: pd.DataFrame) -> pd.DataFrame:
    """
    For demo mode (no real player logs), generate plausible random
    star-availability data so the rest of the pipeline still works.
    """
    import numpy as np
    rng = np.random.default_rng(config.RANDOM_SEED)

    n = len(games)
    # Most games: all 5 stars play. Occasionally 3-4. Rarely 0-2 (heavy injury).
    home = rng.choice([5, 5, 5, 4, 4, 3, 2], size=n)
    away = rng.choice([5, 5, 5, 4, 4, 3, 2], size=n)

    return pd.DataFrame({
        "GAME_ID":          games["GAME_ID"].values,
        "home_stars_avail": home,
        "away_stars_avail": away,
        "stars_avail_diff": home - away,
    })
