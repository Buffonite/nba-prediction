"""
Feature engineering for the NBA game-prediction model.

For each game, we look BACKWARD at each team's recent history and compute
rolling averages — ensuring no data leakage (we never use future games).

Features generated (for both 'home' and 'away' sides):
  - win_pct_last{N}       : fraction of last N games won
  - pts_scored_last{N}    : average points scored
  - pts_allowed_last{N}   : average points allowed
  - net_rating_last{N}    : avg (pts_scored - pts_allowed)  ← offensive/defensive balance
  - rest_days             : days since the team's last game (fatigue proxy)
  - is_b2b                : 1 if playing on back-to-back nights

Windows (N) are set in config.ROLLING_WINDOWS (default: 5 and 10).

Usage:
    from src.preprocessing import build_features
    features_df = build_features(games_df)
"""

import os
import numpy as np
import pandas as pd

import config


# ── Per-team rolling stats ────────────────────────────────────────────────────

def _team_game_log(games: pd.DataFrame) -> pd.DataFrame:
    """
    Explode the per-game table back into a per-team view so we can compute
    each team's rolling stats efficiently.

    Returns a DataFrame with columns:
        GAME_ID, GAME_DATE, team_id, pts_scored, pts_allowed, win
    """
    # Include opponent ELO (when available) so rolling stats can be
    # weighted by strength of schedule.
    has_elo = "home_elo_pre" in games.columns and "away_elo_pre" in games.columns

    home_cols = ["GAME_ID", "GAME_DATE", "home_team_id", "home_pts", "away_pts", "home_win"]
    home_rename = ["GAME_ID", "GAME_DATE", "team_id", "pts_scored", "pts_allowed", "win"]
    if has_elo:
        home_cols += ["away_elo_pre"]
        home_rename += ["opp_elo"]
    home = games[home_cols].copy()
    home.columns = home_rename

    away_cols = ["GAME_ID", "GAME_DATE", "away_team_id", "away_pts", "home_pts", "home_win"]
    away_rename = ["GAME_ID", "GAME_DATE", "team_id", "pts_scored", "pts_allowed", "win"]
    if has_elo:
        away_cols += ["home_elo_pre"]
        away_rename += ["opp_elo"]
    away = games[away_cols].copy()
    away.columns = away_rename
    away["win"] = 1 - away["win"]   # away team wins when home_win == 0

    log = pd.concat([home, away], ignore_index=True)
    log = log.sort_values(["team_id", "GAME_DATE"]).reset_index(drop=True)
    return log


def _rolling_stats(log: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    For each row (team, game), compute rolling stats over the previous `window`
    games using .shift(1) so the current game is NOT included (no leakage).

    If 'opp_elo' is in the log, also computes strength-of-schedule features
    (average opponent ELO) and a quality-adjusted win rate.
    """
    grp = log.groupby("team_id")

    def roll(col):
        return grp[col].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())

    suffix = f"_last{window}"
    result = pd.DataFrame({"GAME_ID": log["GAME_ID"], "team_id": log["team_id"]})
    result[f"win_pct{suffix}"]      = roll("win")
    result[f"pts_scored{suffix}"]   = roll("pts_scored")
    result[f"pts_allowed{suffix}"]  = roll("pts_allowed")
    result[f"net_rating{suffix}"]   = result[f"pts_scored{suffix}"] - result[f"pts_allowed{suffix}"]

    # Strength-of-schedule: average opponent ELO over the window
    if config.USE_SOS_WEIGHTED_FEATURES and "opp_elo" in log.columns:
        result[f"sos{suffix}"] = roll("opp_elo")
        # Quality-adjusted win rate: a win against an elite team counts more.
        # Normalises around 1.0 so multipliers don't blow up.
        log_qa = log.copy()
        log_qa["qa_win"] = log_qa["win"] * (log_qa["opp_elo"] / 1500.0).clip(0.7, 1.3)
        grp_qa = log_qa.groupby("team_id")
        result[f"qa_win_pct{suffix}"] = (
            grp_qa["qa_win"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
    return result


def _rest_features(log: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rest_days and is_b2b (back-to-back) per team per game.
    """
    log = log.sort_values(["team_id", "GAME_DATE"])
    log["prev_date"] = log.groupby("team_id")["GAME_DATE"].shift(1)
    log["rest_days"] = (log["GAME_DATE"] - log["prev_date"]).dt.days.fillna(7).clip(upper=14)
    log["is_b2b"]    = (log["rest_days"] == 1).astype(int)
    return log[["GAME_ID", "team_id", "rest_days", "is_b2b"]]


# ── Main feature builder ──────────────────────────────────────────────────────

def build_features(
    games: pd.DataFrame,
    star_avail: pd.DataFrame | None = None,
    odds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Takes the game-level table (one row per game) from data_fetch.build_game_table()
    and returns a feature matrix ready for model training.

    Args:
        games:      per-game table (may include ELO columns from src.elo)
        star_avail: optional DataFrame from src.injuries.compute_star_availability
                    with columns [GAME_ID, home_stars_avail, away_stars_avail, stars_avail_diff]
        odds:       optional DataFrame from src.odds (synthetic or real)
                    with columns [GAME_ID, home_implied_prob, away_implied_prob, market_edge]

    Target column: 'home_win'  (1 = home team wins, 0 = away team wins)
    """
    log = _team_game_log(games)

    # Rolling stats for each configured window
    stat_frames = [_rolling_stats(log, w) for w in config.ROLLING_WINDOWS]

    # Merge all rolling stats on (GAME_ID, team_id)
    stats = stat_frames[0]
    for frame in stat_frames[1:]:
        stats = stats.merge(frame, on=["GAME_ID", "team_id"])

    rest = _rest_features(log)
    stats = stats.merge(rest, on=["GAME_ID", "team_id"])

    # Attach features back to the game table for home and away teams
    def side_features(side: str, team_col: str) -> pd.DataFrame:
        side_stats = stats.rename(
            columns={c: f"{side}_{c}" for c in stats.columns if c not in ("GAME_ID", "team_id")}
        )
        merged = games[["GAME_ID", "GAME_DATE", "home_win", team_col]].merge(
            side_stats, left_on=["GAME_ID", team_col], right_on=["GAME_ID", "team_id"], how="left"
        ).drop(columns=[team_col, "team_id"])
        return merged

    home_feats = side_features("home", "home_team_id")
    away_feats = side_features("away", "away_team_id").drop(columns=["GAME_DATE", "home_win"])

    features = home_feats.merge(away_feats, on="GAME_ID")

    # Difference features: home advantage expressed as deltas
    for window in config.ROLLING_WINDOWS:
        for stat in config.ROLLING_STATS:
            h_col = f"home_{stat}_last{window}"
            a_col = f"away_{stat}_last{window}"
            if h_col in features.columns and a_col in features.columns:
                features[f"diff_{stat}_last{window}"] = features[h_col] - features[a_col]

    # ── Attach ELO features (if present in the games DataFrame) ──────────────
    elo_cols = [c for c in ("home_elo_pre", "away_elo_pre", "elo_diff") if c in games.columns]
    if elo_cols:
        features = features.merge(games[["GAME_ID"] + elo_cols], on="GAME_ID", how="left")
        print(f"  + ELO columns merged: {elo_cols}")

    # ── Attach star-availability (injury proxy) features ─────────────────────
    if star_avail is not None:
        features = features.merge(star_avail, on="GAME_ID", how="left")
        print(f"  + Star-availability columns merged: "
              f"{[c for c in star_avail.columns if c != 'GAME_ID']}")

    # ── Attach betting-odds features (market consensus) ─────────────────────
    if odds is not None:
        features = features.merge(odds, on="GAME_ID", how="left")
        print(f"  + Odds columns merged: "
              f"{[c for c in odds.columns if c != 'GAME_ID']}")

    # Drop rows where rolling stats are NaN (very early games with no history)
    features = features.dropna().reset_index(drop=True)

    os.makedirs(os.path.dirname(config.PROCESSED_DATA_PATH), exist_ok=True)
    features.to_csv(config.PROCESSED_DATA_PATH, index=False)
    print(f"Feature matrix: {features.shape[0]:,} games × {features.shape[1]} columns → '{config.PROCESSED_DATA_PATH}'")
    return features


def get_feature_columns(features: pd.DataFrame) -> list[str]:
    """Return the list of model input columns (everything except metadata/target)."""
    exclude = {"GAME_ID", "GAME_DATE", "home_win"}
    return [c for c in features.columns if c not in exclude]
