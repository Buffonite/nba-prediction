"""
Inference module: predict the outcome of a single upcoming NBA game.

This is the bridge between the trained model (saved by train.py) and real-world
use. Given a matchup like "LAL @ GSW on 2024-03-15", it:
  1. Loads the trained model + scaler + feature column list
  2. Pulls each team's recent history from cached data
  3. Computes the SAME features used during training (rolling stats + ELO + stars)
  4. Runs the model and returns a probability + readable summary

Usage:
    from src.predict import predict_game
    result = predict_game("LAL", "GSW")
    print(result["home_win_probability"])
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from tensorflow import keras

import config
from src.elo import compute_elo_features
from src.preprocessing import _team_game_log

FEATURE_COLS_PATH = "outputs/models/feature_cols.json"
CALIBRATION_PATH  = "outputs/models/calibration.json"
XGB_MODEL_PATH    = "outputs/models/xgb.json"


# ── Loading saved artifacts ──────────────────────────────────────────────────

def load_artifacts():
    """
    Load all trained artifacts: NN, scaler, feature columns, optional
    XGBoost ensemble model, and optional temperature scaling parameter.

    Returns a dict (v2 — used to return a tuple; tuple form kept for backward
    compatibility via load_artifacts_tuple()).
    """
    missing = [
        p for p in (config.MODEL_SAVE_PATH, config.SCALER_SAVE_PATH, FEATURE_COLS_PATH)
        if not os.path.exists(p)
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing trained artifacts: {missing}\n"
            f"Run `python main.py` first to train the model."
        )

    model = keras.models.load_model(config.MODEL_SAVE_PATH)
    with open(config.SCALER_SAVE_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(FEATURE_COLS_PATH) as f:
        feature_cols = json.load(f)

    # Optional v2 components
    temperature = 1.0
    if os.path.exists(CALIBRATION_PATH):
        with open(CALIBRATION_PATH) as f:
            temperature = json.load(f).get("temperature", 1.0)

    xgb_model = None
    if os.path.exists(XGB_MODEL_PATH):
        try:
            import xgboost as xgb
            xgb_model = xgb.XGBClassifier()
            xgb_model.load_model(XGB_MODEL_PATH)
        except Exception as e:
            print(f"  ⚠ Failed to load XGBoost: {e}")

    # Backward-compat: many callers expect (model, scaler, feature_cols)
    # We return a tuple for them but stash the v2 components as attributes.
    out = (model, scaler, feature_cols)
    # Attach extras via a dict for callers that want them
    load_artifacts.last_temperature = temperature
    load_artifacts.last_xgb         = xgb_model
    return out


def predict_calibrated(model, scaler, X_scaled, xgb_model=None, temperature: float = 1.0) -> float:
    """
    Run model prediction with v2 improvements applied:
      1. Get raw NN probability
      2. Apply temperature scaling to reduce over-confidence
      3. Ensemble with XGBoost if available
    Returns final calibrated probability in [0, 1].
    """
    nn_prob = float(model.predict(X_scaled, verbose=0)[0, 0])

    # Step 1: temperature scaling on NN output
    if temperature and temperature != 1.0:
        nn_prob = float(np.clip(nn_prob, 1e-6, 1 - 1e-6))
        nn_logit = np.log(nn_prob / (1 - nn_prob))
        nn_prob = 1.0 / (1.0 + np.exp(-nn_logit / temperature))

    # Step 2: ensemble with XGBoost
    if xgb_model is not None:
        xgb_prob = float(xgb_model.predict_proba(X_scaled)[0, 1])
        w = config.ENSEMBLE_NN_WEIGHT
        return w * nn_prob + (1 - w) * xgb_prob

    return nn_prob


def team_id_from_abbr(abbr: str) -> tuple[int, str]:
    """Convert a team abbreviation like 'LAL' to (team_id, full_name)."""
    from nba_api.stats.static import teams
    abbr = abbr.upper()
    for team in teams.get_teams():
        if team["abbreviation"] == abbr:
            return team["id"], team["full_name"]
    raise ValueError(f"Unknown team abbreviation '{abbr}'. "
                     f"Use standard NBA abbreviations like LAL, GSW, BOS.")


# ── Per-team feature helpers ─────────────────────────────────────────────────

def _rolling_stats_for_team(log: pd.DataFrame, team_id: int,
                             as_of_date: pd.Timestamp, window: int) -> dict:
    """
    Stats from a team's last N games BEFORE as_of_date.
    v2: also returns SOS (avg opponent ELO) and quality-adjusted win rate
    if 'opp_elo' is in the team game log.
    """
    team_games = log[(log["team_id"] == team_id) & (log["GAME_DATE"] < as_of_date)]
    recent = team_games.tail(window)
    if len(recent) == 0:
        return None
    pts_avg = recent["pts_scored"].mean()
    pts_allowed = recent["pts_allowed"].mean()
    stats = {
        f"win_pct_last{window}":     recent["win"].mean(),
        f"pts_scored_last{window}":  pts_avg,
        f"pts_allowed_last{window}": pts_allowed,
        f"net_rating_last{window}":  pts_avg - pts_allowed,
    }
    # v2: SOS-weighted features
    if config.USE_SOS_WEIGHTED_FEATURES and "opp_elo" in recent.columns:
        stats[f"sos_last{window}"]        = recent["opp_elo"].mean()
        quality_weights                   = (recent["opp_elo"] / 1500.0).clip(0.7, 1.3)
        stats[f"qa_win_pct_last{window}"] = (recent["win"] * quality_weights).mean()
    return stats


def _rest_for_team(log: pd.DataFrame, team_id: int,
                    as_of_date: pd.Timestamp) -> dict:
    """Days since the team's last game, capped at 14."""
    team_games = log[(log["team_id"] == team_id) & (log["GAME_DATE"] < as_of_date)]
    if len(team_games) == 0:
        return {"rest_days": 7, "is_b2b": 0}
    last_date = team_games["GAME_DATE"].max()
    rest = min((as_of_date - last_date).days, 14)
    return {"rest_days": rest, "is_b2b": int(rest == 1)}


# ── Main feature builder for a single matchup ────────────────────────────────

def build_prediction_features(
    home_team_id: int,
    away_team_id: int,
    game_date: pd.Timestamp,
    historical_games: pd.DataFrame,
    home_stars_avail: int = 5,
    away_stars_avail: int = 5,
    home_ml: int | None = None,
    away_ml: int | None = None,
) -> pd.Series:
    """
    Build the same feature set used during training, but for ONE upcoming game.
    Uses only games strictly before game_date — no leakage.
    """
    history = historical_games[historical_games["GAME_DATE"] < game_date].copy()
    if len(history) == 0:
        raise ValueError(f"No historical games before {game_date.date()}")

    # ELO ratings as of game_date — keep BOTH the augmented games table
    # (with home_elo_pre / away_elo_pre columns) AND the final ratings dict,
    # so the team game log can include opponent-ELO for SOS features.
    history_with_elo, ratings = compute_elo_features(history, return_final_ratings=True)
    home_elo = ratings.get(home_team_id, config.ELO_INITIAL)
    away_elo = ratings.get(away_team_id, config.ELO_INITIAL)

    # Per-team rolling stats from team game log (includes opp_elo when ELO exists)
    log = _team_game_log(history_with_elo)
    log["GAME_DATE"] = pd.to_datetime(log["GAME_DATE"])

    features: dict = {}
    for team_id, side in [(home_team_id, "home"), (away_team_id, "away")]:
        for window in config.ROLLING_WINDOWS:
            stats = _rolling_stats_for_team(log, team_id, game_date, window)
            if stats is None:
                raise ValueError(
                    f"No game history found for team_id {team_id} "
                    f"before {game_date.date()}. Make sure cached data covers "
                    f"this team's recent games."
                )
            for k, v in stats.items():
                features[f"{side}_{k}"] = v
        rest = _rest_for_team(log, team_id, game_date)
        features[f"{side}_rest_days"] = rest["rest_days"]
        features[f"{side}_is_b2b"]    = rest["is_b2b"]

    # Difference features (home minus away)
    for window in config.ROLLING_WINDOWS:
        for stat in config.ROLLING_STATS:
            features[f"diff_{stat}_last{window}"] = (
                features[f"home_{stat}_last{window}"]
                - features[f"away_{stat}_last{window}"]
            )

    # ELO + injury features
    features["home_elo_pre"] = home_elo
    features["away_elo_pre"] = away_elo
    features["elo_diff"]     = home_elo - away_elo

    features["home_stars_avail"] = home_stars_avail
    features["away_stars_avail"] = away_stars_avail
    features["stars_avail_diff"] = home_stars_avail - away_stars_avail

    # Optional: betting odds features (only populated if the trained model
    # used them — predict.py checks feature_cols and fills these in only
    # when needed).
    if home_ml is not None and away_ml is not None:
        from src.odds import american_to_prob, remove_vig
        h_raw = american_to_prob(home_ml)
        a_raw = american_to_prob(away_ml)
        h_fair, a_fair = remove_vig(h_raw, a_raw)
        features["home_implied_prob"] = h_fair
        features["away_implied_prob"] = a_fair
        features["market_edge"]       = h_fair - 0.5

    return pd.Series(features)


# ── Top-level prediction function ────────────────────────────────────────────

def predict_game(
    home_team: str,
    away_team: str,
    game_date: str | None = None,
    home_stars_avail: int = 5,
    away_stars_avail: int = 5,
    home_ml: int | None = None,
    away_ml: int | None = None,
) -> dict:
    """
    Predict the probability that the home team wins.

    Args:
        home_team:        team abbreviation, e.g. "LAL"
        away_team:        team abbreviation, e.g. "GSW"
        game_date:        "YYYY-MM-DD" string, default = today
        home_stars_avail: 0-5, how many top players will play for home
        away_stars_avail: 0-5, same for away
        home_ml:          optional home moneyline (American odds, e.g. -150)
        away_ml:          optional away moneyline (American odds, e.g. +130)
                          Ignored if the trained model didn't use odds features.

    Returns:
        Dict with prediction details, ready to print or use programmatically.
    """
    # 1. Load saved model artifacts
    model, scaler, feature_cols = load_artifacts()

    # 2. Resolve team IDs
    home_id, home_full = team_id_from_abbr(home_team)
    away_id, away_full = team_id_from_abbr(away_team)

    # 3. Resolve date
    game_dt = pd.Timestamp(game_date) if game_date else pd.Timestamp.now().normalize()

    # 4. Load cached historical games
    from src.data_fetch import fetch_all_seasons, build_game_table
    raw   = fetch_all_seasons()
    games = build_game_table(raw)

    # 5. Build feature vector
    feats = build_prediction_features(
        home_id, away_id, game_dt, games,
        home_stars_avail=home_stars_avail,
        away_stars_avail=away_stars_avail,
        home_ml=home_ml, away_ml=away_ml,
    )

    # 6. Select in saved order, scale, predict (with v2 calibration + ensemble)
    X = feats[feature_cols].values.reshape(1, -1)
    X_scaled = scaler.transform(X)
    prob = predict_calibrated(
        model, scaler, X_scaled,
        xgb_model=load_artifacts.last_xgb,
        temperature=load_artifacts.last_temperature,
    )

    # 7. Friendly summary
    return {
        "home_team":            home_team.upper(),
        "home_team_full":       home_full,
        "away_team":            away_team.upper(),
        "away_team_full":       away_full,
        "date":                 game_dt.strftime("%Y-%m-%d"),
        "home_win_probability": prob,
        "away_win_probability": 1 - prob,
        "predicted_winner":     home_team.upper() if prob > 0.5 else away_team.upper(),
        "confidence":           _confidence_label(prob),
        "home_elo":             feats["home_elo_pre"],
        "away_elo":             feats["away_elo_pre"],
        "home_recent_winpct":   feats["home_win_pct_last10"],
        "away_recent_winpct":   feats["away_win_pct_last10"],
        "home_rest_days":       feats["home_rest_days"],
        "away_rest_days":       feats["away_rest_days"],
        "home_stars":           home_stars_avail,
        "away_stars":           away_stars_avail,
    }


def _confidence_label(prob: float) -> str:
    """Translate prob distance from 0.5 into a verbal label."""
    margin = abs(prob - 0.5)
    if margin >= 0.20: return "high"
    if margin >= 0.10: return "moderate"
    return "low"
