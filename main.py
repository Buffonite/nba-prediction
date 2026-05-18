"""
NBA Game Prediction – entry point.

Run modes
---------
  python main.py            # full pipeline (fetch -> preprocess -> train -> evaluate)
  python main.py --skip-fetch   # reuse existing raw data file
  python main.py --demo         # use synthetic data (no internet required)

The --demo flag is handy for quick testing or if stats.nba.com is rate-limiting you.
"""

import argparse
import os
import sys

# Make stdout UTF-8 on Windows so Unicode chars in print statements don't crash
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd

import config


def demo_dataset(n_games: int = 3000) -> pd.DataFrame:
    """
    Generate a synthetic dataset that mimics real NBA game structure.
    Useful for testing the pipeline without an internet connection.

    Uses REAL NBA team IDs (from the static lookup baked into nba_api),
    so downstream prediction CLI calls like `predict.py LAL GSW` will work.
    """
    from nba_api.stats.static import teams as nba_teams_module

    rng = np.random.default_rng(config.RANDOM_SEED)
    print(f"[DEMO] Generating {n_games:,} synthetic games (real team IDs) …")

    real_teams = nba_teams_module.get_teams()
    team_ids   = [t["id"] for t in real_teams]
    abbr_map   = {t["id"]: t["abbreviation"] for t in real_teams}
    # Per-team strength so synthetic data has consistent quality differences
    strengths  = {tid: rng.normal(0, 5) for tid in team_ids}

    dates      = pd.date_range("2021-10-01", periods=n_games, freq="12h")
    home_teams = rng.choice(team_ids, n_games)
    away_teams = rng.choice(team_ids, n_games)
    # Make sure home != away
    same = home_teams == away_teams
    while same.any():
        away_teams = np.where(same, rng.choice(team_ids, n_games), away_teams)
        same = home_teams == away_teams

    # Score driven by team strength + home court bonus + noise
    home_strength = np.array([strengths[t] for t in home_teams])
    away_strength = np.array([strengths[t] for t in away_teams])
    home_court = 3.0
    margin = (home_strength - away_strength) + home_court + rng.normal(0, 10, n_games)

    home_pts = rng.integers(95, 125, n_games) + (margin / 2).astype(int)
    away_pts = home_pts - margin.astype(int)
    home_pts = np.clip(home_pts, 80, 145)
    away_pts = np.clip(away_pts, 80, 145)
    home_win = (home_pts > away_pts).astype(int)

    df = pd.DataFrame({
        "GAME_ID":        [f"G{i:05d}" for i in range(n_games)],
        "GAME_DATE":      dates,
        "home_team_id":   home_teams,
        "home_team_abbr": [abbr_map[t] for t in home_teams],
        "away_team_id":   away_teams,
        "away_team_abbr": [abbr_map[t] for t in away_teams],
        "home_pts":       home_pts.astype(int),
        "away_pts":       away_pts.astype(int),
        "home_win":       home_win,
    })

    # Cache to data/raw so predict.py can load it the same way as real data
    os.makedirs(os.path.dirname(config.RAW_DATA_PATH), exist_ok=True)
    df.to_csv(config.RAW_DATA_PATH, index=False)

    print(f"[DEMO] Home-win rate: {home_win.mean():.1%}, saved to {config.RAW_DATA_PATH}")
    return df


def main():
    parser = argparse.ArgumentParser(description="NBA Game Prediction Pipeline")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Load raw data from disk instead of calling the API")
    parser.add_argument("--demo", action="store_true",
                        help="Use synthetic data (no internet required)")
    parser.add_argument("--source", choices=["nba_api", "bref"], default="nba_api",
                        help="Data source: 'nba_api' (default) or 'bref' (Basketball Reference, "
                             "use this if stats.nba.com is blocked)")
    parser.add_argument("--odds", choices=["off", "csv", "synthetic"], default="off",
                        help="Betting odds feature: 'off' (default), 'csv' (load real historical "
                             "odds from config.ODDS_CSV_PATH), or 'synthetic' (demo, inflates AUC)")
    args = parser.parse_args()

    print("=" * 60)
    print("  NBA GAME PREDICTION – Neural Network Pipeline")
    print("=" * 60)

    # ── Step 1: Data acquisition ──────────────────────────────────────────────
    print("\n[1/4] Data acquisition")
    if args.demo:
        games = demo_dataset()
        player_logs = None
    elif args.source == "bref":
        # Basketball Reference path — no player game logs available, so
        # injury features are disabled automatically.
        from src.data_fetch_bref import fetch_all_seasons_bref
        games = fetch_all_seasons_bref()
        player_logs = None
        if config.USE_INJURY_FEATURES:
            print("  ℹ Injury features disabled (Basketball Reference doesn't expose player game logs)")
            config.USE_INJURY_FEATURES = False
    else:
        from src.data_fetch import fetch_all_seasons, build_game_table, fetch_player_logs
        raw   = fetch_all_seasons()
        games = build_game_table(raw)
        player_logs = fetch_player_logs() if config.USE_INJURY_FEATURES else None

    print(f"      Total games: {len(games):,}")

    # ── Step 2: Feature engineering ───────────────────────────────────────────
    print("\n[2/4] Feature engineering")

    # 2a. ELO ratings (works on any per-game table, including demo data)
    if config.USE_ELO_FEATURES:
        from src.elo import compute_elo_features
        games = compute_elo_features(games)

    # 2b. Star availability (real player logs in normal mode, synthetic in demo)
    star_avail = None
    if config.USE_INJURY_FEATURES:
        from src.injuries import (
            identify_stars, compute_star_availability, synthetic_star_availability,
        )
        if args.demo:
            star_avail = synthetic_star_availability(games)
        else:
            stars = identify_stars(player_logs)
            star_avail = compute_star_availability(player_logs, stars, games)

    # 2c. Betting odds (off / csv / synthetic)
    odds = None
    if args.odds == "csv":
        from src.odds import load_odds_csv
        odds_raw = load_odds_csv()
        # Join on (date, home_abbr, away_abbr) to attach GAME_ID
        odds = games.merge(
            odds_raw,
            left_on=["GAME_DATE", "home_team_abbr", "away_team_abbr"],
            right_on=["game_date", "home_team_abbr", "away_team_abbr"],
            how="inner",
        )[["GAME_ID", "home_implied_prob", "away_implied_prob", "market_edge"]]
        print(f"  + Loaded real odds for {len(odds):,} games")
    elif args.odds == "synthetic":
        from src.odds import synthetic_odds
        odds = synthetic_odds(games)

    # 2d. Build the final feature matrix (rolling stats + ELO + injuries + odds)
    from src.preprocessing import build_features
    features = build_features(games, star_avail=star_avail, odds=odds)

    # ── Step 3: Training ──────────────────────────────────────────────────────
    print("\n[3/4] Model training")
    from src.train import run_training
    results = run_training(features)

    # ── Step 4: Evaluation ────────────────────────────────────────────────────
    print("\n[4/4] Evaluation")
    from src.evaluate import run_evaluation
    metrics = run_evaluation(results)

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print(f"  Neural Net  accuracy: {metrics['nn']['accuracy']:.1%}   AUC: {metrics['nn']['roc_auc']:.3f}")
    print(f"  Baseline    accuracy: {metrics['baseline']['accuracy']:.1%}   AUC: {metrics['baseline']['roc_auc']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
