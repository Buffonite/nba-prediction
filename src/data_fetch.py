"""
Fetches historical NBA game logs using the nba_api package.

The free `nba_api` library talks directly to stats.nba.com — no API key needed.
Results are saved to data/raw/games.csv so you only need to download once.

Usage (called from main.py or standalone):
    python -m src.data_fetch
"""

import time
import os
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, leaguegamelog
from nba_api.stats.static import teams as nba_teams_module

import config


def fetch_season(season: str) -> pd.DataFrame:
    """
    Download all regular-season games for one season (e.g. '2022-23').
    Returns a raw DataFrame from the API.
    """
    print(f"  Fetching season {season} …")
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable="Regular Season",
        league_id_nullable="00",   # NBA
    )
    df = finder.get_data_frames()[0]
    # Polite delay so we don't hammer stats.nba.com
    time.sleep(1.5)
    return df


def fetch_all_seasons(seasons: list[str] = config.SEASONS) -> pd.DataFrame:
    """
    Download all specified seasons and concatenate into one DataFrame.
    Skips download if the raw file already exists.
    """
    if os.path.exists(config.RAW_DATA_PATH):
        print(f"Raw data already exists at '{config.RAW_DATA_PATH}'. Loading from disk.")
        return pd.read_csv(config.RAW_DATA_PATH, parse_dates=["GAME_DATE"])

    print("Downloading game logs …")
    frames = []
    for season in seasons:
        df = fetch_season(season)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(config.RAW_DATA_PATH), exist_ok=True)
    combined.to_csv(config.RAW_DATA_PATH, index=False)
    print(f"Saved {len(combined):,} rows → '{config.RAW_DATA_PATH}'")
    return combined


def build_game_table(raw: pd.DataFrame) -> pd.DataFrame:
    """
    The raw API response has ONE ROW PER TEAM PER GAME.
    This function pivots it into ONE ROW PER GAME with both teams' stats.

    Returns a DataFrame with columns:
        GAME_ID, GAME_DATE,
        home_team_id, home_team_abbr,
        away_team_id, away_team_abbr,
        home_pts, away_pts,
        home_win  ← target label (1 = home team won)

    If the input is already in per-game format (e.g. cached demo data),
    just normalise types and return it unchanged.
    """
    # Detect already-built per-game data and short-circuit
    if {"home_team_id", "away_team_id", "home_pts", "away_pts", "home_win"}.issubset(raw.columns):
        out = raw.copy()
        out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"])
        out = out.sort_values("GAME_DATE").reset_index(drop=True)
        print(f"Loaded pre-built game table: {len(out):,} games, "
              f"home-win rate = {out['home_win'].mean():.1%}")
        return out

    df = raw.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    # MATCHUP contains e.g. "LAL vs. GSW" (home) or "LAL @ GSW" (away)
    df["is_home"] = df["MATCHUP"].str.contains(" vs\\. ")

    home = df[df["is_home"]].rename(columns={
        "TEAM_ID": "home_team_id",
        "TEAM_ABBREVIATION": "home_team_abbr",
        "PTS": "home_pts",
        "WL": "home_wl",
    })[["GAME_ID", "GAME_DATE", "home_team_id", "home_team_abbr", "home_pts", "home_wl"]]

    away = df[~df["is_home"]].rename(columns={
        "TEAM_ID": "away_team_id",
        "TEAM_ABBREVIATION": "away_team_abbr",
        "PTS": "away_pts",
    })[["GAME_ID", "away_team_id", "away_team_abbr", "away_pts"]]

    games = home.merge(away, on="GAME_ID", how="inner")
    games["home_win"] = (games["home_wl"] == "W").astype(int)
    games = games.drop(columns=["home_wl"])
    games = games.sort_values("GAME_DATE").reset_index(drop=True)

    print(f"Built game table: {len(games):,} games, home-win rate = {games['home_win'].mean():.1%}")
    return games


# ── Player game logs (used for star-availability / injury features) ─────────

def fetch_player_logs(seasons: list[str] = config.SEASONS) -> pd.DataFrame:
    """
    Download every player's game log for the given seasons in ONE call per
    season. Used by src.injuries to detect which star players actually played.

    Caches to data/raw/player_logs.csv so repeat runs are instant.
    """
    if os.path.exists(config.PLAYER_LOGS_PATH):
        print(f"Player logs already exist at '{config.PLAYER_LOGS_PATH}'. Loading from disk.")
        return pd.read_csv(config.PLAYER_LOGS_PATH)

    print("Downloading player game logs …")
    frames = []
    for season in seasons:
        print(f"  Fetching player logs for {season} …")
        df = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star="Regular Season",
            player_or_team_abbreviation="P",   # 'P' = player-level rows
        ).get_data_frames()[0]
        frames.append(df)
        time.sleep(1.5)

    combined = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(config.PLAYER_LOGS_PATH), exist_ok=True)
    combined.to_csv(config.PLAYER_LOGS_PATH, index=False)
    print(f"Saved {len(combined):,} player-game rows → '{config.PLAYER_LOGS_PATH}'")
    return combined


if __name__ == "__main__":
    raw = fetch_all_seasons()
    games = build_game_table(raw)
    print(games.head())
    if config.USE_INJURY_FEATURES:
        player_logs = fetch_player_logs()
        print(player_logs.head())
