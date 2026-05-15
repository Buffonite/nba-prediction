"""
Alternative data source: scrape schedules + scores from Basketball Reference.

Used when stats.nba.com is blocked (e.g. mainland China without VPN).
basketball-reference.com is hosted in the US on different infrastructure and
is usually reachable when stats.nba.com is not.

Strategy
────────
We use pandas.read_html() to parse Basketball Reference's monthly schedule
tables. The URL pattern is:
    https://www.basketball-reference.com/leagues/NBA_{end_year}_games-{month}.html

Where end_year is the year the season ends (e.g. NBA_2024 = 2023-24 season),
and month is the lowercase month name. We respect BR's rate limit (1 req / 3s).

The output matches the format produced by data_fetch.build_game_table() so
the rest of the pipeline doesn't care which source the data came from.
"""

import os
import time
import re
import pandas as pd
from nba_api.stats.static import teams as nba_teams_module

import config

# Months that contain games (regular season + playoffs)
REGULAR_SEASON_MONTHS = [
    "october", "november", "december",
    "january", "february", "march", "april",
]
PLAYOFF_MONTHS = ["may", "june"]

# All months we fetch — including playoffs means ELO ratings stay up-to-date
# through completed playoff series, so predictions for remaining rounds are sharper.
ALL_MONTHS = REGULAR_SEASON_MONTHS + PLAYOFF_MONTHS

# Polite delay between BR page fetches
BR_DELAY_SECONDS = 3.0


def _season_to_end_year(season: str) -> int:
    """'2023-24' → 2024.  '2025-26' → 2026."""
    start_yy = int(season.split("-")[0])
    return start_yy + 1


def _fetch_month_table(end_year: int, month: str) -> pd.DataFrame | None:
    """Pull a single month's schedule from Basketball Reference."""
    url = f"https://www.basketball-reference.com/leagues/NBA_{end_year}_games-{month}.html"
    try:
        tables = pd.read_html(url)
    except (ValueError, Exception) as e:
        # 404 or empty page — month doesn't have games this season
        return None
    if not tables:
        return None
    df = tables[0]
    # Some monthly pages include a "Playoffs" separator row — drop header repeats
    if "Date" in df.columns:
        df = df[df["Date"] != "Date"].copy()
    return df


def fetch_season_schedule(season: str) -> pd.DataFrame:
    """
    Fetch all regular-season games for one season from Basketball Reference.

    Args:
        season: in our standard 'YYYY-YY' format, e.g. '2023-24'

    Returns:
        A combined DataFrame across all months.
    """
    end_year = _season_to_end_year(season)
    print(f"  Fetching season {season} from Basketball Reference (end year {end_year}) …")

    frames = []
    for month in ALL_MONTHS:
        df = _fetch_month_table(end_year, month)
        if df is not None and len(df) > 0:
            df["_source_month"] = month
            df["_season_end_year"] = end_year
            frames.append(df)
            tag = "playoffs" if month in PLAYOFF_MONTHS else "reg"
            print(f"    ✓ {month:<10}  {len(df):>4} rows  ({tag})")
        else:
            print(f"    ⊘ {month:<10}  (no data — pre-season or off-season)")
        time.sleep(BR_DELAY_SECONDS)

    if not frames:
        raise RuntimeError(f"No games found for season {season} on Basketball Reference.")
    return pd.concat(frames, ignore_index=True)


def fetch_all_seasons_bref(seasons: list[str] = config.SEASONS) -> pd.DataFrame:
    """Drop-in replacement for data_fetch.fetch_all_seasons but using BR."""
    if os.path.exists(config.RAW_DATA_PATH):
        print(f"Raw data already exists at '{config.RAW_DATA_PATH}'. Loading from disk.")
        return pd.read_csv(config.RAW_DATA_PATH, parse_dates=["GAME_DATE"])

    print(f"Downloading game logs from Basketball Reference for {len(seasons)} seasons …")
    print(f"  (Polite delay: {BR_DELAY_SECONDS}s between page requests)")

    frames = []
    for season in seasons:
        df = fetch_season_schedule(season)
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    # Convert to our standard game-table format right away
    games = _to_game_table(raw)

    os.makedirs(os.path.dirname(config.RAW_DATA_PATH), exist_ok=True)
    games.to_csv(config.RAW_DATA_PATH, index=False)
    print(f"Saved {len(games):,} games → '{config.RAW_DATA_PATH}'")
    return games


# ── Conversion: BR schedule → our standard per-game table ────────────────────

def _to_game_table(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert BR's monthly-schedule format into our standard game table.

    BR columns we care about:
      'Date', 'Visitor/Neutral', 'PTS', 'Home/Neutral', 'PTS.1'

    Output columns:
      GAME_ID, GAME_DATE,
      home_team_id, home_team_abbr,
      away_team_id, away_team_abbr,
      home_pts, away_pts, home_win
    """
    nba_teams   = nba_teams_module.get_teams()
    name_to_id   = {t["full_name"]: t["id"]            for t in nba_teams}
    name_to_abbr = {t["full_name"]: t["abbreviation"]  for t in nba_teams}

    df = raw.copy()
    df["GAME_DATE"] = pd.to_datetime(df["Date"], errors="coerce")
    # Drop rows that aren't real games (header repeats, future games)
    df = df.dropna(subset=["GAME_DATE"])
    df = df.dropna(subset=["PTS", "PTS.1"])

    # PTS = visitor (away) points, PTS.1 = home points  (BR's ordering)
    df["home_pts"] = pd.to_numeric(df["PTS.1"], errors="coerce")
    df["away_pts"] = pd.to_numeric(df["PTS"],   errors="coerce")
    df = df.dropna(subset=["home_pts", "away_pts"])

    df["home_team_id"]   = df["Home/Neutral"].map(name_to_id)
    df["away_team_id"]   = df["Visitor/Neutral"].map(name_to_id)
    df["home_team_abbr"] = df["Home/Neutral"].map(name_to_abbr)
    df["away_team_abbr"] = df["Visitor/Neutral"].map(name_to_abbr)

    # Warn about unmapped teams (rare — usually means a name change like
    # "New Orleans Hornets" → "New Orleans Pelicans" in old seasons)
    unmapped = df[df["home_team_id"].isna() | df["away_team_id"].isna()]
    if len(unmapped) > 0:
        bad_names = set(unmapped["Home/Neutral"]) | set(unmapped["Visitor/Neutral"])
        bad_names = {n for n in bad_names if n not in name_to_id}
        print(f"  ⚠ Dropping {len(unmapped)} rows with unmapped team names: {bad_names}")
        df = df.dropna(subset=["home_team_id", "away_team_id"])

    df["home_team_id"] = df["home_team_id"].astype(int)
    df["away_team_id"] = df["away_team_id"].astype(int)
    df["home_pts"]     = df["home_pts"].astype(int)
    df["away_pts"]     = df["away_pts"].astype(int)
    df["home_win"]     = (df["home_pts"] > df["away_pts"]).astype(int)

    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    df["GAME_ID"] = [f"BR{i:06d}" for i in range(len(df))]

    keep_cols = ["GAME_ID", "GAME_DATE", "home_team_id", "home_team_abbr",
                 "away_team_id", "away_team_abbr", "home_pts", "away_pts", "home_win"]
    out = df[keep_cols].copy()
    print(f"  Built game table: {len(out):,} games, "
          f"home-win rate = {out['home_win'].mean():.1%}")
    return out


if __name__ == "__main__":
    games = fetch_all_seasons_bref()
    print(games.head())
    print(f"\nTotal: {len(games):,} games across {games['GAME_DATE'].dt.year.nunique()} years")
