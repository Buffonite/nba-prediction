"""
Betting-odds integration: turn bookmaker prices into model features.

Why this exists
───────────────
Bookmakers aggregate information our model can't see: injury reports, lineup
news, sharp money, public bias, player-tracking analytics. Their implied
probabilities are one of the strongest single signals for predicting NBA
games — typically achieving ~67% straight-up accuracy by themselves.

This module provides:
  1. Odds-format utilities (American ↔ decimal ↔ implied probability,
     vig removal for fair probabilities)
  2. Loaders for historical odds CSVs (Kaggle / scraped datasets)
  3. Live-odds fetcher placeholder for The Odds API (requires free key)
  4. Synthetic odds generator for demoing the feature integration when
     real historical odds aren't available

Honest disclaimer about synthetic odds
──────────────────────────────────────
The synthetic generator (synthetic_odds) creates noisy probabilities that
are correlated with actual outcomes — it's calibrated so the synthetic
market has ~67% straight-up accuracy (matching real Vegas). This lets you
WIRE UP the feature pipeline and see how the model integrates the input.

BUT: synthetic odds are derived from the labels. Training on them inflates
metrics in a way that doesn't reflect real performance. Treat synthetic
results as a CEILING for what real odds could contribute, not a reflection
of what they actually would.

For honest evaluation, use load_odds_csv() with a real historical odds file.
"""

import os
import numpy as np
import pandas as pd

import config


# ── Format utilities ─────────────────────────────────────────────────────────

def american_to_prob(odds: float) -> float:
    """
    Convert American moneyline odds to implied probability.

    Examples:
        american_to_prob(-150)  → 0.600   (favorite: bet $150 to win $100)
        american_to_prob(+200)  → 0.333   (underdog: bet $100 to win $200)
        american_to_prob(-110)  → 0.524   (pick'em with standard juice)
    """
    if odds < 0:
        return -odds / (-odds + 100)
    return 100 / (odds + 100)


def decimal_to_prob(odds: float) -> float:
    """
    Convert European decimal odds to implied probability.

    Examples:
        decimal_to_prob(1.67) → 0.599   (favorite)
        decimal_to_prob(3.00) → 0.333   (underdog)
    """
    if odds <= 1.0:
        raise ValueError(f"Decimal odds must be > 1.0, got {odds}")
    return 1.0 / odds


def remove_vig(home_prob: float, away_prob: float) -> tuple[float, float]:
    """
    Strip the bookmaker's margin (vig / juice) from raw implied probabilities
    so they sum to 1.0 — the "fair" probabilities.

    Standard bookmakers price 2-way markets at ~104-106% combined (i.e.
    4-6% vig). Normalizing by the sum is the simplest fair-odds method;
    more sophisticated methods (Shin, power, etc.) exist but matter little
    for 2-way moneylines.

    Example:
        home_raw = 0.55  (line: -122)
        away_raw = 0.49  (line: +104)
        sum = 1.04 (4% vig)
        → fair_home = 0.529, fair_away = 0.471
    """
    total = home_prob + away_prob
    if total <= 0:
        raise ValueError("Both probabilities are zero or negative")
    return home_prob / total, away_prob / total


# ── CSV loader for historical odds data ──────────────────────────────────────

def load_odds_csv(path: str = config.ODDS_CSV_PATH) -> pd.DataFrame:
    """
    Load historical odds from a CSV file.

    Expected columns (case-insensitive, flexible):
        GAME_DATE        - date of the game (YYYY-MM-DD)
        home_team_abbr   - 3-letter abbreviation (LAL, GSW, ...)
        away_team_abbr   - 3-letter abbreviation
        home_ml          - home moneyline (American odds, e.g. -150)
        away_ml          - away moneyline (American odds)

    Returns a DataFrame with GAME_DATE + home_team_abbr + away_team_abbr
    plus home_implied_prob, away_implied_prob (vig removed).

    Suggested data sources:
      - Kaggle: search "NBA odds historical"
      - https://www.sportsbookreviewsonline.com/ (downloadable XLS files)
      - https://www.basketball-reference.com (has some odds data per game)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No odds CSV at '{path}'. Download a historical odds dataset "
            f"(e.g. from Kaggle) and save it there, or use synthetic_odds() "
            f"for demo purposes."
        )

    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    # Normalize date
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Convert moneylines to fair implied probabilities
    home_raw = df["home_ml"].apply(american_to_prob)
    away_raw = df["away_ml"].apply(american_to_prob)
    fair = [remove_vig(h, a) for h, a in zip(home_raw, away_raw)]
    df["home_implied_prob"] = [f[0] for f in fair]
    df["away_implied_prob"] = [f[1] for f in fair]
    df["market_edge"]       = df["home_implied_prob"] - 0.5

    print(f"Loaded {len(df):,} historical odds rows from '{path}'")
    return df[["game_date", "home_team_abbr", "away_team_abbr",
               "home_implied_prob", "away_implied_prob", "market_edge"]]


# ── Synthetic odds for demo / pipeline verification ──────────────────────────

def synthetic_odds(games: pd.DataFrame, target_accuracy: float = 0.67,
                   seed: int | None = None) -> pd.DataFrame:
    """
    Generate plausible-looking synthetic odds for demo purposes.

    The generator is calibrated so the resulting "market" achieves the
    specified straight-up accuracy (default 67%, matching real NBA Vegas).

    ⚠️  WARNING: these odds are derived from actual outcomes. Using them
    as training features inflates metrics. They're intended for:
      - Verifying the odds-feature pipeline works end-to-end
      - Demonstrating the architecture in your portfolio
      - Setting an UPPER BOUND on what real odds could contribute

    For honest evaluation, use load_odds_csv() with real historical data.
    """
    rng = np.random.default_rng(seed if seed is not None else config.RANDOM_SEED)
    n = len(games)

    # Strategy: start from the true label (0 or 1), then add Gaussian noise
    # on the logit scale. Tune the noise magnitude to hit target accuracy.
    truth = games["home_win"].astype(float).values
    truth_logit = np.where(truth == 1, +3.0, -3.0)  # strong starting signal

    # Binary search for noise that gives target accuracy
    def _accuracy_at(noise_std: float) -> float:
        logits = truth_logit + rng.normal(0, noise_std, n)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return (np.round(probs) == truth).mean()

    # Quick calibration: try a range of noise levels
    best_std, best_diff = 2.5, 1.0
    for std in np.linspace(0.5, 5.0, 30):
        acc = _accuracy_at(std)
        diff = abs(acc - target_accuracy)
        if diff < best_diff:
            best_diff, best_std = diff, std

    # Generate final odds with the calibrated noise level
    logits = truth_logit + rng.normal(0, best_std, n)
    home_prob = 1.0 / (1.0 + np.exp(-logits))
    # Realistic moneylines don't go past ~95% / 5%
    home_prob = np.clip(home_prob, 0.15, 0.85)

    # Add ~4% vig to make the raw market sum > 1.0, then strip it
    raw_home = home_prob * 1.02
    raw_away = (1 - home_prob) * 1.02
    fair_home = raw_home / (raw_home + raw_away)
    fair_away = 1.0 - fair_home

    actual_acc = (np.round(fair_home) == truth).mean()
    print(f"⚠ Generated SYNTHETIC odds (acc = {actual_acc:.1%}, "
          f"noise_std = {best_std:.2f}) — for pipeline demo only.")

    return pd.DataFrame({
        "GAME_ID":           games["GAME_ID"].values,
        "home_implied_prob": fair_home,
        "away_implied_prob": fair_away,
        "market_edge":       fair_home - 0.5,
    })


# ── Live odds fetcher (The Odds API) ─────────────────────────────────────────

def fetch_live_odds(api_key: str = "") -> pd.DataFrame:
    """
    Fetch upcoming NBA games' moneylines from The Odds API.

    Setup:
        1. Sign up free at https://the-odds-api.com/  (500 reqs/month)
        2. Set ODDS_API_KEY in config.py or pass via api_key arg
        3. Run this function — returns today's NBA slate with odds

    Returns DataFrame with columns:
        commence_time, home_team, away_team,
        home_implied_prob, away_implied_prob (vig-stripped, averaged across books)
    """
    import urllib.request, json

    key = api_key or config.ODDS_API_KEY
    if not key:
        raise ValueError(
            "No Odds API key. Get a free one at https://the-odds-api.com/, "
            "then set ODDS_API_KEY in config.py."
        )

    url = (
        "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
        f"?apiKey={key}&regions=us&markets=h2h&oddsFormat=american"
    )
    with urllib.request.urlopen(url, timeout=15) as r:
        data = json.loads(r.read())

    rows = []
    for game in data:
        # Average prices across bookmakers (more robust than single book)
        home_probs, away_probs = [], []
        for book in game.get("bookmakers", []):
            for market in book.get("markets", []):
                if market["key"] != "h2h":
                    continue
                outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                if game["home_team"] not in outcomes or game["away_team"] not in outcomes:
                    continue
                h_raw = american_to_prob(outcomes[game["home_team"]])
                a_raw = american_to_prob(outcomes[game["away_team"]])
                h_fair, a_fair = remove_vig(h_raw, a_raw)
                home_probs.append(h_fair)
                away_probs.append(a_fair)
        if not home_probs:
            continue
        rows.append({
            "commence_time":     game["commence_time"],
            "home_team":         game["home_team"],
            "away_team":         game["away_team"],
            "home_implied_prob": float(np.mean(home_probs)),
            "away_implied_prob": float(np.mean(away_probs)),
        })

    return pd.DataFrame(rows)
