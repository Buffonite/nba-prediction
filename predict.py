"""
Command-line predictor for NBA games.

Examples:
    python predict.py LAL GSW                       # Lakers (home) vs Warriors today
    python predict.py BOS NYK --date 2024-12-25     # Christmas Day game
    python predict.py LAL GSW --home-stars 4        # Lakers missing 1 star

The first argument is the HOME team, second is the AWAY team.
Team codes are standard NBA abbreviations: LAL, GSW, BOS, MIA, etc.
"""

import argparse
import sys

# Make stdout UTF-8 on Windows for the box-drawing characters in the output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.predict import predict_game


def format_result(r: dict) -> str:
    """Pretty-print the prediction with a probability bar."""
    prob = r["home_win_probability"]

    bar_len = 30
    home_bar_filled = int(round(prob * bar_len))
    away_bar_filled = bar_len - home_bar_filled

    home_bar = "█" * home_bar_filled + "░" * away_bar_filled
    away_bar = "░" * home_bar_filled + "█" * away_bar_filled

    return f"""
══════════════════════════════════════════════════════════════
  {r['away_team_full']:>22}  @  {r['home_team_full']:<22}
  Date: {r['date']}
──────────────────────────────────────────────────────────────
  Recent form (last 10 games):
    {r['home_team']:>4}  {r['home_recent_winpct']:>5.0%} win-rate   ELO {r['home_elo']:>5.0f}   rest {r['home_rest_days']}d   stars {r['home_stars']}/5
    {r['away_team']:>4}  {r['away_recent_winpct']:>5.0%} win-rate   ELO {r['away_elo']:>5.0f}   rest {r['away_rest_days']}d   stars {r['away_stars']}/5
──────────────────────────────────────────────────────────────
  Predicted winner:   {r['predicted_winner']}   (confidence: {r['confidence']})

    {r['home_team']:>4}  {prob:>5.1%}  {home_bar}
    {r['away_team']:>4}  {1 - prob:>5.1%}  {away_bar}
══════════════════════════════════════════════════════════════
"""


def main():
    parser = argparse.ArgumentParser(
        description="Predict the outcome of an NBA game.",
        epilog="Example: python predict.py LAL GSW --date 2024-12-25",
    )
    parser.add_argument("home", help="Home team abbreviation (e.g. LAL)")
    parser.add_argument("away", help="Away team abbreviation (e.g. GSW)")
    parser.add_argument("--date", default=None,
                        help="Game date in YYYY-MM-DD format (default: today)")
    parser.add_argument("--home-stars", type=int, default=5, metavar="N",
                        help="Number of star players (0-5) playing for home team")
    parser.add_argument("--away-stars", type=int, default=5, metavar="N",
                        help="Number of star players (0-5) playing for away team")
    parser.add_argument("--home-ml", type=int, default=None, metavar="ODDS",
                        help="Optional home team moneyline (American odds, e.g. -150). "
                             "Only used if the trained model includes odds features.")
    parser.add_argument("--away-ml", type=int, default=None, metavar="ODDS",
                        help="Optional away team moneyline (American odds, e.g. +130)")

    args = parser.parse_args()

    try:
        result = predict_game(
            home_team=args.home,
            away_team=args.away,
            game_date=args.date,
            home_stars_avail=args.home_stars,
            away_stars_avail=args.away_stars,
            home_ml=args.home_ml,
            away_ml=args.away_ml,
        )
    except FileNotFoundError as e:
        print(f"\n❌ {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ {e}", file=sys.stderr)
        sys.exit(1)

    print(format_result(result))


if __name__ == "__main__":
    main()
