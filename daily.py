"""
Command-line: predict every NBA game on a given date.

Usage:
    python daily.py                          # today's games
    python daily.py --date 2024-12-25        # Christmas Day slate
    python daily.py --date 2024-12-25 --csv  # also save results as CSV
"""

import argparse
import sys
from datetime import date as date_cls

# Make stdout UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.daily import predict_daily_slate


def main():
    parser = argparse.ArgumentParser(description="Predict every NBA game scheduled on a date.")
    parser.add_argument("--date", default=None,
                        help="Game date in YYYY-MM-DD format (default: today)")
    parser.add_argument("--csv", action="store_true",
                        help="Save results to predictions_<date>.csv")
    args = parser.parse_args()

    target_date = args.date or date_cls.today().isoformat()
    print(f"\nFetching NBA slate for {target_date} …")

    df = predict_daily_slate(target_date)
    if len(df) == 0:
        print(f"No NBA games scheduled on {target_date} (off-season or break?).")
        sys.exit(0)

    # Pretty-print to console
    display = df.copy()
    display["Win Prob"] = display["Win Prob"].apply(
        lambda p: f"{p:.1%}" if p is not None else "—"
    )

    print(f"\nPredictions for {target_date}  ({len(df)} games):")
    print("─" * 90)
    print(display.to_string(index=False))
    print("─" * 90)

    if args.csv:
        out_path = f"predictions_{target_date}.csv"
        df.to_csv(out_path, index=False)
        print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
