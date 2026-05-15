"""
Simulate an NBA playoff bracket and print championship odds.

Usage:
    python playoffs.py
    python playoffs.py --east BOS NYK MIL CLE PHI MIA ORL IND --west OKC DEN MIN LAL GSW MEM PHX NOP
    python playoffs.py --sims 5000

Output: probability that each team advances to each round and wins the title.
"""

import argparse
import sys
from datetime import date as date_cls

# Make stdout UTF-8 on Windows for the box drawings
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.playoffs import (
    simulate_bracket, simulate_from_round_2, SeriesState, MatchupPredictor,
)


# Default illustrative bracket (typical recent contenders — adjust as needed)
DEFAULT_EAST = ["BOS", "NYK", "MIL", "CLE", "PHI", "MIA", "ORL", "IND"]
DEFAULT_WEST = ["OKC", "DEN", "MIN", "LAL", "GSW", "MEM", "PHX", "NOP"]

# Current 2025-26 Round 2 state (as of 2026-05-11)
# Format: SeriesState(higher_seed, lower_seed, higher_wins, lower_wins)
DEFAULT_R2_EAST = [
    SeriesState("DET", "CLE", 2, 1),   # top quadrant (1v8 winner vs 4v5 winner)
    SeriesState("NYK", "PHI", 3, 0),   # bottom quadrant (3v6 winner vs 2v7 winner)
]
DEFAULT_R2_WEST = [
    SeriesState("OKC", "LAL", 3, 0),   # top quadrant
    SeriesState("SAS", "MIN", 2, 1),   # bottom quadrant
]


def _parse_series_spec(spec: str) -> SeriesState:
    """Parse a string like 'NYK:PHI:3-0' into a SeriesState."""
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid series spec '{spec}'. Use HIGHER:LOWER:W-L (e.g. NYK:PHI:3-0)")
    higher, lower, score = parts
    h_wins, l_wins = map(int, score.split("-"))
    return SeriesState(higher.upper(), lower.upper(), h_wins, l_wins)


def print_round(title: str, probs: dict, top_n: int = 8) -> None:
    print(f"\n{title}")
    print("─" * 50)
    for i, (team, p) in enumerate(list(probs.items())[:top_n]):
        bar = "█" * int(p * 30)
        print(f"  {i+1:>2}. {team:<5} {p:>5.1%}  {bar}")


def main():
    parser = argparse.ArgumentParser(
        description="Simulate an NBA playoff bracket.",
        epilog="Examples:\n"
               "  # Full bracket from Round 1:\n"
               "  python playoffs.py --east DET BOS NYK CLE ORL PHI ATL TOR \\\n"
               "                     --west OKC SAS DEN LAL HOU MIN PHX POR\n\n"
               "  # Continue from current Round 2 state:\n"
               "  python playoffs.py --from-round 2 \\\n"
               "                     --r2-east DET:CLE:2-1 NYK:PHI:3-0 \\\n"
               "                     --r2-west OKC:LAL:3-0 SAS:MIN:2-1\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--from-round", type=int, default=1, choices=[1, 2],
                        help="Start simulation from this round (default 1)")
    parser.add_argument("--east", nargs=8, default=DEFAULT_EAST,
                        metavar="ABBR",
                        help="Eastern conference seeds 1-8 (used when --from-round=1)")
    parser.add_argument("--west", nargs=8, default=DEFAULT_WEST,
                        metavar="ABBR",
                        help="Western conference seeds 1-8 (used when --from-round=1)")
    parser.add_argument("--r2-east", nargs=2, default=None, metavar="SERIES",
                        help="2 East R2 series in HIGHER:LOWER:W-L form, "
                             "top-quadrant first. E.g. DET:CLE:2-1 NYK:PHI:3-0")
    parser.add_argument("--r2-west", nargs=2, default=None, metavar="SERIES",
                        help="2 West R2 series, same format")
    parser.add_argument("--sims", type=int, default=2000,
                        help="Number of full-bracket Monte Carlo simulations (default 2000)")
    parser.add_argument("--date", default=None,
                        help="Reference date for features (default: today)")
    args = parser.parse_args()

    print("=" * 60)
    print("  NBA PLAYOFF BRACKET SIMULATION")
    print("=" * 60)

    # Load predictor once (caches model + data + per-matchup probs)
    print("\nLoading model …")
    predictor = MatchupPredictor(game_date=args.date)

    if args.from_round == 2:
        # ── Mid-bracket mode: continue from current Round 2 state ────────────
        r2_east = (
            [_parse_series_spec(s) for s in args.r2_east]
            if args.r2_east else DEFAULT_R2_EAST
        )
        r2_west = (
            [_parse_series_spec(s) for s in args.r2_west]
            if args.r2_west else DEFAULT_R2_WEST
        )

        print("\n  Mode: continue from Round 2 (Conference Semi-finals)")
        print("  East R2:")
        for s in r2_east:
            print(f"    · {s.describe()}")
        print("  West R2:")
        for s in r2_west:
            print(f"    · {s.describe()}")
        print(f"  Simulations: {args.sims:,}")

        results = simulate_from_round_2(
            r2_east=r2_east, r2_west=r2_west,
            predictor=predictor, n_sims=args.sims,
        )

        print("\n" + "=" * 60)
        print("  RESULTS")
        print("=" * 60)
        print_round("🏆 Conference Finals appearance", results["conf_finals_probs"])
        print_round("🥇 NBA Finals appearance",        results["finals_probs"])
        print_round("👑 NBA Champion",                  results["champion_probs"], top_n=8)
    else:
        # ── Default: full bracket from Round 1 ───────────────────────────────
        print(f"\n  East (seeds 1-8):  {' → '.join(args.east)}")
        print(f"  West (seeds 1-8):  {' → '.join(args.west)}")
        print(f"  Simulations:       {args.sims:,}")

        results = simulate_bracket(
            east_seeds=args.east,
            west_seeds=args.west,
            n_sims=args.sims,
            predictor=predictor,
        )

        print("\n" + "=" * 60)
        print("  RESULTS")
        print("=" * 60)
        print_round("🏀 Advance past Round 1 (Conference Semis)", results["round2_probs"])
        print_round("🏆 Conference Finals appearance",            results["conf_finals_probs"])
        print_round("🥇 NBA Finals appearance",                  results["finals_probs"])
        print_round("👑 NBA Champion",                            results["champion_probs"], top_n=5)

    # Pick the most likely winner
    if results["champion_probs"]:
        champ, prob = next(iter(results["champion_probs"].items()))
        print("\n" + "═" * 60)
        print(f"  Most likely champion: {champ}  ({prob:.1%})")
        print("═" * 60)


if __name__ == "__main__":
    main()
