#!/bin/bash
# ── NBA Prediction Dashboard refresh (macOS / Linux) ──────────────────────
# Run this to re-fetch latest games, retrain, and open the dashboard.
#
# Usage:   ./refresh.sh
# First time:  chmod +x refresh.sh  (then double-click or run)

set -e
cd "$(dirname "$0")"

echo ""
echo "============================================================"
echo "  NBA Prediction Dashboard - Live Refresh"
echo "============================================================"
echo ""

# Activate virtualenv if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "Activated virtual environment."
    echo ""
fi

python dashboard.py --refresh --open

echo ""
echo "============================================================"
echo "  Done! Dashboard opened in your browser."
echo "============================================================"
