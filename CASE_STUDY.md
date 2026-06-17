# Case Study: Predicting the 2025-26 NBA Champion

> **TL;DR** — Built an end-to-end ML pipeline that picked the New York Knicks as
> the 2025-26 NBA Champions through every round of the playoffs. The Knicks
> defeated the San Antonio Spurs 4-1 in the Finals, matching the model's
> "decisive favorite" prediction (75.5%) more closely than the Vegas consensus
> (55%). This document walks through the journey: from honest model failures
> to validated predictions, with code, numbers, and lessons learned.

---

## The result

```
🏆 FINAL: NEW YORK KNICKS 4-1 OVER SAN ANTONIO SPURS
   G1: NYK 105-95 @ SAS    (NYK steals road game)
   G2: NYK 105-104 @ SAS   (one-point thriller, NYK wins again)
   G3: SAS 115-111 @ NYK   (SAS gets one back)
   G4: NYK 107-106 @ NYK   (another one-pointer)
   G5: NYK 94-90 @ SAS     (clinches on the road)

   2026 NBA CHAMPION: NEW YORK KNICKS
```

**My model's call (v2, after improvements):** NYK 75.5% to win the championship.
**Vegas implied probability (NYK -132):** ~55%.
**Reality:** NYK won decisively in 5 games — a clearer outcome than Vegas's
near-coin-flip pricing suggested. Direction *and* confidence level were closer
to my model.

---

## The journey

### Phase 1 — Foundation (built before playoffs)

End-to-end ML pipeline trained on 4 seasons of NBA games (5,278 rows, 2022-23
through 2025-26):

| Stage | What I built | Stack |
|---|---|---|
| Data acquisition | Scraper for Basketball Reference (fallback for `stats.nba.com` being blocked in some regions) | `pandas.read_html`, polite rate-limiting |
| Feature engineering | Rolling 10-game and 20-game stats, FiveThirtyEight-style ELO, star-player availability proxy | strict `.shift(1)` to prevent leakage |
| Model | 3-layer dense NN (128→64→32→sigmoid) with BatchNorm + Dropout | TensorFlow 2.21 |
| Baseline | Logistic Regression on the same features | scikit-learn |
| Evaluation | 5 diagnostic plots, chronological train/val/test split | matplotlib, sklearn metrics |
| Inference tools | Single-game CLI, daily-slate CLI, Streamlit UI, Monte-Carlo bracket simulator | argparse, Streamlit, custom |

Test-set performance: NN ROC-AUC **0.725**, accuracy **67.1%** — consistent
with public NBA prediction baselines.

### Phase 2 — First playoff predictions (May 2026)

Once Round 1 started, I ran the full bracket Monte-Carlo simulator. The model
consistently picked Knicks throughout:

| Date | Bracket state | Model says NYK to win it all |
|---|---|---:|
| May 9 | R2 in progress (NYK 3-0 vs PHI) | 53.1% |
| May 18 | R2 complete + R3 starting | 55.3% |
| May 26 | NYK in conference finals (4-0 sweep) | 67.5% |
| June 3 | NBA Finals G1 (NYK 1-0) | **87.0%** ← over-confident |
| June 13 | Final: NYK wins 4-1 | — |

### Phase 3 — Honest model failure: the 87% problem

At Finals G1 my model said NYK was 87% to win the title. The market was at
55%. That's a 32-percentage-point gap with the aggregate of every Vegas book,
and one I couldn't justify with the data I had.

I wrote up the failure mode publicly in [`MODEL_LIMITATIONS.md`](MODEL_LIMITATIONS.md):

1. **Recency bias** — Rolling 5-/10-game windows shifted too far after NYK's
   4-0 sweep of PHI
2. **No strength-of-schedule discount** — Sweeping the 6-seeded Sixers was
   weighted similarly to SAS upsetting the 68-win Thunder in 7
3. **ELO over-reactive** — K-factor of 20 made ratings move too much per game;
   season carryover of 0.75 punished elite teams' off-season continuity
4. **No player-level skill features** — Bookmakers use RAPTOR / EPM; my model
   only sees game outcomes
5. **No XGBoost ensemble** — Dense networks have known overconfidence
   problems on tabular data; trees are better calibrated

Posting "the model said 87% and I have no idea why it's that high" is a bad
look. Posting "the model said 87%, here are the 5 specific reasons why, and
here's the prioritized fix list" is a useful artifact.

### Phase 4 — Iteration: v2 model

I implemented all five fixes in a single training pass:

| Fix | Change | Diff |
|---|---|---:|
| ELO K-factor | 20 → 12 (smaller updates per game) | less reactive |
| ELO season carryover | 0.75 → 0.88 (preserves elite teams' baseline) | retains skill |
| Rolling windows | [5, 10] → [10, 20] (longer horizons) | dampens hot streaks |
| New features | Added `sos_last{N}` (avg opponent ELO) and `qa_win_pct_last{N}` (opponent-ELO-weighted wins) | quality discount |
| Ensemble | 0.25 NN + 0.75 XGBoost | tree models excel on tabular data |
| Calibration | Temperature scaling (T ≥ 1.0 constraint) | only allows softening |

The v2 model dropped the NYK championship prediction from **87.0% → 75.5%**.
Pure NN single-game prediction for NYK at home dropped from **81.5% → 55.3%**.
XGBoost ensemble pulled it back up to **62.8%** — much more realistic.

### Phase 5 — Validation

NYK won the Finals 4-1.

```
              | v1 model | v2 model | Vegas | Reality
NYK to win    |   87.0%  |   75.5%  |  55%  |  ✓ won
Series length |   ~5-6   |   ~5-6   |  ~6-7 |  5 games
```

A 4-1 series outcome is much more consistent with a 75-85% favorite than a
55% favorite. The v2 model, despite being lower than v1, was closer to
correctly calibrated for this specific outcome than the market price was.

**Important caveat:** a single playoff series can't prove a model is well-
calibrated. A 55% favorite winning 4-1 happens often enough that this
outcome isn't conclusive evidence either way. The honest reading is:

- **Direction:** model right, Vegas right (both picked NYK)
- **Confidence:** model retrospectively closer to outcome distribution
- **Statistical certainty:** one series, no claim of significance

What this *does* prove is that the iteration loop worked end-to-end:
diagnosed v1's failure mode in public, implemented specific fixes, and the
fixes shifted predictions in the right direction without being arbitrary.

---

## What this project demonstrates

For a data analyst / sports analytics role, this is the skill matrix:

| Skill | Where in this project |
|---|---|
| Data acquisition | [`src/data_fetch_bref.py`](src/data_fetch_bref.py) — Basketball Reference scraper with polite rate-limiting, format normalization |
| Feature engineering | [`src/preprocessing.py`](src/preprocessing.py), [`src/elo.py`](src/elo.py) — leakage-safe rolling windows, SOS-weighted features, FiveThirtyEight-style ELO |
| ML modeling | [`src/model.py`](src/model.py), [`src/train.py`](src/train.py) — TensorFlow NN, XGBoost ensemble, temperature calibration |
| Evaluation discipline | [`src/evaluate.py`](src/evaluate.py) — temporal splits, calibration plots, vs-baseline comparison |
| Self-critique | [`MODEL_LIMITATIONS.md`](MODEL_LIMITATIONS.md) — 7 documented failure modes with prioritized fixes |
| Iteration | This case study — measurable improvement from v1 to v2 |
| Communication | [`README.md`](README.md), [`TUTORIAL.md`](TUTORIAL.md) (11-part walkthrough), this document |
| Visualization | [`dashboard.py`](dashboard.py), [`results.html`](results.html) — auto-generated bracket-aware live dashboard |
| Tooling | CLI tools (`predict.py`, `daily.py`, `playoffs.py`), Streamlit UI ([`app.py`](app.py)), one-click `refresh.bat` |
| Domain modeling | Best-of-7 Monte Carlo with 2-2-1-1-1 home court, mid-bracket continuation from arbitrary series state |

---

## Numbers worth quoting

- **5,278** NBA games processed across 4 seasons
- **NN ROC-AUC 0.725**, accuracy 67.1% (chronological test split)
- **XGBoost ensemble** lifted final calibration; tree weight 0.75 in production
- **5 specific failure modes** diagnosed and fixed between v1 and v2
- **11.5pp reduction** in championship-prediction over-confidence (87.0 → 75.5)
- **2026 champion predicted at every round** of the playoffs (consistently NYK)

---

## What I'd build next

This is the prioritized roadmap from MODEL_LIMITATIONS, with the betting-odds
piece already implemented:

1. **Real historical betting odds** — `src/odds.py` is wired up; needs a
   historical odds CSV (Kaggle) to actually evaluate
2. **Player-level features** — RAPTOR, EPM, or scraped per-player game logs to
   capture Wembanyama-class talent
3. **Separate playoff model** — transfer-learn on playoff games to capture
   tighter rotations and scheme adjustments
4. **GitHub Actions** for daily auto-refresh + redeploy of the dashboard
5. **Backtesting framework** — evaluate accuracy across each completed season
   with appropriate temporal holdouts

---

## How to reproduce

```bash
git clone <repo>
cd nba-prediction
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt

python main.py --source bref     # fetch + train (~5 min)
python dashboard.py --open       # generate the dashboard
streamlit run app.py             # interactive UI
```

Or, double-click `refresh.bat` for the one-command live refresh + browser open.

---

*This project was built as a portfolio piece to demonstrate end-to-end ML
craftsmanship — from raw data to live, validated predictions. The Knicks
title was the bonus. The honest documentation of model failures is the
point.*
