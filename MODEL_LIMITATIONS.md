# Model Limitations & Discrepancy from the Market

> An honest analysis of why this model's predictions differ from bookmakers'
> consensus, and what would be needed to close the gap.

This document exists because **good data science means knowing what your model
gets wrong**. Skipping this section would let you ship a model whose
weaknesses you can't articulate — a faster way to lose interviewer credibility
than any low metric.

---

## 1. The discrepancy

| | This model | Vegas / major prediction sites |
|---|---:|---:|
| Most likely champion (May 2026) | **NYK 53.1%** | **OKC ~40-45%** |
| Second favorite | OKC 26.1% | NYK or SAS |
| Third favorite | SAS 17.5% | SAS |

The model is **directionally wrong about the favorite**. This isn't a small
calibration error — it's a meaningful divergence that demands explanation.

Bookmakers aren't infallible (they shade lines to manage exposure), but their
aggregate prior reflects decades of analyst work, player-tracking data, and
the collective wisdom of sharps who eat losses for being wrong. When a hobby
model disagrees with the market, **the burden of proof is on the model**.

---

## 2. Seven reasons the model favors NYK over OKC

### 2.1 🔥 Recency bias from rolling-window features

The model's strongest features are rolling 5- and 10-game averages plus ELO.
After NYK's 4-0 sweep of PHI (including a 137-98 demolition in Game 1),
NYK's recent-form values look like:

```
last_10_win_rate:     ~80%
ELO:                  1761  (up from ~1700 pre-playoffs)
last game point diff: +39
```

A single dominant series produced a step-change in the inputs the model
trusts most. The market doesn't update this hard on four games against a
mid-tier opponent.

**Fix:** weight rolling windows over a longer horizon (last 30+ games) or
blend with prior-season ratings that decay slower.

---

### 2.2 📊 Opponent quality isn't discounted enough

Both NYK and OKC went 4-0 in Round 2:

| Sweep | Opponent | Opponent's regular-season record |
|---|---|---|
| NYK 4-0 PHI | PHI | 49-40 (6 seed) |
| OKC 4-0 LAL | LAL | 56-31 (4 seed) |

OKC beat a stronger opponent. The ELO formula has a margin-of-victory
multiplier and accounts for opponent rating, but **the asymmetry between
"sweeping a mediocre team" and "sweeping a strong team" is not large enough**
in the implementation.

**Fix:** add a "strength-of-schedule adjusted" rolling stat — weight each
recent game by opponent ELO.

---

### 2.3 🛣️ Path-to-title analysis is naive

The model reasons:
- NYK in East needs to beat 1 elite team to win the title (OKC or SAS)
- OKC in West must beat SAS-MIN winner, **then** the East champion

This shorter path inflates NYK's odds. But the model treats OKC vs SAS as
roughly a coin flip, when in reality OKC's regular-season net rating likely
makes them a ~60-65% favorite even against a 66-win SAS team.

**Fix:** model net rating differential explicitly, not just win record + ELO.

---

### 2.4 🏀 Missing "true skill" priors that bookmakers use

The market's inputs include features this model has **none** of:

| Feature | What it captures | Whether we have it |
|---|---|---|
| Player Impact (RAPTOR, EPM, BPM) | Individual player value | ❌ |
| Multi-season weighted ratings | Skill stability across years | ❌ (we have 25% season-carryover, not enough) |
| Coaching quality | Playoff schemes & adjustments | ❌ |
| Playoff experience | Veteran composure | ❌ |
| Offense/defense efficiency split | Type-of-team mismatch | ❌ (we aggregate to net rating) |
| Tracking data (player speed, shot quality) | Underlying skill, not outcomes | ❌ |

Our model sees **only what happened in games**. The market sees **why** it
happened. That's a big information gap.

**Fix:** join external player-level data (Basketball Reference has PER,
WS, BPM; NBA Stats has player tracking). Even just adding team net rating
splits (offensive / defensive) would help.

---

### 2.5 🎯 Trained on regular season, applied to playoffs

The playoffs are structurally different:

- Rotations shorten from 10-11 players to 8 → star-heavy teams gain
- Coaches have 2-3 days between games to scheme → matchup-specific tactics
- Refs let more contact go → physical/defensive teams gain
- Pressure → veteran teams (OKC's title defense) gain composure edge

The model has seen ~250 playoff games (4 years × ~85 games) out of 5,257
total. That's 5% of training data trying to capture an 11-week period that
behaves nothing like the 6-month regular season.

**Fix:** train a separate "playoff mode" model, or use transfer learning —
fine-tune the base model on playoff games with a higher weight.

---

### 2.6 📉 Season carryover decays elite teams too aggressively

```python
# From src/elo.py
ELO_SEASON_CARRYOVER = 0.75   # New season: keep 75%, pull 25% toward 1500
```

OKC's 2024-25 ELO of (say) 1750 → 2025-26 starting ELO of 1687. They have
to "re-prove" themselves every October. Bookmakers don't do this — they
treat a team's ratings as continuous unless the roster turns over.

**Fix:** make the carryover roster-aware. If the core 5 returned, keep
90% of last season's rating. If they lost their best player, keep less.

---

### 2.7 🎲 Independent-game assumption in the Monte Carlo

Each game's outcome is sampled independently. In reality:

- A team that loses Game 1 will adjust strategy for Game 2
- Momentum is real (debated but non-zero effect)
- Star players develop chemistry across a series
- Coaches exploit revealed matchup advantages

The model treats Game 7 as if the prior 6 games never happened. This
under-rewards teams good at in-series adjustments (mostly: experienced
coaches and veterans = OKC, SAS).

**Fix:** introduce a series-state-dependent probability adjustment (e.g.,
slight Bayesian update after each game).

---

## 3. What the model gets right

Worth defending the genuine strengths:

✅ **Strict no-leakage feature engineering** — `.shift(1)` guards mean
   no game uses its own outcome.

✅ **Honest temporal split** — train/val/test by date, not random.
   Many tutorials mess this up; reported metrics are real.

✅ **Calibrated probability output** — sigmoid + binary cross-entropy +
   the calibration plot in `outputs/plots/`.

✅ **Logistic regression baseline** — explicit reality check that a
   simpler model is competitive (LR AUC 0.728 vs NN 0.721).

✅ **Mid-bracket simulation from current state** — most public bracket
   simulators force you to start from Round 1; this one accepts arbitrary
   in-progress series.

The model is a solid demonstration of ML mechanics. Its weakness is
**feature scope**, not pipeline correctness.

---

## 4. If I had two weeks to close the market gap

A prioritized roadmap:

### Week 1: better features
1. Scrape Basketball Reference for **team-level advanced stats**
   (offensive/defensive rating, pace, eFG%) — adds 8-10 features
2. Add **opponent-strength-weighted rolling stats** — replace plain averages
   with averages weighted by opponent ELO
3. Multi-season weighted ELO — keep 90% carryover for stable rosters

### Week 2: model adjustments
4. Train **separate playoff model** on the ~600 playoff games we have
5. **XGBoost ensemble** — average NN + tree-model predictions (typically
   1-2% AUC gain on tabular data)
6. **Calibrate to market** as a sanity check — compare predicted
   championship odds with consensus odds; if a team is off by > 10pp,
   investigate why

Expected outcome: champion-odds RMSE vs market drops from ~15pp to ~5pp,
likely flipping the favorite from NYK to OKC.

---

## 5. Why this document exists

In a data science interview, "what's wrong with your model?" is a
**make-or-break question**. The right answer isn't "nothing" — it's a
specific list of failure modes, prioritized by impact, with concrete fixes.

This document is the answer.

> A model you can't critique is a model you don't understand.

— A useful interview principle

---

## 6. Reading list

- **Nate Silver, "The Signal and the Noise"** — chapters on sports models
  cover exactly this trade-off between fancy models and good priors
- **538's NBA RAPTOR methodology** — what "good" player ratings look like
- **Kaggle "Don't Overfit" competition discussions** — wisdom on prior +
  feature quality > model complexity
- **Andrew Gelman's blog** — Bayesian framing of "your prior should beat
  any model trained on small data"

---

*This document is intentionally written to be linked from interviews. If
you'd like to discuss the failure modes here in more depth, the [TUTORIAL](TUTORIAL.md)
goes deeper on the underlying mechanics.*
