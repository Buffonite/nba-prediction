# 🏀 NBA Game & Playoff Prediction

> End-to-end machine-learning project: from scraping 5,200+ NBA games to a neural-network model, daily-slate predictions, and a Monte-Carlo playoff bracket simulator — wrapped in a Streamlit web UI and a one-command live dashboard.

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/status-portfolio--ready-brightgreen.svg)

![Live dashboard preview](docs/screenshots/dashboard-preview.png)

## 🏆 Validated prediction — 2025-26 NBA Champion

> **Predicted: New York Knicks** (model said 75.5% to win)
> **Actual: New York Knicks defeated San Antonio Spurs 4-1 in the Finals (June 13, 2026)** ✅
>
> Model picked NYK as the favorite at every round of the playoffs. The 4-1 series outcome
> aligned more closely with the model's confident prediction than with Vegas (-132 ≈ 55%) —
> a near-coin-flip line that didn't fit a decisive 5-game series. See the full journey in
> [**CASE_STUDY.md**](CASE_STUDY.md).

---

## 🎯 Portfolio summary (recruiter 30-second read)

**What:** Predicted the 2025-26 NBA Champion (Knicks) before and during the playoffs — picking
them at every round, with the 4-1 Finals result validating the model's "decisive favorite"
confidence over Vegas's near-coin-flip line.

**Built end-to-end alone:**
- **Data engineering** — Scraped 5,278 games (4 seasons) from Basketball Reference with rate-limit-aware fetcher + normalization layer
- **Feature engineering** — Leakage-safe rolling stats, strength-of-schedule weighting, FiveThirtyEight-style ELO, star-availability proxy, betting-odds integration
- **Modeling** — TensorFlow neural network + XGBoost ensemble with temperature scaling, chronological train/val/test split, logistic-regression baseline for sanity
- **Honest evaluation** — Documented 7 failure modes in [`MODEL_LIMITATIONS.md`](MODEL_LIMITATIONS.md), iterated v1 → v2 with measurable improvement (87% → 75.5% over-confidence reduction)
- **Productization** — CLI tools (single-game, daily slate, playoff bracket), Streamlit web UI, one-click live dashboard generator
- **Communication** — Polished README, 11-part [TUTORIAL.md](TUTORIAL.md), [CASE_STUDY.md](CASE_STUDY.md) narrative, model-vs-Vegas analysis

**Skills shown:** Python, TensorFlow, XGBoost, scikit-learn, pandas, web scraping, Monte-Carlo simulation, model calibration, ensemble learning, data visualization, technical writing.

**Relevance for hiring:**
- Data analyst — full pipeline from raw HTML to validated probability outputs
- Sports analytics — applied ML with NBA-specific domain understanding (ELO, playoff format, home court)
- ML engineer — productionised inference with proper artifact versioning, dashboard, and live refresh tooling

📖 Start with [**CASE_STUDY.md**](CASE_STUDY.md) for the full story, or jump straight to the [live dashboard](dashboard.html).

---

## 📊 At a glance

| | |
|---|---|
| **Task**       | Predict the winner of any NBA game (binary classification) |
| **Data**       | 4 NBA seasons (2022-23 → 2025-26), 5,257 games, scraped from Basketball Reference |
| **Model**      | TensorFlow neural network — 3 dense layers with BatchNorm + Dropout |
| **Baseline**   | Logistic Regression (for honest comparison) |
| **Performance**| Neural Network AUC **0.721** · accuracy **67.8%** (chronological test split) |
| **Extensions** | ELO ratings, star-player availability proxy, daily-slate batch predict, playoff bracket Monte-Carlo simulator |
| **Interface**  | CLI (3 scripts) + Streamlit web UI (4 tabs) |

---

## 🎯 Why this project?

Built as a portfolio piece to demonstrate the full ML lifecycle:

- 🔍 **Data engineering**: scraping, cleaning, leakage-free feature pipelines
- 🧠 **Model design**: choosing architecture, regularisation, optimization
- 📈 **Evaluation discipline**: temporal splits, multiple metrics, calibration checks
- 🚢 **Productionisation**: saved artifacts, inference CLI, web UI, simulation tools
- 📝 **Communication**: detailed [TUTORIAL.md](TUTORIAL.md) explaining every design choice

---

## 🚀 Quickstart

```bash
git clone <repo-url>
cd nba-prediction

# 1. Install dependencies (~3 min)
python -m venv .venv
source .venv/Scripts/activate            # Windows
# source .venv/bin/activate              # macOS/Linux
pip install -r requirements.txt

# 2. Train the model (~5 min — fetches data + trains NN + makes 5 plots)
python main.py --source bref

# 3. Explore — pick one:
streamlit run app.py                     # web UI (recommended)
python predict.py LAL GSW                # single-game prediction
python daily.py                          # all of today's games
python playoffs.py --from-round 2        # current 2025-26 playoff projection
```

> The `--source bref` flag scrapes from Basketball Reference. Use the default `nba_api` source if `stats.nba.com` is reachable from your network.

---

## 🖥️ The four ways to use it

### 1. Single-game prediction (CLI)

```bash
python predict.py NYK DET --date 2026-05-20 --home-stars 5 --away-stars 4
```

```
══════════════════════════════════════════════════════════════
         Detroit Pistons  @  New York Knicks
  Date: 2026-05-20
──────────────────────────────────────────────────────────────
  Recent form (last 10 games):
     NYK    80% win-rate   ELO  1761   rest 3d   stars 5/5
     DET    70% win-rate   ELO  1703   rest 4d   stars 4/5
──────────────────────────────────────────────────────────────
  Predicted winner:   NYK   (confidence: moderate)

     NYK  68.0%  ████████████████████░░░░░░░░░░
     DET  32.0%  ░░░░░░░░░░░░░░░░░░░░██████████
══════════════════════════════════════════════════════════════
```

### 2. Daily slate (CLI)

```bash
python daily.py --date 2026-05-15 --csv
```

Fetches all NBA games on a date and predicts every matchup.

### 3. Playoff bracket simulator (CLI)

```bash
# Full bracket from Round 1
python playoffs.py --east DET BOS NYK CLE ORL PHI ATL TOR \
                   --west OKC SAS DEN LAL HOU MIN PHX POR --sims 5000

# Continue from current Round 2 state
python playoffs.py --from-round 2 \
                   --r2-east "NYK:PHI:4-0" "DET:CLE:2-2" \
                   --r2-west "OKC:LAL:4-0" "SAS:MIN:2-2"
```

Each best-of-7 series respects the 2-2-1-1-1 home-court format. Monte-Carlo runs return per-team probabilities for each round.

### 4. One-command live dashboard ⭐

```bash
python dashboard.py --refresh --open
```

Or just **double-click `refresh.bat`** (Windows) / `refresh.sh` (Mac/Linux).

![Live dashboard preview](docs/screenshots/dashboard-preview.png)

Generates a beautiful self-contained [`dashboard.html`](dashboard.html) with everything:
- 🏆 Hero champion prediction with Monte Carlo probability
- 📅 Current bracket state (all completed and in-progress series shown as cards)
- 🛣️ Modal path to the title (round-by-round predicted winner & confidence)
- 📊 Championship + Finals odds bar charts (10,000-iteration Monte Carlo)
- 🏀 Most recent playoff games table
- 🅴🅆 Both conferences' regular-season seeding
- 📈 Model performance metrics

Auto-detects the playoff state from your cached data — no manual updates needed.
On refresh, existing data is backed up so a transient network error never wipes your cache.

### 5. Streamlit web UI

```bash
streamlit run app.py
```

Four tabs in the browser:
- 🔮 **Predict a Game** — team dropdowns, date picker, injury sliders, probability bars
- 📅 **Daily Slate** — all games on a date, downloadable as CSV
- 👑 **Playoff Bracket** — full bracket simulation with progress bars per round
- 📊 **Model & Project** — methodology, all 5 diagnostic plots, network architecture

---

## 📁 Project structure

```
nba-prediction/
├── 📋 CLI entrypoints
│   ├── main.py           Train pipeline: fetch → features → train → evaluate
│   ├── predict.py        Predict one matchup
│   ├── daily.py          Predict every game on a date
│   └── playoffs.py       Bracket Monte-Carlo simulator
│
├── 🖥️ Web UI
│   └── app.py            Streamlit app — 4 interactive tabs
│
├── 🧠 Source modules (src/)
│   ├── data_fetch.py     nba_api downloader
│   ├── data_fetch_bref.py Basketball Reference scraper (works when nba_api is blocked)
│   ├── preprocessing.py  Rolling-window features, leakage-safe with .shift(1)
│   ├── elo.py            FiveThirtyEight-style ELO ratings
│   ├── injuries.py       Star-availability proxy (top-5 minutes leaders)
│   ├── model.py          Neural network + logistic regression baseline
│   ├── train.py          Train pipeline with temporal split + scaling
│   ├── evaluate.py       5 metrics + 5 diagnostic plots
│   ├── predict.py        Inference for a single matchup
│   ├── daily.py          Batch prediction wrapper
│   └── playoffs.py       Series & bracket simulation + SeriesState
│
├── 📚 Documentation
│   ├── README.md         (this file)
│   ├── TUTORIAL.md       In-depth Chinese walk-through, 11 parts
│   └── results.html      Standalone results showcase page
│
├── ⚙️ Configuration
│   └── config.py         All hyperparameters in one place
│
└── 🎯 Outputs (gitignored except plots)
    ├── plots/            5 PNG diagnostic plots
    └── models/           Saved Keras model + scaler + feature columns
```

---

## 📈 Results

### Model metrics (chronological test set, last 20% of games)

| Metric | Neural Net | Logistic Reg |
|---|---:|---:|
| Accuracy | 67.8% | 67.8% |
| ROC-AUC | 0.721 | 0.728 |
| Precision | 67.4% | 69.4% |
| Recall | 73.8% | 75.3% |
| F1 | 70.4% | 72.2% |

> The logistic regression baseline edges out the neural network on AUC by a small margin. This is honest: NBA win prediction is mostly a linear signal problem on this feature set. The neural network's value comes from learning small non-linear interactions, but for production a tree-based model (XGBoost) would likely outperform both. See [TUTORIAL Part 9.5](TUTORIAL.md) for analysis.

### Diagnostic plots

5 PNGs saved to `outputs/plots/` after running `python main.py`:

| Plot | What it shows |
|---|---|
| `training_curves.png` | Loss & AUC per epoch — over/underfitting check |
| `roc_curve.png` | NN vs. Logistic Regression vs. random |
| `confusion_matrix.png` | True vs. predicted labels |
| `calibration.png` | Are predicted probabilities trustworthy? |
| `feature_importance.png` | Most influential features (via LR coefficients) |

### Current 2025-26 playoff projection

As of mid-May 2026, Round 2 in progress:

```
👑 NBA Champion (10,000 simulations)
   1. NYK   53.1%  ████████████████
   2. OKC   26.1%  ████████
   3. SAS   17.5%  █████
   4. DET    2.4%
   5. CLE    0.7%
```

**Open [`results.html`](results.html) in a browser for the full visual report.**

---

## 🧠 Methodology highlights

### Leakage-safe feature engineering

For each game, features come from each team's **past games only**:

- Rolling 5- and 10-game averages: win %, points scored/allowed, net rating
- Rest days, back-to-back flag
- Pre-game ELO ratings (FiveThirtyEight-style with home advantage, MOV multiplier, season carryover)
- Star-player availability (top-5 minutes leaders)
- Difference features: `home − away` for each rolling stat

```python
# The leakage guard — never include the current game in the rolling window
df.groupby("team_id")["pts_scored"].transform(
    lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
)
```

### Temporal train/val/test split

Random splits leak future information into training. We sort by date and split chronologically:

```
[─────── train (65%) ──────][── val (15%) ──][── test (20%) ──]
       earlier games                              most recent
```

### Neural network architecture

```
Input (~34 features)
    ↓
Dense(128) → BatchNorm → ReLU → Dropout(0.30)
    ↓
Dense(64)  → BatchNorm → ReLU → Dropout(0.30)
    ↓
Dense(32)  →             ReLU → Dropout(0.15)
    ↓
Dense(1)   →           Sigmoid  →  P(home win)
```

Trained with binary cross-entropy + Adam (lr=1e-3) + early stopping on validation AUC.

### Playoff bracket simulation

Two modes:

**1. From Round 1** — pure forward simulation, given the 8 East + 8 West seeds.

**2. From current state** (`--from-round 2`) — pass in current series scores like `NYK:PHI:3-0`, and the simulator continues from there using:
- Remaining-games Monte Carlo for each in-progress series
- Standard best-of-7 simulation for future hypothetical matchups
- Aggregated 10,000-iteration full-bracket draw to produce championship odds

See [`src/playoffs.py`](src/playoffs.py) for the implementation.

---

## 🛠️ Tech stack

- **Python** 3.10+
- **TensorFlow / Keras** — neural network training and inference
- **scikit-learn** — logistic regression baseline, scaling, metrics
- **pandas / numpy** — data manipulation
- **matplotlib** — diagnostic plots
- **Streamlit** — interactive web UI
- **Basketball Reference (scraped)** — data source (HTML tables parsed with `pandas.read_html`)
- **nba_api** — alternative data source (when `stats.nba.com` is reachable)

---

## 📖 Detailed tutorial

For a from-scratch walk-through covering every design decision — including the mathematics behind ReLU, dropout, ELO ratings, gradient descent, and the Monte-Carlo simulation — see [`TUTORIAL.md`](TUTORIAL.md).

The tutorial has 11 parts:

1. Global picture
2. Data acquisition
3. Feature engineering (the most important part)
4. Neural network model — every layer explained
5. Training pipeline
6. Evaluation & visualisation
7. How to read the outputs
8. Interview Q&A
9. Advanced features — ELO + injuries
10. Deep-dive topics (backprop, vanishing gradients, overfitting, Adam internals, alternative architectures)
11. Playoff bracket simulation
12. Basketball Reference scraping fallback

---

## ⚠️ Honest limitations

| Limitation | Impact |
|---|---|
| Trained on regular-season games | Playoff-specific dynamics (rotations, scheme adjustments) not captured |
| Star-availability is a proxy | Real injury impact varies — we count only "did they play", not "how impactful" |
| ~5,200 training samples | Relatively small for deep learning; XGBoost might fit better |
| No player-level skill features | Bookmakers use RAPTOR / EPM — this model only sees game outcomes |
| Recency bias in rolling features | A single dominant playoff series over-shifts predictions |
| Network access | Default `nba_api` source blocked in some regions; use `--source bref` |

📖 **See [MODEL_LIMITATIONS.md](MODEL_LIMITATIONS.md) for an in-depth analysis** with a prioritized
roadmap to close the gap with bookmakers. Written specifically to discuss in interviews.

---

## 🔬 Live case study — when the model disagrees with Vegas

As of the 2026 NBA Finals (NYK leads 1-0 over SAS), there's a striking gap:

| | This model | Vegas (NYK -132 / SAS +112) |
|---|---:|---:|
| **NYK wins championship** | **87.0%** | **~55%** (vig-stripped) |
| Implied per-game win probability | ~63% | ~50% |

**The model is over-confident by ~30 percentage points.** Working out *why* this happens is more
educational than the prediction itself, and exercises every failure mode listed in
[`MODEL_LIMITATIONS.md`](MODEL_LIMITATIONS.md):

1. **Recency bias** — NYK swept PHI and CLE; ELO inflated past historical baselines
2. **Opponent quality not discounted** — NYK beat the 6 seed and a depleted 4 seed; SAS beat the
   68-win OKC in a Game-7 road win, a far harder test. Model treats both as comparable evidence
3. **No player-level skill features** — Wembanyama's defensive impact (likely DPOY) isn't in any
   feature. Vegas's RAPTOR/EPM-based priors see it clearly
4. **Path / battle-tested factor not modeled** — teams emerging from a 7-game grind historically
   over-perform; teams cruising in sweeps under-perform. Model sees neither
5. **Series independence assumption** — Monte Carlo assumes each game is i.i.d., missing
   in-series adjustments

**Historical base rate check:** Game 1 Finals winners win the series ~70% of the time. The
model's 87% is well above that; Vegas's 55% is well below it (they think SAS is the better team
on a per-game basis). Both deviations are informative.

**What this case study demonstrates:** A model you can critique is a model you understand.
Knowing *why* my prediction differs from the market — and being able to point at the specific
feature gap that caused it — is the difference between "I built an ML model" and "I built an ML
model and know what would make it better." The full failure-mode taxonomy + fix roadmap is in
[`MODEL_LIMITATIONS.md`](MODEL_LIMITATIONS.md).

**One-line fix that would shrink the gap:** ensemble with the betting-odds feature
([`src/odds.py`](src/odds.py) is already wired up — see the section below). Plugging in
historical odds typically pulls model predictions back toward market consensus by 10-15
percentage points.

---

## 💰 Betting odds integration (optional feature)

The model supports betting odds as input features ([`src/odds.py`](src/odds.py)).
Three modes via `python main.py --odds {off|csv|synthetic}`:

| Mode | Source | Use when |
|---|---|---|
| `off` (default) | None | Honest baseline; what you see in published metrics |
| `csv` | `data/raw/odds.csv` from Kaggle / SBR / scraper | Real evaluation with market features |
| `synthetic` | Generated from outcomes + noise | Pipeline demo only — inflates AUC |

Effect when synthetic odds are added (illustrative only):
- NN AUC: **0.72 → 0.85** (+13 pp)
- Confirms the integration works and shows the ceiling of what real odds could contribute

For live predictions, supply moneylines at the CLI:

```bash
python predict.py NYK DET --home-ml -180 --away-ml +155
```

Or fetch live odds programmatically via The Odds API (free 500 reqs/month;
see [`src/odds.py:fetch_live_odds`](src/odds.py)).

## 🔮 Future work

- [x] ~~Betting-odds feature integration~~ ([`src/odds.py`](src/odds.py))
- [ ] **Real historical odds dataset** — currently relies on user-supplied CSV
- [ ] **XGBoost ensemble** — average NN + tree predictions (typical 1-2% AUC gain)
- [ ] **Player-level ELO** — track individual players, not just teams
- [ ] **Real injury reports** — scrape Rotoworld or `nba_api` injury endpoint
- [ ] **Deploy Streamlit app** to `share.streamlit.io` for live public demo
- [ ] **Backtest** prediction accuracy across each completed season
- [ ] **Hyperparameter search** via Optuna

---

## 📜 License

MIT. Use freely for learning and portfolios.

---

## 🙏 Acknowledgements

- [Basketball Reference](https://www.basketball-reference.com) — all historical NBA data
- [FiveThirtyEight](https://fivethirtyeight.com/methodology/how-our-nba-predictions-work/) — ELO methodology
- [`nba_api`](https://github.com/swar/nba_api) — alternative free data source

---

*Predictions are for educational and entertainment purposes only. Past performance does not guarantee future results.*
