# 🏀 NBA Game & Playoff Prediction

> End-to-end machine-learning project: from scraping 5,200+ NBA games to a neural-network model, daily-slate predictions, and a Monte-Carlo playoff bracket simulator — wrapped in a Streamlit web UI.

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/status-portfolio--ready-brightgreen.svg)

**Current prediction:** New York Knicks to win the 2025-26 NBA Championship (**53.1%** modelled probability) — [view the full results page](results.html).

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

### 4. One-command live dashboard

```bash
python dashboard.py --refresh --open
```

Or just **double-click `refresh.bat`** (Windows) / `refresh.sh` (Mac/Linux).

Generates a beautiful self-contained [`dashboard.html`](dashboard.html) with everything:
- 🏆 Hero champion prediction with Monte Carlo probability
- 📅 Current bracket state (all completed and in-progress series)
- 🛣️ Modal path to the title (round-by-round projection)
- 📊 Championship + Finals odds with bar charts
- 🏀 Most recent playoff games
- 🅴🅆 Both conferences' regular-season seeding
- 📈 Model performance metrics

Auto-detects the playoff state from your cached data — no manual updates needed.

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

📖 **See [MODEL_LIMITATIONS.md](MODEL_LIMITATIONS.md) for an in-depth analysis** of why this
model's predictions diverge from bookmakers (it currently favors NYK while the market favors
OKC), and a prioritized roadmap to close that gap. Written specifically to discuss in interviews.

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
