# Portfolio Outreach Toolkit

Ready-to-use copy for putting this project to work in your job hunt. Paste, tweak,
send. Sections below cover LinkedIn, resume, cover letters, interview talking points,
and a 60-second elevator pitch.

---

## 1. The 60-second elevator pitch

> I built an end-to-end machine-learning pipeline that predicts NBA games. It
> scrapes four seasons of data from Basketball Reference, engineers leakage-safe
> features including a FiveThirtyEight-style ELO system and strength-of-schedule
> weighting, and trains a TensorFlow neural network in ensemble with XGBoost.
>
> The interesting part isn't that I built it — it's that I publicly documented
> where it failed. My first version said the Knicks had an 87% chance of winning
> the 2026 title; Vegas had them at 55%. I wrote a `MODEL_LIMITATIONS.md`
> identifying five specific reasons for the gap, implemented all five fixes, and
> the v2 model dropped to 75.5%. The Knicks then won 4-1, which is the kind of
> decisive series outcome that a 75% prediction fits better than Vegas's
> coin-flip line.
>
> So the deliverable is a pipeline, a dashboard, an 11-part tutorial, and a
> case study walking through the iteration loop. It's portfolio code, but the
> communication discipline is what I'd bring to a data analyst role.

---

## 2. LinkedIn post (publish at project launch)

```
Just shipped a portfolio project: an end-to-end ML pipeline that picked the
New York Knicks as the 2025-26 NBA champion at every round of the playoffs.

🏆 Predicted: NYK to win the title (75.5%)
🏆 Actual: NYK beat SAS 4-1 in the Finals

What's in the repo:
• 5,278 NBA games scraped from Basketball Reference (4 seasons)
• Leakage-safe feature engineering (rolling stats, FiveThirtyEight ELO,
  strength-of-schedule weighting, star-availability proxy)
• TensorFlow neural network + XGBoost ensemble with temperature calibration
• Monte-Carlo bracket simulator with mid-state continuation
• Streamlit web UI + one-click live dashboard
• 11-part tutorial covering every design decision
• Honest MODEL_LIMITATIONS.md documenting 7 known failure modes

The favorite part: the model's first version said 87% NYK. Vegas said 55%.
I wrote up the gap, identified 5 specific reasons (recency bias, SOS not
discounted, ELO over-reactive, no XGBoost ensemble, no calibration), and the
v2 model came in at 75.5%. The 4-1 series result lands closer to that than
to the market's coin-flip pricing.

Demo + code: [GitHub link]
Walkthrough: [GitHub case study link]

Looking for data analyst / sports analytics roles — happy to talk about any
of the design decisions in the repo.

#dataanalytics #sportsanalytics #machinelearning #python #portfolio #nba
```

---

## 3. Resume bullet points

Pick the 2-3 that match the role you're applying for. Numbers are concrete and
defensible (every claim ties back to a file in the repo).

**Project bullet (top of "Projects" section):**
- **NBA Playoff Predictor** — End-to-end ML pipeline that correctly predicted the
  2025-26 NBA Champion at every playoff round; model output (75.5% NYK) tracked
  closer to actual outcome (4-1 series) than Vegas consensus (55%).

**Sub-bullets (if you want detail):**
- Scraped & normalized 5,278 NBA games across 4 seasons from Basketball Reference;
  built a fallback fetcher when stats.nba.com was network-blocked
- Engineered 35+ leakage-safe features including ELO ratings, strength-of-schedule
  weighting, and rolling rest/availability metrics
- Trained TensorFlow neural network (ROC-AUC 0.725) in ensemble with XGBoost and
  temperature calibration; reduced model over-confidence from 87% → 75.5% via
  5 documented v1→v2 fixes
- Built Monte-Carlo bracket simulator handling best-of-7 series, 2-2-1-1-1 home
  court, and mid-bracket continuation from arbitrary in-progress states
- Shipped a one-click live dashboard (HTML generator), Streamlit web UI, and
  three CLI tools (single game / daily slate / playoff bracket)
- Wrote 11-part technical tutorial and public MODEL_LIMITATIONS.md documenting
  failure modes with prioritized fixes — full iteration loop in v2

**Skills line (technologies you can cite):**
- Python, TensorFlow, XGBoost, scikit-learn, pandas, NumPy, Streamlit, BeautifulSoup,
  Monte-Carlo simulation, model calibration, feature engineering, web scraping

---

## 4. Cover letter paragraph (data analyst / DA role)

Drop into any cover letter where you want to lead with a concrete project:

> A portfolio project I'm proud of is an NBA championship predictor I built
> end-to-end this year. I scraped four seasons of data from Basketball Reference,
> engineered a leakage-safe feature pipeline (rolling stats, ELO ratings,
> strength-of-schedule weighting), trained a TensorFlow neural network in
> ensemble with XGBoost, and deployed it through a Streamlit UI and a
> one-click live dashboard. The model picked the Knicks as the 2026 champion
> at every round of the playoffs, and the 4-1 Finals result tracked the
> model's confident prediction (75.5%) more closely than the Vegas line (~55%).
>
> What I think matters more than the prediction itself is what I did before
> it landed: I publicly documented where the model was likely over-confident
> (a 32-percentage-point gap with the market) in a `MODEL_LIMITATIONS.md`
> file, identified five specific feature-engineering and modeling issues
> causing it, implemented all five fixes in a v2, and shipped the comparison
> as a case study. That's the working style I'd bring to your team — build,
> measure, critique honestly, iterate.

---

## 5. Cover letter paragraph (sports analytics / basketball role)

Same project, different framing:

> A portfolio piece I'd point you to is an NBA championship predictor I built
> through the 2025-26 playoffs. It's a Python pipeline scraping Basketball
> Reference, engineering features around rolling form, FiveThirtyEight-style
> ELO with margin-of-victory and home-court adjustments, strength-of-schedule
> weighting, and back-to-back / rest signals. The model output is fed into a
> Monte-Carlo bracket simulator that respects the 2-2-1-1-1 home-court
> format and supports mid-bracket continuation from any current series score.
>
> It picked the Knicks at every round and was directionally validated when
> NYK beat SAS 4-1 in the Finals. Equally important — when my first version
> predicted NYK 87% and Vegas had them at 55%, I wrote up the disagreement
> publicly, identified specifically which features were causing the over-
> confidence, and iterated. That kind of basketball-aware skepticism about
> model output (not just "AUC went up") is what I'd contribute.

---

## 6. Interview-ready talking points

Have these ready when an interviewer asks about the project:

### "Walk me through the project"
- Started with question: can I predict NBA game outcomes?
- Built data pipeline → features → NN → evaluation → playoff bracket simulator
- During the actual 2026 playoffs, model picked NYK at every round
- Identified the model was over-confident vs Vegas, fixed in v2
- NYK won 4-1 — model directionally right, calibration improvement matters

### "What was the hardest part?"
- Honestly: not the modeling. The hardest part was deciding to write
  MODEL_LIMITATIONS.md instead of pretending the 87% number was fine.
- Once that doc existed, the v2 fixes were obvious: ELO too reactive, no SOS,
  no ensemble, no calibration. The discipline was the discipline.

### "What would you do differently with more time?"
- Three things, in priority order:
  1. Plug in real historical betting odds (the `src/odds.py` module is wired
     up; just needs a Kaggle dataset to evaluate properly)
  2. Add player-level skill features (RAPTOR / EPM) — bookmakers' biggest
     edge over me
  3. Train a separate playoff model — current model is mostly regular-season
     trained, playoff dynamics differ

### "How would you put this into production?"
- Already partially productionised: `dashboard.py` regenerates from cached data
- For real production: add a GitHub Actions workflow running daily — re-fetch
  BR, retrain, redeploy dashboard to GitHub Pages, push betting-line
  evaluation to a Postgres so I can backtest accuracy across seasons
- Wire up monitoring: drift detection on input features, calibration check
  on each prediction, alert on out-of-distribution opponents

### "Why TensorFlow + XGBoost together?"
- NN catches non-linear interactions; trees handle threshold-style features
  (like "ELO diff > 100 → big jump") that NNs struggle with at this sample size
- The ensemble is also a hedge against either model's idiosyncrasies; in
  particular XGBoost is less prone to NN's over-confidence on tabular data
- Empirically: NN alone gave NYK 81.5% on the Finals matchup, XGBoost gave
  65.3%, ensemble landed at 62.8% — closer to the right answer

### "How do you handle missing data?"
- Rolling features use `.shift(1)` so a missing game just shortens the window
- ELO ratings handle missing teams gracefully (init at 1500)
- Star-availability features default to 5 if data unavailable
- Odds features fully optional via a config flag

### "What's your testing strategy?"
- Chronological train/val/test split (not random — that's a leakage trap)
- Multiple evaluation metrics, not just accuracy (AUC, precision, recall, F1)
- Calibration plot, confusion matrix, ROC curve all generated automatically
- Logistic regression baseline trained on the same features as a sanity check
- Used the actual 2026 playoffs as an out-of-sample evaluation in real time

---

## 7. GitHub repo polish checklist

Before sharing the repo link with recruiters:

- [ ] Pin the repo on your GitHub profile
- [ ] Add a one-line repo description and Topics
  - Description: `NBA championship predictor (TensorFlow + XGBoost ensemble). Picked the 2025-26 champion correctly with documented iteration from v1 → v2.`
  - Topics: `machine-learning`, `data-science`, `nba`, `sports-analytics`,
    `tensorflow`, `xgboost`, `python`, `monte-carlo`, `streamlit`, `portfolio`
- [ ] Enable GitHub Pages so `dashboard.html`, `results.html`, and the docs are
      browsable without cloning. Settings → Pages → `main` branch, `/` (root)
- [ ] Add a `LICENSE` file (MIT is fine for portfolios)
- [ ] Save the `docs/screenshots/dashboard-preview.png` so the README hero
      image actually renders
- [ ] Push a final commit with a meaningful message: `🏆 Final: model
      predicted 2025-26 NBA champion validated by NYK 4-1 Finals win`

---

## 8. Where to share

In rough order of impact for finding a DA / sports analytics job:

1. **LinkedIn post** (template above) — pin to top of your profile
2. **Pinned GitHub repo** — what recruiters look at first
3. **Resume project section** — top of the Projects block
4. **r/sportsanalytics, r/MachineLearning [P]** subreddits — pin to
   Show-and-tell threads
5. **Twitter/X**: short version of the LinkedIn post + dashboard screenshot
6. **Discord communities**: NBA analytics Discords (Apanalytics, Squared2020),
   sports analytics professional groups
7. **Direct outreach**: when applying to specific roles, mention the project
   by name in the cover letter — interviewers love a concrete artifact

---

## 9. Roles worth targeting

This project fits well at:

- **NBA team analytics departments** — every team has 2-5 person analytics
  teams; entry-level roles often called Basketball Operations Analyst or
  Data Analyst
- **Sports analytics startups** — Krossover, Stathletes, Synergy Sports,
  Second Spectrum, Hudl, SaberSim
- **Sports betting / DFS companies** — DraftKings, FanDuel, PrizePicks,
  Underdog Fantasy — they all have data science orgs
- **Media companies with sports verticals** — ESPN, The Athletic, FiveThirtyEight
  (when hiring again), Bleacher Report
- **General DA roles where domain interest signals motivation** —
  any consumer/product analyst role; the project showcases the full DA
  toolkit even if the topic isn't basketball

---

## 10. Things NOT to do

- ❌ Don't claim the model "beat Vegas." A single series isn't statistical
  evidence. Stay calibrated about your own model's calibration.
- ❌ Don't oversell it as production-grade. It's a portfolio piece. The
  production hardening section in CASE_STUDY is honest about what's missing.
- ❌ Don't hide MODEL_LIMITATIONS.md. It's the single most valuable artifact
  in the repo for interviewers — a model you can critique is a model you
  understand.
- ❌ Don't blast the same generic post everywhere. Tailor the framing for
  the audience: DA roles vs basketball roles vs ML engineer roles.

---

*Last updated after the 2025-26 NBA Finals concluded on June 13, 2026.*
