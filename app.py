"""
Streamlit web UI for the NBA prediction model.

Run locally:
    streamlit run app.py

Three modes available in the sidebar:
  1. Predict a Game   – pick two teams + date, get probability
  2. Daily Slate      – pick a date, predict every scheduled game
  3. Model & Project  – view training metrics and diagnostic plots
"""

import os
from datetime import date

import pandas as pd
import streamlit as st

import config


# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NBA Game Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light custom styling
st.markdown(
    """
    <style>
    .big-prob {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
    }
    .team-name {
        font-size: 1.2rem;
        font-weight: 600;
        text-align: center;
    }
    .winner-badge {
        background: #ff6b35;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading trained model …")
def cached_load_artifacts():
    from src.predict import load_artifacts
    return load_artifacts()


@st.cache_data
def cached_team_list():
    from nba_api.stats.static import teams
    return sorted(teams.get_teams(), key=lambda t: t["full_name"])


@st.cache_data(show_spinner="Predicting …")
def cached_predict(home: str, away: str, game_date: str,
                   home_stars: int, away_stars: int):
    from src.predict import predict_game
    return predict_game(home, away, game_date, home_stars, away_stars)


@st.cache_data(show_spinner="Fetching slate …", ttl=600)
def cached_daily_slate(game_date: str):
    from src.daily import predict_daily_slate
    return predict_daily_slate(game_date)


# ── Helper: rendering a single prediction ────────────────────────────────────

def render_prediction(result: dict):
    """Pretty-print a single prediction result."""
    prob_home = result["home_win_probability"]
    prob_away = result["away_win_probability"]

    st.subheader(f"{result['away_team_full']} @ {result['home_team_full']}")
    st.caption(f"📅 {result['date']}")

    col_h, col_vs, col_a = st.columns([5, 1, 5])

    with col_h:
        st.markdown(f"<div class='team-name'>🏠 {result['home_team']}</div>",
                    unsafe_allow_html=True)
        is_winner = prob_home > 0.5
        prob_color = "#ff6b35" if is_winner else "#888"
        st.markdown(
            f"<div class='big-prob' style='color:{prob_color}'>{prob_home:.1%}</div>",
            unsafe_allow_html=True,
        )
        st.progress(prob_home)
        if is_winner:
            st.markdown("<div class='winner-badge'>★ Predicted Winner</div>",
                        unsafe_allow_html=True)

    with col_vs:
        st.markdown("<div style='text-align:center; padding-top:30px; "
                    "font-size:1.5rem; color:#888'>VS</div>",
                    unsafe_allow_html=True)

    with col_a:
        st.markdown(f"<div class='team-name'>✈️ {result['away_team']}</div>",
                    unsafe_allow_html=True)
        is_winner = prob_away > 0.5
        prob_color = "#ff6b35" if is_winner else "#888"
        st.markdown(
            f"<div class='big-prob' style='color:{prob_color}'>{prob_away:.1%}</div>",
            unsafe_allow_html=True,
        )
        st.progress(prob_away)
        if is_winner:
            st.markdown("<div class='winner-badge'>★ Predicted Winner</div>",
                        unsafe_allow_html=True)

    st.divider()

    # Confidence level
    conf_emoji = {"high": "🟢", "moderate": "🟡", "low": "🔴"}
    st.markdown(
        f"**Confidence:** {conf_emoji[result['confidence']]} "
        f"`{result['confidence'].upper()}`"
    )

    # Detailed stats table
    st.markdown("**Team Details**")
    details = pd.DataFrame({
        "Metric": ["ELO Rating", "Last 10 Win %", "Rest (days)", "Stars Available"],
        result["home_team"]: [
            f"{result['home_elo']:.0f}",
            f"{result['home_recent_winpct']:.0%}",
            f"{result['home_rest_days']}",
            f"{result['home_stars']}/5",
        ],
        result["away_team"]: [
            f"{result['away_elo']:.0f}",
            f"{result['away_recent_winpct']:.0%}",
            f"{result['away_rest_days']}",
            f"{result['away_stars']}/5",
        ],
    })
    st.dataframe(details, hide_index=True, use_container_width=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🏀 NBA Predictor")
st.sidebar.caption("Neural network trained on 3 NBA seasons")

mode = st.sidebar.radio(
    "Mode",
    ["Predict a Game", "Daily Slate", "Playoff Bracket", "Model & Project"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.markdown(
    "**About**\n\n"
    "This is a portfolio project demonstrating end-to-end ML: "
    "data acquisition → feature engineering → neural network training → "
    "deployment via this UI."
)


# ── Verify model is trained ──────────────────────────────────────────────────

if not os.path.exists(config.MODEL_SAVE_PATH):
    st.error(
        "⚠️ No trained model found.\n\n"
        "Please run `python main.py` first to train the model."
    )
    st.stop()


# ── Mode 1: Predict a Single Game ────────────────────────────────────────────

if mode == "Predict a Game":
    st.title("🔮 Predict a Game")
    st.caption("Pick a matchup and get the model's win probability.")

    teams = cached_team_list()
    team_names = [t["full_name"] for t in teams]
    abbr_lookup = {t["full_name"]: t["abbreviation"] for t in teams}

    # Find sensible defaults (LAL home, GSW away)
    default_home = team_names.index("Los Angeles Lakers") if "Los Angeles Lakers" in team_names else 0
    default_away = team_names.index("Golden State Warriors") if "Golden State Warriors" in team_names else 1

    col_home, col_away = st.columns(2)

    with col_home:
        st.subheader("🏠 Home Team")
        home_name = st.selectbox("Home team", team_names,
                                 index=default_home, label_visibility="collapsed")
        home_stars = st.slider(
            "Stars available (out of 5)",
            min_value=0, max_value=5, value=5, key="home_stars",
            help="Top-5 minutes leaders who will play. Lower if star players are injured.",
        )

    with col_away:
        st.subheader("✈️ Away Team")
        away_name = st.selectbox("Away team", team_names,
                                 index=default_away, label_visibility="collapsed")
        away_stars = st.slider(
            "Stars available (out of 5)",
            min_value=0, max_value=5, value=5, key="away_stars",
        )

    game_date = st.date_input("Game date", value=date.today())

    if home_name == away_name:
        st.warning("Home and away team must differ.")
    else:
        if st.button("🔮 Predict", type="primary", use_container_width=True):
            try:
                result = cached_predict(
                    home=abbr_lookup[home_name],
                    away=abbr_lookup[away_name],
                    game_date=game_date.strftime("%Y-%m-%d"),
                    home_stars=home_stars,
                    away_stars=away_stars,
                )
                st.divider()
                render_prediction(result)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.exception(e)


# ── Mode 2: Daily Slate ──────────────────────────────────────────────────────

elif mode == "Daily Slate":
    st.title("📅 Daily Slate")
    st.caption("Get predictions for every NBA game scheduled on a date.")

    selected_date = st.date_input("Select a date", value=date.today())

    if st.button("Load slate", type="primary"):
        date_str = selected_date.strftime("%Y-%m-%d")
        df = cached_daily_slate(date_str)

        if len(df) == 0:
            st.warning(
                f"No NBA games scheduled on {date_str}. "
                f"This is normal during the off-season (July–September) or All-Star break."
            )
        else:
            st.success(f"Found {len(df)} games on {date_str}.")

            # Format the Win Prob column for display
            display = df.copy()
            display["Win Prob"] = display["Win Prob"].apply(
                lambda p: f"{p:.1%}" if pd.notna(p) else "—"
            )

            st.dataframe(
                display,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Confidence": st.column_config.TextColumn(width="small"),
                    "Home ELO":   st.column_config.NumberColumn(format="%d"),
                    "Away ELO":   st.column_config.NumberColumn(format="%d"),
                },
            )

            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download as CSV",
                data=csv,
                file_name=f"predictions_{date_str}.csv",
                mime="text/csv",
            )


# ── Mode 3: Playoff Bracket Simulation ───────────────────────────────────────

elif mode == "Playoff Bracket":
    st.title("👑 Playoff Bracket Simulator")
    st.caption(
        "Monte-Carlo simulate the full 16-team NBA playoff bracket. "
        "Each best-of-7 series is simulated thousands of times based on the "
        "model's single-game probabilities."
    )

    st.info(
        "**How it works:** "
        "(1) For each round, the higher seed gets home-court in the 2-2-1-1-1 format. "
        "(2) Series outcomes are sampled from single-game probabilities. "
        "(3) The full bracket is repeated N times to get championship odds."
    )

    col_east, col_west = st.columns(2)

    with col_east:
        st.subheader("🅴 Eastern Conference")
        st.caption("Seeds 1-8 (one per line, top to bottom)")
        east_text = st.text_area(
            "East seeds",
            value="BOS\nNYK\nMIL\nCLE\nPHI\nMIA\nORL\nIND",
            height=240,
            label_visibility="collapsed",
            key="east_seeds_input",
        )

    with col_west:
        st.subheader("🅆 Western Conference")
        st.caption("Seeds 1-8 (one per line, top to bottom)")
        west_text = st.text_area(
            "West seeds",
            value="OKC\nDEN\nMIN\nLAL\nGSW\nMEM\nPHX\nNOP",
            height=240,
            label_visibility="collapsed",
            key="west_seeds_input",
        )

    n_sims = st.slider(
        "Bracket simulations", min_value=500, max_value=5000, value=2000, step=500,
        help="More simulations = more stable probabilities but slower",
    )

    if st.button("🏆 Simulate playoffs", type="primary", use_container_width=True):
        east = [s.strip().upper() for s in east_text.splitlines() if s.strip()]
        west = [s.strip().upper() for s in west_text.splitlines() if s.strip()]

        if len(east) != 8 or len(west) != 8:
            st.error(f"Need exactly 8 teams per conference. Got East={len(east)}, West={len(west)}.")
        else:
            try:
                from src.playoffs import simulate_bracket, MatchupPredictor
                with st.spinner("Loading model & computing matchup probabilities …"):
                    predictor = MatchupPredictor()

                with st.spinner(f"Running {n_sims} bracket simulations …"):
                    results = simulate_bracket(east, west, n_sims=n_sims, predictor=predictor)

                # ── Display results ──
                st.success(f"Completed {n_sims} simulations ✓")

                st.divider()

                # Champion (the headline result)
                if results["champion_probs"]:
                    champ, prob = next(iter(results["champion_probs"].items()))
                    st.markdown(f"### 👑 Most Likely Champion: **{champ}**")
                    st.metric("Championship probability", f"{prob:.1%}")

                st.divider()

                # All four rounds side-by-side
                tabs = st.tabs([
                    "Champion",
                    "NBA Finals",
                    "Conference Finals",
                    "Conf Semis (R2)",
                ])
                rounds_data = [
                    ("champion_probs",    "👑 NBA Champion"),
                    ("finals_probs",      "🥇 Reach NBA Finals"),
                    ("conf_finals_probs", "🏆 Reach Conference Finals"),
                    ("round2_probs",      "🏀 Advance past Round 1"),
                ]

                for tab, (key, label) in zip(tabs, rounds_data):
                    with tab:
                        probs = results[key]
                        if not probs:
                            st.write("—")
                            continue
                        df_round = pd.DataFrame(
                            [{"Team": t, "Probability": p} for t, p in probs.items()]
                        )
                        st.dataframe(
                            df_round,
                            hide_index=True,
                            use_container_width=True,
                            column_config={
                                "Probability": st.column_config.ProgressColumn(
                                    "Probability",
                                    format="%.1f%%",
                                    min_value=0,
                                    max_value=1,
                                ),
                            },
                        )

            except Exception as e:
                st.error(f"Simulation failed: {e}")
                st.exception(e)


# ── Mode 4: About / Model Performance ────────────────────────────────────────

elif mode == "Model & Project":
    st.title("📊 Model & Project")

    tab_overview, tab_plots, tab_arch = st.tabs(
        ["Overview", "Diagnostic Plots", "Architecture"]
    )

    with tab_overview:
        st.markdown(
            """
            ### NBA Game Outcome Prediction

            **Task:** Binary classification — predict whether the home team wins.

            **Data:** 3 NBA seasons (~3,600 regular-season games) from `nba_api`.

            **Features (~34 total):**
            - Rolling 5/10-game team stats: win %, points scored/allowed, net rating
            - Rest days + back-to-back flag
            - ELO ratings (FiveThirtyEight-style with home advantage and MOV)
            - Star-player availability (top-5 by minutes, injury proxy)

            **Model:** 3-layer dense neural network (128 → 64 → 32 → sigmoid)
            with BatchNorm, Dropout, and EarlyStopping.

            **Baseline:** Logistic Regression (for comparison).

            **Split:** Chronological train/val/test (80/15/20 by date).
            """
        )

    with tab_plots:
        plot_dir = "outputs/plots"
        plots = [
            ("training_curves.png",   "Training Curves",
             "Loss & AUC over epochs. Look for train/val gap to detect over-fitting."),
            ("roc_curve.png",         "ROC Curve",
             "NN vs. Logistic Regression vs. random. Higher curve = better model."),
            ("confusion_matrix.png",  "Confusion Matrix",
             "True vs. predicted labels. Diagonal = correct."),
            ("calibration.png",       "Calibration Curve",
             "Are the predicted probabilities trustworthy? Diagonal = perfect."),
            ("feature_importance.png","Feature Importance (LR proxy)",
             "Which features matter most. Based on Logistic Regression coefficients."),
        ]
        for filename, title, caption in plots:
            path = os.path.join(plot_dir, filename)
            if os.path.exists(path):
                st.subheader(title)
                st.image(path, caption=caption)
            else:
                st.info(f"{title} not yet generated. Run `python main.py` first.")

    with tab_arch:
        st.markdown(
            """
            ### Network Architecture

            ```
            Input (~34 features)
                │
            Dense(128) → BatchNorm → ReLU → Dropout(0.3)
                │
            Dense(64)  → BatchNorm → ReLU → Dropout(0.3)
                │
            Dense(32)  →           ReLU → Dropout(0.15)
                │
            Dense(1)   → Sigmoid  →  P(home win)
            ```

            **Loss:** Binary cross-entropy
            **Optimizer:** Adam (lr = 1e-3)
            **Regularization:** Dropout + BatchNorm + EarlyStopping (patience=15)

            ### Why this architecture
            - **Pyramid (128→64→32):** progressive abstraction; fewer params than uniform width
            - **BatchNorm:** stabilizes training; tolerates wider learning rates
            - **Dropout:** essential — we have ~3,500 samples vs ~15,600 parameters
            - **Sigmoid:** probabilistic output, pairs nicely with cross-entropy

            See [`TUTORIAL.md`](TUTORIAL.md) for a deep dive on each component.
            """
        )

# ── Footer ────────────────────────────────────────────────────────────────────

st.sidebar.divider()
st.sidebar.caption("Built with TensorFlow + Streamlit")
