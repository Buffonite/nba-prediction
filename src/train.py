"""
Training pipeline: splits data, scales features, trains NN + baseline.

Temporal split strategy
───────────────────────
We split the data CHRONOLOGICALLY (not randomly) because NBA games are
time-ordered. Random splitting would leak future information into training —
a common mistake that inflates metrics but fails in real deployment.

  [─────── train ────────][── val ──][── test ──]
        (earliest)                    (most recent)

Usage:
    from src.train import run_training
    results = run_training(features_df)
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import config
from src.model import build_nn, build_baseline, get_callbacks
from src.preprocessing import get_feature_columns

# Saved alongside model + scaler so predict.py knows the exact column order
FEATURE_COLS_PATH    = "outputs/models/feature_cols.json"
CALIBRATION_PATH     = "outputs/models/calibration.json"
XGB_MODEL_PATH       = "outputs/models/xgb.json"


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """
    Fit a single temperature scalar T to minimise binary cross-entropy on
    validation data:  p_calibrated = sigmoid(logit / T).

    T > 1 makes the model LESS confident (shrinks probabilities toward 0.5).
    Typical NN over-confidence problem maps to T in [1.5, 3.0].
    """
    def neg_log_likelihood(T: float) -> float:
        T = max(T, 0.05)
        scaled_logits = logits / T
        # Numerically stable log-loss
        log_p = -np.logaddexp(0, -scaled_logits)
        log_1mp = -np.logaddexp(0, scaled_logits)
        return -np.mean(y_true * log_p + (1 - y_true) * log_1mp)

    # Constrain T >= 1.0: only allow SOFTENING, never sharpening.
    # Validation NLL may have small T as optimum, but extreme predictions
    # (especially compounded across a 7-game playoff series) need to be
    # damped — not amplified.
    result = minimize_scalar(neg_log_likelihood, bounds=(1.0, 10.0), method="bounded")
    return float(result.x)


def temporal_split(features: pd.DataFrame):
    """
    Split data chronologically to avoid data leakage.
    Returns (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    features = features.sort_values("GAME_DATE").reset_index(drop=True)

    feature_cols = get_feature_columns(features)
    X = features[feature_cols].values
    y = features["home_win"].values

    n = len(X)
    test_idx  = int(n * (1 - config.TEST_SIZE))
    val_idx   = int(test_idx * (1 - config.VAL_SIZE))

    X_train, y_train = X[:val_idx],              y[:val_idx]
    X_val,   y_val   = X[val_idx:test_idx],      y[val_idx:test_idx]
    X_test,  y_test  = X[test_idx:],             y[test_idx:]

    print(f"Split sizes → train: {len(X_train):,}  val: {len(X_val):,}  test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    """
    StandardScaler: zero mean, unit variance.
    Fit ONLY on training data, then apply to val/test (prevents leakage).
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    os.makedirs(os.path.dirname(config.SCALER_SAVE_PATH), exist_ok=True)
    with open(config.SCALER_SAVE_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved → '{config.SCALER_SAVE_PATH}'")

    return X_train_s, X_val_s, X_test_s, scaler


def run_training(features: pd.DataFrame) -> dict:
    """
    Full training run. Returns a dict with trained models and test data.

    Dict keys:
        nn_model    : trained Keras model
        baseline    : trained LogisticRegression
        history     : Keras training History object
        X_test      : scaled test features
        y_test      : test labels
        feature_cols: list of feature names
    """
    feature_cols = get_feature_columns(features)
    print(f"\nInput features ({len(feature_cols)}): {feature_cols[:5]} … +{len(feature_cols)-5} more")

    # Persist column order for predict.py to use later
    os.makedirs(os.path.dirname(FEATURE_COLS_PATH), exist_ok=True)
    with open(FEATURE_COLS_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"Feature columns saved → '{FEATURE_COLS_PATH}'")

    X_train, X_val, X_test, y_train, y_val, y_test = temporal_split(features)
    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    # ── Neural network ────────────────────────────────────────────────────────
    print("\n── Training neural network ──")
    nn = build_nn(input_dim=X_train_s.shape[1])
    nn.summary()

    history = nn.fit(
        X_train_s, y_train,
        validation_data=(X_val_s, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=get_callbacks(),
        verbose=1,
    )
    print(f"Best epoch: {np.argmax(history.history['val_auc']) + 1}")

    # ── Logistic regression baseline ─────────────────────────────────────────
    print("\n── Training logistic regression baseline ──")
    baseline = build_baseline()
    X_trainval = np.vstack([X_train_s, X_val_s])
    y_trainval = np.concatenate([y_train, y_val])
    baseline.fit(X_trainval, y_trainval)
    print(f"Baseline train accuracy: {baseline.score(X_trainval, y_trainval):.3f}")

    # ── Temperature scaling (v2: fix over-confidence) ─────────────────────────
    temperature = 1.0
    if config.USE_TEMPERATURE_SCALING:
        print("\n── Fitting temperature scaling on validation set ──")
        val_probs = nn.predict(X_val_s, verbose=0).flatten()
        val_probs = np.clip(val_probs, 1e-6, 1 - 1e-6)
        val_logits = np.log(val_probs / (1 - val_probs))
        temperature = fit_temperature(val_logits, y_val)
        print(f"  Fitted temperature: T = {temperature:.3f}  "
              f"(T > 1 → less confident; reduces extreme probabilities)")
        with open(CALIBRATION_PATH, "w") as f:
            json.dump({"temperature": temperature}, f)
        print(f"  Saved calibration → '{CALIBRATION_PATH}'")

    # ── XGBoost ensemble (v2: tabular data sweet spot) ────────────────────────
    xgb_model = None
    if config.USE_XGBOOST_ENSEMBLE:
        print("\n── Training XGBoost ensemble ──")
        import xgboost as xgb
        xgb_model = xgb.XGBClassifier(
            n_estimators     = 300,
            max_depth        = 4,
            learning_rate    = 0.05,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            eval_metric      = "auc",
            random_state     = config.RANDOM_SEED,
            tree_method      = "hist",
            early_stopping_rounds = 25,
        )
        xgb_model.fit(
            X_train_s, y_train,
            eval_set=[(X_val_s, y_val)],
            verbose=False,
        )
        best_iter = xgb_model.best_iteration
        val_auc_xgb = float(np.mean(xgb_model.predict_proba(X_val_s)[:, 1] >= 0.5) == y_val) if False else None
        print(f"  XGBoost trained — best iteration: {best_iter}")
        xgb_model.save_model(XGB_MODEL_PATH)
        print(f"  Saved XGBoost → '{XGB_MODEL_PATH}'")

    return {
        "nn_model":     nn,
        "baseline":     baseline,
        "xgb_model":    xgb_model,
        "temperature":  temperature,
        "history":      history,
        "X_test":       X_test_s,
        "y_test":       y_test,
        "feature_cols": feature_cols,
        "scaler":       scaler,
    }
