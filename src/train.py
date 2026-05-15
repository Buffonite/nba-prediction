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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import config
from src.model import build_nn, build_baseline, get_callbacks
from src.preprocessing import get_feature_columns

# Saved alongside model + scaler so predict.py knows the exact column order
FEATURE_COLS_PATH = "outputs/models/feature_cols.json"


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
    # Combine train + val for the baseline (it doesn't use val for early stopping)
    X_trainval = np.vstack([X_train_s, X_val_s])
    y_trainval = np.concatenate([y_train, y_val])
    baseline.fit(X_trainval, y_trainval)
    print(f"Baseline train accuracy: {baseline.score(X_trainval, y_trainval):.3f}")

    return {
        "nn_model":     nn,
        "baseline":     baseline,
        "history":      history,
        "X_test":       X_test_s,
        "y_test":       y_test,
        "feature_cols": feature_cols,
        "scaler":       scaler,
    }
