"""
Evaluation metrics and visualisations.

Plots saved to outputs/plots/:
  1. training_curves.png   – loss & AUC over epochs
  2. confusion_matrix.png  – NN predictions vs true labels
  3. roc_curve.png         – ROC curve: NN vs baseline vs random
  4. calibration.png       – how well predicted probabilities match actual rates
  5. feature_importance.png – baseline LR coefficients (proxy for feature impact)

Metrics printed to console:
  Accuracy, Precision, Recall, F1-score, ROC-AUC  (for both NN and baseline)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless backend — works on any machine, no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
    ConfusionMatrixDisplay,
)
from sklearn.calibration import calibration_curve

PLOT_DIR = "outputs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

STYLE = {
    "nn_color":       "#1f77b4",
    "baseline_color": "#ff7f0e",
    "random_color":   "#999999",
}


# ── Core metric reporting ─────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_prob),
    }


def print_metrics(metrics_nn: dict, metrics_lr: dict) -> None:
    header = f"{'Metric':<12} {'Neural Net':>12} {'Logistic Reg':>14}"
    print("\n" + "═" * len(header))
    print(header)
    print("─" * len(header))
    for key in metrics_nn:
        print(f"{key:<12} {metrics_nn[key]:>12.4f} {metrics_lr[key]:>14.4f}")
    print("═" * len(header))


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_training_curves(history, save: bool = True) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history.history["loss"],     label="Train",      color=STYLE["nn_color"])
    axes[0].plot(history.history["val_loss"], label="Validation", color=STYLE["baseline_color"])
    axes[0].set_title("Binary Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # AUC
    axes[1].plot(history.history["auc"],     label="Train",      color=STYLE["nn_color"])
    axes[1].plot(history.history["val_auc"], label="Validation", color=STYLE["baseline_color"])
    axes[1].set_title("ROC-AUC Score")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].legend()

    fig.suptitle("Neural Network Training Curves", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save:
        path = os.path.join(PLOT_DIR, "training_curves.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray, y_prob: np.ndarray, save: bool = True) -> None:
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Away Win", "Home Win"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Neural Network – Confusion Matrix", fontweight="bold")
    fig.tight_layout()

    if save:
        path = os.path.join(PLOT_DIR, "confusion_matrix.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob_nn: np.ndarray,
    y_prob_lr: np.ndarray,
    save: bool = True,
) -> None:
    fpr_nn, tpr_nn, _ = roc_curve(y_true, y_prob_nn)
    fpr_lr, tpr_lr, _ = roc_curve(y_true, y_prob_lr)

    auc_nn = roc_auc_score(y_true, y_prob_nn)
    auc_lr = roc_auc_score(y_true, y_prob_lr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_nn, tpr_nn, label=f"Neural Net (AUC = {auc_nn:.3f})", color=STYLE["nn_color"], lw=2)
    ax.plot(fpr_lr, tpr_lr, label=f"Logistic Reg (AUC = {auc_lr:.3f})", color=STYLE["baseline_color"], lw=2)
    ax.plot([0, 1], [0, 1], "--", label="Random (AUC = 0.500)", color=STYLE["random_color"])

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save:
        path = os.path.join(PLOT_DIR, "roc_curve.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def plot_calibration(
    y_true: np.ndarray,
    y_prob_nn: np.ndarray,
    y_prob_lr: np.ndarray,
    save: bool = True,
) -> None:
    """
    Calibration plot: if a model says "70% chance home wins", does it really
    win ~70% of the time? Well-calibrated models follow the diagonal.
    """
    prob_true_nn, prob_pred_nn = calibration_curve(y_true, y_prob_nn, n_bins=10, strategy="quantile")
    prob_true_lr, prob_pred_lr = calibration_curve(y_true, y_prob_lr, n_bins=10, strategy="quantile")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color=STYLE["random_color"], label="Perfect calibration")
    ax.plot(prob_pred_nn, prob_true_nn, "o-", color=STYLE["nn_color"],       label="Neural Net")
    ax.plot(prob_pred_lr, prob_true_lr, "s-", color=STYLE["baseline_color"], label="Logistic Reg")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Actual Win Rate)")
    ax.set_title("Calibration Curve", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save:
        path = os.path.join(PLOT_DIR, "calibration.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def plot_feature_importance(baseline, feature_cols: list[str], top_n: int = 20, save: bool = True) -> None:
    """
    Logistic regression coefficients as a proxy for feature importance.
    Positive coefficient → feature pushes toward home team winning.
    """
    coefs = baseline.coef_[0]
    importance = pd.Series(coefs, index=feature_cols).abs().sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [STYLE["nn_color"] if c >= 0 else STYLE["baseline_color"]
              for c in coefs[importance.index.map(lambda x: feature_cols.index(x))]]

    importance.sort_values().plot(kind="barh", ax=ax, color=colors[::-1])
    ax.set_title(f"Top {top_n} Feature Importances (LR |coefficient|)", fontweight="bold")
    ax.set_xlabel("|Coefficient|")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    if save:
        path = os.path.join(PLOT_DIR, "feature_importance.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


# ── Full evaluation run ───────────────────────────────────────────────────────

def run_evaluation(results: dict) -> None:
    """
    Accepts the dict returned by train.run_training() and produces all plots
    plus a printed metrics table.
    """
    nn      = results["nn_model"]
    baseline = results["baseline"]
    history  = results["history"]
    X_test   = results["X_test"]
    y_test   = results["y_test"]
    feat_cols = results["feature_cols"]

    # Predictions
    y_prob_nn = nn.predict(X_test, verbose=0).flatten()
    y_prob_lr = baseline.predict_proba(X_test)[:, 1]

    # Metrics
    metrics_nn = compute_metrics(y_test, y_prob_nn)
    metrics_lr = compute_metrics(y_test, y_prob_lr)
    print_metrics(metrics_nn, metrics_lr)

    # Plots
    print("\nGenerating plots …")
    plot_training_curves(history)
    plot_confusion_matrix(y_test, y_prob_nn)
    plot_roc_curve(y_test, y_prob_nn, y_prob_lr)
    plot_calibration(y_test, y_prob_nn, y_prob_lr)
    plot_feature_importance(baseline, feat_cols)

    print(f"\nAll plots saved to '{PLOT_DIR}/'")
    return {"nn": metrics_nn, "baseline": metrics_lr}
