"""Generate publication-quality 2×3 baseline comparison figure.

Layout:
  Row 0 — Amino-Acid Freq (composition_lr):  ROC | PR | Confusion matrix
  Row 1 — Dipeptide Freq  (dipeptide_lr):    ROC | PR | Confusion matrix
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

from protein_alpha_classifier._utils import configure_logging
from protein_alpha_classifier.models.baselines import BASELINE_CONFIGS, DECISION_THRESHOLD

# Display names aligned with user request
_ROW_LABELS = {
    "composition_lr": "Amino-Acid Freq",
    "dipeptide_lr": "Dipeptide Freq",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "legend.fontsize": 8,
})


def _draw_roc(ax, y_true, y_prob, row, auroc):
    RocCurveDisplay.from_predictions(
        y_true, y_prob, ax=ax, name=f"AUROC = {auroc:.3f}", color="#2166ac",
    )
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    if row == 0:
        ax.set_title("ROC Curve", fontweight="bold", pad=6)


def _draw_pr(ax, y_true, y_prob, row, auprc):
    prevalence = y_true.mean()
    PrecisionRecallDisplay.from_predictions(
        y_true, y_prob, ax=ax, name=f"AUPRC = {auprc:.3f}", color="#d6604d",
    )
    ax.axhline(prevalence, ls="--", color="grey", lw=0.8, label=f"No skill ({prevalence:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    if row == 0:
        ax.set_title("PR Curve", fontweight="bold", pad=6)


def _draw_cm(ax, y_true, y_prob, row, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    thresh = 0.5
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm_norm[i, j]:.2f}",
                ha="center", va="center", fontsize=9,
                color="white" if cm_norm[i, j] > thresh else "black",
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Not Alpha", "Alpha"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Not Alpha", "Alpha"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    # restore spines for the heatmap border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
    if row == 0:
        ax.set_title("Confusion Matrix", fontweight="bold", pad=6)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--val", required=True, type=Path)
    parser.add_argument("--out-dir", default="figures", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=DECISION_THRESHOLD)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)

    log = configure_logging(__name__)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(args.train)
    val_df = pd.read_parquet(args.val)
    X_train = train_df["sequence"].tolist()
    y_train = train_df["label"].to_numpy()
    X_val = val_df["sequence"].tolist()
    y_val = val_df["label"].to_numpy()
    log.info("Train: %d  Val: %d", len(train_df), len(val_df))

    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)

    for row, cfg in enumerate(BASELINE_CONFIGS):
        model_name = cfg["name"]
        row_label = _ROW_LABELS[model_name]
        log.info("Fitting %s (%s)...", model_name, row_label)

        model = cfg["factory"](seed=args.seed, **cfg["params"])
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]

        auroc = roc_auc_score(y_val, y_prob)
        auprc = average_precision_score(y_val, y_prob)
        log.info("  AUROC=%.4f  AUPRC=%.4f", auroc, auprc)

        _draw_roc(axes[row, 0], y_val, y_prob, row, auroc)
        _draw_pr(axes[row, 1], y_val, y_prob, row, auprc)
        _draw_cm(axes[row, 2], y_val, y_prob, row, args.threshold)

        # Bold row label on the y-axis of the leftmost panel
        axes[row, 0].set_ylabel(
            f"{row_label}\n\nTrue Positive Rate",
            fontweight="bold",
            multialignment="center",
        )

    for fmt in ("pdf", "png"):
        out = args.out_dir / f"baseline_comparison.{fmt}"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        log.info("Saved %s", out)

    plt.close(fig)


if __name__ == "__main__":
    main()
