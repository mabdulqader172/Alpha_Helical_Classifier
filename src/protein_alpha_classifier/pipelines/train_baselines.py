"""Fit and evaluate baseline models; log all runs to MLflow.

Evaluation order per docs/evaluation-protocol.md:
  1. 5-fold StratifiedGroupKFold CV on the train split (model selection / reporting)
  2. Refit on the full train split; evaluate on the val split
  3. Test split is NOT touched here — reserved for final comparison only.

Primary metric: AUPRC (alpha class is ~23% of data, so it is uncommon).
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from protein_alpha_classifier._utils import configure_logging
from protein_alpha_classifier.models.baselines import BASELINE_CONFIGS, DECISION_THRESHOLD

PRIMARY_METRIC = "auprc"
CV_FOLDS = 5
LABELING_RULE = "CL=1000000 -> alpha (label=1); all other CL -> not_alpha (label=0)"


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    report = classification_report(y_true, y_pred, labels=[0, 1],
                                   target_names=["not_alpha", "alpha"],
                                   output_dict=True, zero_division=0)
    return {
        "auprc": average_precision_score(y_true, y_prob),
        "auroc": roc_auc_score(y_true, y_prob),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "precision_alpha": report["alpha"]["precision"],
        "recall_alpha": report["alpha"]["recall"],
        "f1_alpha": report["alpha"]["f1-score"],
        "precision_not_alpha": report["not_alpha"]["precision"],
        "recall_not_alpha": report["not_alpha"]["recall"],
        "f1_not_alpha": report["not_alpha"]["f1-score"],
    }


def _cv_metrics(model, X: list, y: np.ndarray, groups: np.ndarray, n_splits: int) -> dict:
    """Return per-fold and mean/std metrics from StratifiedGroupKFold CV."""
    skf = StratifiedGroupKFold(n_splits=n_splits)
    fold_metrics: list[dict] = []
    for train_idx, val_idx in skf.split(X, y, groups):
        X_tr = [X[i] for i in train_idx]
        X_va = [X[i] for i in val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_va)[:, 1]
        fold_metrics.append(_compute_metrics(y_va, y_prob, DECISION_THRESHOLD))

    summary: dict = {}
    for key in fold_metrics[0]:
        vals = np.array([fm[key] for fm in fold_metrics])
        summary[f"cv_mean_{key}"] = float(vals.mean())
        summary[f"cv_std_{key}"] = float(vals.std())
    summary["cv_fold_auprc"] = [fm["auprc"] for fm in fold_metrics]
    return summary


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _pr_figure(y_true, y_prob, name: str) -> plt.Figure:
    fig, ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax, name=name)
    ax.set_title(f"PR curve — {name}")
    return fig


def _roc_figure(y_true, y_prob, name: str) -> plt.Figure:
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax, name=name)
    ax.set_title(f"ROC curve — {name}")
    return fig


def _cm_figure(y_true, y_prob, name: str, threshold: float) -> plt.Figure:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(cm, display_labels=["not_alpha", "alpha"])
    disp.plot(ax=ax)
    ax.set_title(f"Confusion matrix (thr={threshold}) — {name}")
    return fig


def _calibration_figure(y_true, y_prob, name: str) -> plt.Figure:
    fig, ax = plt.subplots()
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", label="perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction positive")
    ax.set_title(f"Calibration — {name}")
    ax.legend()
    return fig


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "dirty_worktree"


def _package_versions() -> dict:
    import sklearn, mlflow as mf, pandas as pd_, Bio
    return {
        "sklearn": sklearn.__version__,
        "mlflow": mf.__version__,
        "pandas": pd_.__version__,
        "biopython": Bio.__version__,
        "python": sys.version,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--val", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--source-manifest", type=Path, default=None,
                        help="data/raw/scop/source_manifest.json for full provenance")
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=DECISION_THRESHOLD)
    parser.add_argument("--mlflow-experiment", default="protein-alpha-classifier-baselines")
    args = parser.parse_args(argv)

    log = configure_logging(__name__)

    train_df = pd.read_parquet(args.train)
    val_df = pd.read_parquet(args.val)
    log.info("Train: %d sequences  Val: %d sequences", len(train_df), len(val_df))

    X_train: list[str] = train_df["sequence"].tolist()
    y_train = train_df["label"].to_numpy()
    groups_train = train_df["cluster_id"].to_numpy()
    X_val: list[str] = val_df["sequence"].tolist()
    y_val = val_df["label"].to_numpy()

    dataset_version = train_df["dataset_version"].iloc[0]
    split_manifest = json.loads(args.split_manifest.read_text())
    source_manifest = (
        json.loads(args.source_manifest.read_text()) if args.source_manifest else {}
    )
    git_commit = _git_commit()
    pkg_versions = _package_versions()

    mlflow.set_experiment(args.mlflow_experiment)
    log.info("MLflow experiment: %s", args.mlflow_experiment)

    for cfg in BASELINE_CONFIGS:
        model_name: str = cfg["name"]
        log.info("--- Fitting %s ---", model_name)

        model = cfg["factory"](seed=args.seed, **cfg["params"]) if cfg["params"] else cfg["factory"]()

        # CV on train
        log.info("  CV (%d folds)...", args.cv_folds)
        cv_summary = _cv_metrics(model, X_train, y_train, groups_train, args.cv_folds)
        log.info("  cv_mean_auprc=%.4f ± %.4f", cv_summary["cv_mean_auprc"], cv_summary["cv_std_auprc"])

        # Refit on full train
        model.fit(X_train, y_train)
        y_prob_train = model.predict_proba(X_train)[:, 1]
        y_prob_val = model.predict_proba(X_val)[:, 1]

        train_metrics = _compute_metrics(y_train, y_prob_train, args.threshold)
        val_metrics = _compute_metrics(y_val, y_prob_val, args.threshold)
        log.info("  val_auprc=%.4f  val_auroc=%.4f", val_metrics["auprc"], val_metrics["auroc"])

        with mlflow.start_run(run_name=model_name):
            # --- Parameters ---
            mlflow.log_params({
                "model_name": model_name,
                "feature_type": cfg["feature_type"],
                "classifier": type(model).__name__ if not hasattr(model, "steps") else type(model[-1]).__name__,
                "C": cfg["params"].get("C", "n/a"),
                "class_weight": cfg["params"].get("class_weight", "none"),
                "cv_folds": args.cv_folds,
                "threshold": args.threshold,
                "seed": args.seed,
                "primary_metric": PRIMARY_METRIC,
                "labeling_rule": LABELING_RULE,
                "dataset_version": dataset_version,
                "split_seed": split_manifest.get("seed"),
                "train_n_sequences": len(train_df),
                "train_n_alpha": int(y_train.sum()),
                "train_n_not_alpha": int((y_train == 0).sum()),
                "val_n_sequences": len(val_df),
                "val_n_alpha": int(y_val.sum()),
                "git_commit": git_commit,
            })
            mlflow.log_params({f"pkg_{k}": v for k, v in pkg_versions.items()})

            # --- CV metrics ---
            mlflow.log_metrics({k: v for k, v in cv_summary.items() if not isinstance(v, list)})

            # --- Train metrics ---
            mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})

            # --- Val metrics ---
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

            # --- Figures ---
            for fig, fname in [
                (_pr_figure(y_val, y_prob_val, model_name), "val_pr_curve.png"),
                (_roc_figure(y_val, y_prob_val, model_name), "val_roc_curve.png"),
                (_cm_figure(y_val, y_prob_val, model_name, args.threshold), "val_confusion_matrix.png"),
                (_calibration_figure(y_val, y_prob_val, model_name), "val_calibration.png"),
            ]:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    fig.savefig(tmp.name, bbox_inches="tight")
                    mlflow.log_artifact(tmp.name, artifact_path=fname.replace(".png", ""))
                plt.close(fig)

            # --- Val predictions artifact ---
            preds_df = pd.DataFrame({
                "sequence_id": val_df["sequence_id"].values,
                "y_true": y_val,
                "y_prob_alpha": y_prob_val,
                "y_pred": (y_prob_val >= args.threshold).astype(int),
            })
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                preds_df.to_parquet(tmp.name, index=False)
                mlflow.log_artifact(tmp.name, artifact_path="val_predictions")

            # --- Manifests ---
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
                json.dump(split_manifest, tmp, indent=2)
                mlflow.log_artifact(tmp.name, artifact_path="split_manifest")
            if source_manifest:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
                    json.dump(source_manifest, tmp, indent=2)
                    mlflow.log_artifact(tmp.name, artifact_path="source_manifest")

            # --- Model artifact ---
            mlflow.sklearn.log_model(
                model,
                name="model",
                skops_trusted_types=[
                    "protein_alpha_classifier.features.composition.AACompositionTransformer",
                    "protein_alpha_classifier.features.composition.DipeptideCompositionTransformer",
                    "sklearn.linear_model._logistic.LogisticRegression",
                    "sklearn.preprocessing._data.StandardScaler",
                    "sklearn.pipeline.Pipeline",
                ],
            )

        log.info("  Run logged to MLflow.")

    log.info("All baselines complete. View runs: mlflow ui --backend-store-uri mlruns/")


if __name__ == "__main__":
    main()
