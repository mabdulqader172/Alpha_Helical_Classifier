"""Baseline sklearn pipelines per docs/evaluation-protocol.md."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from protein_alpha_classifier.features.composition import (
    AACompositionTransformer,
    DipeptideCompositionTransformer,
)

# Pre-specified threshold declared before any evaluation.
DECISION_THRESHOLD = 0.5


def composition_lr_pipeline(C: float = 1.0, seed: int = 42) -> Pipeline:
    """L2-regularised logistic regression on 20-dim AA fractional composition."""
    return Pipeline([
        ("features", AACompositionTransformer()),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=2000, random_state=seed, solver="lbfgs")),
    ])


def dipeptide_lr_pipeline(C: float = 1.0, seed: int = 42) -> Pipeline:
    """L2-regularised logistic regression on 400-dim dipeptide fractional composition."""
    return Pipeline([
        ("features", DipeptideCompositionTransformer()),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=2000, random_state=seed, solver="lbfgs")),
    ])


BASELINE_CONFIGS: list[dict] = [
    {
        "name": "composition_lr",
        "feature_type": "aa_composition",
        "factory": composition_lr_pipeline,
        "params": {"C": 1.0},
    },
    {
        "name": "dipeptide_lr",
        "feature_type": "dipeptide_composition",
        "factory": dipeptide_lr_pipeline,
        "params": {"C": 1.0},
    },
]
