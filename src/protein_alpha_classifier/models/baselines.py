"""Baseline sklearn pipelines required by docs/evaluation-protocol.md."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from protein_alpha_classifier.features.composition import (
    AACompositionTransformer,
    LengthTransformer,
)

# Pre-specified threshold declared before any evaluation.
DECISION_THRESHOLD = 0.5


def majority_class_pipeline() -> DummyClassifier:
    """Always predict the majority class (not_alpha). No sequence features used."""
    return DummyClassifier(strategy="most_frequent")


def length_lr_pipeline(C: float = 1.0, seed: int = 42) -> Pipeline:
    """Logistic regression on sequence length only."""
    return Pipeline([
        ("features", LengthTransformer()),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=2000, random_state=seed, solver="lbfgs")),
    ])


def composition_lr_pipeline(C: float = 1.0, seed: int = 42) -> Pipeline:
    """L2-regularised logistic regression on 20-dim AA fractional composition."""
    return Pipeline([
        ("features", AACompositionTransformer()),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=2000, random_state=seed, solver="lbfgs")),
    ])


BASELINE_CONFIGS: list[dict] = [
    {
        "name": "majority_class",
        "feature_type": "none",
        "factory": majority_class_pipeline,
        "params": {},
    },
    {
        "name": "length_lr",
        "feature_type": "length",
        "factory": length_lr_pipeline,
        "params": {"C": 1.0},
    },
    {
        "name": "composition_lr",
        "feature_type": "aa_composition",
        "factory": composition_lr_pipeline,
        "params": {"C": 1.0},
    },
]
