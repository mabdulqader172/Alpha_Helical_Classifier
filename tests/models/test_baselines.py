import numpy as np
import pytest

from protein_alpha_classifier.models.baselines import (
    BASELINE_CONFIGS,
    DECISION_THRESHOLD,
    composition_lr_pipeline,
    dipeptide_lr_pipeline,
)


def _make_data(n_alpha=40, n_not_alpha=160, seq_len=30, seed=0):
    rng = np.random.default_rng(seed)
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    alpha_seqs = ["".join(rng.choice(aas, seq_len)) for _ in range(n_alpha)]
    other_seqs = ["".join(rng.choice(aas, seq_len)) for _ in range(n_not_alpha)]
    X = alpha_seqs + other_seqs
    y = np.array([1] * n_alpha + [0] * n_not_alpha)
    perm = rng.permutation(len(y))
    return [X[i] for i in perm], y[perm]


# ---------------------------------------------------------------------------
# Composition LR baseline
# ---------------------------------------------------------------------------

def test_composition_lr_fits_and_predicts():
    X, y = _make_data()
    model = composition_lr_pipeline()
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_composition_lr_probabilities_in_range():
    X, y = _make_data()
    model = composition_lr_pipeline()
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_composition_lr_deterministic():
    X, y = _make_data()
    m1 = composition_lr_pipeline(seed=7)
    m2 = composition_lr_pipeline(seed=7)
    m1.fit(X, y)
    m2.fit(X, y)
    np.testing.assert_allclose(m1.predict_proba(X), m2.predict_proba(X))


# ---------------------------------------------------------------------------
# Dipeptide LR baseline
# ---------------------------------------------------------------------------

def test_dipeptide_lr_fits_and_predicts():
    X, y = _make_data()
    model = dipeptide_lr_pipeline()
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_dipeptide_lr_probabilities_in_range():
    X, y = _make_data()
    model = dipeptide_lr_pipeline()
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_dipeptide_lr_deterministic():
    X, y = _make_data()
    m1 = dipeptide_lr_pipeline(seed=7)
    m2 = dipeptide_lr_pipeline(seed=7)
    m1.fit(X, y)
    m2.fit(X, y)
    np.testing.assert_allclose(m1.predict_proba(X), m2.predict_proba(X))


# ---------------------------------------------------------------------------
# BASELINE_CONFIGS registry
# ---------------------------------------------------------------------------

def test_baseline_configs_has_two_entries():
    assert len(BASELINE_CONFIGS) == 2


def test_baseline_configs_names():
    names = {cfg["name"] for cfg in BASELINE_CONFIGS}
    assert names == {"composition_lr", "dipeptide_lr"}


def test_decision_threshold_is_declared():
    assert 0 < DECISION_THRESHOLD < 1
