import numpy as np
import pandas as pd
import pytest

from protein_alpha_classifier.pipelines.split_dataset import (
    stratified_cluster_split,
    _partition_summary,
)


def _make_clusters(n_alpha: int, n_not_alpha: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Build synthetic cluster_ids and labels for testing."""
    rng = np.random.default_rng(seed)
    labels = np.array([1] * n_alpha + [0] * n_not_alpha)
    cluster_ids = np.array([f"c{i:04d}" for i in range(len(labels))])
    perm = rng.permutation(len(labels))
    return cluster_ids[perm], labels[perm]


# ---------------------------------------------------------------------------
# No cluster leaks across partitions
# ---------------------------------------------------------------------------

def test_no_cluster_appears_in_two_partitions():
    cluster_ids, labels = _make_clusters(n_alpha=200, n_not_alpha=600)
    train_mask, val_mask, test_mask = stratified_cluster_split(
        cluster_ids, labels, test_frac=0.15, val_frac=0.15, seed=42
    )
    train_set = set(cluster_ids[train_mask])
    val_set = set(cluster_ids[val_mask])
    test_set = set(cluster_ids[test_mask])
    assert not (train_set & val_set)
    assert not (train_set & test_set)
    assert not (val_set & test_set)


def test_all_clusters_assigned():
    cluster_ids, labels = _make_clusters(n_alpha=200, n_not_alpha=600)
    train_mask, val_mask, test_mask = stratified_cluster_split(
        cluster_ids, labels, test_frac=0.15, val_frac=0.15, seed=42
    )
    assert (train_mask | val_mask | test_mask).all()


# ---------------------------------------------------------------------------
# Stratification: alpha fraction preserved across partitions
# ---------------------------------------------------------------------------

def test_alpha_fraction_approximately_preserved():
    cluster_ids, labels = _make_clusters(n_alpha=500, n_not_alpha=1500)
    overall_alpha_frac = labels.mean()

    train_mask, val_mask, test_mask = stratified_cluster_split(
        cluster_ids, labels, test_frac=0.15, val_frac=0.15, seed=42
    )
    for mask, name in [(train_mask, "train"), (val_mask, "val"), (test_mask, "test")]:
        frac = labels[mask].mean()
        assert abs(frac - overall_alpha_frac) < 0.04, (
            f"{name} alpha fraction {frac:.3f} too far from overall {overall_alpha_frac:.3f}"
        )


# ---------------------------------------------------------------------------
# Approximate size fractions
# ---------------------------------------------------------------------------

def test_partition_sizes_approximate_fractions():
    n = 1000
    cluster_ids, labels = _make_clusters(n_alpha=250, n_not_alpha=750)
    train_mask, val_mask, test_mask = stratified_cluster_split(
        cluster_ids, labels, test_frac=0.15, val_frac=0.15, seed=42
    )
    assert abs(test_mask.sum() / n - 0.15) < 0.03
    assert abs(val_mask.sum() / n - 0.15) < 0.03
    assert abs(train_mask.sum() / n - 0.70) < 0.03


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_gives_same_split():
    cluster_ids, labels = _make_clusters(n_alpha=200, n_not_alpha=600)
    masks_a = stratified_cluster_split(cluster_ids, labels, 0.15, 0.15, seed=7)
    masks_b = stratified_cluster_split(cluster_ids, labels, 0.15, 0.15, seed=7)
    for a, b in zip(masks_a, masks_b):
        np.testing.assert_array_equal(a, b)


def test_different_seeds_give_different_splits():
    cluster_ids, labels = _make_clusters(n_alpha=200, n_not_alpha=600)
    train_a, _, _ = stratified_cluster_split(cluster_ids, labels, 0.15, 0.15, seed=1)
    train_b, _, _ = stratified_cluster_split(cluster_ids, labels, 0.15, 0.15, seed=2)
    assert not np.array_equal(train_a, train_b)


# ---------------------------------------------------------------------------
# Partition summary helper
# ---------------------------------------------------------------------------

def test_partition_summary_counts():
    df = pd.DataFrame({
        "label": [1, 1, 0, 0, 0],
        "cluster_id": ["a", "b", "c", "d", "e"],
    })
    s = _partition_summary(df, "train")
    assert s["n_sequences"] == 5
    assert s["n_alpha"] == 2
    assert s["n_not_alpha"] == 3
    assert abs(s["alpha_frac"] - 0.4) < 1e-9
