"""Split ML-ready dataset into train / val / test partitions.

Splitting is cluster-group-aware (a cluster appears in exactly one partition)
and stratified by label. The test set is reserved before any model selection
per docs/evaluation-protocol.md.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from protein_alpha_classifier._utils import configure_logging

PIPELINE_VERSION = "1.0.0"


def stratified_cluster_split(
    cluster_ids: np.ndarray,
    labels: np.ndarray,
    test_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boolean masks (train, val, test) with group-awareness and stratification.

    Each cluster_id appears in exactly one partition. Stratification is done
    within each class before merging, so label prevalence is preserved.
    """
    rng = np.random.default_rng(seed)
    train_ids: list = []
    val_ids: list = []
    test_ids: list = []

    for cls in np.unique(labels):
        cls_mask = labels == cls
        cls_clusters = cluster_ids[cls_mask]
        n = len(cls_clusters)
        shuffled = rng.permutation(cls_clusters)

        n_test = round(n * test_frac)
        n_val = round(n * val_frac)

        test_ids.extend(shuffled[:n_test])
        val_ids.extend(shuffled[n_test : n_test + n_val])
        train_ids.extend(shuffled[n_test + n_val :])

    test_set = set(test_ids)
    val_set = set(val_ids)
    test_mask = np.isin(cluster_ids, list(test_set))
    val_mask = np.isin(cluster_ids, list(val_set))
    train_mask = ~test_mask & ~val_mask
    return train_mask, val_mask, test_mask


def _partition_summary(df: pd.DataFrame, name: str) -> dict:
    alpha = int((df["label"] == 1).sum())
    not_alpha = int((df["label"] == 0).sum())
    return {
        "partition": name,
        "n_sequences": len(df),
        "n_clusters": df["cluster_id"].nunique(),
        "n_alpha": alpha,
        "n_not_alpha": not_alpha,
        "alpha_frac": round(alpha / len(df), 4) if len(df) else 0.0,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    log = configure_logging(__name__)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading dataset from %s", args.dataset)
    df = pd.read_parquet(args.dataset)
    log.info("Total sequences: %d  clusters: %d", len(df), df["cluster_id"].nunique())

    assert df["is_cluster_representative"].all(), "Dataset contains non-representative rows"
    assert df["sequence_id"].is_unique, "Duplicate sequence_id in dataset"

    cluster_ids = df["cluster_id"].to_numpy()
    labels = df["label"].to_numpy()

    train_mask, val_mask, test_mask = stratified_cluster_split(
        cluster_ids, labels, args.test_frac, args.val_frac, args.seed
    )

    # Verify no cluster leaks across partitions
    train_clusters = set(df.loc[train_mask, "cluster_id"])
    val_clusters = set(df.loc[val_mask, "cluster_id"])
    test_clusters = set(df.loc[test_mask, "cluster_id"])
    assert not (train_clusters & val_clusters), "Cluster leak between train and val"
    assert not (train_clusters & test_clusters), "Cluster leak between train and test"
    assert not (val_clusters & test_clusters), "Cluster leak between val and test"
    assert len(train_clusters) + len(val_clusters) + len(test_clusters) == df["cluster_id"].nunique()

    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    train_df.to_parquet(args.out_dir / "train.parquet", index=False)
    val_df.to_parquet(args.out_dir / "val.parquet", index=False)
    test_df.to_parquet(args.out_dir / "test.parquet", index=False)

    summaries = [
        _partition_summary(train_df, "train"),
        _partition_summary(val_df, "val"),
        _partition_summary(test_df, "test"),
    ]

    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "seed": args.seed,
        "test_frac": args.test_frac,
        "val_frac": args.val_frac,
        "source_dataset": str(args.dataset),
        "run_at": datetime.now(timezone.utc).isoformat(),
        "partitions": summaries,
    }
    manifest_path = args.out_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    for s in summaries:
        log.info(
            "%-6s  sequences=%d  clusters=%d  alpha=%d (%.1f%%)",
            s["partition"], s["n_sequences"], s["n_clusters"],
            s["n_alpha"], 100 * s["alpha_frac"],
        )
    log.info("Split manifest -> %s", manifest_path)


if __name__ == "__main__":
    main()
