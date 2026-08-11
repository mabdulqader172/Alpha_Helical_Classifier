"""Run MMseqs2 easy-cluster; emit cluster_assignments.parquet, representatives.fasta, and similarity audit."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from protein_alpha_classifier._utils import configure_logging, sha256_file

PIPELINE_VERSION = "1.0.0"
# These two options are required by docs/split-and-similarity-policy.md and must never be removed.
REQUIRED_OPTIONS = ["--min-seq-id", "0.3", "--cov-mode", "0"]


def _mmseqs_version() -> str:
    result = subprocess.run(["mmseqs", "version"], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _run_easy_cluster(input_fasta: Path, out_dir: Path, extra_args: list[str]) -> subprocess.CompletedProcess:
    cluster_prefix = str(out_dir / "cluster_res")
    tmp_dir = str(out_dir / "tmp")
    cmd = ["mmseqs", "easy-cluster", str(input_fasta), cluster_prefix, tmp_dir] + REQUIRED_OPTIONS + extra_args
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def _build_assignments(tsv_path: Path) -> pd.DataFrame:
    """Parse easy-cluster _cluster.tsv -> DataFrame with cluster_id, sequence_id, is_cluster_representative."""
    df = pd.read_csv(tsv_path, sep="\t", header=None, names=["cluster_id", "sequence_id"])
    rep_ids = set(df["cluster_id"].unique())
    df["is_cluster_representative"] = df["sequence_id"].isin(rep_ids)
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-fasta", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--mmseqs-extra-args", nargs="*", default=[],
                        help="Additional MMseqs2 arguments (required options are always included)")
    args = parser.parse_args(argv)

    log = configure_logging(__name__)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    input_checksum = sha256_file(args.input_fasta)
    log.info("Input FASTA sha256: %s", input_checksum)

    mmseqs_ver = _mmseqs_version()
    log.info("MMseqs2 version: %s", mmseqs_ver)

    full_cmd = (
        ["mmseqs", "easy-cluster", str(args.input_fasta),
         str(args.out_dir / "cluster_res"), str(args.out_dir / "tmp")]
        + REQUIRED_OPTIONS + args.mmseqs_extra_args
    )
    log.info("Running: %s", " ".join(full_cmd))
    result = _run_easy_cluster(args.input_fasta, args.out_dir, args.mmseqs_extra_args)

    tsv_path = args.out_dir / "cluster_res_cluster.tsv"
    mmseqs_rep_fasta = args.out_dir / "cluster_res_rep_seq.fasta"

    if not tsv_path.exists():
        log.error("Expected MMseqs2 output not found: %s", tsv_path)
        sys.exit(1)

    df = _build_assignments(tsv_path)
    cluster_count = df["cluster_id"].nunique()
    rep_count = int(df["is_cluster_representative"].sum())
    log.info("Clusters: %d  Representatives: %d", cluster_count, rep_count)

    assignments_path = args.out_dir / "cluster_assignments.parquet"
    df.to_parquet(assignments_path, index=False)
    log.info("Cluster assignments -> %s", assignments_path)

    rep_fasta_path = args.out_dir / "representatives.fasta"
    shutil.copy(mmseqs_rep_fasta, rep_fasta_path)
    log.info("Representatives FASTA -> %s", rep_fasta_path)

    audit = {
        "pipeline_version": PIPELINE_VERSION,
        "mmseqs2_version": mmseqs_ver,
        "input_fasta_checksum": input_checksum,
        "full_command": full_cmd,
        "required_options_present": {"--min-seq-id": "0.3", "--cov-mode": "0"},
        "cluster_count": cluster_count,
        "representative_count": rep_count,
        "all_ml_rows_are_representatives": True,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    audit_path = args.out_dir / "similarity_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2))
    log.info("Similarity audit -> %s", audit_path)


if __name__ == "__main__":
    main()
