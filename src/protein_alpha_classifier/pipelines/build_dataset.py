"""Join cluster representatives to annotations; write canonical ML-ready dataset.parquet."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from Bio import SeqIO

from protein_alpha_classifier._utils import configure_logging, sha256_file

PIPELINE_VERSION = "1.0.0"
CL_ALPHA = "1000000"
CL_RE = re.compile(r"CL=(\d+)")
UNIREG_RE = re.compile(r"^(\d+)-(\d+)$")
SEQUENCE_ID_RE = re.compile(r"^([A-Z0-9]+)_([1-9][0-9]*)_([1-9][0-9]*)$")

CANONICAL_COLUMNS = [
    "sequence_id", "uniprot_id", "start", "end", "sequence",
    "label", "label_name", "source_cl", "label_source",
    "cluster_id", "is_cluster_representative", "dataset_version",
]


def _parse_annotations(path: Path, ann_checksum: str, ann_url: str) -> tuple[pd.DataFrame, int]:
    """Return (DataFrame, n_duplicates_dropped)."""
    rows: list[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("FA-DOMID"):
                continue
            parts = line.split("\t")
            if len(parts) < 11:
                continue
            uni_id = parts[3].strip()
            uni_reg = parts[4].strip()
            scopcla = parts[10].strip()

            m_reg = UNIREG_RE.match(uni_reg)
            if not m_reg:
                continue
            start, end = int(m_reg.group(1)), int(m_reg.group(2))

            m_cl = CL_RE.search(scopcla)
            if not m_cl:
                continue

            rows.append({
                "sequence_id": f"{uni_id}_{start}_{end}",
                "uniprot_id": uni_id,
                "start": start,
                "end": end,
                "source_cl": m_cl.group(1),
                "label_source": f"url={ann_url} sha256={ann_checksum} scopcla={scopcla}",
            })

    df = pd.DataFrame(rows)
    n_before = len(df)
    df = df.drop_duplicates(subset=["sequence_id"], keep="first")
    return df, n_before - len(df)


def _derive_label(source_cl: str) -> tuple[int, str]:
    """Labeling rule per docs/decisions.md: CL=1000000 -> alpha (1); all other -> not_alpha (0)."""
    if source_cl == CL_ALPHA:
        return 1, "alpha"
    return 0, "not_alpha"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representatives", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--cluster-assignments", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--annotations-url",
        default="https://www.ebi.ac.uk/pdbe/scop/files/scop-cla-latest.txt",
    )
    args = parser.parse_args(argv)

    log = configure_logging(__name__)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rep_checksum = sha256_file(args.representatives)
    ann_checksum = sha256_file(args.annotations)
    dataset_version = (
        f"reps_sha256={rep_checksum[:16]} ann_sha256={ann_checksum[:16]} pipeline={PIPELINE_VERSION}"
    )
    log.info("Dataset version: %s", dataset_version)

    log.info("Parsing annotations from %s", args.annotations)
    ann_df, n_dup = _parse_annotations(args.annotations, ann_checksum, args.annotations_url)
    if n_dup:
        log.warning("Dropped %d duplicate annotation records (kept first occurrence)", n_dup)
    log.info("Annotation records: %d", len(ann_df))

    log.info("Loading cluster assignments from %s", args.cluster_assignments)
    cluster_df = pd.read_parquet(args.cluster_assignments)[
        ["sequence_id", "cluster_id", "is_cluster_representative"]
    ]

    log.info("Parsing representatives FASTA from %s", args.representatives)
    rep_rows: list[dict] = []
    malformed = 0
    for rec in SeqIO.parse(str(args.representatives), "fasta"):
        if not SEQUENCE_ID_RE.match(rec.id):
            log.warning("Skipping malformed sequence_id in FASTA: %r", rec.id)
            malformed += 1
            continue
        rep_rows.append({"sequence_id": rec.id, "sequence": str(rec.seq).upper()})
    if malformed:
        log.warning("Skipped %d records with malformed sequence_id", malformed)
    rep_df = pd.DataFrame(rep_rows)
    log.info("Representatives parsed: %d", len(rep_df))

    df = rep_df.merge(ann_df, on="sequence_id", how="inner")
    unmatched = len(rep_df) - len(df)
    if unmatched:
        log.warning("%d representative sequences had no annotation match and are excluded", unmatched)

    df = df.merge(cluster_df, on="sequence_id", how="left")

    labels = df["source_cl"].apply(_derive_label)
    df["label"] = [lbl[0] for lbl in labels]
    df["label_name"] = [lbl[1] for lbl in labels]
    df["is_cluster_representative"] = df["is_cluster_representative"].fillna(True).astype(bool)
    df["dataset_version"] = dataset_version

    assert df["is_cluster_representative"].all(), "Non-representative rows found — violates data contract"
    assert df["sequence_id"].is_unique, "Duplicate sequence_id in final dataset — violates data contract"

    df = df[CANONICAL_COLUMNS]
    df.to_parquet(args.output, index=False)

    alpha = int((df["label"] == 1).sum())
    not_alpha = int((df["label"] == 0).sum())
    lengths = df["sequence"].str.len()
    log.info("--- Dataset summary ---")
    log.info("Total sequences:  %d", len(df))
    log.info("  alpha:          %d  (%.1f%%)", alpha, 100 * alpha / len(df))
    log.info("  not_alpha:      %d  (%.1f%%)", not_alpha, 100 * not_alpha / len(df))
    log.info("Length  min=%d  median=%d  max=%d", lengths.min(), int(lengths.median()), lengths.max())
    log.info("Output -> %s", args.output)


if __name__ == "__main__":
    main()
