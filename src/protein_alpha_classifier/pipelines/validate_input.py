"""Filter eligible sequences; emit labeled FASTA for MMseqs2 and omission records."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from protein_alpha_classifier._utils import configure_logging, sha256_file

PIPELINE_VERSION = "1.0.0"
CANONICAL_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
SEQUENCE_ID_RE = re.compile(r"^[A-Z0-9]+_[1-9][0-9]*_[1-9][0-9]*$")
CL_RE = re.compile(r"CL=(\d+)")
UNIREG_RE = re.compile(r"^(\d+)-(\d+)$")


def _parse_annotations(path: Path) -> tuple[dict[str, tuple[str, str, str, str, str]], int]:
    """Return ({fa_domid: (source_cl, scopcla, uni_id, start, end)}, n_skipped).

    Annotation file format (space-separated, comment lines start with #):
      col 0  FA-DOMID  — numeric domain identifier; joins to FASTA record.id
      col 3  FA-UNIID  — UniProt accession
      col 4  FA-UNIREG — residue range "start-end"
      col 10 SCOPCLA   — e.g. "TP=1,CL=1000000,CF=...,SF=...,FA=..."
    """
    records: dict[str, tuple[str, str, str, str, str]] = {}
    skipped = 0
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 11:
                skipped += 1
                continue
            fa_domid = parts[0]
            uni_id = parts[3]
            uni_reg = parts[4]
            scopcla = parts[10]

            m_reg = UNIREG_RE.match(uni_reg)
            if not m_reg:
                skipped += 1
                continue
            start, end = m_reg.group(1), m_reg.group(2)

            m_cl = CL_RE.search(scopcla)
            if not m_cl:
                skipped += 1
                continue

            records[fa_domid] = (m_cl.group(1), scopcla, uni_id, start, end)
    return records, skipped


def _invalid_symbols(seq: str) -> list[str]:
    return sorted(set(seq) - CANONICAL_ALPHABET)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output-fasta", required=True, type=Path)
    parser.add_argument("--invalid-records", required=True, type=Path)
    args = parser.parse_args(argv)

    log = configure_logging(__name__)
    args.output_fasta.parent.mkdir(parents=True, exist_ok=True)
    args.invalid_records.parent.mkdir(parents=True, exist_ok=True)

    fasta_checksum = sha256_file(args.fasta)
    ann_checksum = sha256_file(args.annotations)
    log.info("Input FASTA sha256: %s", fasta_checksum)
    log.info("Annotations sha256: %s", ann_checksum)

    log.info("Parsing annotations from %s", args.annotations)
    annotations, ann_skipped = _parse_annotations(args.annotations)
    log.info("Annotation records loaded: %d  (unparseable lines skipped: %d)", len(annotations), ann_skipped)

    eligible: list[SeqRecord] = []
    omitted: list[dict] = []
    raw_count = 0

    for rec in SeqIO.parse(str(args.fasta), "fasta"):
        raw_count += 1
        fa_domid = rec.id
        seq = str(rec.seq).upper()

        if fa_domid not in annotations:
            omitted.append({
                "sequence_id": fa_domid,
                "rejection_reason": "no_annotation_match",
                "invalid_symbols": "",
                "fasta_checksum": fasta_checksum,
                "ann_checksum": ann_checksum,
                "pipeline_version": PIPELINE_VERSION,
            })
            continue

        source_cl, scopcla, uni_id, start, end = annotations[fa_domid]
        seq_id = f"{uni_id}_{start}_{end}"

        if not SEQUENCE_ID_RE.match(seq_id):
            omitted.append({
                "sequence_id": seq_id,
                "rejection_reason": "malformed_sequence_id",
                "invalid_symbols": "",
                "fasta_checksum": fasta_checksum,
                "ann_checksum": ann_checksum,
                "pipeline_version": PIPELINE_VERSION,
            })
            continue

        bad = _invalid_symbols(seq)
        if bad:
            omitted.append({
                "sequence_id": seq_id,
                "rejection_reason": "noncanonical_residues",
                "invalid_symbols": ",".join(bad),
                "fasta_checksum": fasta_checksum,
                "ann_checksum": ann_checksum,
                "pipeline_version": PIPELINE_VERSION,
            })
            continue

        # Rewrite header to sequence_id so downstream steps (MMseqs2, build_dataset) use it directly.
        eligible.append(SeqRecord(Seq(seq), id=seq_id, description=""))

    SeqIO.write(eligible, str(args.output_fasta), "fasta")
    pd.DataFrame(omitted).to_parquet(args.invalid_records, index=False)

    # alpha_count uses the fa_domid -> annotations lookup via seq_id reconstruction
    seq_id_to_cl = {
        f"{v[2]}_{v[3]}_{v[4]}": v[0] for v in annotations.values()
    }
    alpha_count = sum(1 for r in eligible if seq_id_to_cl.get(r.id) == "1000000")
    not_alpha_count = len(eligible) - alpha_count
    omit_reasons: dict[str, int] = {}
    for row in omitted:
        omit_reasons[row["rejection_reason"]] = omit_reasons.get(row["rejection_reason"], 0) + 1

    log.info("--- Summary ---")
    log.info("Raw FASTA records:       %d", raw_count)
    log.info("Omitted:                 %d  %s", len(omitted), omit_reasons)
    log.info("Eligible sequences:      %d", len(eligible))
    log.info("  alpha  (CL=1000000):   %d  (%.1f%%)", alpha_count, 100 * alpha_count / max(len(eligible), 1))
    log.info("  not_alpha:             %d  (%.1f%%)", not_alpha_count, 100 * not_alpha_count / max(len(eligible), 1))
    log.info("Eligible FASTA  -> %s", args.output_fasta)
    log.info("Omission parquet-> %s", args.invalid_records)


if __name__ == "__main__":
    main()
