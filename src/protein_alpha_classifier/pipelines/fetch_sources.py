"""Download SCOP annotation and representative-sequence FASTA; write source manifest."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import requests
from Bio import SeqIO

from protein_alpha_classifier._utils import configure_logging, sha256_file

PIPELINE_VERSION = "1.0.0"


def _download(url: str, dest: Path) -> dict:
    resp = requests.get(url, stream=True, timeout=180)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return {
        "url": url,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "bytes": dest.stat().st_size,
        "sha256": sha256_file(dest),
        "http_status": resp.status_code,
        "content_type": resp.headers.get("Content-Type"),
    }


def _first_comment_line(path: Path) -> str:
    """Return the first `#` line from the annotation file (contains SCOP release info)."""
    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                return line.strip()
    return ""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-url", required=True)
    parser.add_argument("--fasta-url", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    log = configure_logging(__name__)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ann_path = args.out_dir / "scop-cla-latest.txt"
    fa_path = args.out_dir / "scop_fa_represeq_lib_latest.fa"

    log.info("Downloading annotations -> %s", ann_path)
    ann_prov = _download(args.annotations_url, ann_path)
    ann_prov["scop_release_header"] = _first_comment_line(ann_path)

    log.info("Downloading FASTA -> %s", fa_path)
    fa_prov = _download(args.fasta_url, fa_path)
    fasta_count = sum(1 for _ in SeqIO.parse(str(fa_path), "fasta"))
    fa_prov["record_count"] = fasta_count
    log.info("FASTA record count: %d", fasta_count)

    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "annotations": ann_prov,
        "fasta": fa_prov,
    }
    manifest_path = args.out_dir / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("Source manifest written to %s", manifest_path)


if __name__ == "__main__":
    main()
