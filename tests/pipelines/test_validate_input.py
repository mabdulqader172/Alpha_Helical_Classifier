import io
import textwrap
from pathlib import Path

import pandas as pd
import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from protein_alpha_classifier.pipelines.validate_input import (
    CANONICAL_ALPHABET,
    SEQUENCE_ID_RE,
    _invalid_symbols,
    _parse_annotations,
)


# ---------------------------------------------------------------------------
# Alphabet / residue checks
# ---------------------------------------------------------------------------

def test_canonical_alphabet_has_20_residues():
    assert len(CANONICAL_ALPHABET) == 20


def test_invalid_symbols_clean_sequence():
    assert _invalid_symbols("ACDEFGHIKLMNPQRSTVWY") == []


def test_invalid_symbols_detects_ambiguous():
    bad = _invalid_symbols("ACXBZJO")
    assert set(bad) == {"X", "B", "Z", "J", "O"}


def test_invalid_symbols_detects_lowercase():
    # sequences are uppercased before calling this; lowercase is still invalid
    assert _invalid_symbols("acde") == ["a", "c", "d", "e"]


def test_invalid_symbols_empty_sequence():
    assert _invalid_symbols("") == []


# ---------------------------------------------------------------------------
# sequence_id regex
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sid", ["P0A7V8_2_44", "A0A000_1_1000", "P12345_10_200"])
def test_sequence_id_regex_valid(sid):
    assert SEQUENCE_ID_RE.match(sid)


@pytest.mark.parametrize("sid", [
    "P0A7V8_0_44",      # start = 0 (must be >= 1)
    "P0A7V8_2",         # missing end
    "P0A7V8",           # missing both positions
    "p0a7v8_2_44",      # lowercase UniProt ID
    "P0A7V8_2_0",       # end = 0
    "_2_44",            # no UniProt prefix
])
def test_sequence_id_regex_rejects_invalid(sid):
    assert not SEQUENCE_ID_RE.match(sid)


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------

def _make_annotation_file(tmp_path: Path, lines: list[str]) -> Path:
    content = "# SCOP-CLA test\nFA-DOMID\tFA-PDBID\tFA-PDBREG\tFA-UNIID\tFA-UNIREG\tSF-DOMID\tSF-PDBID\tSF-PDBREG\tSF-UNIID\tSF-UNIREG\tSCOPCLA\n"
    content += "\n".join(lines) + "\n"
    p = tmp_path / "scop-cla.txt"
    p.write_text(content)
    return p


def test_parse_annotations_basic(tmp_path):
    p = _make_annotation_file(tmp_path, [
        "1\t1dlw\tA:2-45\tP0A7V8\t2-44\t1\t1dlw\tA:2-45\tP0A7V8\t2-44\tTP=PK,CL=1000000,CF=1000000,SF=1000000,FA=1000000",
        "2\t2abc\tA:1-50\tQ9XYZ1\t1-50\t2\t2abc\tA:1-50\tQ9XYZ1\t1-50\tTP=PK,CL=2000000,CF=2000000,SF=2000000,FA=2000000",
    ])
    records, skipped = _parse_annotations(p)
    assert "P0A7V8_2_44" in records
    assert records["P0A7V8_2_44"][0] == "1000000"
    assert "Q9XYZ1_1_50" in records
    assert records["Q9XYZ1_1_50"][0] == "2000000"
    assert skipped == 0


def test_parse_annotations_skips_malformed_unireg(tmp_path):
    p = _make_annotation_file(tmp_path, [
        "1\t1dlw\tA:2-45\tP0A7V8\tBAD\t1\t1dlw\tA:2-45\tP0A7V8\tBAD\tTP=PK,CL=1000000,CF=1000000,SF=1000000,FA=1000000",
    ])
    records, skipped = _parse_annotations(p)
    assert len(records) == 0
    assert skipped == 1


def test_parse_annotations_skips_missing_cl(tmp_path):
    p = _make_annotation_file(tmp_path, [
        "1\t1dlw\tA:2-45\tP0A7V8\t2-44\t1\t1dlw\tA:2-45\tP0A7V8\t2-44\tTP=PK",
    ])
    records, skipped = _parse_annotations(p)
    assert len(records) == 0
    assert skipped == 1
