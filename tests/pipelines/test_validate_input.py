import pytest
from pathlib import Path

import pandas as pd

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
    "P0A7V8_0_44",   # start = 0
    "P0A7V8_2",      # missing end
    "P0A7V8",        # missing both positions
    "p0a7v8_2_44",   # lowercase
    "P0A7V8_2_0",    # end = 0
    "_2_44",         # no prefix
])
def test_sequence_id_regex_rejects_invalid(sid):
    assert not SEQUENCE_ID_RE.match(sid)


# ---------------------------------------------------------------------------
# Annotation parsing (space-separated, keyed by FA-DOMID)
# ---------------------------------------------------------------------------

def _make_annotation_file(tmp_path: Path, data_lines: list[str]) -> Path:
    """Write a minimal SCOP annotation file (space-separated, comment header)."""
    content = "# SCOP-CLA test\n" + "\n".join(data_lines) + "\n"
    p = tmp_path / "scop-cla.txt"
    p.write_text(content)
    return p


def _ann_line(fa_domid, uni_id, uni_reg, cl):
    """Build a valid annotation line with the given key fields."""
    scopcla = f"TP=1,CL={cl},CF=2000000,SF=3000000,FA=4000000"
    # 11 space-separated columns; SF columns mirror FA columns
    return f"{fa_domid} PDBID A:1-10 {uni_id} {uni_reg} 9000000 PDBID A:1-10 {uni_id} {uni_reg} {scopcla}"


def test_parse_annotations_basic(tmp_path):
    p = _make_annotation_file(tmp_path, [
        _ann_line("8000001", "P0A7V8", "2-44", "1000000"),
        _ann_line("8000002", "Q9XYZ1", "1-50", "2000000"),
    ])
    records, skipped = _parse_annotations(p)
    assert skipped == 0
    assert "8000001" in records
    assert records["8000001"] == ("1000000", records["8000001"][1], "P0A7V8", "2", "44")
    assert "8000002" in records
    assert records["8000002"][0] == "2000000"


def test_parse_annotations_skips_malformed_unireg(tmp_path):
    line = "8000001 PDBID A:1-10 P0A7V8 BAD 9 PDBID A:1-10 P0A7V8 BAD TP=1,CL=1000000,CF=2,SF=3,FA=4"
    p = _make_annotation_file(tmp_path, [line])
    records, skipped = _parse_annotations(p)
    assert len(records) == 0
    assert skipped == 1


def test_parse_annotations_skips_missing_cl(tmp_path):
    line = "8000001 PDBID A:1-10 P0A7V8 2-44 9 PDBID A:1-10 P0A7V8 2-44 TP=1"
    p = _make_annotation_file(tmp_path, [line])
    records, skipped = _parse_annotations(p)
    assert len(records) == 0
    assert skipped == 1


def test_parse_annotations_skips_comment_lines(tmp_path):
    p = _make_annotation_file(tmp_path, [
        "# this is a comment",
        _ann_line("8000001", "P0A7V8", "2-44", "1000000"),
    ])
    records, skipped = _parse_annotations(p)
    assert len(records) == 1
    assert skipped == 0
