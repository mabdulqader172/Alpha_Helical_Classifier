import pytest
from pathlib import Path

from protein_alpha_classifier.pipelines.build_dataset import (
    CANONICAL_COLUMNS,
    CL_ALPHA,
    _derive_label,
    _parse_annotations,
)


# ---------------------------------------------------------------------------
# Label derivation — non-negotiable rule
# ---------------------------------------------------------------------------

def test_cl_1000000_is_alpha():
    label, name = _derive_label("1000000")
    assert label == 1
    assert name == "alpha"


@pytest.mark.parametrize("cl", ["2000000", "3000000", "0", "999999", "1000001", "9999999"])
def test_any_other_cl_is_not_alpha(cl):
    label, name = _derive_label(cl)
    assert label == 0
    assert name == "not_alpha"


def test_cl_alpha_constant_matches_rule():
    assert CL_ALPHA == "1000000"


# ---------------------------------------------------------------------------
# Canonical columns
# ---------------------------------------------------------------------------

def test_canonical_columns_present():
    expected = {
        "sequence_id", "uniprot_id", "start", "end", "sequence",
        "label", "label_name", "source_cl", "label_source",
        "cluster_id", "is_cluster_representative", "dataset_version",
    }
    assert set(CANONICAL_COLUMNS) == expected


# ---------------------------------------------------------------------------
# Annotation parsing (space-separated, keyed by sequence_id)
# ---------------------------------------------------------------------------

def _make_ann(tmp_path: Path, data_lines: list[str]) -> Path:
    """Write a minimal SCOP annotation file (space-separated)."""
    p = tmp_path / "ann.txt"
    p.write_text("# test\n" + "\n".join(data_lines) + "\n")
    return p


def _ann_line(fa_domid, uni_id, uni_reg, cl):
    scopcla = f"TP=1,CL={cl},CF=2000000,SF=3000000,FA=4000000"
    return f"{fa_domid} PDBID A:1-10 {uni_id} {uni_reg} 9000000 PDBID A:1-10 {uni_id} {uni_reg} {scopcla}"


def test_parse_annotations_drops_duplicates(tmp_path):
    row = _ann_line("8000001", "P0A7V8", "2-44", "1000000")
    p = _make_ann(tmp_path, [row, row])
    df, n_dup = _parse_annotations(p, "abc123", "http://example.com")
    assert n_dup == 1
    assert len(df) == 1
    assert df.iloc[0]["sequence_id"] == "P0A7V8_2_44"


def test_parse_annotations_label_source_records_provenance(tmp_path):
    row = _ann_line("8000001", "P0A7V8", "2-44", "1000000")
    p = _make_ann(tmp_path, [row])
    df, _ = _parse_annotations(p, "deadbeef", "http://example.com/ann.txt")
    label_source = df.iloc[0]["label_source"]
    assert "deadbeef" in label_source
    assert "http://example.com/ann.txt" in label_source


def test_parse_annotations_both_labels(tmp_path):
    p = _make_ann(tmp_path, [
        _ann_line("8000001", "P0A7V8", "2-44", "1000000"),
        _ann_line("8000002", "Q9XYZ1", "1-50", "2000000"),
    ])
    df, _ = _parse_annotations(p, "x", "http://x")
    alpha_rows = df[df["source_cl"] == "1000000"]
    not_alpha_rows = df[df["source_cl"] != "1000000"]
    assert len(alpha_rows) == 1
    assert len(not_alpha_rows) == 1
