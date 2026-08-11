import pytest
from protein_alpha_classifier.pipelines.build_dataset import (
    CANONICAL_COLUMNS,
    CL_ALPHA,
    _derive_label,
    _parse_annotations,
)
from pathlib import Path


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
# Annotation parsing (duplicates)
# ---------------------------------------------------------------------------

def _make_ann(tmp_path: Path, lines: list[str]) -> Path:
    header = (
        "# test\n"
        "FA-DOMID\tFA-PDBID\tFA-PDBREG\tFA-UNIID\tFA-UNIREG\t"
        "SF-DOMID\tSF-PDBID\tSF-PDBREG\tSF-UNIID\tSF-UNIREG\tSCOPCLA\n"
    )
    p = tmp_path / "ann.txt"
    p.write_text(header + "\n".join(lines) + "\n")
    return p


def test_parse_annotations_drops_duplicates(tmp_path):
    row = "1\t1dlw\tA:2-45\tP0A7V8\t2-44\t1\t1dlw\tA:2-45\tP0A7V8\t2-44\tTP=PK,CL=1000000,CF=1000000,SF=1000000,FA=1000000"
    p = _make_ann(tmp_path, [row, row])
    df, n_dup = _parse_annotations(p, "abc123", "http://example.com")
    assert n_dup == 1
    assert len(df) == 1
    assert df.iloc[0]["sequence_id"] == "P0A7V8_2_44"


def test_parse_annotations_label_source_records_provenance(tmp_path):
    row = "1\t1dlw\tA:2-45\tP0A7V8\t2-44\t1\t1dlw\tA:2-45\tP0A7V8\t2-44\tTP=PK,CL=1000000,CF=1000000,SF=1000000,FA=1000000"
    p = _make_ann(tmp_path, [row])
    df, _ = _parse_annotations(p, "deadbeef", "http://example.com/ann.txt")
    label_source = df.iloc[0]["label_source"]
    assert "deadbeef" in label_source
    assert "http://example.com/ann.txt" in label_source
