import numpy as np
import pytest

from protein_alpha_classifier.features.composition import (
    CANONICAL_AA,
    DIPEPTIDE_ORDER,
    AACompositionTransformer,
    DipeptideCompositionTransformer,
    LengthTransformer,
)


# ---------------------------------------------------------------------------
# AACompositionTransformer
# ---------------------------------------------------------------------------

def test_composition_pure_alanine():
    t = AACompositionTransformer()
    result = t.fit_transform(["AAAA"])
    assert result.shape == (1, 20)
    a_idx = CANONICAL_AA.index("A")
    assert result[0, a_idx] == pytest.approx(1.0)
    assert result[0].sum() == pytest.approx(1.0)


def test_composition_equal_mix():
    t = AACompositionTransformer()
    result = t.fit_transform(["AC"])
    a_idx = CANONICAL_AA.index("A")
    c_idx = CANONICAL_AA.index("C")
    assert result[0, a_idx] == pytest.approx(0.5)
    assert result[0, c_idx] == pytest.approx(0.5)


def test_composition_sums_to_one():
    t = AACompositionTransformer()
    seqs = ["ACDEFGHIKLMNPQRSTVWY", "AAACCC", "MMMM"]
    result = t.fit_transform(seqs)
    for i, row in enumerate(result):
        assert row.sum() == pytest.approx(1.0), f"Row {i} sums to {row.sum()}"


def test_composition_empty_sequence():
    t = AACompositionTransformer()
    result = t.fit_transform([""])
    assert result.shape == (1, 20)
    assert (result == 0).all()


def test_composition_ignores_noncanonical():
    t = AACompositionTransformer()
    result = t.fit_transform(["AX"])
    a_idx = CANONICAL_AA.index("A")
    assert result[0, a_idx] == pytest.approx(0.5)


def test_composition_feature_names():
    t = AACompositionTransformer()
    names = t.get_feature_names_out()
    assert len(names) == 20
    assert names[0] == "frac_A"


def test_composition_fit_is_noop():
    t = AACompositionTransformer()
    t2 = t.fit(["ACDE"], [1])
    assert t2 is t


# ---------------------------------------------------------------------------
# LengthTransformer
# ---------------------------------------------------------------------------

def test_length_basic():
    t = LengthTransformer()
    result = t.fit_transform(["ACE", "M", "ACDEFG"])
    assert result.shape == (3, 1)
    np.testing.assert_array_equal(result.flatten(), [3, 1, 6])


def test_length_empty():
    t = LengthTransformer()
    result = t.fit_transform([""])
    assert result[0, 0] == 0


def test_length_feature_name():
    t = LengthTransformer()
    assert t.get_feature_names_out()[0] == "length"


# ---------------------------------------------------------------------------
# DipeptideCompositionTransformer
# ---------------------------------------------------------------------------

def test_dipeptide_shape():
    t = DipeptideCompositionTransformer()
    assert t.fit_transform(["ACDE"]).shape == (1, 400)


def test_dipeptide_400_features():
    assert len(DIPEPTIDE_ORDER) == 400


def test_dipeptide_sums_to_one():
    t = DipeptideCompositionTransformer()
    for seq in ["ACDE", "MMMM", "ACDEFGHIKLMNPQRSTVWY"]:
        result = t.fit_transform([seq])
        assert result[0].sum() == pytest.approx(1.0), f"Failed for seq={seq!r}"


def test_dipeptide_correct_pair_count():
    # "ACA" → dipeptides are AC and CA, each with fraction 0.5
    t = DipeptideCompositionTransformer()
    result = t.fit_transform(["ACA"])
    ac_idx = DIPEPTIDE_ORDER.index("AC")
    ca_idx = DIPEPTIDE_ORDER.index("CA")
    assert result[0, ac_idx] == pytest.approx(0.5)
    assert result[0, ca_idx] == pytest.approx(0.5)
    # All other positions should be zero
    mask = np.ones(400, dtype=bool)
    mask[ac_idx] = False
    mask[ca_idx] = False
    assert (result[0, mask] == 0).all()


def test_dipeptide_repeated_pair():
    # "AAA" → two AA dipeptides, frac_AA = 1.0
    t = DipeptideCompositionTransformer()
    result = t.fit_transform(["AAA"])
    aa_idx = DIPEPTIDE_ORDER.index("AA")
    assert result[0, aa_idx] == pytest.approx(1.0)


def test_dipeptide_short_sequence_zero():
    t = DipeptideCompositionTransformer()
    for seq in ["", "A"]:
        result = t.fit_transform([seq])
        assert (result == 0).all(), f"Expected zeros for seq={seq!r}"


def test_dipeptide_feature_names():
    t = DipeptideCompositionTransformer()
    names = t.get_feature_names_out()
    assert len(names) == 400
    assert names[0] == "frac_AA"
    assert names[-1] == "frac_YY"


def test_dipeptide_fit_is_noop():
    t = DipeptideCompositionTransformer()
    t2 = t.fit(["ACDE"], [1])
    assert t2 is t
