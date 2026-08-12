"""Sequence-derived feature transformers.

All transformers are stateless (fit is a no-op). Learned transformations
(scaling) belong in the sklearn Pipeline that wraps these.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Canonical 20 amino acids in fixed alphabetical order.
CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"
_AA_INDEX: dict[str, int] = {aa: i for i, aa in enumerate(CANONICAL_AA)}


class AACompositionTransformer(BaseEstimator, TransformerMixin):
    """Transform a collection of sequences into 20-dim fractional AA-composition vectors."""

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        seqs = list(X)
        out = np.zeros((len(seqs), 20), dtype=np.float64)
        for i, seq in enumerate(seqs):
            n = len(seq)
            if n == 0:
                continue
            for aa in seq:
                j = _AA_INDEX.get(aa)
                if j is not None:
                    out[i, j] += 1
            out[i] /= n
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"frac_{aa}" for aa in CANONICAL_AA])


class LengthTransformer(BaseEstimator, TransformerMixin):
    """Transform a collection of sequences into a 1-dim sequence-length vector."""

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        return np.array([[len(seq)] for seq in X], dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.array(["length"])


# All 400 ordered AA pairs in fixed alphabetical order.
DIPEPTIDE_ORDER: list[str] = [a + b for a in CANONICAL_AA for b in CANONICAL_AA]
_DIPEPTIDE_INDEX: dict[str, int] = {dp: i for i, dp in enumerate(DIPEPTIDE_ORDER)}


class DipeptideCompositionTransformer(BaseEstimator, TransformerMixin):
    """Transform sequences into 400-dim fractional dipeptide-composition vectors.

    Counts every consecutive ordered AA pair; divides by (len - 1).
    Sequences shorter than 2 residues produce a zero vector.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        seqs = list(X)
        out = np.zeros((len(seqs), 400), dtype=np.float64)
        for i, seq in enumerate(seqs):
            n = len(seq)
            if n < 2:
                continue
            for k in range(n - 1):
                j = _DIPEPTIDE_INDEX.get(seq[k : k + 2])
                if j is not None:
                    out[i, j] += 1
            out[i] /= (n - 1)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"frac_{dp}" for dp in DIPEPTIDE_ORDER])
