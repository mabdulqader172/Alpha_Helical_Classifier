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
