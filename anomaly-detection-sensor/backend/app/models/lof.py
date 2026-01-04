"""Local Outlier Factor: density-based local anomaly detection."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from .base import BaseDetector


class LOFDetector(BaseDetector):
    name = "lof"
    label = "Local Outlier Factor"
    description = (
        "Density-based. Compares each point's local density to that of its "
        "k nearest neighbours; points in much sparser regions than their "
        "neighbours score higher. Catches local anomalies a global model misses."
    )

    def __init__(self, contamination: float = 0.06, n_neighbors: int = 20):
        self.contamination = contamination
        self.n_neighbors = n_neighbors

    def fit_score(self, X: np.ndarray) -> np.ndarray:
        n_neighbors = min(self.n_neighbors, max(2, X.shape[0] - 1))
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors, contamination=self.contamination
        )
        model.fit_predict(X)
        # negative_outlier_factor_: more negative == more outlying. Negate.
        return -model.negative_outlier_factor_
