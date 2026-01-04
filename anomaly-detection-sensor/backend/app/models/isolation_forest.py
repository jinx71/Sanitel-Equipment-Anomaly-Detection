"""Isolation Forest: isolates anomalies via short random-partition paths."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from .base import BaseDetector


class IsolationForestDetector(BaseDetector):
    name = "isolation_forest"
    label = "Isolation Forest"
    description = (
        "Tree ensemble. Anomalies require fewer random splits to isolate, so a "
        "shorter average path length means a higher anomaly score. Scale-robust "
        "and effective in higher dimensions."
    )

    def __init__(self, contamination: float = 0.06, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state

    def fit_score(self, X: np.ndarray) -> np.ndarray:
        model = IsolationForest(
            n_estimators=200,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        model.fit(X)
        # decision_function: higher == more normal. Negate for an anomaly score.
        return -model.decision_function(X)
