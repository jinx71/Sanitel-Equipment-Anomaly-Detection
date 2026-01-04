"""Shared detector interface and scoring helpers.

Every detector returns a *raw anomaly score* where a higher value means more
anomalous. The pipeline then min-max normalises scores to [0, 1] so models on
different scales can be compared and averaged into an ensemble, and applies a
common contamination-budget threshold so each model flags the same number of
points (a fair, operationally-meaningful comparison).
"""

from __future__ import annotations

import numpy as np


class BaseDetector:
    name: str = "base"
    label: str = "Base detector"
    description: str = ""

    def fit_score(self, X: np.ndarray) -> np.ndarray:
        """Fit on (unlabelled) X and return one raw anomaly score per row."""
        raise NotImplementedError


def normalize01(scores: np.ndarray) -> np.ndarray:
    """Min-max scale scores to [0, 1]; returns zeros if scores are constant."""
    scores = np.asarray(scores, dtype=float)
    lo, hi = float(np.min(scores)), float(np.max(scores))
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def threshold_by_contamination(scores: np.ndarray, contamination: float):
    """Flag the top ``contamination`` fraction of scores as anomalies.

    Returns ``(predictions, threshold)`` where predictions is a 0/1 array.
    """
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    k = max(1, int(round(float(contamination) * n)))
    k = min(k, n)
    # k-th largest score is the cut point.
    threshold = float(np.partition(scores, n - k)[n - k])
    preds = (scores >= threshold).astype(int)
    return preds, threshold
