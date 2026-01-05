"""One-Class SVM: learns a boundary around the normal operating region."""

from __future__ import annotations

import numpy as np
from sklearn.svm import OneClassSVM

from .base import BaseDetector


class OneClassSVMDetector(BaseDetector):
    name = "ocsvm"
    label = "One-Class SVM"
    description = (
        "Boundary-based. Fits an RBF-kernel frontier enclosing the dense normal "
        "region; points outside the frontier score higher. Flexible non-linear "
        "decision surface, sensitive to feature scaling."
    )

    def __init__(self, nu: float = 0.06, gamma: str = "scale"):
        self.nu = nu
        self.gamma = gamma

    def fit_score(self, X: np.ndarray) -> np.ndarray:
        model = OneClassSVM(kernel="rbf", nu=self.nu, gamma=self.gamma)
        model.fit(X)
        # decision_function: higher == more normal. Negate for an anomaly score.
        return -model.decision_function(X)
