"""PCA reconstruction error: a linear subspace anomaly detector.

Projects readings onto the leading principal components that capture the normal
operating subspace, then reconstructs them. In-control points lie close to that
subspace and reconstruct well; anomalies (especially correlation-breaks that
leave the subspace) reconstruct poorly, giving a high residual.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from .base import BaseDetector


class PCAReconstructionDetector(BaseDetector):
    name = "pca_reconstruction"
    label = "PCA Reconstruction"
    description = (
        "Linear subspace model. Keeps the principal components that explain the "
        "normal operating subspace; the squared reconstruction residual is the "
        "anomaly score. A fast, interpretable linear cousin of the autoencoder."
    )

    def __init__(self, variance: float = 0.90):
        self.variance = variance

    def _n_components(self, X: np.ndarray) -> int:
        n_features = X.shape[1]
        # Number of components reaching the variance target, capped below full
        # rank so reconstruction is genuinely lossy.
        full = PCA(n_components=min(X.shape)).fit(X)
        cum = np.cumsum(full.explained_variance_ratio_)
        k = int(np.searchsorted(cum, self.variance) + 1)
        return int(np.clip(k, 1, max(1, n_features - 1)))

    def fit_score(self, X: np.ndarray) -> np.ndarray:
        model = PCA(n_components=self._n_components(X))
        Z = model.fit_transform(X)
        X_reconstructed = model.inverse_transform(Z)
        return np.sum((X - X_reconstructed) ** 2, axis=1)
