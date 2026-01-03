"""Project standardised sensor vectors to 2D for the latent-space map.

UMAP gives better separation of anomaly clusters but pulls in numba, so it is
imported lazily and the function transparently falls back to PCA when UMAP is
unavailable or fails. The method actually used is returned so the UI can label
the axes honestly.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def _pca_projection(X: np.ndarray) -> np.ndarray:
    return PCA(n_components=2, random_state=42).fit_transform(X)


def project(X_scaled: np.ndarray, method: str = "pca"):
    """Return (method_used, points) where points is an (n, 2) float array."""
    method = (method or "pca").lower()

    if method == "umap":
        try:
            import umap  # type: ignore

            n_neighbors = min(15, max(2, X_scaled.shape[0] - 1))
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                random_state=42,
            )
            points = reducer.fit_transform(X_scaled)
            return "umap", np.asarray(points, dtype=float)
        except Exception:
            # numba/umap not installed or failed - degrade gracefully to PCA.
            return "pca", _pca_projection(X_scaled)

    return "pca", _pca_projection(X_scaled)
