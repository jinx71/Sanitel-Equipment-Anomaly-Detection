"""Cluster flagged anomalies into distinct signatures with DBSCAN.

Operates in standardised feature space, so each cluster's mean coordinate on a
feature is directly its deviation from the dataset mean in standard-deviation
units (a z-score). That lets every anomaly cluster be summarised by the sensors
that characterise it (e.g. "conductivity +3.4 SD"), which maps cleanly onto a
likely failure mode for an engineer reviewing the console.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Sentinels used in the per-point cluster vector.
NORMAL = -1   # not flagged by the ensemble
NOISE = -2    # flagged, but not part of any dense anomaly cluster


def _adaptive_eps(points: np.ndarray, min_samples: int) -> float:
    """Estimate a DBSCAN eps from the k-nearest-neighbour distance distribution."""
    k = min(min_samples, points.shape[0] - 1)
    if k < 1:
        return 1.0
    nn = NearestNeighbors(n_neighbors=k).fit(points)
    distances, _ = nn.kneighbors(points)
    return float(np.median(distances[:, -1]) * 1.6) + 1e-6


def cluster_anomalies(
    X_scaled: np.ndarray,
    ensemble_pred: np.ndarray,
    true_types: list,
    feature_keys: list[str],
    feature_labels: list[str],
):
    """Return (cluster_labels_per_point, cluster_summaries)."""
    n = X_scaled.shape[0]
    cluster_labels = np.full(n, NORMAL, dtype=int)

    flagged_idx = np.flatnonzero(ensemble_pred == 1)
    if flagged_idx.size == 0:
        return cluster_labels.tolist(), []

    A = X_scaled[flagged_idx]
    min_samples = 3 if A.shape[0] >= 6 else 2
    eps = _adaptive_eps(A, min_samples)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(A)

    # Map DBSCAN output back onto the full-length cluster vector.
    for local_i, global_i in enumerate(flagged_idx):
        lab = labels[local_i]
        cluster_labels[global_i] = NOISE if lab == -1 else int(lab)

    summaries = []
    for cid in sorted(c for c in set(labels) if c != -1):
        members = flagged_idx[labels == cid]
        centroid = X_scaled[members].mean(axis=0)  # mean z-score per feature
        order = np.argsort(np.abs(centroid))[::-1][:3]
        top_features = [
            {
                "key": feature_keys[j],
                "label": feature_labels[j],
                "deviation": round(float(centroid[j]), 2),
                "direction": "high" if centroid[j] >= 0 else "low",
            }
            for j in order
        ]
        member_types = [true_types[i] for i in members if true_types[i] is not None]
        dominant = max(set(member_types), key=member_types.count) if member_types else None
        summaries.append(
            {
                "cluster": int(cid),
                "size": int(members.size),
                "dominant_type": dominant,
                "top_features": top_features,
            }
        )

    return cluster_labels.tolist(), summaries
