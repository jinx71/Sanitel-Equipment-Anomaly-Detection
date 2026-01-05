"""End-to-end analysis pipeline.

generate -> standardise -> detect (per model) -> ensemble -> cluster -> project
-> evaluate, assembling a single payload the dashboard renders. Ground-truth
labels are passed only to the evaluation step; detectors are fit on the bare
feature matrix.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from .analysis.clustering import cluster_anomalies
from .analysis.evaluation import evaluate
from .analysis.projection import project
from .data import generator
from .models.base import normalize01, threshold_by_contamination
from .models.registry import ORDER, build_detector


def run_analysis(
    equipment: str,
    n_samples: int = 600,
    anomaly_rate: float = 0.06,
    seed: int | None = None,
    model_names: list[str] | None = None,
    contamination: float = 0.06,
    projection_method: str = "pca",
) -> dict:
    model_names = [m for m in (model_names or ORDER) if m in ORDER] or list(ORDER)
    contamination = float(np.clip(contamination, 0.01, 0.30))

    dataset = generator.generate(
        equipment=equipment,
        n_samples=n_samples,
        anomaly_rate=anomaly_rate,
        seed=seed,
    )
    profile = dataset.profile
    feature_keys = profile.feature_keys
    feature_labels = [s.label for s in profile.sensors]
    X = dataset.features
    y_true = dataset.labels

    # Standardise once; every detector and the projection consume scaled data.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---- per-model detection -------------------------------------------- #
    models_out = []
    normalized_scores = []
    prediction_matrix = []  # (n_models, n_samples)
    for name in model_names:
        detector = build_detector(name, contamination)
        raw = detector.fit_score(X_scaled)
        norm = normalize01(raw)
        preds, threshold = threshold_by_contamination(norm, contamination)
        metrics = evaluate(y_true, norm, preds)
        normalized_scores.append(norm)
        prediction_matrix.append(preds)
        models_out.append(
            {
                "name": detector.name,
                "label": detector.label,
                "description": detector.description,
                "threshold": round(float(threshold), 4),
                "metrics": metrics,
                "scores": [round(float(s), 4) for s in norm],
                "predictions": [int(p) for p in preds],
            }
        )

    normalized_scores = np.vstack(normalized_scores)
    prediction_matrix = np.vstack(prediction_matrix)

    # ---- ensemble (mean of normalised anomaly scores) ------------------ #
    # Averaging min-max-normalised scores preserves the magnitude of each
    # detector's separation (a confidently bimodal detector keeps its signal),
    # which empirically beat rank-averaging on these systems. It needs no
    # labels, so the ensemble stays fully unsupervised. Note that a naive equal-
    # weight ensemble does not always beat the single best detector.
    ensemble_score = normalize01(normalized_scores.mean(axis=0))
    ensemble_pred, ensemble_threshold = threshold_by_contamination(
        ensemble_score, contamination
    )
    ensemble_metrics = evaluate(y_true, ensemble_score, ensemble_pred)
    models_agree = prediction_matrix.sum(axis=0)  # votes per sample

    # ---- clustering of flagged anomalies -------------------------------- #
    true_types = dataset.frame["anomaly_type"].tolist()
    cluster_labels, cluster_summaries = cluster_anomalies(
        X_scaled, ensemble_pred, true_types, feature_keys, feature_labels
    )

    # ---- 2D projection -------------------------------------------------- #
    method_used, points = project(X_scaled, projection_method)

    # ---- assemble per-sample series ------------------------------------- #
    frame = dataset.frame
    series = []
    for i in range(len(frame)):
        series.append(
            {
                "index": int(frame["index"].iloc[i]),
                "timestamp": frame["timestamp"].iloc[i],
                "sensors": {k: float(frame[k].iloc[i]) for k in feature_keys},
                "is_true_anomaly": bool(frame["is_anomaly"].iloc[i]),
                "anomaly_type": frame["anomaly_type"].iloc[i],
                "ensemble_score": round(float(ensemble_score[i]), 4),
                "ensemble_pred": int(ensemble_pred[i]),
                "models_agree": int(models_agree[i]),
                "cluster": int(cluster_labels[i]),
            }
        )

    # ---- projection points --------------------------------------------- #
    projection_points = [
        {
            "index": int(i),
            "x": round(float(points[i, 0]), 4),
            "y": round(float(points[i, 1]), 4),
            "is_anomaly": int(ensemble_pred[i]),
            "is_true_anomaly": bool(y_true[i]),
            "cluster": int(cluster_labels[i]),
        }
        for i in range(len(frame))
    ]

    # ---- flagged anomaly table (sorted by ensemble score) --------------- #
    flagged_idx = np.flatnonzero(ensemble_pred == 1)
    flagged_idx = flagged_idx[np.argsort(ensemble_score[flagged_idx])[::-1]]
    flagged = []
    for i in flagged_idx:
        z = X_scaled[i]
        order = np.argsort(np.abs(z))[::-1][:3]
        deviations = [
            {
                "key": feature_keys[j],
                "label": feature_labels[j],
                "z": round(float(z[j]), 2),
                "direction": "high" if z[j] >= 0 else "low",
            }
            for j in order
        ]
        flagged.append(
            {
                "index": int(frame["index"].iloc[i]),
                "timestamp": frame["timestamp"].iloc[i],
                "ensemble_score": round(float(ensemble_score[i]), 4),
                "models_agree": int(models_agree[i]),
                "n_models": len(model_names),
                "anomaly_type": frame["anomaly_type"].iloc[i],
                "true_anomaly": bool(y_true[i]),
                "cluster": int(cluster_labels[i]),
                "values": {k: float(frame[k].iloc[i]) for k in feature_keys},
                "deviations": deviations,
            }
        )

    n_flagged = int(ensemble_pred.sum())
    return {
        "dataset": {
            "equipment": profile.key,
            "label": profile.label,
            "description": profile.description,
            "n_samples": len(frame),
            "n_features": len(feature_keys),
            "feature_names": [
                {
                    "key": s.key,
                    "label": s.label,
                    "unit": s.unit,
                    "low": s.low,
                    "high": s.high,
                    "decimals": s.decimals,
                }
                for s in profile.sensors
            ],
            "anomaly_rate": anomaly_rate,
            "n_true_anomalies": int(y_true.sum()),
            "seed": dataset.seed,
            "sample_interval_s": profile.sample_interval_s,
        },
        "series": series,
        "models": models_out,
        "ensemble": {
            "label": "Ensemble (mean score)",
            "model_names": model_names,
            "threshold": round(float(ensemble_threshold), 4),
            "metrics": ensemble_metrics,
            "scores": [round(float(s), 4) for s in ensemble_score],
            "predictions": [int(p) for p in ensemble_pred],
        },
        "projection": {"method": method_used, "points": projection_points},
        "clusters": cluster_summaries,
        "flagged": flagged,
        "summary": {
            "n_flagged": n_flagged,
            "n_models": len(model_names),
            "contamination": contamination,
        },
    }
