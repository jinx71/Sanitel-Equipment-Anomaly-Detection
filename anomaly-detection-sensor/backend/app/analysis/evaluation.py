"""Detector evaluation against the synthetic ground-truth labels.

These labels are used here for *measurement only*. No detector ever sees them
during training, which is the whole point of an unsupervised pipeline: the
labels exist purely so we can quantify how well a label-free method recovers the
injected anomalies.

Threshold-free metrics (ROC-AUC, PR-AUC) judge the ranking quality of the
continuous scores; precision/recall/F1 judge the 0/1 decisions at the chosen
contamination budget. PR-AUC (average precision) is the headline metric because
anomalies are rare and class-imbalanced.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def evaluate(y_true: np.ndarray, scores: np.ndarray, preds: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    preds = np.asarray(preds, dtype=int)

    has_both_classes = 0 < int(y_true.sum()) < y_true.size

    roc_auc = float(roc_auc_score(y_true, scores)) if has_both_classes else None
    pr_auc = float(average_precision_score(y_true, scores)) if has_both_classes else None

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0
    )

    tp = int(np.sum((preds == 1) & (y_true == 1)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
