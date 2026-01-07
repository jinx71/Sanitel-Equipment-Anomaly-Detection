"""Tests for the detectors and the shared scoring helpers."""

import numpy as np
import pytest

from app.data import generator
from app.models.autoencoder import AutoencoderDetector
from app.models.base import normalize01, threshold_by_contamination
from app.models.registry import ORDER, build_detector, model_metadata


@pytest.fixture
def scaled_data():
    from sklearn.preprocessing import StandardScaler

    ds = generator.generate("wfi_loop", n_samples=300, anomaly_rate=0.06, seed=11)
    X = StandardScaler().fit_transform(ds.features)
    return X, ds.labels


def test_normalize01_bounds():
    out = normalize01(np.array([5.0, 10.0, 15.0, 20.0]))
    assert out.min() == 0.0 and out.max() == 1.0


def test_normalize01_constant_is_zero():
    out = normalize01(np.array([3.0, 3.0, 3.0]))
    assert np.all(out == 0.0)


def test_threshold_flags_expected_count():
    scores = np.linspace(0, 1, 100)
    preds, _ = threshold_by_contamination(scores, contamination=0.1)
    assert preds.sum() == 10


@pytest.mark.parametrize("name", ORDER)
def test_detector_scores_shape_and_finite(name, scaled_data):
    X, _ = scaled_data
    detector = build_detector(name, contamination=0.06)
    scores = detector.fit_score(X)
    assert scores.shape == (X.shape[0],)
    assert np.isfinite(scores).all()


def test_strong_detector_and_ensemble_separate_anomalies(scaled_data):
    """Isolation Forest (reliably) and most detectors should rank true
    anomalies above normals. Not every unsupervised detector separates well on
    every system - that is an honest property of the problem, not a bug."""
    X, y = scaled_data
    separating = 0
    for name in ORDER:
        scores = normalize01(build_detector(name, 0.06).fit_score(X))
        if scores[y == 1].mean() > scores[y == 0].mean():
            separating += 1
        if name == "isolation_forest":
            assert scores[y == 1].mean() > scores[y == 0].mean()
    assert separating >= 3


def test_autoencoder_learns_to_reconstruct():
    """Reconstruction error should fall substantially during training."""
    from sklearn.preprocessing import StandardScaler

    ds = generator.generate("bioreactor", n_samples=300, anomaly_rate=0.0, seed=2)
    X = StandardScaler().fit_transform(ds.features)
    ae = AutoencoderDetector(epochs=1, seed=0)
    err_start = ae.fit_score(X).mean()
    ae_trained = AutoencoderDetector(epochs=400, seed=0)
    err_end = ae_trained.fit_score(X).mean()
    assert err_end < err_start * 0.6


def test_model_metadata_complete():
    meta = model_metadata()
    assert len(meta) == len(ORDER)
    for m in meta:
        assert {"name", "label", "family", "description"} <= set(m)
