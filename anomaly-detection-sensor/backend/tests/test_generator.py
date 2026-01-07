"""Tests for the synthetic sensor data generator."""

import numpy as np
import pytest

from app.data import generator


def test_profiles_registered():
    keys = {p.key for p in generator.list_profiles()}
    assert keys == {"wfi_loop", "cleanroom_hvac", "bioreactor"}


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        generator.get_profile("nope")


@pytest.mark.parametrize("equipment", ["wfi_loop", "cleanroom_hvac", "bioreactor"])
def test_generate_shape_and_labels(equipment):
    ds = generator.generate(equipment, n_samples=400, anomaly_rate=0.06, seed=1)
    profile = ds.profile
    assert len(ds.frame) == 400
    assert ds.features.shape == (400, len(profile.feature_keys))
    assert set(ds.labels.tolist()) <= {0, 1}
    # 'index' and 'timestamp' columns are present alongside sensors + labels.
    for col in ("index", "timestamp", "is_anomaly", "anomaly_type"):
        assert col in ds.frame.columns
    assert np.isfinite(ds.features).all()


def test_anomaly_rate_is_approximate_sample_fraction():
    ds = generator.generate("wfi_loop", n_samples=1000, anomaly_rate=0.08, seed=3)
    frac = ds.labels.mean()
    # Target is on anomalous samples; allow a small overshoot from the final window.
    assert 0.07 <= frac <= 0.11


def test_zero_anomaly_rate_gives_clean_data():
    ds = generator.generate("bioreactor", n_samples=300, anomaly_rate=0.0, seed=5)
    assert ds.labels.sum() == 0
    assert ds.frame["anomaly_type"].isna().all()


def test_seed_is_reproducible():
    a = generator.generate("cleanroom_hvac", n_samples=300, anomaly_rate=0.06, seed=42)
    b = generator.generate("cleanroom_hvac", n_samples=300, anomaly_rate=0.06, seed=42)
    np.testing.assert_array_equal(a.features, b.features)
    np.testing.assert_array_equal(a.labels, b.labels)


def test_all_anomaly_families_appear_over_many_samples():
    ds = generator.generate("wfi_loop", n_samples=2000, anomaly_rate=0.15, seed=9)
    types = set(t for t in ds.frame["anomaly_type"].tolist() if t is not None)
    # With a large budget every family should be exercised at least once.
    assert types == set(generator.ANOMALY_TYPES)
