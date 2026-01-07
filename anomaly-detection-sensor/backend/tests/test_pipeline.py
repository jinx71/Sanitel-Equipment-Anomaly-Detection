"""Tests for the end-to-end analysis pipeline."""

import pytest

from app.pipeline import run_analysis

EXPECTED_KEYS = {
    "dataset",
    "series",
    "models",
    "ensemble",
    "projection",
    "clusters",
    "flagged",
    "summary",
}


def test_pipeline_returns_expected_structure():
    r = run_analysis("wfi_loop", n_samples=300, anomaly_rate=0.06, seed=1)
    assert EXPECTED_KEYS <= set(r)
    n = r["dataset"]["n_samples"]
    assert n == 300
    assert len(r["series"]) == n
    assert len(r["projection"]["points"]) == n
    assert len(r["ensemble"]["scores"]) == n
    for model in r["models"]:
        assert len(model["scores"]) == n
        assert len(model["predictions"]) == n
        assert {"precision", "recall", "f1", "roc_auc", "pr_auc"} <= set(model["metrics"])


def test_pipeline_respects_model_subset():
    r = run_analysis(
        "bioreactor", n_samples=300, seed=1, model_names=["isolation_forest", "lof"]
    )
    names = {m["name"] for m in r["models"]}
    assert names == {"isolation_forest", "lof"}
    assert r["summary"]["n_models"] == 2


def test_pipeline_flag_count_matches_contamination():
    r = run_analysis("cleanroom_hvac", n_samples=500, contamination=0.06, seed=4)
    expected = round(0.06 * 500)
    assert r["summary"]["n_flagged"] == expected
    assert len(r["flagged"]) == expected


def test_projection_method_selection():
    # PCA is always available and must be honoured exactly.
    r_pca = run_analysis("wfi_loop", n_samples=200, projection_method="pca", seed=1)
    assert r_pca["projection"]["method"] == "pca"
    # Requesting UMAP yields UMAP when installed, else a transparent PCA fallback.
    r_umap = run_analysis("wfi_loop", n_samples=200, projection_method="umap", seed=1)
    assert r_umap["projection"]["method"] in {"umap", "pca"}


def test_unknown_equipment_raises_keyerror():
    with pytest.raises(KeyError):
        run_analysis("does_not_exist", n_samples=200)


def test_flagged_entries_have_deviation_breakdown():
    r = run_analysis("wfi_loop", n_samples=300, seed=1)
    assert r["flagged"], "expected at least one flagged anomaly"
    first = r["flagged"][0]
    assert {"index", "ensemble_score", "models_agree", "values", "deviations"} <= set(first)
    assert len(first["deviations"]) <= 3
