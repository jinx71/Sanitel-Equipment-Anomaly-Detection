"""API tests via FastAPI's TestClient."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["status"] == "healthy"


def test_envelope_shape():
    body = client.get("/").json()
    assert set(body) == {"success", "data", "message"}


def test_list_equipment():
    body = client.get("/api/equipment").json()
    assert body["success"] is True
    keys = {p["key"] for p in body["data"]}
    assert keys == {"wfi_loop", "cleanroom_hvac", "bioreactor"}


def test_list_models():
    body = client.get("/api/models").json()
    assert len(body["data"]) == 5


def test_analyze_endpoint():
    resp = client.post(
        "/api/analyze",
        json={"equipment": "wfi_loop", "n_samples": 300, "seed": 1, "contamination": 0.06},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["dataset"]["n_samples"] == 300
    assert len(data["models"]) == 5
    assert data["projection"]["method"] in {"pca", "umap"}


def test_analyze_unknown_equipment_returns_404():
    resp = client.post("/api/analyze", json={"equipment": "nope", "n_samples": 200})
    assert resp.status_code == 404


def test_analyze_validation_error_on_bad_samples():
    resp = client.post("/api/analyze", json={"equipment": "wfi_loop", "n_samples": 10})
    assert resp.status_code == 422  # below ge=100 bound


def test_generate_endpoint():
    resp = client.post("/api/generate", json={"equipment": "bioreactor", "n_samples": 200, "seed": 2})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data["records"]) == 200
