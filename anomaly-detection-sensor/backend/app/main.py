"""FastAPI application exposing the anomaly-detection pipeline.

All responses use the project-standard envelope: {success, data, message}.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .data import generator
from .models.registry import model_metadata
from .pipeline import run_analysis
from .schemas import AnalyzeRequest, GenerateRequest

app = FastAPI(title=config.APP_NAME, version=config.APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def ok(data, message: str = ""):
    return {"success": True, "data": data, "message": message}


@app.get("/")
def root():
    return ok(
        {"name": config.APP_NAME, "version": config.APP_VERSION},
        "Unsupervised anomaly detection for pharmaceutical equipment telemetry.",
    )


@app.get("/api/health")
def health():
    return ok({"status": "healthy"}, "Service is up.")


@app.get("/api/equipment")
def list_equipment():
    profiles = [
        {
            "key": p.key,
            "label": p.label,
            "description": p.description,
            "n_features": len(p.sensors),
            "sensors": [
                {"key": s.key, "label": s.label, "unit": s.unit} for s in p.sensors
            ],
        }
        for p in generator.list_profiles()
    ]
    return ok(profiles, f"{len(profiles)} equipment profiles available.")


@app.get("/api/models")
def list_models():
    meta = model_metadata()
    return ok(meta, f"{len(meta)} unsupervised detectors available.")


@app.post("/api/generate")
def generate_dataset(req: GenerateRequest):
    try:
        ds = generator.generate(
            equipment=req.equipment,
            n_samples=req.n_samples,
            anomaly_rate=req.anomaly_rate,
            seed=req.seed,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return ok(
        {
            "equipment": ds.profile.key,
            "label": ds.profile.label,
            "n_samples": len(ds.frame),
            "n_true_anomalies": int(ds.labels.sum()),
            "seed": ds.seed,
            "records": ds.frame.to_dict(orient="records"),
        },
        f"Generated {len(ds.frame)} readings ({int(ds.labels.sum())} anomalies).",
    )


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    try:
        result = run_analysis(
            equipment=req.equipment,
            n_samples=req.n_samples,
            anomaly_rate=req.anomaly_rate,
            seed=req.seed,
            model_names=req.models,
            contamination=req.contamination,
            projection_method=req.projection_method,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    s = result["summary"]
    return ok(
        result,
        f"Analysed {result['dataset']['n_samples']} readings with "
        f"{s['n_models']} detectors; ensemble flagged {s['n_flagged']} anomalies.",
    )
