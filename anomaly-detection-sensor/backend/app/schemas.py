"""Request schemas and validation for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    equipment: str = Field(default="wfi_loop", description="Equipment profile key")
    n_samples: int = Field(default=600, ge=100, le=5000)
    anomaly_rate: float = Field(default=0.06, ge=0.0, le=0.30)
    contamination: float = Field(default=0.06, ge=0.01, le=0.30)
    seed: int | None = Field(default=None)
    models: list[str] | None = Field(
        default=None, description="Subset of detector names; null runs all"
    )
    projection_method: str = Field(default="pca", pattern="^(pca|umap)$")


class GenerateRequest(BaseModel):
    equipment: str = Field(default="wfi_loop")
    n_samples: int = Field(default=600, ge=100, le=5000)
    anomaly_rate: float = Field(default=0.06, ge=0.0, le=0.30)
    seed: int | None = Field(default=None)
