"""Registry and factory for the available unsupervised detectors."""

from __future__ import annotations

from .autoencoder import AutoencoderDetector
from .base import BaseDetector
from .isolation_forest import IsolationForestDetector
from .lof import LOFDetector
from .ocsvm import OneClassSVMDetector
from .pca_reconstruction import PCAReconstructionDetector

# Canonical display order.
ORDER = ["isolation_forest", "lof", "ocsvm", "pca_reconstruction", "autoencoder"]

_CLASSES = {
    "isolation_forest": IsolationForestDetector,
    "lof": LOFDetector,
    "ocsvm": OneClassSVMDetector,
    "pca_reconstruction": PCAReconstructionDetector,
    "autoencoder": AutoencoderDetector,
}

# Coarse family label shown in the UI to highlight methodological diversity.
_FAMILY = {
    "isolation_forest": "Tree-based",
    "lof": "Density-based",
    "ocsvm": "Boundary-based",
    "pca_reconstruction": "Reconstruction (linear)",
    "autoencoder": "Reconstruction (neural)",
}


def available_models() -> list[str]:
    return list(ORDER)


def build_detector(name: str, contamination: float) -> BaseDetector:
    """Instantiate a detector, wiring the contamination budget where it applies."""
    if name not in _CLASSES:
        raise KeyError(f"Unknown model: {name!r}")
    if name == "isolation_forest":
        return IsolationForestDetector(contamination=contamination)
    if name == "lof":
        return LOFDetector(contamination=contamination)
    if name == "ocsvm":
        return OneClassSVMDetector(nu=contamination)
    if name == "pca_reconstruction":
        return PCAReconstructionDetector()
    return AutoencoderDetector()


def model_metadata() -> list[dict]:
    """Static metadata for the model picker (no fitting required)."""
    meta = []
    for name in ORDER:
        cls = _CLASSES[name]
        meta.append(
            {
                "name": cls.name,
                "label": cls.label,
                "family": _FAMILY[name],
                "description": cls.description,
            }
        )
    return meta
