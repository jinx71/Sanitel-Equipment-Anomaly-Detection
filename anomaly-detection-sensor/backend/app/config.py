"""Application configuration sourced from environment variables."""

from __future__ import annotations

import os

APP_NAME = "Sentinel - Equipment Anomaly Detection"
APP_VERSION = "1.0.0"


def cors_origins() -> list[str]:
    """Comma-separated allowed origins; defaults to local Vite dev servers."""
    raw = os.environ.get(
        "CLIENT_URL",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]
