"""Synthetic pharmaceutical equipment sensor telemetry generator.

Produces multivariate time-series readings for three GMP-relevant systems and
injects four distinct anomaly families. The generator deliberately builds
*correlations* between sensors (e.g. flow drives pressure, agitation drives
dissolved oxygen) so that "correlation-break" anomalies are physically
meaningful and invisible to per-sensor threshold alarms.

Ground-truth labels (``is_anomaly`` / ``anomaly_type``) are produced for
*evaluation only* and are never passed to any detector during training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Anomaly families injected by the generator. Spikes and correlation-breaks are
# single-sample (point) events; drift and excursion are sustained (window) events.
ANOMALY_TYPES = ("spike", "drift", "correlation_break", "excursion")

# Window length (in samples) for sustained anomalies.
_WINDOW_MIN = 6
_WINDOW_MAX = 14


@dataclass(frozen=True)
class SensorSpec:
    key: str
    label: str
    unit: str
    # Plausible operating band, used by the UI for trend-chart reference lines.
    low: float
    high: float
    decimals: int = 2


@dataclass(frozen=True)
class EquipmentProfile:
    key: str
    label: str
    description: str
    sensors: tuple[SensorSpec, ...]
    sample_interval_s: int = 60

    @property
    def feature_keys(self) -> list[str]:
        return [s.key for s in self.sensors]


# --------------------------------------------------------------------------- #
# Equipment profiles
# --------------------------------------------------------------------------- #

WFI_LOOP = EquipmentProfile(
    key="wfi_loop",
    label="WFI Distribution Loop",
    description=(
        "Hot Water-for-Injection storage and distribution loop. Held >80 C to "
        "inhibit microbial growth; conductivity and TOC are compendial quality "
        "attributes (USP <645>/<643>)."
    ),
    sensors=(
        SensorSpec("supply_temp", "Supply temp", "C", 80.0, 85.0, 2),
        SensorSpec("return_temp", "Return temp", "C", 77.5, 83.0, 2),
        SensorSpec("conductivity", "Conductivity", "uS/cm", 0.5, 1.3, 3),
        SensorSpec("toc", "TOC", "ppb", 120.0, 500.0, 0),
        SensorSpec("flow_rate", "Flow rate", "L/min", 75.0, 110.0, 1),
        SensorSpec("pressure", "Loop pressure", "bar", 1.3, 2.0, 3),
    ),
)

CLEANROOM_HVAC = EquipmentProfile(
    key="cleanroom_hvac",
    label="Cleanroom HVAC",
    description=(
        "Grade C/D cleanroom air handling. Positive differential pressure "
        "cascade keeps particles out; particle counts rise when the pressure "
        "cascade is lost (EU GMP Annex 1)."
    ),
    sensors=(
        SensorSpec("room_temp", "Room temp", "C", 18.0, 24.0, 2),
        SensorSpec("humidity", "Humidity", "%RH", 30.0, 60.0, 1),
        SensorSpec("diff_pressure", "Diff. pressure", "Pa", 8.0, 18.0, 1),
        SensorSpec("particles_05", "Particles >=0.5um", "ct/m3", 0.0, 200000.0, 0),
        SensorSpec("particles_5", "Particles >=5.0um", "ct/m3", 0.0, 4000.0, 0),
    ),
)

BIOREACTOR = EquipmentProfile(
    key="bioreactor",
    label="Bioreactor",
    description=(
        "Mammalian cell-culture bioreactor at steady state. Dissolved oxygen is "
        "sustained by agitation; pH and temperature are tightly controlled "
        "critical process parameters."
    ),
    sensors=(
        SensorSpec("temp", "Temperature", "C", 36.5, 37.5, 2),
        SensorSpec("ph", "pH", "", 6.9, 7.3, 3),
        SensorSpec("dissolved_oxygen", "Dissolved O2", "%sat", 30.0, 50.0, 1),
        SensorSpec("agitation", "Agitation", "rpm", 130.0, 170.0, 0),
        SensorSpec("optical_density", "Optical density", "OD600", 1.5, 4.0, 2),
    ),
)

PROFILES: dict[str, EquipmentProfile] = {
    p.key: p for p in (WFI_LOOP, CLEANROOM_HVAC, BIOREACTOR)
}


def list_profiles() -> list[EquipmentProfile]:
    return list(PROFILES.values())


def get_profile(key: str) -> EquipmentProfile:
    if key not in PROFILES:
        raise KeyError(f"Unknown equipment profile: {key!r}")
    return PROFILES[key]


# --------------------------------------------------------------------------- #
# Normal (in-control) data generation
# --------------------------------------------------------------------------- #

def _normal_wfi_loop(rng: np.random.Generator, n: int) -> pd.DataFrame:
    supply_temp = rng.normal(82.0, 0.7, n)
    # Return temp tracks supply with a small, stable delta across the loop.
    return_temp = supply_temp - rng.normal(2.4, 0.35, n)
    flow_rate = rng.normal(92.0, 7.0, n)
    # Pressure is driven by flow (turbulent loop): higher flow -> higher pressure.
    pressure = 1.60 + 0.012 * (flow_rate - 92.0) + rng.normal(0.0, 0.04, n)
    conductivity = rng.normal(1.00, 0.12, n)
    toc = rng.normal(210.0, 28.0, n)
    return pd.DataFrame(
        {
            "supply_temp": supply_temp,
            "return_temp": return_temp,
            "conductivity": conductivity,
            "toc": toc,
            "flow_rate": flow_rate,
            "pressure": pressure,
        }
    )


def _normal_cleanroom_hvac(rng: np.random.Generator, n: int) -> pd.DataFrame:
    room_temp = rng.normal(21.0, 0.4, n)
    humidity = rng.normal(45.0, 3.0, n)
    diff_pressure = rng.normal(12.5, 1.0, n)
    # Particle load falls as the pressure cascade strengthens.
    particles_05 = np.clip(
        rng.normal(80000.0, 12000.0, n) - 2200.0 * (diff_pressure - 12.5),
        500.0,
        None,
    )
    # 5um counts are a small, noisy fraction of 0.5um counts.
    particles_5 = np.clip(
        particles_05 * rng.normal(0.020, 0.004, n) + rng.normal(0.0, 30.0, n),
        0.0,
        None,
    )
    return pd.DataFrame(
        {
            "room_temp": room_temp,
            "humidity": humidity,
            "diff_pressure": diff_pressure,
            "particles_05": particles_05,
            "particles_5": particles_5,
        }
    )


def _normal_bioreactor(rng: np.random.Generator, n: int) -> pd.DataFrame:
    temp = rng.normal(37.0, 0.25, n)
    ph = rng.normal(7.10, 0.04, n)
    agitation = rng.normal(150.0, 8.0, n)
    # Dissolved oxygen is sustained by agitation (gas-liquid mass transfer).
    dissolved_oxygen = 40.0 + 0.15 * (agitation - 150.0) + rng.normal(0.0, 1.6, n)
    optical_density = rng.normal(2.5, 0.30, n)
    return pd.DataFrame(
        {
            "temp": temp,
            "ph": ph,
            "dissolved_oxygen": dissolved_oxygen,
            "agitation": agitation,
            "optical_density": optical_density,
        }
    )


_NORMAL_BUILDERS = {
    "wfi_loop": _normal_wfi_loop,
    "cleanroom_hvac": _normal_cleanroom_hvac,
    "bioreactor": _normal_bioreactor,
}


# --------------------------------------------------------------------------- #
# Anomaly injection (equipment-specific physics)
# --------------------------------------------------------------------------- #

def _inject_wfi_loop(rng, df, idx, kind, window):
    if kind == "spike":
        choice = rng.integers(0, 4)
        if choice == 0:
            df.loc[idx, "supply_temp"] += rng.uniform(9.0, 13.0)   # probe / heater glitch
        elif choice == 1:
            df.loc[idx, "conductivity"] = rng.uniform(2.6, 4.2)    # ion breakthrough
        elif choice == 2:
            df.loc[idx, "pressure"] += rng.uniform(0.5, 0.8)       # water-hammer
        else:
            df.loc[idx, "toc"] = rng.uniform(620.0, 900.0)         # organic excursion
    elif kind == "drift":
        # Gradual fouling: conductivity (or TOC) ramps over the window.
        ramp = np.linspace(0.0, rng.uniform(1.8, 2.6), len(window))
        if rng.random() < 0.5:
            df.loc[window, "conductivity"] = df.loc[window, "conductivity"].to_numpy() + ramp
        else:
            df.loc[window, "toc"] = df.loc[window, "toc"].to_numpy() + ramp * 200.0
    elif kind == "correlation_break":
        # Marginals stay plausible but the joint relationship is broken.
        choice = rng.integers(0, 2)
        if choice == 0:
            # Flow falls but pressure stays high (downstream blockage).
            df.loc[idx, "flow_rate"] = rng.uniform(64.0, 70.0)
            df.loc[idx, "pressure"] = rng.uniform(1.62, 1.72)
        else:
            # Return temp exceeds supply temp (impossible heat flow -> sensor fault).
            df.loc[idx, "return_temp"] = df.loc[idx, "supply_temp"] + rng.uniform(0.8, 1.8)
    elif kind == "excursion":
        # Sustained low circulation (loop pump degradation).
        df.loc[window, "flow_rate"] = rng.uniform(58.0, 64.0, len(window))
        df.loc[window, "pressure"] = 1.60 + 0.012 * (
            df.loc[window, "flow_rate"].to_numpy() - 92.0
        ) + rng.normal(0.0, 0.04, len(window))


def _inject_cleanroom_hvac(rng, df, idx, kind, window):
    if kind == "spike":
        burst = rng.uniform(5.0, 8.0)
        df.loc[idx, "particles_05"] *= burst
        df.loc[idx, "particles_5"] *= burst
    elif kind == "drift":
        # Dehumidifier degradation: humidity climbs over the window.
        ramp = np.linspace(0.0, rng.uniform(14.0, 20.0), len(window))
        df.loc[window, "humidity"] = df.loc[window, "humidity"].to_numpy() + ramp
    elif kind == "correlation_break":
        # Particle excursion with an intact pressure cascade (internal source).
        df.loc[idx, "particles_05"] *= rng.uniform(4.0, 6.0)
        df.loc[idx, "particles_5"] *= rng.uniform(8.0, 12.0)
        # diff_pressure left at its normal value -> pressure-only alarm misses it.
    elif kind == "excursion":
        # Lost pressure cascade (door held open / fan fault) with particle ingress.
        df.loc[window, "diff_pressure"] = rng.uniform(-2.0, 3.0, len(window))
        df.loc[window, "particles_05"] = df.loc[window, "particles_05"].to_numpy() * rng.uniform(
            2.5, 4.0, len(window)
        )


def _inject_bioreactor(rng, df, idx, kind, window):
    if kind == "spike":
        if rng.random() < 0.5:
            df.loc[idx, "temp"] += rng.uniform(1.6, 2.4)          # probe glitch
        else:
            df.loc[idx, "ph"] -= rng.uniform(0.35, 0.55)          # transient acidosis
    elif kind == "drift":
        # Acid accumulation / contamination: pH drifts down over the window.
        ramp = np.linspace(0.0, rng.uniform(0.30, 0.45), len(window))
        df.loc[window, "ph"] = df.loc[window, "ph"].to_numpy() - ramp
    elif kind == "correlation_break":
        # Oxygen demand surge: DO collapses while agitation stays normal.
        df.loc[idx, "dissolved_oxygen"] = rng.uniform(22.0, 28.0)
        # agitation untouched -> DO-vs-agitation relationship broken.
    elif kind == "excursion":
        # Jacket control fault: sustained temperature elevation.
        df.loc[window, "temp"] = rng.uniform(37.8, 38.3, len(window))


_INJECTORS = {
    "wfi_loop": _inject_wfi_loop,
    "cleanroom_hvac": _inject_cleanroom_hvac,
    "bioreactor": _inject_bioreactor,
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

@dataclass
class Dataset:
    profile: EquipmentProfile
    frame: pd.DataFrame  # sensor columns + index/timestamp + is_anomaly + anomaly_type
    seed: int

    @property
    def features(self) -> np.ndarray:
        return self.frame[self.profile.feature_keys].to_numpy(dtype=float)

    @property
    def labels(self) -> np.ndarray:
        return self.frame["is_anomaly"].to_numpy(dtype=int)


def generate(
    equipment: str,
    n_samples: int = 500,
    anomaly_rate: float = 0.06,
    seed: int | None = None,
) -> Dataset:
    """Generate a synthetic telemetry dataset with injected anomalies."""
    profile = get_profile(equipment)
    n_samples = int(np.clip(n_samples, 100, 5000))
    anomaly_rate = float(np.clip(anomaly_rate, 0.0, 0.30))
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    rng = np.random.default_rng(seed)

    df = _NORMAL_BUILDERS[equipment](rng, n_samples)
    df["is_anomaly"] = False
    df["anomaly_type"] = None

    inject = _INJECTORS[equipment]
    target_anomalies = int(round(anomaly_rate * n_samples))

    # Inject a mix of point and window anomalies until the target fraction of
    # anomalous *samples* is reached, so anomaly_rate is directly comparable to
    # the detector contamination budget.
    taken = np.zeros(n_samples, dtype=bool)
    attempts = 0
    max_attempts = max(50, target_anomalies * 50)
    while int(taken.sum()) < target_anomalies and attempts < max_attempts:
        attempts += 1
        remaining = target_anomalies - int(taken.sum())
        # Near the end of the budget, only point anomalies can fit cleanly.
        force_point = remaining < _WINDOW_MIN
        kind = (
            rng.choice(["spike", "correlation_break"])
            if force_point
            else rng.choice(ANOMALY_TYPES)
        )
        if kind in ("drift", "excursion"):
            w = min(int(rng.integers(_WINDOW_MIN, _WINDOW_MAX + 1)), remaining)
            start = int(rng.integers(0, max(1, n_samples - w)))
            window = list(range(start, start + w))
            if taken[window].any():
                continue  # avoid overlapping sustained events
            taken[window] = True
            inject(rng, df, start, kind, window)
            df.loc[window, "is_anomaly"] = True
            df.loc[window, "anomaly_type"] = kind
        else:
            free = np.flatnonzero(~taken)
            if free.size == 0:
                break
            idx = int(rng.choice(free))
            taken[idx] = True
            inject(rng, df, idx, kind, [idx])
            df.loc[idx, "is_anomaly"] = True
            df.loc[idx, "anomaly_type"] = kind

    # Clamp physically-bounded sensors to non-negative ranges after injection.
    for col in ("conductivity", "toc", "flow_rate", "particles_05", "particles_5", "optical_density"):
        if col in df.columns:
            df[col] = df[col].clip(lower=0.0)

    base_ts = datetime(2025, 1, 1, 0, 0, 0)
    df.insert(0, "index", np.arange(n_samples))
    df.insert(
        1,
        "timestamp",
        [
            (base_ts + timedelta(seconds=i * profile.sample_interval_s)).isoformat()
            for i in range(n_samples)
        ],
    )

    # Round sensor columns to their display precision for clean payloads.
    for spec in profile.sensors:
        df[spec.key] = df[spec.key].round(spec.decimals + 2)

    return Dataset(profile=profile, frame=df, seed=seed)
