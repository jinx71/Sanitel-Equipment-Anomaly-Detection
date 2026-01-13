# Sentinel — Unsupervised Anomaly Detection on Equipment Sensor Data

Sentinel detects anomalies in pharmaceutical equipment telemetry **without labels**.
It runs five complementary unsupervised detectors over multivariate sensor streams
(WFI distribution loops, cleanroom HVAC, bioreactors), blends them into an ensemble,
clusters the flagged readings into distinct failure signatures, and visualises
everything in a dark instrument-console dashboard.

The motivating problem comes from real GMP manufacturing: an Environmental Monitoring
System raises an alarm only when a *single* tag breaches a *fixed* limit. It cannot see
a **correlation break** — e.g. dissolved oxygen collapsing while agitation looks normal,
or loop flow and pressure drifting apart — even though those are exactly the early signs
of equipment trouble. Sentinel is built to catch the multivariate and drift anomalies a
univariate alarm misses.

> **Detection is fully unsupervised.** Ground-truth labels are generated alongside the
> synthetic data and used *only* to score the detectors after the fact. No model ever
> sees a label during training. This is enforced in code and asserted in the test suite.

---

## Contents

- [Architecture](#architecture)
- [The five detectors](#the-five-detectors)
- [Key design decisions](#key-design-decisions)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Configuration](#configuration)
- [API reference](#api-reference)
- [Testing](#testing)
- [Deployment](#deployment)
- [Interview talking points](#interview-talking-points)
- [Limitations & honest notes](#limitations--honest-notes)

---

## Architecture

```
┌─────────────────────────────┐         ┌──────────────────────────────────────┐
│  React + TypeScript (Vite)  │  HTTP   │            FastAPI backend             │
│                             │ ───────▶│                                        │
│  • Control rail             │  /api   │  generate → standardise → detect (×5)  │
│  • Latent operating map     │◀─────── │     → ensemble → cluster → project     │
│  • Model comparison         │  JSON   │     → evaluate                         │
│  • Score separation         │         │                                        │
│  • Sensor trends            │         │  scikit-learn · NumPy autoencoder      │
│  • Cluster signatures       │         │  DBSCAN · PCA / UMAP                    │
│  • Flagged-reading table    │         │                                        │
└─────────────────────────────┘         └──────────────────────────────────────┘
```

The backend is a single analysis pipeline (`app/pipeline.py`) exposed over a thin
FastAPI layer. One `POST /api/analyze` call returns everything the dashboard needs:
per-sample series, per-model scores and metrics, the ensemble, a 2D projection, the
anomaly clusters, and a ranked table of flagged readings.

### The data

There is no public, labelled pharma-telemetry dataset, so Sentinel ships a physically
plausible **synthetic generator** (`app/data/generator.py`) with three equipment
profiles. Each profile encodes realistic multivariate correlations between sensors
(flow → pressure, supply → return temperature, agitation → dissolved oxygen,
differential pressure → particle count), then injects four families of anomaly:

| Family | What it models | Why it matters |
|---|---|---|
| **Spike** | A single-sample transient on one sensor | The easy case; even a univariate alarm catches it |
| **Drift** | A slow ramp over a window | Sensor fouling / calibration loss; below alarm limits for a long time |
| **Correlation break** | Two coupled sensors decouple | **Invisible to any single-tag alarm** — the core motivation |
| **Sustained excursion** | A multi-sample out-of-band event | A genuine process upset |

---

## The five detectors

The detectors are deliberately chosen to span the methodological taxonomy, so the
project demonstrates breadth rather than five variations on one idea.

| Detector | Family | Core idea |
|---|---|---|
| **Isolation Forest** | Tree-based | Anomalies are isolated in fewer random splits |
| **Local Outlier Factor** | Density-based | Anomalies sit in lower-density neighbourhoods than their neighbours |
| **One-Class SVM** | Boundary-based | Learn a frontier enclosing normal data; score by distance outside it |
| **PCA reconstruction** | Reconstruction (linear) | Project to top components and back; large residual ⇒ anomaly |
| **Autoencoder** | Reconstruction (neural) | **Hand-written NumPy** autoencoder; large reconstruction error ⇒ anomaly |

Every detector implements the same tiny interface (`fit_score(X) → higher = more
anomalous`), scores are min-max normalised, and a contamination budget converts scores
to 0/1 decisions. The **ensemble** is the mean of the normalised scores.

---

## Key design decisions

**Unsupervised by construction; labels for evaluation only.**
Detectors are fit on the bare feature matrix. Labels flow exclusively into the metrics
step (`app/analysis/evaluation.py`). *Why:* real anomaly detection rarely has labels —
the interesting question is how well a label-free method recovers known anomalies, and
the architecture makes accidental leakage impossible.

**The autoencoder is implemented from scratch in NumPy.**
Manual forward pass, backprop, and Adam optimiser; architecture `d → 8 → bottleneck → 8
→ d` with tanh activations (`app/models/autoencoder.py`). *Why:* it shows the mechanics
of reconstruction-based detection rather than hiding them behind a framework, and it
keeps the image lean and deployable on a free tier (no TensorFlow/PyTorch download).

**`anomaly_rate` targets the fraction of anomalous *samples*, not events.**
Window anomalies (drift, excursion) span many samples, so counting events makes the true
anomaly fraction unpredictable and structurally caps recall. *Why:* targeting samples
makes `anomaly_rate` directly comparable to the detector `contamination` budget, so
precision/recall are meaningful instead of artefacts of window length.

**The ensemble averages normalised scores, not ranks.**
Rank-averaging was implemented and benchmarked — it **regressed**, because it discards
the confident bimodal magnitude separation that a strong detector (Isolation Forest)
produces. *Why:* the honest, measured choice beat the "sophisticated" one. See the
talking points for the full story.

**UMAP is optional with a transparent PCA fallback.**
The 2D projection prefers UMAP for cluster separation but lazily imports it and falls
back to PCA when it (and its heavy `numba` dependency) is absent (`app/analysis/
projection.py`). The method actually used is returned and labelled in the UI. *Why:*
better visuals where the dependency is available, without bloating the default install
or breaking deploys where it is not.

**Flagged anomalies are clustered with DBSCAN into signatures.**
Clustering runs in standardised space, so each cluster's mean coordinate on a feature is
its z-score deviation. *Why:* it turns a pile of flagged points into a handful of named
failure modes ("differential pressure −5.5σ + particles +2.8σ"), which is what an
engineer actually needs.

---

## Tech stack

**Backend:** Python 3.12 · FastAPI · scikit-learn · NumPy · pandas · pydantic
**Frontend:** React 18 · TypeScript · Vite · Tailwind CSS · Recharts · Axios
**Tooling:** pytest · Docker · docker-compose · nginx

---

## Project structure

```
anomaly-detection-sensor/
├── backend/
│   ├── app/
│   │   ├── data/generator.py        # synthetic equipment telemetry + anomalies
│   │   ├── models/                  # 5 detectors + base interface + registry
│   │   │   ├── base.py
│   │   │   ├── isolation_forest.py
│   │   │   ├── lof.py
│   │   │   ├── ocsvm.py
│   │   │   ├── pca_reconstruction.py
│   │   │   ├── autoencoder.py        # from-scratch NumPy autoencoder
│   │   │   └── registry.py
│   │   ├── analysis/
│   │   │   ├── clustering.py         # DBSCAN anomaly signatures
│   │   │   ├── projection.py         # UMAP → PCA fallback
│   │   │   └── evaluation.py         # metrics (eval only)
│   │   ├── pipeline.py               # orchestration
│   │   ├── schemas.py · config.py · main.py
│   ├── tests/                        # 34 pytest tests
│   ├── requirements.txt · requirements-dev.txt · Dockerfile · .env.example
├── frontend/
│   ├── src/
│   │   ├── components/               # ControlPanel, LatentMap, MetricsBar, …
│   │   ├── api/client.ts · types/ · lib/format.ts
│   │   ├── App.tsx · main.tsx · index.css
│   ├── Dockerfile · nginx.conf · vite/tailwind/postcss configs
├── docker-compose.yml
└── README.md
```

---

## Getting started

### Option A — Docker (one command)

```bash
docker compose up --build
```

- Dashboard: <http://localhost:8080>
- API + interactive docs: <http://localhost:8000/docs>

nginx serves the built frontend and proxies `/api` to the backend, so there is no CORS
configuration to worry about locally.

### Option B — run the two services directly

**Backend**

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend** (in a second terminal)

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173, proxies /api to :8000
```

The dashboard auto-runs one analysis on load, so it is populated immediately.

---

## Configuration

**Backend** (`backend/.env`)

| Variable | Default | Purpose |
|---|---|---|
| `CLIENT_URL` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated CORS origins (no trailing slash) |
| `PORT` | `8000` | Port to listen on (Render/Railway inject this) |

**Frontend** (`frontend/.env`)

| Variable | Default | Purpose |
|---|---|---|
| `VITE_API_URL` | *(empty)* | Deployed API origin in production. Empty in dev so Vite proxies `/api`. **Baked at build time — changing it requires a rebuild.** |

To enable UMAP projections, uncomment `umap-learn` in `backend/requirements.txt` and
reinstall. Without it, requesting UMAP transparently returns a PCA projection.

---

## API reference

All responses use the envelope `{ "success": bool, "data": ..., "message": str }`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Liveness check |
| `GET` | `/api/equipment` | List equipment profiles and their sensors |
| `GET` | `/api/models` | List the available detectors and their families |
| `POST` | `/api/generate` | Generate a raw labelled dataset (no detection) |
| `POST` | `/api/analyze` | Run the full detection pipeline |

### `POST /api/analyze`

| Field | Type | Default | Bounds |
|---|---|---|---|
| `equipment` | string | `wfi_loop` | one of the profile keys |
| `n_samples` | int | `600` | 100–5000 |
| `anomaly_rate` | float | `0.06` | 0.0–0.30 (fraction of anomalous samples) |
| `contamination` | float | `0.06` | 0.01–0.30 (detector budget) |
| `seed` | int \| null | `null` | — |
| `models` | string[] \| null | `null` (all) | subset of detector names |
| `projection_method` | string | `pca` | `pca` \| `umap` |

The response `data` contains `dataset`, `series`, `models[]`, `ensemble`, `projection`,
`clusters[]`, `flagged[]`, and `summary`.

---

## Testing

```bash
cd backend
pip install -r requirements-dev.txt
pytest -q
```

34 tests cover the generator (shape, reproducibility, anomaly-rate calibration, all
families appearing), each detector (finite scores, separation, that the autoencoder
actually learns), the pipeline (structure, model subsetting, flag counts, projection
fallback), and every API endpoint (envelope, validation, 404s).

A core invariant under test: detectors are never passed labels, and the strong detectors
plus the ensemble rank true anomalies above normal readings.

---

## Deployment

Mirrors a standard split deploy:

- **Backend → Render / Railway.** Containerised; reads `$PORT`. Set `CLIENT_URL` to the
  deployed frontend origin.
- **Frontend → Vercel.** Build `npm run build`, output `dist/`. Set `VITE_API_URL` to the
  deployed API origin and redeploy (Vite inlines env vars at build time).
- **Both → any Docker host.** `docker compose up --build` brings up the full stack behind
  nginx with same-origin `/api`.

---

## Interview talking points

**Supervised vs unsupervised — and why this is the unsupervised half.**
Classification needs labelled failures, which manufacturing almost never has. Sentinel
learns the shape of *normal* and flags departures from it, so it works on day one with
zero labelled anomalies. The synthetic labels exist only to *measure* the detectors —
demonstrating the methodology while still quantifying it.

**The anomaly a univariate alarm cannot see.**
A SCADA/EMS alarm fires when one tag crosses one limit. A **correlation break** — DO
falling while agitation holds, or flow and pressure decoupling — keeps every individual
tag inside its band, so no alarm fires, yet it is a real fault. Standardising and scoring
the *joint* distribution is what surfaces it. The trend panel shows flagged readings
sitting inside every single-sensor band — a concrete picture of what multivariate
detection buys you. This is a problem I lived during GMP audits.

**Reconstruction-error detection, hand-built.**
The autoencoder is raw NumPy: forward pass, backprop, Adam. It compresses normal data
through a bottleneck and reconstructs it; because it only ever saw normal patterns,
anomalies reconstruct poorly and the residual is the anomaly score. Implementing it by
hand means I can explain every line — gradients, learning rate, why tanh — rather than
pointing at `model.fit`.

**Why PR-AUC is the headline, not accuracy or ROC-AUC.**
Anomalies are rare (~6%), so accuracy is useless (predict "normal" → 94% accurate, 0
anomalies caught). ROC-AUC is optimistic under heavy class imbalance because the huge
normal class inflates the true-negative rate. **PR-AUC (average precision)** focuses on
the positive class and is the honest summary for rare-event detection.

**Threshold-free vs thresholded metrics.**
ROC-AUC and PR-AUC judge the *ranking* the scores produce — independent of any cutoff.
Precision/recall/F1 judge the 0/1 decisions at the chosen contamination budget. Reporting
both separates "is the model's ordering good?" from "is the operating point right?", which
are different questions an operator cares about independently.

**Why a naive ensemble does not always win — a real result, not a slogan.**
I benchmarked rank-averaging against mean-of-normalised-scores. Rank-averaging *lost*,
because it throws away the confident, bimodal magnitude separation that Isolation Forest
produces and lets weak detectors drag the strong one down. On some equipment the best
single detector still beats the equal-weight ensemble. The lesson: ensembles help when
members are *diverse and comparably good*; blindly averaging in a weak member hurts. I
kept the choice that the measurements supported and documented the finding instead of
hiding it.

**DBSCAN turns alerts into failure modes.**
A list of 30 flagged timestamps is noise to an engineer. Clustering them in standardised
space groups readings with the same signature and labels each by its dominant sensor
deviations ("differential pressure −5.5σ, particles +2.8σ"), which maps directly to a
likely root cause and tells you which alerts are the same underlying event.

**Engineering choices that make it production-shaped.**
A `fit_score` interface every detector shares; an `app.js`/server split so endpoints test
without a live server; a lean dependency set and from-scratch autoencoder so the container
deploys on a free tier; an optional heavy dependency (UMAP) behind a graceful fallback;
and 34 tests that pin the behaviour, including the no-label-leakage invariant.

---

## Limitations & honest notes

- The data is **synthetic**. The correlations and anomaly families are physically
  motivated and make the detection problem realistic, but results on real plant data
  would differ.
- Detection is **unsupervised and point/window-based**; it does not model long-range
  temporal dependence (no RNN/transformer over the sequence). That is a deliberate scope
  choice — the focus is the unsupervised multivariate toolkit.
- The ensemble is intentionally simple (equal-weight mean). Weighting by per-detector
  reliability would need labels or a proxy, which would compromise the unsupervised
  guarantee.
- Metrics depend on the chosen `contamination` budget; the threshold-free PR-AUC/ROC-AUC
  are the more robust way to compare detectors.
