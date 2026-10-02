# ADR-002 Addendum 1: Kaggle-specific version pins (mlflow, protobuf, tensorflow)

**Date:** 2026-09-09
**Status:** proposed
**Amends:** ADR-001 (environment versions), ADR-002 (Kaggle deployment)
**Scope:** environment-versions on Kaggle only

---

## Context

ADR-001 pins `mlflow==3.4.0` and `flwr==1.21.0`. On the Kaggle Notebook
environment chosen for G1–G4 by ADR-002, these three transitive constraints
collide simultaneously and cannot all be satisfied:

| Constraint | Source |
|---|---|
| `protobuf >= 5.0` | `mlflow>=3.0` — verified from `mlflow-3.4.0` PyPI metadata |
| `protobuf < 5` | `flwr==1.21.0` — verified via `pip show flwr` on Kaggle session 2026-09-09 |
| `protobuf >= 5.0` (transitive) | Kaggle base image ships `tensorflow` + `keras` which import `protobuf>=5` |

Additionally the Kaggle base image ships `numpy>=2.0`, which the original
notebook install cell attempted to downgrade to `<2.0`. Ultralytics 8.3.253
and PyTorch 2.7.1 both natively support numpy 2.x, so the downgrade was
gratuitous and broke wheels on Kaggle.

CLAUDE.md §5 forbids silent version deviation from a single source of truth.
This ADR records the deviation explicitly so `environment.lock` produced on
Kaggle remains an authoritative record.

---

## Decision

For **Kaggle sessions only** (execution target set by ADR-002), the version
pins for the following packages deviate from ADR-001:

| Package | ADR-001 pin | ADR-002-A1 pin (Kaggle) | Reason |
|---|---|---|---|
| `mlflow` | `==3.4.0` | `>=2.0,<3.0` | last major line compatible with `protobuf<5` |
| `protobuf` | not pinned | `>=3.20,<5.0` | explicit ceiling to prevent transitive upgrade |
| `tensorflow` / `tensorflow-cpu` / `keras` / `tf-keras` | not installed | **uninstalled at bootstrap** | transitively pull `protobuf>=5`, break flwr |
| `numpy` | `>=1.26,<2.0` | Kaggle default (`>=2.0`) | Ultralytics + PyTorch support numpy 2.x |

All other ADR-001 pins remain in force unchanged:

- `torch==2.7.1`, `torchvision==0.22.0` (CUDA 12.8 wheel)
- `ultralytics==8.3.253`
- `flwr==1.21.0`

Docker Compose deployment (local GPU, ADR-001 §Container Architecture) is
NOT affected and keeps `mlflow==3.4.0`. The Docker base image is TensorFlow-free
and can pin `protobuf>=5` cleanly.

---

## Evidence

- `[FL-DOC]` `flwr-1.21.0` metadata: `Requires-Dist: protobuf<5,>=3.20`
  (verified via `pip show flwr` after install on Kaggle, session 2026-09-09).
- `[ENGINEERING]` `mlflow-3.4.0` metadata: `Requires-Dist: protobuf<7,>=5.0.dev0`
  (verified via `pip show mlflow` after `pip install mlflow==3.4.0` in a
  clean venv, 2026-09-09).
- `[ENGINEERING]` Kaggle base image `tensorflow-2.x` + `keras-3.x` transitively
  require `protobuf>=5`. Removed at bootstrap in
  `notebooks/01_smoke_G1_G3.py` Cell 1.
- `[YOLO-DOC]` `ultralytics-8.3.253` supports numpy 2.x per Ultralytics
  release notes for the 8.3 series (release 2026-01-13).

---

## Consequences

### Preserved
- Full MLflow classic tracking (params / metrics / artifacts / runs) —
  present in 2.x and unchanged in 3.x. §21 artefact contract satisfied.
- `environment.lock` remains the single canonical record per §5; deviation
  is captured *inside* the lock file rather than in code.
- FL algorithm semantics (FedAvg / SCAFFOLD / FedNova) untouched — only
  logging/telemetry backend version differs.

### Constrained
- MLflow 3.x LLM-tracking and prompt-registry features unavailable on
  Kaggle. Not used by the current research pipeline.
- Kaggle-produced runs cannot be re-ingested by a `mlflow>=3.0` server
  without an MLflow-side upgrade path. Acceptable — Kaggle runs stay in
  the Kaggle Output Dataset until analysis-time.

### Reproducibility contract
Every experiment `environment.json` MUST include a `deploy_target` field
with one of `"kaggle"` or `"docker"`. `src/experiments/runner.py::prepare_run`
reads this from environment.lock or defaults to `"kaggle"` when the lock
was produced on Kaggle. (Follow-up: enforce this field in `save_environment`
once G1 passes.)

---

## Alternatives considered

| Option | Rejected because |
|---|---|
| Downgrade `flwr` to a `protobuf>=5`-compatible line | flwr `>=1.22` requires re-verifying all H2/H3 code against a newer Flower API. Not worth the risk at G1. |
| Upgrade `flwr` to 1.33+ and keep `mlflow==3.4.0` | Same API-regression risk; deviates from thesis Chapter 3 pin. |
| Move Kaggle → Colab Pro / rented GPU now | ADR-002 already rejected paid options at G1. Not revisiting until G6/G7. |
| Silently accept whichever version Kaggle installs | Violates CLAUDE.md §5 (canonical env). Explicitly forbidden. |
| Keep `mlflow==3.4.0` and uninstall `flwr` protobuf checks | Would require patching Flower internals — brittle, hard to reproduce. |

---

## Rollout

- `notebooks/01_smoke_G1_G3.py` Cell 1 already implements the new stack
  (2026-09-09 diff).
- `notebooks/02_baseline_G2_G3.py` Cell 1 aligned in this ADR's rollout commit.
- `notebooks/03_missing_class_G4.py` Cell 1 aligned in this ADR's rollout commit.
- `docs/KAGGLE_SETUP.md` references this ADR in Section 5.

---

## Related

- ADR-001 — canonical environment versions
- ADR-002 — Kaggle deployment for G1–G4
- CLAUDE.md §5 — single source of truth for versions
- CLAUDE.md §21 — artefact contract (`environment.json` records deployed versions)
