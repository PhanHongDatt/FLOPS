# FLOPS — Federated Learning for Object Detection under Missing-Class Non-IID

Research project investigating whether client-side knowledge preservation and server-side class-aware aggregation can reduce class-level degradation in federated object detection when some clients have **zero positive samples** for one or more traffic classes.

**Target classes:** `car` · `bus` · `truck` · `motorcycle`  
**Dataset:** BDD100K  
**Model:** YOLOv8n  
**FL framework:** Flower 1.21.0

---

## Research Questions

| Hypothesis | Question |
|-----------|---------|
| **H1** | Do Missing-Class clients degrade global detection of that class? |
| **H2** | Does reducing updates to class-associated parameters preserve prior knowledge? |
| **H3** | Does per-class server aggregation outperform uniform weighting under Non-IID? |

---

## Stage Progress

| Gate | Description | Status |
|------|-------------|--------|
| G0 | Evidence/source registry | ✅ passed |
| G1 | Environment reproducible | 🔄 in progress |
| G2 | Centralized YOLOv8 baseline | ⬜ |
| G3 | FedAvg baseline | ⬜ |
| G4 | Missing-Class effect established | ⬜ |
| G5 | YOLO feasibility (F1/F2/F3) | ⬜ |
| G6–G11 | Method development & evaluation | 🔒 blocked |

---

## Environment

| Component | Version |
|-----------|---------|
| Python | 3.11 |
| PyTorch | 2.7.1 + CUDA 12.8 |
| Ultralytics | 8.3.253 |
| Flower (flwr) | 1.21.0 |
| MLflow | 3.4.0 |

See `research/decisions/ADR-001-environment-versions.md` for full rationale.

---

## Quick Start

### 1. Prerequisites

- Docker + Docker Compose
- NVIDIA Container Toolkit (driver ≥ 525 for CUDA 12.8)

### 2. Build research image

```bash
docker compose -f docker/docker-compose.yml build
```

### 3. Verify environment (smoke test)

```bash
docker run --rm --gpus all flops-research:latest python -c "
import torch, ultralytics, flwr, mlflow
print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('Ultralytics:', ultralytics.__version__)
print('Flower:', flwr.__version__)
print('MLflow:', mlflow.__version__)
"
```

### 4. Freeze environment

```bash
docker run --rm flops-research:latest pip freeze > environment.lock
```

### 5. Run smoke experiment

```bash
docker compose -f docker/docker-compose.yml up
```

MLflow UI: [http://localhost:5000](http://localhost:5000)

---

## Project Structure

```
src/
  data/           # BDD100K loading, partitioning, class statistics
  model/          # YOLOv8n wrapper (get/set parameters, train, eval)
  federated/      # Flower client/server, FL strategies
    strategies/   # FedAvg ✅ | FedProx 🔄 | SCAFFOLD 🔄 | FedNova 🔄
  preservation/   # H2 implementation — locked until G5
  evaluation/     # per-class mAP, AP, Precision, Recall, ΔAP
  experiments/    # experiment runner, artifact management
  utils/          # config, logging, MLflow, artifact helpers
configs/
  base_config.yaml          # canonical hyperparameters
  partition/                # S0, S1, S1-Control partition configs
  experiments/              # smoke, feasibility, main
research/
  gates.yaml                # G0–G11 status
  evidence/
    literature_registry.yaml  # verified paper sources
  decisions/                # ADR-001, ADR-002, ...
  feasibility/F1/ F2/ F3/   # YOLO parameter analysis
  experiment_registry/      # all run records
docker/
  Dockerfile.research
  docker-compose.yml        # SuperLink + SuperNode + MLflow
tests/
artifacts/                  # experiment outputs (gitignored)
```

---

## Running Experiments

### Run classes

| Class | Purpose | Command |
|-------|---------|---------|
| `smoke` | Correctness only (1 epoch, 2 rounds) | `python -m src.experiments.runner --run-class smoke` |
| `feasibility` | Reduced scale (10 epochs, 5 rounds) | `--run-class feasibility` |
| `main` | Full experiment (100 epochs, 10 rounds, ≥3 seeds) | `--run-class main` |

> Do NOT launch `main` before estimating compute cost and passing G3.

### Reproduce a run

Every completed run records `config.yaml`, `environment.json`, `partition_manifest.yaml`, and `git_commit` in `artifacts/runs/<run_id>/`. To reproduce:

```bash
git checkout <git_commit>
docker compose build
python -m src.experiments.runner --config artifacts/runs/<run_id>/config.yaml
```

---

## Experimental Scenarios

| Scenario | Description |
|----------|-------------|
| **S0** | IID — approximate class balance across clients |
| **S1** | Missing-Class Non-IID — ≥1 client has zero positives for ≥1 class |
| **S1-Control** | Matched control — same image/bbox counts, all classes present |

---

## FL Baselines

| Algorithm | Role | Status |
|-----------|------|--------|
| FedAvg | Primary baseline | ✅ implemented |
| FedProx | Heterogeneous FL baseline | 🔄 verify built-in availability |
| SCAFFOLD | Client drift correction baseline | 🔄 custom impl needed |
| FedNova | Objective inconsistency baseline | 🔄 custom impl needed |

---

## Evidence Policy

Every non-trivial research decision is tagged:

- `[LITERATURE]` — peer-reviewed paper
- `[YOLO-DOC]` — verified from pinned Ultralytics source
- `[FL-DOC]` — verified from Flower source
- `[THESIS-HYPOTHESIS]` — must be experimentally validated
- `[ENGINEERING]` — implementation choice, no scientific claim
- `[NEEDS-VERIFICATION]` — source not yet confirmed

See `research/evidence/literature_registry.yaml` for all verified sources.

---

## Anti-Cherry-Picking Policy

- All seeds reported (mean ± std), never best-seed only
- Negative results retained in `research/experiment_registry/`
- Rare classes defined **before** inspecting model outcomes
- Same evaluator and partition used across all compared methods
