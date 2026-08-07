# ADR-001: Environment Versions & Container Architecture

**Date:** 2026-08-08 (updated after thesis config extracted)
**Status:** accepted — 2026-08-08 (user authorized proceed)
**Scope:** environment-versions, container-architecture

---

## Context

G1 requires a reproducible, canonical environment recorded in one file.
The project runs on a GPU-enabled system and prioritises containerised deployment
via Flower's SuperLink/SuperNode (FLOps) architecture.

Version source (CLAUDE.md §5 resolution order):
- Thesis document (Chapter 3) specifies Ultralytics 8.3.x, Flower 1.21.x, MLflow 3.4.x → treated as **experiment metadata**.
- PyTorch / CUDA / Python not specified in thesis → proposed from Docker Hub.

Every YOLO parameter assumption is valid only for the pinned version (CLAUDE.md §5).
No version may be chosen silently.

---

## Proposed Version Matrix

| Component | Proposed version | Source | Evidence |
|-----------|-----------------|--------|----------|
| Python | **3.11** | derived | Required by Flower 1.21.0 (`Python >=3.9`) and consistent with PyTorch image `[FL-DOC]` |
| PyTorch | **2.7.1** | proposed | Stable (2025-04-23); CUDA 12.8 image confirmed on Docker Hub `[ENGINEERING]` |
| CUDA | **12.8** | proposed | Bundled in `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime` `[ENGINEERING]` |
| cuDNN | **9** | proposed | Bundled in above image `[ENGINEERING]` |
| Ultralytics | **8.3.253** | thesis (8.3.x) | Latest patch of 8.3.x series; released 2026-01-13; verified on PyPI `[YOLO-DOC]` |
| Flower (flwr) | **1.21.0** | thesis (1.21.x) | Only patch in 1.21.x; released 2025-09-10; verified on PyPI `[FL-DOC]` |
| MLflow | **3.4.0** | thesis (3.4.x) | Released 2025-09-17; verified on mlflow.org `[ENGINEERING]` |
| torchvision | **0.22.0** | proposed | Paired with PyTorch 2.7.x per pytorch.org version table `[ENGINEERING]` |

---

## ⚠️ Conflict: Flower 1.21.0 vs 1.33.0

| | Flower 1.21.0 | Flower 1.33.0 |
|--|--------------|--------------|
| Source | Thesis document | Originally proposed (latest stable) |
| Release date | 2025-09-10 | 2026-08-05 |
| Docker images | `flwr/superlink:1.21.0` | `flwr/superlink:1.33.0` |
| SuperLink/SuperNode API | Present (1.x series) | Same architecture, newer API |
| Python requirement | ≥3.9 | ≥3.11 |

**Decision needed:** Use **thesis version (1.21.0)** for reproducibility alignment, or **upgrade to 1.33.0** for latest fixes and Docker image support?

Recommendation: **1.21.0** — matches thesis, avoids introducing unreported API changes into the experiment baseline. If Flower API breaks are found during G2/G3, open ADR-002-flower-upgrade.

---

## Training Hyperparameters (from thesis — fixed, not for approval)

These are recorded as config, not version decisions:

| Parameter | Value |
|-----------|-------|
| Model | YOLOv8n |
| Pretrained weights | yolov8n.pt |
| Image size | 640 × 640 |
| Epochs (centralized) | 100 |
| Batch size | 16 |
| Optimizer | SGD |
| Learning rate | 0.01 |
| Confidence threshold | 0.25 |
| IoU threshold | 0.70 |
| FL rounds | 10 |
| Local epochs per round | 1 |
| Fraction fit | 1.0 |
| Fraction evaluate | 1.0 |
| FL clients | 4 |
| FL algorithms | FedAvg, FedProx, SCAFFOLD, FedNova |

---

## Container Architecture (unchanged — SuperLink/SuperNode)

```
┌─────────────────────────────────────────────────────────┐
│  Docker network: flwr-network                           │
│                                                         │
│  ┌──────────────────┐   ports 9091/9092/9093            │
│  │   superlink      │ ← flwr/superlink:1.21.0           │
│  │  (coordinator)   │                                   │
│  └────────┬─────────┘                                   │
│           │                                             │
│  ┌────────┴────────┐                                    │
│  │   serverapp     │ ← research:latest                  │
│  │  (aggregation)  │   (PyTorch 2.7.1 + YOLO + flwr)   │
│  └─────────────────┘                                    │
│                                                         │
│  supernode-0 … supernode-3  ← flwr/supernode:1.21.0    │
│  clientapp-0 … clientapp-3  ← research:latest          │
│                                                         │
│  mlflow-server               ← mlflow:3.4.0 tracking   │
│  GPU: NVIDIA Container Toolkit (shared / host)          │
└─────────────────────────────────────────────────────────┘
```

---

## Approval Required

**Confirm before pinning:**

- [ ] PyTorch **2.7.1 + CUDA 12.8** — NVIDIA driver trên host GPU ≥ 525?
- [ ] Flower **1.21.0** (thesis) hay upgrade lên **1.33.0**?
- [ ] GPU sharing: 1 GPU shared across all 4 client containers, hay dedicated per client?
- [ ] MLflow server: chạy trong Docker Compose hay external?

---

## Consequences

1. Ultralytics pinned tại **8.3.253** — F1 parameter mapping chỉ valid cho version này.
2. Nếu Flower 1.21.0 approved: Docker images dùng `flwr/*:1.21.0`.
3. MLflow 3.4.0 thêm vào requirements và docker-compose.
4. `environment.lock` điền sau smoke test trên GPU host.

---

## Related ADRs

- ADR-002 (planned): YOLO class-parameter mapping (sau F1)
- ADR-003 (planned): G5 feasibility gate decision (A/B/C)
