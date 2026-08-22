# ADR-002: Kaggle Notebook Deployment (Deviation from ADR-001 Container Architecture)

**Date:** 2026-08-21
**Status:** proposed
**Scope:** deployment-architecture, execution-environment
**Supersedes:** none
**Related:** ADR-001 (environment versions), CLAUDE.md §5 (single source of truth), §14 (compute budget), §17 (multi-agent contracts)

---

## Context

ADR-001 pins the deployment architecture as a Docker Compose stack running Flower SuperLink / SuperNode / serverapp / clientapp / MLflow tracking server on a GPU-enabled host with the NVIDIA Container Toolkit.

The user does not currently have a local GPU host meeting those prerequisites. To make progress on gates **G1 (environment) → G2 (centralized baseline) → G3 (FedAvg baseline) → G4 (Missing-Class effect)** without blocking on hardware procurement, the project needs an alternative execution environment.

Kaggle Notebooks offer:
- Free NVIDIA T4 x2 or P100 GPU (~30 h/week quota)
- Preinstalled scientific Python stack
- Public dataset mounting (BDD100K uploaded as private Kaggle Dataset)
- 12 h max session, 20 GB `/kaggle/working` disk

This is a **deployment-only deviation**. All research contracts remain unchanged: versions in ADR-001, evidence policy §4, gate order §7, artifact contract §21, anti-cherry-picking §20.

---

## Decision

**Use Kaggle Notebooks as the execution environment for gates G1–G4** (setup, smoke, centralized baseline, FedAvg baseline, Missing-Class effect) under the following constraints:

1. **Flower simulation mode** (`fl.simulation.start_simulation`) — no SuperLink/SuperNode deployment. This is already how `src/federated/server.py:82` is implemented, so no code change is needed.

2. **MLflow local file store** — `file:///kaggle/working/mlruns` passed via `--mlflow-uri`. No external MLflow server. Artifacts copied to a Kaggle Output dataset at end of each session for persistence across the 12 h reset.

3. **Versions pinned per ADR-001** — every notebook session begins with:
   ```
   pip install torch==2.7.1 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
   pip install -r requirements.txt
   pip freeze > /kaggle/working/environment.lock
   ```
   The generated `environment.lock` is exported and committed after the first successful session. Kaggle preinstalled versions are **not** accepted (user choice, recorded 2026-08-21).

4. **BDD100K provisioning** — upload the official BDD100K release (`bdd100k_images_100k.zip` + `bdd100k_det_20_labels_trainval.zip`) as a private Kaggle Dataset with the folder structure expected by `scripts/prepare_bdd100k.py`:
   ```
   bdd100k/
     images/100k/{train,val}/*.jpg
     labels/det_20/det_{train,val}.json
   ```
   Mount read-only at `/kaggle/input/bdd100k/`. YOLO-converted output written to `/kaggle/working/data/bdd100k_yolo/` (writable, non-persistent).

5. **Session persistence pattern** — at the end of each run:
   - `environment.lock`, `research/experiment_registry/*.yaml`, and `artifacts/runs/<run_id>/` are saved to a Kaggle Output dataset ("FLOPS-artifacts").
   - The next session mounts that dataset at `/kaggle/input/flops-artifacts/` to resume.

6. **Main multi-seed experiments (G10)** — deferred. Kaggle 12 h session limit makes a full 3-seed × 4-algorithm × 10-round run tight. Revisit deployment target (Kaggle vs. Colab Pro vs. rented GPU) when G6/G7 are validated. Do not use Kaggle for G10 without a new ADR.

7. **Reproducibility** — every experiment records the Kaggle session ID and mounted dataset version in `environment.json` in addition to the ADR-001 fields.

---

## Evidence

- `[ENGINEERING]` `src/federated/server.py:82` already uses `fl.simulation.start_simulation` with `client_resources={"num_cpus": 1, "num_gpus": 0.25}` — simulation mode does not require SuperLink/SuperNode. Verified by direct file read 2026-08-21.
- `[ENGINEERING]` `src/utils/logger.py:30` calls `mlflow.set_tracking_uri(tracking_uri)`. MLflow supports `file://` scheme natively as documented in MLflow 3.4.0 tracking guide. Verified by direct file read.
- `[ENGINEERING]` `scripts/prepare_bdd100k.py:118-124` accepts `--train-ann` and `--val-ann` as CLI overrides, allowing structure adjustment if the Kaggle dataset layout differs from official.
- `[NEEDS-VERIFICATION]` Kaggle preinstalled PyTorch/CUDA versions and reinstall behavior — must be confirmed in the first session and recorded in `environment.lock`.

---

## Consequences

### Enabled
- G1/G2/G3/G4 unblocked without local GPU procurement.
- Zero code change to Flower or MLflow layer; only CLI flag overrides.
- BDD100K provisioning is a one-time upload cost.

### Constrained
- Docker Compose stack (`docker/docker-compose.yml`, `docker/Dockerfile.research`) is **not deleted** — it remains the canonical local deployment. It is simply not used until a GPU host is available.
- Kaggle session ephemerality forces artifact export at the end of every session; missing this step loses the run.
- 12 h session cap forbids G10 main experiments on Kaggle.
- MLflow UI is not live during a session (file store only); post-hoc inspection via `mlflow ui --backend-store-uri file:///path` locally after downloading the artifact dataset.

### Open
- If Kaggle Notebook API changes (deprecation of Internet access, dataset limits, GPU quota reduction), this ADR must be revisited.
- Version reinstall time per session (~5–10 min) is overhead accepted per user decision 2026-08-21.

---

## Alternatives Considered

| Option | Rejected because |
|---|---|
| **Wait for local GPU** | Blocks G1–G4 indefinitely; no timeline commitment from user. |
| **Google Colab Free** | 12 h session, similar disk, but no persistent Kaggle-Dataset-equivalent for artifact reuse across sessions without Google Drive mount. Kaggle Datasets are cleaner for reproducibility. |
| **Colab Pro / Pro+** | Requires paid subscription — not chosen at G1 stage. Revisit for G10. |
| **Rented cloud GPU (Vast.ai, RunPod, Lambda)** | Adds billing + setup complexity at G1 stage. Revisit for G10. |
| **Accept Kaggle-preinstalled torch/CUDA** | Version deviates from ADR-001, breaks §5 canonical env single-source rule. Rejected by user 2026-08-21. |
| **Rewrite deployment to SuperLink/SuperNode on Kaggle** | Kaggle notebook is a single container; running multiple Flower nodes as separate processes is possible but adds complexity with no research benefit — simulation mode is functionally equivalent for our 4-client experiments. |

---

## Notebook Template Contract

To be provided in a separate deliverable (`notebooks/flops_kaggle_smoke.ipynb`). Required cells in order:

1. GPU verification (`nvidia-smi`)
2. Repo clone / upload (user's private repo mirror or Kaggle upload)
3. Reinstall pinned versions + `pip freeze > environment.lock`
4. Mount BDD100K dataset, verify structure
5. Run `scripts/prepare_bdd100k.py` → YOLO format
6. Run `scripts/generate_partition.py` with `configs/partition/s0_iid.yaml`
7. Run `scripts/run_fl_experiment.py --run-class smoke --algorithm FedAvg --mlflow-uri file:///kaggle/working/mlruns`
8. Verify artifacts against §21 contract
9. Copy artifacts to Kaggle Output dataset

---

## Approval Required

- [ ] User confirms Kaggle as the G1–G4 execution environment.
- [ ] User confirms version-reinstall policy (torch 2.7.1+cu128, ultralytics 8.3.253, flwr 1.21.0).
- [ ] User confirms BDD100K will be uploaded as a private Kaggle Dataset with the official folder structure.
- [ ] User confirms G10 will use a different environment (to be decided later).

---

## Related ADRs

- ADR-001 — environment versions & container architecture (unchanged; deployment method superseded for G1–G4 only)
- ADR-003 (planned) — G5 feasibility gate decision (A/B/C)
- ADR-004 (planned, if needed) — G10 main-experiment execution environment
