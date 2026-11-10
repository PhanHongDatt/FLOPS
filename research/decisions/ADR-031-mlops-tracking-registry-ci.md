# ADR-031: MLOps — experiment tracking, model registry and CI

**Date:** 2026-11-10
**Status:** accepted (scope approved by the author, 2026-11-10)
**Scope:** module-contract | mlops

## Context

The 2026-11-10 code audit found MLOps limited to partial tracking: MLflow received only the
server's evaluation metrics, `log_params`/`log_artifact` were never called on the FL path,
`setup_mlflow` crashed a run when the tracking server was unreachable (default
`http://mlflow:5000` only resolves inside the compose network), a resumed run opened a second
MLflow run, there was no model registry, no CI, and `docker/docker-compose.yml` used an MLflow
3.4.0 server image while every client is pinned to `mlflow<3` (ADR-002-A1: `flwr==1.21.0`
needs `protobuf<5`; `environment.lock` has `mlflow==2.22.5`).

CLAUDE.md §2 lists production serving as OUT of scope. The author approved this scope
extension on 2026-11-10: **tracking + model registry + CI/CD, MLflow on local Docker.**
Deployment, monitoring and retraining remain out of scope.

## Decision

All items are `[ENGINEERING]`; none changes a research equation, the evaluator, the partitions
or the aggregation rules. CSV artifacts in the run directory remain the results of record
(CLAUDE.md §21); MLflow is an index over them.

1. **Tracking** — `src/mlops/tracking.py` (`start_tracking` → `Tracker`).
   * A tracking failure never ends a run (`--mlflow-required` opts into failing).
   * HTTP URIs are probed at `<uri>/health` (5 s) before use, instead of MLflow's retry loop.
   * `--mlflow-uri none` disables tracking; the default is `$MLFLOW_TRACKING_URI`, else
     `http://localhost:5000` (the compose service published on the host).
   * The MLflow run id is stored in `<run_dir>/mlflow_run.json`; `--resume` reopens that run.
   * Logged: flattened effective config (after arm overrides), CLI arm identity as params/tags,
     git commit, server evaluation metrics per round, client fit metrics (below), and the
     run's §21 files under artifact path `run/`.
2. **Fit tracking** — `src/mlops/fit_tracking.py` wraps any strategy's `configure_fit` /
   `aggregate_fit` (no strategy code changes) and writes `fit_metrics.csv` (one row per client
   per round): `num_examples`, `uplink_bytes` (serialized client parameters),
   `downlink_bytes` (serialized global parameters sent; config payloads such as SCAFFOLD's
   `server_c` are not counted), `fit_phase_s`, `aggregation_s` (includes the B2 server
   fine-tune when enabled), and the client's local training loss.
   * Local training loss: `train_one_round` now also returns
     `trainer.label_loss_items(trainer.tloss, prefix="train")` captured at
     `on_train_epoch_end`. `[NEEDS-VERIFICATION]` against the pinned Ultralytics 8.3.253
     source; the capture is wrapped so a missing attribute only drops the loss columns.
   * Keys `metrics/*` and `fitness` from Ultralytics' results are **not** tracked: with
     `run_val=False` they come from the 8-image validation stub and are meaningless.
3. **Model registry** — `src/mlops/registry.py`, `src/mlops/model_card.py`,
   `scripts/mlops_registry.py`, policy in `configs/mlops/registry.yaml`.
   * Package = `final_params.npz` (ordered `state_dict`, the format already used by
     checkpoints) + `model_meta.json` (base weights incl. `coco:` prefix, class names, image
     size, sha256, metrics, protocol, run identity). `load_packaged_model` rebuilds it with
     `build_model` + `set_parameters` (strict count check).
   * `eval_protocol.json` is written at run start with the **effective** weights (the saved
     `config.yaml` predates the `coco:` rewrite) and a fingerprint of the global val image set.
   * Every registration creates a version and moves alias `candidate` to it.
     `scripts/train_centralized.py` now also saves `checkpoint/final_params.npz`.
   * Promotion to alias `champion` is a separate, explicit step. Rule (pre-configured, applied
     mechanically): candidate `run_class` allowed (default `main`); protocol keys present and
     equal to the champion's; primary metric (`mAP50`) > champion + `min_primary_gain` (0.0);
     no target class `AP50_<c>` below champion − `per_class_max_drop` (0.0). A forced promotion
     requires a reason and records the failed checks as version tags.
   * A registry decision is a **deployment** choice, not a research claim: it never replaces
     the multi-seed, same-partition comparison protocol (CLAUDE.md §14, §20).
4. **MLflow server** — `docker/Dockerfile.mlflow` installs `mlflow==2.22.5` (same version as the
   clients, from `environment.lock`), backend `sqlite:////mlflow/db/mlflow.db`, artifacts served
   by the server (`--artifacts-destination`). Only the `mlflow` compose service is in scope;
   the Flower SuperLink/SuperNode services stay non-functional (no ServerApp/ClientApp exists).
5. **CI** — `.github/workflows/ci.yml`: CPU PyTorch 2.7.1 + `requirements.txt`, `ruff` with
   error-class rules only (`E9,F63,F7,F82`), `pytest -m "not slow"`, and a build + `/health`
   check of the MLflow image.

## Evidence

* `[ENGINEERING]` throughout. Versions from `environment.lock` (mlflow 2.22.5) and ADR-002-A1.
* MLflow client APIs used: `create_registered_model`, `create_model_version`,
  `set_registered_model_alias`, `get_model_version_by_alias`, `set_model_version_tag`,
  `start_run(run_id=...)` — available in MLflow 2.x (aliases since 2.3).
  `[NEEDS-VERIFICATION]` by running the tests on the pinned stack: they were written on a
  machine without mlflow/torch/flwr installed and have not been executed yet.

## Consequences

* `run_fl_server(...)` takes `tracker=` instead of `mlflow_experiment=`/`mlflow_uri=`;
  `_build_centralized_evaluate_fn(...)` takes an optional `tracker=` (None → no MLflow call,
  which also stops tests from creating a stray `./mlruns`).
* New run files: `fit_metrics.csv`, `eval_protocol.json`, `model_meta.json`,
  `mlflow_run.json`. They are not added to the required §21 set, so older runs still verify.
* `src/utils/logger.py` MLflow helpers are kept for compatibility but no longer used by the
  scripts.
* Out of scope (unchanged): deployment/serving, monitoring, retraining, Flower deployment mode.

## Alternatives Considered

* MLflow pyfunc model flavor — needs a custom PythonModel wrapping Ultralytics; heavier and
  unnecessary while no serving exists. The artifact package can be wrapped later.
* MLflow 3.x server — incompatible client pin (protobuf<5); mixed client/server majors risk
  registry API mismatches.
* Automatic promotion at the end of a run — rejected: promotion must be a deliberate,
  auditable step.

## Related ADRs

ADR-001, ADR-002-A1 (versions), ADR-009 (evaluation protocol), ADR-018 (COCO head / `coco:`).
