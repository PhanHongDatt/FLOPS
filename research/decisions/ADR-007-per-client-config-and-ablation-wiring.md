# ADR-007: Per-client client config, ablation wiring, and resumable runs

**Date:** 2026-10-02
**Status:** accepted — 2026-10-02 (user authorised the fix order)
**Scope:** module contracts (`src/federated/client.py`, `src/federated/strategies/*`,
`scripts/run_fl_experiment.py`, `src/experiments/*`), experiment matrix execution
**Supersedes:** nothing. **Amends:** ADR-002 operationally (adds resume + resource
config to the Kaggle execution path; the Flower-simulation decision is unchanged).

---

## Context

A source review on 2026-10-01 (recorded in `research/plan/plan.md` §6.5 and §13)
found that the experiment matrix had no execution path:

1. `scripts/run_fl_experiment.py` exposed only `FedAvg | FedProx | SCAFFOLD |
   FedNova`. `ClassAwareAgg` existed in the server registry but was not an
   `--algorithm` choice, and `PreservationClient` was in no client registry.
   **A1, A2a, A2b, A3, A4b were unreachable.**
2. `_load_strategy_builder` forwarded `num_rounds` into flwr's `FedAvg.__init__`,
   which accepts neither `num_rounds` nor `**kwargs` (verified by introspection).
   **SCAFFOLD, FedNova and ClassAwareAgg raised `TypeError` at construction.**
3. `ClassAwareAggregation` never held `θ^t`, so its `S_c = ∅` branch logged
   `action: keep_global` while leaving the FedAvg average in place — violating
   CLAUDE.md §12 and writing a trace that stated something untrue (§19/§20).
4. `class_counts_json` was emitted only by `PreservationClient`, so the
   server-only arm (A3) received no class counts and silently degraded to
   FedAvg. `missing_classes` / `class_counts` / `rho` were read from
   `train_config`, **one dict shared by all clients**, so they could not hold
   per-client values at all.
5. Local training received neither the experiment seed nor `warmup_epochs`,
   `workers` or `val` settings: the three experiment seeds never influenced
   local training, every round was pure warm-up (`warmup_epochs` default 3.0 >
   `local_epochs` 1), and each client ran a full validation of the **shared
   global val set** per epoch — which also meant Ultralytics selected `best.pt`
   by fitness on the test set.
6. Runs could not resume: `run_id` embedded a timestamp and only the final round
   was checkpointed, making Kaggle's 12 h session cap fatal for Tier 1.

Per CLAUDE.md §17, changing another module's public contract requires an ADR.

---

## Decision

### D7.1 — Per-client values become explicit constructor arguments

`YOLOFlowerClient.__init__` gains `class_counts`, `missing_classes`, `rho`,
`base_seed`. They are **no longer read from `train_config`**. `train_config`
keeps only values that are genuinely shared (hyper-parameters).

`_build_client_fn` fills them from the partition manifest, which already records
`class_counts` and `missing_classes` per client.

### D7.2 — `class_counts_json` is reported by the base client

Moved from `PreservationClient` into `YOLOFlowerClient._fit_metrics`, so every
client reports it regardless of algorithm. Required for A3, where the client
applies no preservation.

### D7.3 — Two orthogonal CLI axes plus ablation presets

`--algorithm` selects the **server strategy** (now including `ClassCountFedAvg`
and `ClassAwareAgg`); `--client-mechanism` selects the **client mechanism**
(`none | preservation_param | preservation_loss`); `--ablation {A0,A1,A2a,A2b,A3,A4a,A4b}`
is a preset filling both. `--rho` is **mandatory** whenever a mechanism is set —
there is no literature-supported default (plan.md §7.3).

Invalid combinations are refused, not silently degraded: `SCAFFOLD`/`FedNova`
plus a preservation mechanism raises, because those clients override `fit()` and
the mechanism would not be applied.

All nine pre-existing CLI flags keep their names and meanings so the Kaggle
notebooks continue to work unchanged (plan.md §13 D5).

### D7.4 — Uniform strategy builder signature

Every strategy exposes `build_*(num_rounds, **kwargs)` which **consumes**
`num_rounds`. Added `build_scaffold`, `build_fednova`, `build_class_count_fedavg`.

### D7.5 — A1 and A3 are one class with one flag

`ClassAwareAggregation(no_contributor_rule=...)`: `True` → A3 (keeps `θ^t`),
`False` → A1 (class-count weighting, FedAvg fallback). This makes the plan.md
§6.4 finding explicit in code — with `τ_elig = 1` and full participation the two
arms are **numerically identical except in the no-contributor branch** — and it
is pinned by `test_a1_equals_a3_when_every_class_has_a_contributor`.

`eligibility_threshold` (`τ_elig`) is a constructor argument: primary value 1,
pre-registered sensitivity variant 50 (`--tau-elig`).

### D7.6 — Missing metadata fails loudly

A client that reports no `class_counts_json` raises by default
(`require_class_counts=True`). A parameter map matching **zero** class-head keys
raises (`find_cls_head_indices`). Both previously degraded silently into FedAvg
while the trace claimed class-aware behaviour.

The class-head regex no longer hard-codes the Detect index
(`model\.\d+\.cv3\.\d+\.2\.(weight|bias)$`) and warns when the match count is
not 6 (3 scales × weight/bias).

### D7.7 — Resumable runs, deterministic run ids

`default_run_id()` drops the timestamp: `{exp_id}_{run_class}_seed{seed}_{partition_id}`.
`src/experiments/checkpoint.py` saves `checkpoint/global_round_NNN.npz` **before**
evaluating each round, prunes to `keep_last_checkpoints`, and deletes that
round's per-client Ultralytics weights (≈0.5 GB/run against a 20 GB
`/kaggle/working` budget).

`run_fl_server(resume=...)` loads the newest checkpoint as `initial_parameters`,
runs only the remaining rounds, and offsets round numbers so CSV/MLflow steps
stay **absolute**. Without `--resume` it **refuses** to start when checkpoints
exist, rather than restarting at round 1 over existing metrics.

### D7.8 — Local-training knobs move into config `[ENGINEERING]`

`train.workers` (2), `train.warmup_epochs` (0.0), `train.close_mosaic` (0),
`train.client_run_val` (false), `train.deterministic` (true), `train.nbs` (null),
`federated.client_num_cpus` (1), `federated.client_num_gpus` (**1.0**, was 0.25),
`federated.keep_last_checkpoints` (2), `federated.prune_client_weights` (true).

These are **not** thesis hyper-parameters — the thesis specifies none of them.
The thesis-specified values (imgsz 640, batch 16, SGD, lr0 0.01, 10 rounds,
1 local epoch, fraction_fit 1.0, conf 0.25, iou 0.70, seeds 42/123/2024) are
unchanged.

### D7.9 — `fraction_evaluate` default kept; overridden per experiment config

`base_config.yaml` keeps the thesis value `1.0`. `smoke.yaml` and
`feasibility.yaml` set `0.0`, and `--fraction-evaluate` allows explicit override.
Justification: every per-client data yaml points `val:` at the shared global val
set, so client-side evaluation repeats the same validation once per client per
round on top of the server's centralized `evaluate_fn`, which is the actual
reporting path required by §14. **Any main run that deviates from 1.0 must say so
in its `README.md`.**

### D7.10 — SCAFFOLD server step corrected

`aggregate_fit` now implements Algorithm 1 (Option II):
`x_new = x + (η_g/|S|) · Σ(y_i − x)` — an unweighted mean of client deltas.
The previous code did `num_examples`-weighted averaging of `y_i` and never used
`eta_global`, i.e. it was FedAvg under a SCAFFOLD label (§22). No results had
been produced with the old behaviour, so nothing is invalidated.

### D7.11 — A2b is wired but refuses to run

`LossPreservationClient` exists so the CLI matrix is complete, but `fit()`
raises `NotImplementedError` naming its three blockers: F1 runtime verification,
`src/preservation/rho_loss.py`, and ADR-006. CLAUDE.md §4 forbids implementing
research logic whose justification is still `[NEEDS-VERIFICATION]`; the
`FedProxClient` precedent in this repo is to block rather than fall back.

---

## Consequences

### Enabled
- A0, A1, A3 and A4a are executable; A2a is executable once G5 permits.
- SCAFFOLD and FedNova are constructible for the first time.
- Kaggle runs survive session interruption (`--resume`).
- `aggregation_trace.yaml` is actually written (`trace_dir` now passed).
- Per-class FP/FN are produced (`yolo_wrapper._per_class_fp_fn`).
- Tests: 40 → 107, including the two contribution modules that previously had none.

### Constrained
- `YOLOFlowerClient` has a wider constructor; any caller must pass per-client
  values. The buggy `make_client_fn` helper (which hard-coded `num_examples=1`)
  was **removed** rather than fixed — it was dead except in an archived notebook.
- Re-running an existing arm requires `--resume` or a new `--run-id`.
- `client_num_gpus=1.0` serialises clients, so a round now takes the sum of
  client times rather than their max. On a 4 GB GPU this is mandatory; on a 16 GB
  T4/P100 it may be relaxed after measuring real VRAM.
- Numbers from SCAFFOLD change relative to the old (incorrect) implementation.

### Still blocked
- `[NEEDS-VERIFICATION]` confusion-matrix orientation for per-class FP/FN; the
  AP-level and threshold-level FP/FN are different quantities and the protocol
  must state which is reported.
- Confidence statistics (§13) are **not** implemented — they need a dedicated
  prediction pass, planned with F2/F3.
- Resume for SCAFFOLD/FedNova is approximate: server-side control state is not
  checkpointed and resets. A warning is logged; do not report a resumed
  SCAFFOLD/FedNova run without noting it.
- A2b (`rho_loss`) remains unimplemented by design.

---

## Validation

- `pytest tests/ -q` → **107 passed** on a machine with **no GPU, no torch-CUDA,
  no ultralytics** (every new test is numpy/flwr-level). This is the §13.3
  criterion 1.
- Criteria 2–6 of §13.3 (Kaggle smoke per arm; `verify_artifacts`; kill-and-resume;
  populated `environment.json`; disk budget) are **not yet satisfied** — they
  require the Kaggle session that also closes G1.

---

## Related

- `research/plan/plan.md` §6.5 (B/H findings), §7.1 (matrix), §13 (Kaggle contract)
- ADR-001 (versions), ADR-002 + A1 (Kaggle execution)
- ADR-003 (reserved: G5 Gate A/B/C), ADR-004 (main-run environment),
  ADR-005 (vacant class = bus), ADR-006 (ρ mechanism)
