# ADR-006: A2b — loss-level preservation via per-class BCE weight rho

**Date:** 2026-10-03
**Status:** accepted for exploratory runs (G5 not yet passed — results are not a method claim)
**Scope:** research mechanism H2 (client side) — `src/preservation/rho_loss.py`, `LossPreservationClient`

## Context

H2 needs a client-side mechanism separable from server-side A3. A2a (parameter mask after
training) is numerically absorbed by A3 (plan.md §6.3: A4a ≡ A3), so A2b — acting on the loss and
therefore on the trajectory of shared parameters — is the only separable client mechanism.
Its prerequisites, previously `[NEEDS-VERIFICATION]`, are now met:

- `[YOLO-DOC]` ultralytics 8.3.253 `utils/loss.py`: `v8DetectionLoss.bce =
  BCEWithLogitsLoss(reduction="none")` on `pred_scores` [B, anchors, nc], then
  `.sum() / target_scores_sum`; `DetectionModel.init_criterion()` builds it lazily.
- F1 confirmed on the pinned Kaggle stack (2026-10-03).

## Decision

`[THESIS-HYPOTHESIS]` For a client with zero positives of class c, multiply the BCE term of
channel c by `rho ∈ [0, 1]` during local training: rho = 1 is ordinary training; rho = 0
removes the "class absent here" pressure on c's logits. Implemented by wrapping the criterion's
`bce` at `on_train_start` (restored at `on_train_end`). Box and DFL losses are untouched.
`rho = 0.25` is the single value used in the first comparison (ADR-011), chosen before any
result (CLAUDE.md §6 candidate set {0, 0.25, 1}); a sweep is future work and must be reported.

## Evidence / tests

`tests/test_rho_loss.py` (weights, channel scaling, zero gradient at rho = 0, lazy criterion,
restore); `tests/test_build_model_nc.py::test_rho_zero_freezes_the_missing_class_row_in_real_training`
— real Ultralytics training on CPU: with rho = 0 the bus class-head bias is bit-identical after
training while car's moves; `tests/test_experiment_arms.py` (client builds the weight vector).

## Falsifiable predictions / known limits

- rho < 1 should reduce bus forgetting on clients without bus; predicted side effect: more bus
  false positives (FP/class is therefore a required metric — plan.md §6.1).
- Weight decay still shrinks the protected class-row weights slightly; biases are not decayed.

## Related
ADR-007 (per-client rho/missing classes) · ADR-008 · ADR-011 · plan.md §6.1–§6.3
