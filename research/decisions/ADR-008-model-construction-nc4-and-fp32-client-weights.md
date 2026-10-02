# ADR-008: 4-class model construction and fp32 client weights

**Date:** 2026-10-02
**Status:** proposed (code implemented and tested; awaiting author acceptance)
**Scope:** module contract — `src/model/yolo_wrapper.py` (`build_model`, `train_one_round`)

## Context

F1 runtime verification and a CPU smoke run (2026-10-02, ultralytics 8.3.253) found three
defects that affect every FL experiment, not only F3 (the third was found by code review
and reproduced):

1. **nc=80 vs nc=4.** `build_model("yolov8n.pt")` returned the 80-class COCO model.
   Clients become nc=4 inside their first `model.train()`, but the server's evaluate model
   never trains, so `set_parameters` raised `size mismatch for model.22.cv3.0.0.conv.weight`
   at the first post-round evaluation. Round-0 evaluation ran the 80-class head against
   4-class labels (what Ultralytics reports in that case was not verified), and the initial
   global model was whichever client Flower polled first.
2. **fp16 client weights.** After `train()`, Ultralytics reloads best.pt/last.pt, whose EMA is
   saved with `.half()`. Every client update was rounded to fp16 every round. In the F3
   smoke run the class-row Δθ was bit-identical across both sides of the matched pair and
   across seeds — it was the rounding error, not the update. On CUDA with AMP the final-epoch
   validation additionally runs `trainer.ema.ema.half()` in place before casting back.
3. **In-place fuse.** `model.val()` fuses Conv+BN of `model.model` in place (355 → 127
   state_dict entries). The server reuses one evaluate model across rounds, so the next
   `set_parameters` failed with a parameter-count mismatch (reproduced on CPU).

## Decision

1. `build_model(weights, init_seed=0)` always returns a model with `len(TARGET_CLASSES)`
   classes. A source with another class count is rebuilt as
   `DetectionModel(source.yaml, nc=4)` and receives the source weights through
   `DetectionModel.load` (Ultralytics' own shape-matched `intersect_dicts`). Construction
   runs under `torch.random.fork_rng()` + `manual_seed(init_seed)`, so every Ray actor and
   the server build the identical initial model without disturbing the caller's RNG.
   An nc=4 checkpoint (e.g. G2 best.pt) is returned as loaded.
2. `train_one_round` registers an `on_train_epoch_end` callback that snapshots the fp32 EMA
   at the end of the final epoch — before Ultralytics' final validation and checkpoint
   reload — and loads it into the model after `train()`. The callback is removed in
   `finally`. EMA semantics are unchanged; only fp16 rounding is removed. Raises if no
   snapshot was captured (no silent fallback).
3. `evaluate()` validates a deep copy of `model.model` and restores the original, so callers
   keep an unfused model.

## Evidence

- `[YOLO-DOC]` ultralytics 8.3.253: `nn/modules/head.py` Detect `c3 = max(ch[0], min(nc, 100))`;
  `nn/tasks.py` `DetectionModel.load` → `utils/torch_utils.py intersect_dicts` (shape match);
  `engine/model.py train()` reloads `trainer.best`/`last`; `engine/trainer.py save_model`
  stores `"ema": deepcopy(ema).half()`; `nn/tasks.py load_checkpoint` → `.float()`;
  `engine/trainer.py` epoch end: `on_train_epoch_end` → `ema.update_attr` → `validate()`;
  `engine/validator.py` (training mode) `model.half()` when `trainer.amp` on CUDA.
- `[ENGINEERING]` reproduction + regression tests: `tests/test_build_model_nc.py`
  (10 tests, incl. real CPU training runs and a simulated AMP in-place half; skipped where
  ultralytics is absent). The AMP path itself is only simulated on CPU — confirm on the
  first Kaggle GPU run that uploaded weights are not fp16-representable.

## Consequences

- COCO weights transfer for 319/355 keys; all 36 cv3 keys start fresh at nc=4. This was
  already true inside Ultralytics' trainer; it is now explicit and seeded.
- Round-0 global AP is now honestly ~0 (fresh head) instead of a mislabelled COCO score.
- F3 must start from a checkpoint that already detects the target class (plan.md §6.2,
  `research/feasibility/F3/README.md`).
- Uploaded parameters are fp32 → model/update bytes per round double vs fp16 (record in
  the §13 "update/model bytes" metric; not a correctness issue).
- Any result produced before this change is invalid for FL (it could not get past round 1).

## Alternatives Considered

- **Upload raw (non-EMA) fp32 weights** (`trainer.model`): closer to textbook FedAvg, but
  changes the update semantics for every arm; kept as an open question for the author.
  Ultralytics EMA decay is 0.9999·(1−e^(−u/2000)); for u ≤ 125 optimizer steps per round it is
  ≤ 0.06, so EMA ≈ raw weights at current settings (a new trainer, hence a new EMA, every round).
- **Seed the server's initial parameters from a file**: equivalent once construction is
  seeded; not needed.
- **`YOLO("yolov8n.yaml")` + `load("yolov8n.pt")`**: still builds nc=80 from the YAML.

## Related ADRs

ADR-001 (pinned versions) · ADR-002 (Kaggle deployment) · ADR-007 (per-client config) ·
ADR-003 (planned, Gate A/B/C — consumes F3 output)
