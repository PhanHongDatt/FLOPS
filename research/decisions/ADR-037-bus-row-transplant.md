# ADR-037: is the missing-class gap held in the bus row of the final classification conv? (row transplant)

**Date:** 2026-11-14
**Status:** proposed (pre-registration; not run)
**Scope:** feasibility-gate (G5, F2-style controlled perturbation)

## Context
ADR-036 (s33e): neck features keep bus separable in the missing run (probe Δ +0.003 / +0.018), while the model's
own bus readout is lower at bus boxes (+0.098 / +0.069 ideal − missing) and stays at the COCO starting score.
Post-hoc check on the same features (2026-11-14): the bus logit drops at bus boxes (S26 −2.67 vs S27 −1.88;
S21b −3.51 vs S28 −2.95) but not at truck boxes (−5.50 vs −5.49; −5.66 vs −5.69), so bus-vs-truck AUC of the bus
logit falls (0.795 → 0.759; 0.721 → 0.690) and bus-vs-car does not. A uniform bias shift cannot explain this and
would not change AP (AP is rank-based); the bus weight vector, or its interplay with the head body, differs.
This is the hypothesis `[THESIS-HYPOTHESIS]`: the gap is held in the bus row of `Detect.cv3[l][2]`
(3 levels × (c3 + 1) parameters: 771 on YOLO11m (c3 = 256), 387 on YOLO11s (c3 = 128), checked on the pinned model).

## Decision
Controlled perturbation, no training (CLAUDE.md §8 F2). For each model pair (missing M, ideal I), round 20:
- `M` and `I` re-evaluated in the same session (identity copies through the same script);
- `M+rowI`: M with the bus row (weight and bias, every level) of cv3[l][2] taken from I;
- `I+rowM`: the reverse.
Pairs: YOLO11m S26 (M) / S27 (I); YOLO11s S21b (M) / S28 (I). Everything else in the parameter list is unchanged
(the script refuses any other difference). Evaluator: the server's global evaluator
(`scripts/eval_global_checkpoint.py`, 960 px, the runs' conf / IoU); per-detection dumps with
`scripts/eval_size_split.py` for paired bootstrap CIs (`scripts/bootstrap_ci.py`, 1,000 resamples, bus).

**Rule (fixed before the run)**, with gap = AP50_bus(I) − AP50_bus(M) measured in this session:
- sufficiency R_s = (AP50_bus(M+rowI) − AP50_bus(M)) / gap;
- necessity R_n = (AP50_bus(I) − AP50_bus(I+rowM)) / gap;
- a model is **conclusive** only if its in-session gap ≥ 0.015 AP50 (round 20: 11m 0.030, 11s 0.022 in the runs);
- sanity: car / truck / motorcycle AP50 of a transplanted model within 0.01 of its recipient.

| Outcome (both models conclusive) | Reading | Next |
|---|---|---|
| R_s ≥ 0.5 on both | the bus row is sufficient to carry most of the gap | row-level method (own ADR; must explain why A1c class-count row weighting failed) |
| R_s < 0.5 on both, R_n < 0.5 on both | the gap is not in the row alone: row and head body shifted jointly | output-level method (direction B) |
| anything else | partial / not replicated | reported; no method triggered |
R_n is reported with R_s; it decides nothing alone. A model that is not conclusive makes the outcome "not replicated".

## Evidence
- `[YOLO-DOC]` cv3[l][2] = Conv2d(c3 → nc) with bias; COCO bus id 5 (`src/model/yolo_wrapper.py COCO_IDS`);
  keys `model.23.cv3.{0,1,2}.2.{weight,bias}` on both YOLO11s and YOLO11m.
- `[THESIS-HYPOTHESIS]` row locus as above. `[ENGINEERING]` transplant script and thresholds.

## Consequences / limits
- One training seed; the bootstrap covers val sampling only.
- A transplanted row meets a head body it was not trained with; a negative R_s can mean coupling, not absence.
- Cost `[RESOURCE-BUDGET]`: 8 global evals + 8 size-split dumps (4 per model), two lanes on one T4 x2 session,
  estimated 2-3 h. Authorised by the author 2026-11-14.

## Related
ADR-020 (P5f, CCVR), ADR-021 (A1c), ADR-028 (B1), ADR-036 (probe).

## Result
Not run.
