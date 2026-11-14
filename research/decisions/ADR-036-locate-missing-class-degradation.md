# ADR-036: locate the missing-class degradation (features vs head readout) before choosing a method

**Date:** 2026-11-14
**Status:** proposed (pre-registration; not run)
**Scope:** feasibility-gate (G5 evidence) · diagnostics

## Context
The bus gap (ideal − missing, mean r10/20) is +0.036 AP50 on YOLO11m (S27 − S26) and +0.038 on YOLO11s
(S28 − S21b), recall-driven, seed 42 only. Eleven missing-class-targeted interventions did not close it
(ADR-012/013/020/028/035, A2b/A3/A4b). Two of them touched the classifier, but neither tells where the damage
sits: CCVR failed through its virtual set (ADR-020: "rejects this adaptation, not CCVR"), and P5f froze only
`cv3[l][2]`. D1 (ADR-020) split the excess bus errors into +164 confused with car/truck and +140 missed; YOLO11
has no objectness branch, so "missed" can come from the features or from the head.
Candidate next methods depend on the answer: feature-level alignment (prototype / contrastive, plan
`research/plan/huong_cai_tien_20261114.md` direction A) only makes sense if the shared features lose bus
separability; head/bias calibration only if the features are intact and the readout is shifted.

## Decision
Measure bus/truck/car separability at three depths of the same checkpoints, with a rule fixed now.

**Checkpoints** (round 20, `global_round_020.npz`): COCO initial (round 0), S26 (missing), S27 (ideal) on
YOLO11m; replication S21b (missing), S28 (ideal) on YOLO11s. Pair check done 2026-11-14: configs identical except
the partition; S26/S27 commits have no `src/` difference; S21b/S28 differ only by flag-gated features
(B1, M1-M3) and the S24 smoke reproduced S21b C4 round-1 losses to every digit (ADR-028).

**Samples** `[ENGINEERING]`: every GT box of car / bus / truck in BDD100K val (9,905 images, ~1,597 bus),
letterboxed to 960 px exactly as the evaluator does. One feature vector per box: the cell containing the box
centre, on each of P3/P4/P5 (strides 8/16/32). Primary level fixed by box size in input pixels (sqrt(area)):
< 64 → P3, < 192 → P4, else P5. All three levels are reported; the decision uses the primary level.
No ROI-align (no extra free parameters).

**Depths** (`[YOLO-DOC]` ultralytics 8.3.253 `Detect.cv3[l]` = [DWConv+Conv, DWConv+Conv, Conv2d(c3→nc)]):
1. neck output = Detect input (layers 16/19/22): linear probe;
2. `cv3[l][1]` output (penultimate classification features, 256 ch on YOLO11m): linear probe;
3. readout `cv3[l][2]`: no probe; the model's own sigmoid bus score (COCO row 5) at the bus box cells, and the
   share of bus boxes whose bus logit exceeds both the car (2) and truck (7) logits.

**Probe** `[ENGINEERING]`: multinomial logistic regression (torch, L2 1e-4 fixed, LBFGS, standardised inputs;
no tuning). Split val in halves by image (seeded hash of the image name, seed 0); train on half 1 with classes
balanced by subsampling car and truck to the bus count; evaluate on half 2 (all boxes). Report 3×3 confusion,
macro accuracy, bus recall. CIs: 1,000 paired bootstrap resamples of half-2 images (same resample for both
checkpoints of a pair).

**Decision rule (fixed before the run)**, Δ = ideal − missing on the primary level, metric = probe bus recall:
| Outcome | Reading | Next |
|---|---|---|
| neck Δ ≥ 0.03, 95 % CI excludes 0, on YOLO11m **and** YOLO11s | features degraded | direction A (own ADR) |
| neck Δ < 0.03 (or CI includes 0), `cv3[l][1]` Δ ≥ 0.03 with CI excluding 0, both models | head body degraded | head-level method (own ADR) |
| neither, but readout mean bus score Δ ≥ 0.03 with CI excluding 0, both models | readout shifted | bus bias / logit calibration (own ADR) |
| none of the above | not a classification-separability effect | localisation / threshold analysis |
A result met on one model only is reported as "not replicated" and does not trigger a method.

Threshold 0.03: with ~800 bus boxes in half 2 and probe recall near 0.8, the binomial 95 % half-width is
≈ 0.028; 0.03 is about one half-width on top of the CI-excludes-0 requirement. It is in probe-recall units
and is not comparable with the 0.036 AP50 gap. The COCO initial model is a reference (was separability there
before FL?), not part of the rule.

## Evidence
- `[YOLO-DOC]` Detect / cv3 layout checked on the installed ultralytics (YOLO11m, 3 levels, cv3[l][2] 256→80).
- `[THESIS-HYPOTHESIS]` damage location → method choice (direction A vs head calibration).
- `[ENGINEERING]` probe, pooling, size rule, split, threshold as above.

## Consequences / limits
- One training seed per scenario: the bootstrap covers val sampling only, not training noise.
- Linear separability is a proxy; a non-linear head could read features a linear probe cannot (and vice versa).
- Centre cells ignore the TAL assignment, so the sampled locations are identical across checkpoints by construction.
- Cost `[RESOURCE-BUDGET]`: feature extraction for 5 checkpoints on val, ~1-2 GPU-h on one Kaggle T4 session;
  probes and bootstrap on CPU.

## Alternatives considered
ROI-align pooling (extra parameters), TAL-assigned anchors (differ between checkpoints), non-linear probes
(confound probe capacity with feature quality).

## Related
ADR-020 (D1, CCVR, P5f), ADR-028 (B1), ADR-030/033 (S26/S27), ADR-035 (direction A local steps).

## Result
Not run.
