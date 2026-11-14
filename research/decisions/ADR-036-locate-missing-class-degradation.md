# ADR-036: locate the missing-class degradation (features vs head readout) before choosing a method

**Date:** 2026-11-14
**Status:** accepted (run 2026-11-14, s33e)
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

## Implementation (2026-11-14, fixed before the run)
- Code: `src/evaluation/probe.py` (pure helpers, `tests/test_probe.py`), `scripts/extract_probe_features.py`
  (GPU), `scripts/probe_degradation.py` (CPU), `notebooks/08_probe_features.py`, session `s33e` (phdatt; datasets
  `phdatt/flops-ckpt-y11m-s26-s27-s29` and private `phdatt/flops-ckpt-y11s-s21b-s28`).
- Val boxes: car 102,506, bus 1,597, truck 4,245. Car boxes are sampled before extraction by a seeded hash of
  (image, box index), 78 per mille (~8,000), independent of any model; bus and truck boxes are all kept.
- Letterbox: long side 960, short side padded to a multiple of 32, centred (the evaluator's rect batch shape
  for 1280x720 frames, 960x544).
- Primary-level probe: one probe per level, trained on the half-0 boxes whose primary level it is (feature width
  differs per level), predictions pooled over levels. Readout metrics use the primary level, all boxes.
- The bootstrap resamples half-1 images with the probe fixed (it is not refitted per resample).
- Smoke 2026-11-14 (CPU, 200 images, S26/S27/COCO YOLO11m): pipeline end to end; at the box-centre cell the
  readout argmax equals the GT class for 87 % of boxes (coordinates aligned). Smoke numbers are not results.

## Result (s33e, completed 2026-11-14)
All six extractions completed on the same 13,877 boxes (car 8,035, bus 1,597, truck 4,245); 797 bus boxes in
the evaluation half. Artifacts: `kaggle/output/s33e/flops_export/probe/` (features not committed).

**Formal decision (rule above): not replicated -> no method is triggered.** YOLO11m: readout shifted;
YOLO11s: head body degraded (cv3[l][1] met on 11s only).

Δ = ideal − missing at the primary level (95 % CI):
| Depth | YOLO11m (S27 − S26) | YOLO11s (S28 − S21b) |
|---|---|---|
| neck, probe bus recall | +0.003 [−0.028, +0.032] | +0.018 [−0.014, +0.047] |
| cv3[l][1], probe bus recall | −0.011 [−0.040, +0.021] | **+0.045 [+0.016, +0.074]** |
| readout, mean bus score at bus boxes | **+0.098 [+0.092, +0.105]** | **+0.069 [+0.063, +0.074]** |
| readout, share of bus boxes where bus beats car and truck | +0.138 [+0.122, +0.156] | +0.118 [+0.101, +0.136] |

Per checkpoint (primary level): probe bus recall neck / cv3[l][1]: S26 0.685 / 0.745, S27 0.688 / 0.734,
S21b 0.669 / 0.691, S28 0.686 / 0.737; COCO 11m 0.645 / 0.670, COCO 11s 0.622 / 0.664. Mean bus score:
S26 0.267, S27 0.365, COCO 11m 0.264; S21b 0.211, S28 0.280, COCO 11s 0.202.

Observations (descriptive, not part of the rule):
- Neck: no difference on either model; a linear probe recovers bus from the missing-scenario features as well
  as from the ideal ones. Per-level results outside the primary level are mixed and not consistent between the
  models (11m P5 +0.040, 11s P4 +0.056, each with CI excluding 0).
- Readout: the one effect present on both models, with the largest size and narrowest CI. In the missing
  scenario the bus score at bus boxes stays at the COCO starting value (0.267 vs 0.264; 0.211 vs 0.202), while
  the ideal run raises it. This matches a recall-driven gap and the earlier P5f null (freezing cv3[l][2] at COCO
  values keeps the score where the missing run already is).
- Any method aimed at the readout (bus row of cv3[l][2] / its bias) is a post-hoc hypothesis from this result and
  needs its own ADR and pre-registered rule; this ADR does not trigger it. One training seed per scenario.
