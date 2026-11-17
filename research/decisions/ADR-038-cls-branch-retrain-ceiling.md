# ADR-038: ceiling test 1a, retrain the classification branch on pooled labelled data

**Date:** 2026-11-15
**Status:** accepted (run 2026-11-17, s37e; outcome: not viable)
**Scope:** feasibility-gate (G5) · decides whether head-level calibration (CCVR-like, plan step 1b) is worth building

## Context
ADR-036: neck features keep bus separable in the missing run; the readout is shifted. ADR-037: the bus row of
cv3[l][2] alone carries 14-24 % of the gap, so the shift spans the whole cv3 branch. Plan
(`research/plan/huong_cai_tien_20261114.md`, update 2026-11-15): before building any server-side head calibration
from client feature statistics (1b), measure the best such a method could do: retrain the whole cv3 branch on the
real, labelled neck features of all 15,000 client images. If even that does not close the gap, 1b is dropped.

Control `[THESIS-HYPOTHESIS]` / design: retraining on pooled labelled data is extra centralized training and can
raise AP for any model. The ideal-scenario model gets the identical retrain, and the rule compares the gap after
both are retrained; a gain shared by both is not a missing-class fix.

## Decision
Pairs (round 20): YOLO11m S26 (M) / S27 (I); YOLO11s S21b (M) / S28 (I). Four retrained models M_rt, I_rt plus
identity copies of M and I, all evaluated in one session.

Recipe (fixed; `scripts/retrain_cls_head.py`, `src/model/head_retrain.py`):
- trainable: Detect.cv3 only (all levels, incl. its BatchNorm); backbone, neck, cv2, dfl frozen
  (Ultralytics `freeze`; frozen BN in eval mode `[YOLO-DOC]` 8.3.253 `BaseTrainer._model_train`). Frozen tensors
  are restored to the exact checkpoint values after a check that they moved by rounding only (≤ 1e-5 relative;
  EMA arithmetic moves them ~1e-7).
- data: the 15,000 images of `s1b5_central_3k_allmoto_seed42` (= the union of the clients' images), COCO-id view,
  repeat-factor sampling t = 1.0 (as P4t1), no distillation;
- 2 epochs, SGD, lr0 0.002 (0.2 × the clients' 0.01) with Ultralytics' linear decay to lrf 0.01, warmup 0,
  batch 8 (default nbs 64, as the clients), default augmentation, seed 0, 960 px; EMA weights returned.
- no other recipe is tried; the recipe is not tuned after seeing results.

Evaluation: the server's global evaluator (`eval_global_checkpoint.py`), plus per-detection dumps
(`eval_size_split.py`) and paired bootstrap (`bootstrap_ci.py`, bus, 1,000 resamples) for gap0 and gap_rt.

**Rule (fixed before the run)**, gap0 = AP50_bus(I) − AP50_bus(M), gap_rt = AP50_bus(I_rt) − AP50_bus(M_rt):
- **head calibration viable** if gap_rt ≤ 0.5 × gap0 on both models, and car / truck / motorcycle AP50 of M_rt are
  not below M by more than 0.01;
- otherwise **not viable**: plan step 1b (and CCVR-like head calibration in general) is dropped;
- a model whose in-session gap0 < 0.015 is inconclusive and makes the outcome "not replicated".
Reported alongside (decides nothing): recovery (AP50_bus(M_rt) − AP50_bus(M)) / gap0, I_rt − I, all classes.

Also in the session (descriptive, ADR-036 readout on A0c S32a / S32b round 20): mean bus score at bus boxes for
FedAvg without KD. Hypothesis `[THESIS-HYPOTHESIS]`: the P4t1 missing run sits at the COCO score (0.267 vs 0.264)
because not-true KD from the COCO teacher anchors it there; without KD (A0c) the missing-run score is expected
below the COCO 0.264. Reported, not a rule.

## Evidence
- `[YOLO-DOC]` freeze semantics and frozen-BN eval as above; Detect index = last layer (23 on YOLO11s / YOLO11m).
- `[LITERATURE]` CCVR (Luo et al., NeurIPS 2021) re-fits the classifier on server-side virtual features; this test
  replaces the virtual features by real ones (an upper bound for that family). `[NEEDS-VERIFICATION]` full-text
  details of CCVR are not reused here, only the idea.
- `[ENGINEERING]` recipe, rule thresholds.

## Consequences / limits
- An oracle: it uses pooled labelled data no FL server has. A pass only says a head-level method could work.
- One training seed; bootstrap covers val sampling only. 2 epochs is one fixed budget, not a convergence claim.
- Smoke 2026-11-15 (CPU, 96 images, 320 px, 1 epoch): 81 tensors changed, all in cv3; frozen tensors within
  1e-7 relative before restore.
- Cost `[RESOURCE-BUDGET]`: ~3.5-4 h on one T4 x2 session (11m lane: 2 retrains × 2 epochs + 4 evals + dumps;
  11s lane: the same + A0c readout features).

## Related
ADR-020 (CCVR attempt, P5f), ADR-036, ADR-037.

## Result
s37e (goshihayashi/flops-s37e-cls-retrain, commit 7d2ca79), round 20, global evaluator, AP50:

| Model | M | I | M_rt | I_rt | gap0 | gap_rt | gap_rt / gap0 | recovery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 11m bus | 0.6220 | 0.6521 | 0.6367 | 0.6561 | 0.0301 | 0.0194 | 0.64 | 0.49 |
| 11s bus | 0.5903 | 0.6127 | 0.6066 | 0.6213 | 0.0224 | 0.0147 | 0.66 | 0.73 |

Non-target, M_rt − M: 11m car +0.003, truck +0.022, motorcycle **−0.019**; 11s car +0.005, truck +0.015,
motorcycle **−0.027**. I_rt − I bus: +0.004 (11m), +0.009 (11s).

Paired bootstrap (dumps, bus, 1,000): 11m gap0 0.0318 [0.0245, 0.0390], gap_rt 0.0225 [0.0149, 0.0297],
M→M_rt +0.0142 [0.0076, 0.0202]; 11s gap0 0.0178 [0.0111, 0.0254], gap_rt 0.0125 [0.0047, 0.0199],
M→M_rt +0.0125 [0.0073, 0.0182].

**Decision: not viable** (both gap0 ≥ 0.015, so conclusive). gap_rt > 0.5 × gap0 on both models, and motorcycle
of M_rt drops beyond 0.01 on both. Plan step 1b / CCVR-like head calibration is dropped. Even with real labelled
features of all 15,000 images, re-fitting cv3 leaves about two thirds of the gap and its CI excludes 0: the part
of the gap a head-only method can reach is about one third; the rest sits upstream of cv3 although the ADR-036
probe saw no loss of linear bus separability in the neck (a linear probe on centre cells is a weaker test than AP).

A0c readout (descriptive): mean bus score at bus boxes 0.076 (missing) vs 0.290 (ideal) vs 0.264 (COCO init);
bus wins 0.25 vs 0.52; neck probe delta 0.029 [−0.016, 0.072] (not significant), readout delta 0.214
[0.202, 0.228]. The hypothesis holds: without KD the missing run's bus score falls far below the COCO score, so
not-true KD is what anchors P4t1 at 0.267. The score shift in A0c is ~3× P4t1's yet its AP gap is smaller (0.017):
the shift is mostly a rank-preserving rescale, consistent with AP being insensitive to it.
