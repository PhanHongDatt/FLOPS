# ADR-013: A6 — Equalized Focal Loss on the client, with A6c (plain focal) as its control

**Date:** 2026-10-05
**Status:** accepted for an exploratory run (author approved, exploratory label — CLAUDE.md §7)
**Scope:** `src/preservation/efl_loss.py`, `EFLClient`, `FocalClient`, presets `A6`, `A6c`,
sessions s7a (A6c) / s7b (A6)

## Context

A2b (ADR-006) multiplied the whole BCE of a locally missing class by rho = 0.25. In s5/s6 the
FP bus count rose ~3× with A4b and only ~72 more buses were found. The long-tailed detection
literature reports the same failure for negative-gradient suppression and corrects it:

- `[LITERATURE]` Tan et al., "Equalization Loss v2", CVPR 2021 — gradient-guided
  re-weighting of positive/negative gradients per class (sigmoid). Code checked
  (github.com/tztztztztz/eqlv2 `eqlv2.py`): designed for sampled RoIs of two-stage detectors.
- `[LITERATURE]` Wang et al., "Seesaw Loss", CVPR 2021 — a *compensation factor* raises the
  penalty of misclassified samples to avoid tail-class FPs; softmax-based, so not applicable to
  the YOLOv8 sigmoid head as is.
- `[LITERATURE]` Li, Yao, Tan et al., "Equalized Focal Loss for Dense Long-Tailed Object
  Detection", CVPR 2022 — the one-stage, sigmoid, dense-anchor version. Verified 2026-10-05
  against the official code (github.com/ModelTC/EOD,
  `up/tasks/det/plugins/efl/models/losses/efl.py`):
  `gamma_j = gamma_b + s * (1 - g_j)`, weight `gamma_j / gamma_b`,
  `loss = alpha_t * w_j * (1 - p_t)^gamma_j * CE`, g_j = clamp(Σ|grad_pos| / Σ|grad_neg|, 0, 1)
  accumulated over training (initial 1), all-reduced over GPUs. Defaults gamma_b = 2,
  alpha = 0.25; s = 4 in the authors' YOLOX configs (`configs/det/efl/efl_yolox_small.yaml`).

EFL is chosen because it is the variant built for dense one-stage detectors like YOLOv8.

## Decision

- A6: every client trains with EFL (gamma_b 2, alpha 0.25, s 4 — authors' YOLOX setting, not
  tuned); aggregation FedAvg.
- `[YOLO-DOC]` Soft (task-aligned) targets use ultralytics 8.3.253 `FocalLoss`'s form
  `BCE * (1 - p_t)^gamma * alpha_t`, p_t = t·p + (1−t)(1−p), unreduced (v8DetectionLoss
  reduces).
- `[THESIS-HYPOTHESIS]` Federated adaptation: gradient statistics are accumulated per client and
  kept across rounds (`clients/<id>/efl_state.json`) instead of being all-reduced over the
  federation. A bus-free client then has g_bus = 0 → gamma 6, weight 3: easy "no bus" negatives
  are suppressed, confident wrong bus scores keep (and gain) loss — the compensation A2b lacked.
  Pooling the statistics on the server is the untested alternative.
- A6c: plain focal loss (gamma 2, alpha 0.25) on every client. EFL changes the loss *family*
  (BCE → focal) as well as equalising it; without A6c any A6-vs-A0 difference could be the
  focal loss alone (§11: simpler controls must be reported).

## Implementation notes (code review 2026-10-05)

- g_j is computed from the gradient of the full EFL-weighted loss with respect to the class
  logits, as the official `collect_grad` does (backward hook on the cls output,
  `collect_func_module: post_process.cls_loss`), not from the plain BCE gradient |p − t|.
  With focal weighting the positive/negative ratio of a class the client sees often reaches the
  clamp at 1 (plain focal for that class); that is the method's behaviour, and A6 then differs
  from A6c mainly on scarce or absent classes — which is the intended contrast here.
- Gradients are divided by the AMP GradScaler factor before accumulation, so batches before and
  after a scale change weigh the same; non-finite steps are skipped; state is written atomically.
- Focal loss with alpha 0.25 makes the classification term much smaller relative to box/DFL
  (hyp.cls unchanged). A6c vs A0 therefore mixes "focal instead of BCE" with "smaller cls
  weight"; only A6 vs A6c isolates equalisation. Do not claim "focal helps" from A6c vs A0.

## Falsifiable prediction (pre-declared)

Rule: mean of rounds 20/25/30, seed 42.
- A6 supports the equalisation mechanism only if AP50 bus ≥ A6c + 0.010 **and** ≥ A0, with
  FP bus not above A0 + 50 % and mAP50 not below A0 − 0.005.
- If A6 ≈ A6c (|Δ| < 0.010), any gain over A0 is not from equalisation (focal form and/or the
  smaller classification weight).
- Reject if A6 ≤ A0 on AP50 bus, or FP bus > A0 + 50 %.

## Risks

- Focal loss rescales the classification term relative to box/DFL losses; this is shared by A6
  and A6c and is why A6c exists.
- Per-client statistics differ from the paper's global statistics most for exactly the missing
  class; that is the hypothesis, recorded as such.

## Related
ADR-006 · ADR-011 · ADR-012 · ADR-014 · literature_registry: efl2022, eqlv2_2021, seesaw2021
