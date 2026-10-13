# ADR-020: Missing-class-specific components on top of P4noKD — classifier calibration (N1) and frozen classifier (N2)

**Date:** 2026-10-13
**Status:** accepted for exploratory runs (author: "tiến hành thực hiện đề xuất sau đó tiến hành triển khai")
**Scope:** `src/preservation/classifier_calibration.py`, `scripts/calibrate_classifier.py`, COCO-aware D1,
`freeze_cls_head` (client), preset `P5f`, notebook 06, sessions s14_g / s15_g; YOLO11s, S1b5-3k and its pooled
re-split, seed 42

## Context
P4noKD (ADR-019) is a general improvement but does not shrink the missing-class gap (pooled − S1b5, AP50 bus):
A0c +0.031 → P4noKD +0.056 (EXP-2026-10-11-S13-*-pooled). D1 on YOLOv8n located the missed buses mostly as
**truck/car classifications at the right place** (registry EXP-2026-10-05-S7-D1-errors).

## Hypothesis `[THESIS-HYPOTHESIS]`
Clients without bus bias the **classifier** (the last 1×1 class conv of each Detect level) against bus; the
features still localise buses. Fixing the classifier should shrink the missing-class gap specifically.

## Evidence
- `[LITERATURE]` Luo, Chen, Hu, Zhang, Liang, Feng — "No Fear of Heterogeneity: Classifier Calibration for
  Federated Learning with Non-IID Data", NeurIPS 2021, arXiv:2106.05001 (abstract checked): under non-IID data the
  classifier is the most biased layer; post-hoc calibration with virtual representations sampled from per-class
  Gaussians (CCVR) improves accuracy.
- `[LITERATURE]` Oh, Kim, Yun — "FedBABU", ICLR 2022, arXiv:2106.06042 (abstract checked): updating only the body
  during federated training (head never updated) improves representations under heterogeneity.
- `[YOLO-DOC]` ultralytics 8.3.253, YOLO11s: `model.23` (Detect) has `cv3[l][2] = Conv2d(128 → 80, 1×1)` per level
  l ∈ {stride 8, 16, 32}; targets are COCO ids car 2, motorcycle 3, bus 5, truck 7 (ADR-018).

## Decision
**Step 0 — D1 on P4noKD (round 20, S1b5 and pooled):** decompose missed buses (tp / confused truck / confused
car / low score / missed) with the COCO ids. Go/no-go for N1/N2: if, on S1b5, `confused_*` is not the largest
excess over pooled, the classifier-bias hypothesis is weakened (report; N2 still runs as it is pre-registered).

**N1 — CCVR adapted to detection (post-hoc, no FL retraining):**
- Features: the 128-d input of `cv3[l][2]`, captured with a forward hook, model in eval mode, val-style
  letterboxed 640 images.
- Positives `[ENGINEERING]`: for each ground-truth box, the cell containing its centre on ONE level chosen by size
  (√(w·h) at 640: < 64 px → stride 8, < 128 px → stride 16, else stride 32).
  Background: up to 8 random cells per image and level outside every box.
- Each client sends, per level and per class (4 targets + background): count, mean, covariance — no images.
  Clients use their own images only (bus-free clients send no bus statistics).
- Server merges the statistics (pooled mean/covariance), samples 2,000 virtual features per target class and
  8,000 background per level, and fine-tunes ONLY the rows {2,3,5,7} (weights + biases) of `cv3[l][2]` with
  sigmoid BCE (targets one-hot over the 4 ids; background all zero), Adam lr 1e-3, 300 steps, from the current
  weights. All other parameters unchanged. Fixed before any run.
- Applied to P4noKD round-20 checkpoints of S1b5 and of pooled (seed 42).

**N2 — P5f = P4noKD + frozen classifier (FedBABU-style):** `cv3[l][2]` of every level keeps its COCO values for
the whole federated training (requires_grad False on every client); everything else as P4noKD. Arms on S1b5 and
pooled, 20 rounds, eval 10/20.

## Pre-declared reading (AP50; S1b5 vs pooled; rule = round 20 for N1, mean of rounds 10/20 for N2)
A component is a **missing-class-specific contribution** only if all hold, relative to P4noKD under the same rule:
1. gap (pooled − S1b5, AP50 bus) shrinks by ≥ 0.010;
2. AP50 bus on S1b5 rises by ≥ 0.010;
3. mAP50 not below −0.005 and AP50 car not below −0.010 on S1b5.
Otherwise it is reported as not missing-class specific (a general gain or a loss). Single seed: a passing
component is a candidate for the 3-seed runs.

## Risks
- CCVR was designed for softmax image classification; dense sigmoid detection with a background class is an
  adaptation (hypothesis). Sharing class-conditional feature statistics is more than FedAvg shares.
- Freezing the COCO classifier may cap car/truck (COCO car on BDD: 0.584 zero-shot).

## Related
ADR-014 (D1) · ADR-018 (COCO head) · ADR-019 (P4noKD) · literature_registry: ccvr2021, fedbabu2022

## Results — s14_g (2026-10-14; registry EXP-2026-10-14-S14-*)
**Step 0 (D1, P4noKD round 20, 1,597 GT bus):** S1b5 − pooled excess: confused car + truck **+164**
(264+179 vs 159+120), missed **+140** (435 vs 295), low score −34; S1b5 bus FP 131 vs 585. Confusion is the largest
excess only with car and truck summed — the classifier-bias hypothesis is **weakly** supported; a nearly equal share
of buses get no detection at all.

**N1 (CCVR adaptation): rejected.**

| AP50 (round 20) | S1b5 before | S1b5 after | pooled before | pooled after |
|---|---:|---:|---:|---:|
| mAP50 | 0.557 | 0.448 | 0.564 | 0.454 |
| bus | 0.530 | 0.412 | 0.563 | 0.452 |
| car | 0.746 | 0.592 | 0.748 | 0.605 |
| FP bus | 205 | 364,302 | 739 | 292,793 |

Gap 0.033 → 0.040 (wider); rules 2 and 3 fail. Cause: the virtual set (2,000 positives per class, 8,000 background
per level) does not reflect the dense prior (~8,400 background cells per image for ~10 objects), so re-fitted biases
over-score background. This rejects this detection adaptation, not CCVR. Not re-tuned post hoc; any variant (e.g.
weights only, biases kept) needs a new pre-registration.
