# ADR-018: P3 — keep the COCO head and distil from the frozen COCO model (no server data)

**Date:** 2026-10-08
**Status:** accepted for an exploratory run (the author asked to run the new method on S1b5-3k first,
without FedAvg arms, to save GPU)
**Scope:** `build_model("coco:…")`, `target_class_ids`, COCO-aware `evaluate`, `src/data/coco_view.py`,
presets `A0c` / `P3`, session s10 (goshihayashi)

## Context

- P1 (ADR-015) is the only method that beat its controls (+0.025 / +0.029 AP50 bus), but it needs 1,000
  labelled images on the server.
- The original COCO YOLOv8n, with no BDD training, already scores AP50 bus 0.294 and motorcycle 0.138 on the
  BDD val (registry EXP-2026-10-07-EVAL-COCO-teacher) — more than FedAvg after 30 rounds (0.231 / 0.004).
  Re-initialising the class head (ADR-009 item 5) discards that knowledge.

## Decision

- **Model `[YOLO-DOC]`:** `coco:yolov8n.pt` keeps the 80-class head; the targets are COCO ids car 2, motorcycle 3,
  bus 5, truck 7 (verified against ultralytics 8.3.253 `cfg/datasets/coco.yaml`). Label files are remapped in a
  sibling tree with its own label folders (own Ultralytics cache); images are shared.
- **Evaluation:** unchanged protocol; per-class metrics read at the COCO ids; mAP50 = mean of the four target AP50
  (as for the 4-class model). The 76 other classes have no ground truth and do not enter AP.
- **P3 `[THESIS-HYPOTHESIS]`:** clients train with FedNTD not-true distillation (β = τ = 1, ADR-012 form) from a
  frozen copy of the initial COCO model; aggregation FedAvg; no server data, no BDD pre-training. Rationale: P1
  showed that a fixed, non-degraded teacher stops FedAvg from eroding knowledge; the COCO model is such a
  teacher that the FL setting already has.
- **A0c** (FedAvg with the COCO head, no KD) is the control that separates "keep the head" from "distil".
  Not run now (author's choice to save GPU); without it the P3 result is descriptive only.

## Pre-declared reading (S1b5-3k, seed 42, mean of rounds 20/25/30)

- Before A0c exists: report P3 next to the zero-shot COCO model (round 0 of the same run). P3 below its own round
  0 on AP50 bus would mean FL with KD still erodes the COCO knowledge.
- Once A0c exists: KD contributes only if P3 ≥ A0c + 0.010 AP50 bus with mAP50 not below A0c − 0.005 and AP50 car
  not below A0c − 0.010 (the COCO teacher is weaker on BDD cars, 0.498).

## Risks

- The 76 non-target channels only receive negatives from BDD; harmless for the targets, slightly more compute.
- Results are on a new scenario (S1b5-3k), not comparable number-for-number with S1b-2k runs.

## Related
ADR-009 · ADR-012 · ADR-015 · ADR-016 · ADR-017
