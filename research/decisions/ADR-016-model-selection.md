# ADR-016: Detector for the main experiments — YOLOv8n → s-tier (YOLO11s, or YOLOv8s)

**Date:** 2026-10-08
**Status:** accepted for a trial on 2026-10-09 — the author approved moving to the YOLO11 family
("tiến hành thử đổi sang họ yolo11"), a scope change against project rules §1 ("YOLOv8"). First run: P3 on
YOLO11s (session s11_g). Exploratory runs so far (s2–s10) stay valid for YOLOv8n and are never mixed with
YOLO11s numbers.
**Scope:** model of the main experiments (G10), before 3-seed main runs are spent

## Evidence

**1. Data (measured, BDD100K, 1280×720 resized to 640):** small objects dominate — median √area car 18.6 px,
motorcycle 25.5 px, bus 38.6 px, truck 37.1 px; 70 % of cars and 43 % of buses are < 32 px; 44 % of cars < 16 px.

**2. Official numbers** `[YOLO-DOC]` (ultralytics v8.3.253 `docs/macros/yolo-det-perf.md`, `docs/en/models/yolov8.md`, COCO val, 640):

| Model | params (M) | FLOPs (B) | COCO mAP50-95 |
|---|---:|---:|---:|
| YOLOv8n | 3.2 | 8.7 | 37.3 |
| YOLO11n | 2.6 | 6.5 | 39.5 |
| YOLOv8s | 11.2 | 28.6 | 44.9 |
| YOLO11s | 9.4 | 21.5 | 47.0 |
| YOLOv8m / YOLO11m | 25.9 / 20.1 | 78.9 / 68.0 | 50.2 / 51.5 |

**3. Measured on our task** (COCO weights, no BDD training, full BDD val, AP50; registry EXP-2026-10-08-EVAL-model-zoo;
the local evaluator reproduces the Kaggle evaluation of T exactly):

| Model | bus | car | truck | motorcycle | mAP50 (4 classes) |
|---|---:|---:|---:|---:|---:|
| YOLOv8n | 0.293 | 0.498 | 0.239 | 0.138 | 0.292 |
| YOLO11n | 0.286 | 0.503 | 0.236 | 0.140 | 0.291 |
| YOLOv8s | 0.369 | 0.579 | 0.325 | 0.239 | 0.378 |
| **YOLO11s** | **0.379** | **0.584** | **0.335** | **0.273** | **0.393** |

For reference, the best trained FL arm so far (P1, YOLOv8n) reaches AP50 bus 0.300, mAP50 0.338.

**4. System:** Kaggle T4 ×2, 2 clients in parallel, 30 GPU-h/week per account, 5 h timeout per arm; YOLOv8n arm = 65 min
measured. FLOPs-scaled upper bounds: YOLO11s ≈ 2.7 h, YOLOv8s ≈ 3.5 h per arm; m-tier ≈ 8–10 h (exceeds the per-arm
timeout and a week's quota for a 3-seed comparison) — excluded. Update size per transfer (fp32): 12.8 MB (v8n),
37.6 MB (11s), 44.8 MB (v8s).

## Decision (proposed)

- **n-tier → no change worth making:** YOLO11n's COCO advantage does not transfer to BDD (0.291 vs 0.292).
- **Move the main experiments to the s-tier.** Zero-shot alone gains +0.09–0.10 mAP50 and +0.08 AP50 bus over
  YOLOv8n — about 3–4× the largest method effect measured so far (P1 − B1 = +0.025) — and the data are dominated
  by small objects, where capacity matters.
- **Within the s-tier, YOLO11s** is better than YOLOv8s on every class, with 16 % fewer parameters (less traffic)
  and 25 % fewer FLOPs (faster). **If the author keeps the "YOLOv8" scope of §1, use YOLOv8s** (second best,
  rule-compliant).
- Keep the COCO class head (map car/bus/truck/motorcycle to COCO ids 2/5/7/3) instead of re-initialising it
  (ADR-009 item 5): the zero-shot numbers above are what re-initialisation throws away.

## Consequences / risks

- Every main-experiment arm must be re-baselined on the chosen model (A0, matched control, pooled, T/B1/P1).
  Exploratory YOLOv8n results remain as the study's first phase, never mixed with s-tier numbers.
- Parameter map (F1) and class-aware aggregation key names must be re-verified for YOLO11 (its Detect head
  differs); P1/KD act on outputs and are unaffected.
- Zero-shot ranking may not equal post-training ranking: confirm with one 1-seed FL pilot (A0 on S1b, 30 rounds)
  that also measures the real arm time before committing quota.

## Related
ADR-009 · ADR-011 · ADR-015 · registry EXP-2026-10-07-EVAL-COCO-teacher, EXP-2026-10-08-EVAL-model-zoo

## Update 2026-10-09 (before the YOLO11s run)

- P3 on YOLOv8n, S1b5-3k (EXP-2026-10-08-S10b-P3at-s5-50r) plateaus from round 20 (AP50 bus 0.333 → 0.334 at
  round 50), so the YOLO11s trial runs 20 rounds.
- Measured YOLOv8n P3 cost: 4 h 29 min for 50 rounds → ~4.4 min/round. FLOPs-scaled upper bound for YOLO11s:
  ~11 min/round → ~4.5 h for 20 rounds with evaluations; per-arm timeout raised to 8 h for this session.
- CPU smoke of the full P3 path on `coco:yolo11s.pt` passed (COCO head kept, teacher called, KD active,
  per-class metrics read at the COCO ids).
