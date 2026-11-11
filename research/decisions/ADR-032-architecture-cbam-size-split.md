# ADR-032: architecture change 1 (gated CBAM on the Detect inputs) and AP50 by object size

- Status: accepted for one exploratory run (author's request 2026-11-10: "đề xuất và thử nghiệm" architecture
  changes; CLAUDE.md §2 lists a new YOLO architecture as out of scope unless approved — approved here for this arm)
- Related: ADR-033 (main config YOLO11m), ADR-030 (S26/S27), ADR-029 (rule 2 needs size-split AP)

## Context
The missing-class cost is bus recall (~0.036 AP50, ADR-030) and no architecture change targets it; the author's
goal is higher absolute AP, where motorcycle (0.52) is the weakest class. A 100-image local check of S26
round 20 (images that contain a motorcycle; biased sample, not a result) gave motorcycle AP50 0.28 for
small (< 32² px), 0.80 medium, 0.86 large: small motorcycles look like the main loss. The full val split is
measured first (below); it decides whether a P2 head is worth a later ADR.

## Decision 1 — size-split evaluation `[ENGINEERING]`, COCO ranges `[LITERATURE]`
`src/evaluation/size_split.py` + `scripts/eval_size_split.py`: AP50 and recall per class for COCO area ranges
(original pixels, small < 32², medium 32²-96², large >= 96², Lin et al. ECCV 2014 / pycocotools convention:
out-of-range ground truths ignored). Predictions at the global evaluator's settings (conf 0.001, NMS 0.7,
max_det 300, 960 px). Unit-tested (tests/test_size_split.py). Runs on CPU from downloaded checkpoints, so it
costs no GPU quota; the "all" row is a consistency check against the run's AP50.

## Decision 2 — gated CBAM on P3/P4/P5 `[LITERATURE]` + `[ENGINEERING]`
CBAM (Woo et al., ECCV 2018; Ultralytics `CBAM`, 8.3.253) after each neck output that feeds Detect,
`y = x + γ (CBAM(x) − x)`, γ a learnable scalar per level initialised to 0 (`src/model/arch.py`). At
initialisation the network is exactly the COCO YOLO11m (unit test: identical outputs), so the COCO head,
the not-true KD teacher (= initial model) and round 0 are unchanged; all 649 COCO tensors are loaded
(Detect renamed 23 → 26). Extra parameters: CBAM channel MLPs + spatial convs + 3 gates (< 1 %).
Arm: P4t1 @ s5moto with `configs/experiments/convergence_5c_y11m_cbam_20r_960.yaml` (weights
`yolo11m.pt+cbam`), seed 42; everything else as S26.

## Decision rule (fixed before the run), vs S26, mean of rounds 10/20
Promising only if (1) motorcycle AP50 ≥ +0.010, (2) small-motorcycle AP50 (size split, both rounds) higher,
(3) car, bus, truck AP50 each not lower by more than 0.01. Seed 42 only: descriptive; a pass justifies the
ideal-scenario run and seeds 123/2024, not a claim.

## Cost `[RESOURCE-BUDGET]`
~10-11 h on T4 x 2 (S26 10.0 h; CBAM adds < 1 % parameters but runs at P3 resolution); part 2 if cut at 11 h.
Size split: CPU only.

## Result
s29 (phdatt): 20 of 20 rounds trained (10.8 h, +8 % vs S26); the round-20 eval hit the 11 h arm timeout and was
run afterwards from the saved checkpoint with the same evaluator (s29e, `scripts/eval_global_checkpoint.py`).
Round 0 equals S26 in every metric (identity at init confirmed on GPU). Mean of rounds 10/20 vs S26:

| AP50 | S26 YOLO11m | S29 + gated CBAM | Δ |
|---|---:|---:|---:|
| mAP50 | 0.6422 | 0.6477 | +0.006 |
| motorcycle | 0.5234 | 0.5396 | **+0.016** |
| bus | 0.6020 | 0.6032 | +0.001 |
| truck | 0.6224 | 0.6263 | +0.004 |
| car | 0.8211 | 0.8215 | +0.000 |

Motorcycle recall +0.018, bus precision +0.020 / recall −0.009. Size split (motorcycle, rounds 10/20, n = 135 / 223 / 94
small / medium / large): S26 0.221/0.268, 0.585/0.614, 0.715/0.752; S29 0.224/0.275, 0.589/0.620, 0.725/0.764.

Rule check (seed 42, descriptive): (1) motorcycle +0.016 ≥ 0.010 — met; (2) small-motorcycle AP50 higher at both
rounds — met, but by only +0.003 / +0.007, so the gain is spread over all sizes rather than concentrated on small
objects; (3) car, bus, truck not lower — met. **Promising at seed 42**; not a claim before seeds 123 / 2024.

Size-split protocol note: the s29e numbers used Ultralytics `predict()`, whose NMS keeps one class per box, while the
evaluator's `val()` NMS is multi-label. Bus suffers most (S26 r20 bus "all" 0.536 vs evaluator 0.622; motorcycle
0.538 vs 0.541). Fixed in `scripts/eval_size_split.py` (multi-label predictor); on a 60-image bus subset the fixed
script gives bus 0.737 vs `val()` 0.754 (single-label: 0.538); the remaining ~0.017 is COCO greedy matching vs
Ultralytics matching. The motorcycle size split above is usable; the bus size split of s29e is not and must be rerun.
Small motorcycles (30 % of val motorcycles) stay at AP50 ≈ 0.27 against ≈ 0.75 for large: the case for a P2 head.
