# ADR-019: P4 — P3 + cosine round learning rate + client-side repeat-factor sampling

**Date:** 2026-10-09
**Status:** accepted for an exploratory run (author: "tiến hành thử nghiệm chạy đi")
**Scope:** `cosine_round_lr` (client), `src/preservation/rfs.py`, `_harden_label_cache`, presets `P3LR`, `P4`;
YOLO11s, S1b5-3k, 20 rounds, eval rounds 10 and 20 (`convergence_5c_y11s_20r.yaml`), seed 42

## Context
P3 on YOLO11s (EXP-2026-10-09-S11-P3-y11s-s5): mAP50 0.472, AP50 bus 0.425 at round 20, but bus peaks at round
10 (0.430) and FN bus @0.25 rises with training; motorcycle 0.297 (rare: ~10 % of each client's images).

## Components (added one at a time)
- ② `[LITERATURE]` Li, Huang, Yang, Wang, Zhang — "On the Convergence of FedAvg on Non-IID Data",
  arXiv:1907.02189 (abstract checked): on non-IID data the learning rate must decay. Cosine over the run from
  lr0 = 0.01 to 0.05 × lr0 (`[ENGINEERING]` shape and floor, fixed now).
- ③ `[LITERATURE]` repeat-factor sampling of LVIS (Gupta, Dollár, Girshick); formula checked against the official
  Detectron2 code: r(c) = max(1, √(t/f(c))), r(I) = max over classes in I, stochastic rounding per epoch.
  Per client, from its own images only; loss untouched (negatives kept, unlike A2b/A6). ultralytics 8.3.253 keeps
  duplicate list entries (verified in `get_img_files` / `cache_labels`; CPU smoke: trainer saw the repeated list).
  **t = 0.5**, chosen before any run from the measured client frequencies (manifest of s11_g): motorcycle f ≈ 0.10
  → r 2.24 on every client; truck f 0.27–0.32 → r 1.26–1.37 on C0–C2; bus f 0.33–0.35 → r 1.19–1.23 on C3/C4;
  car r 1. ≈ +20 % images per round. (t = 0.3 would only repeat motorcycle.)
- Engineering: corrupt Ultralytics label caches are rebuilt instead of crashing (s8a B1 r11, s8c_g B2 v1).

## Arms
A0c (COCO head, FedAvg) · P3 (have) · P3LR (+②) · P4 (+② +③). Same partition, model, rounds, evaluator.

## Pre-declared reading (rule: mean of the evals at rounds 10 and 20 — the evaluated rounds of P3)
- KD (P3 vs A0c): keep if P3 ≥ A0c + 0.010 AP50 bus and mAP50 ≥ A0c − 0.005 (ADR-018 rule).
- ② (P3LR vs P3): keep if AP50 bus ≥ P3 + 0.010, or equal within 0.010 with |bus(20) − bus(10)| smaller; and
  mAP50 not below P3 − 0.005.
- ③ (P4 vs P3LR): keep if AP50 motorcycle ≥ P3LR + 0.010 with AP50 bus and truck each not below P3LR − 0.010 and
  mAP50 not below P3LR − 0.005.
- Single seed: a kept component is a candidate for the 3-seed runs, not a result.

## Cost (YOLO11s P3 measured: 2 h 28 min per 20-round arm + ~25 min setup)
A0c ≈ 2.3 h, P3LR ≈ 2.9 h, P4 ≈ 3.4 h → ≈ 8.6 GPU-h on goshihayashi (≈ 15 h left).

## Addendum 2026-10-09 — P4noKD (declared before A0c/P3LR results)
`P4noKD` = A0c + ② + ③ (no distillation). Reading: if P4 − P4noKD < 0.010 AP50 bus (mean of rounds 10/20),
distillation is not needed once ② and ③ are present, and the simpler P4noKD is preferred (no teacher forward,
~20 % cheaper). Launched while P3LR was still running, because it is needed whatever the other results are.
