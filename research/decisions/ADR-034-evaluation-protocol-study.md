# ADR-034: evaluation-protocol study on saved checkpoints (no training)

- Status: accepted (author: "tiến hành thực hiện", 2026-11-11, items A1/A2/A3/A5 of the proposal list)
- Related: ADR-032 (CBAM, size split and its multi-label fix), ADR-033 (main config), verification note V7
  (`max_det` shared by 80 classes)

## Scope
Round-20 checkpoints of S26 (YOLO11m, missing), S27 (YOLO11m, ideal), S29 (YOLO11m + gated CBAM, missing).
No training; session s30e (notebooks/07_eval_protocols.py), two T4 lanes, ~2.5 h.

## Protocols `[ENGINEERING]`
- P0 = the registered evaluator: Ultralytics `val()` at 960 px, conf 0.001, NMS IoU 0.7, multi-label, max_det 300 over
  all 80 COCO classes. All registered numbers stay P0; nothing already registered is re-labelled.
- P1 = P0 with NMS over the four target classes only (`src/model/class_filter.py`: a forward hook sets the 76 other
  class scores to 0 before NMS; trained weights unchanged; unit-tested).
- P2 = P0 at 1280 px (BDD frames are 1280 x 720; training stays at 960).
- Size split (COCO area ranges, `scripts/eval_size_split.py`, multi-label NMS) for P0 and P2, plus P1 for S26;
  per-detection TP flags dumped for bootstrap.
- A1: paired percentile bootstrap over val images, 1,000 resamples (`scripts/bootstrap_ci.py`, Efron & Tibshirani
  1993 `[LITERATURE]`). It covers val-set sampling only, not training-seed variation.

## Decision rules (fixed before the session)
1. P1: if |AP50(P1) − AP50(P0)| ≤ 0.005 for every class on S26 and S29, P0 stays the only protocol. Otherwise every
   main-matrix run is reported under P0 (primary, comparable with all earlier runs) and P1 (secondary).
2. P2: if small-motorcycle AP50 rises by ≥ 0.02 at 1280 for both S26 and S29 with a bootstrap CI excluding 0, a
   training run at 1280 px (own ADR) is the next motorcycle step; otherwise the P2 detection head is.
3. A1: the CBAM − S26 round-20 difference counts as "beyond val-set noise" for a class only if its 95 % CI excludes 0.
   This does not replace the 3-seed requirement.
4. A5: the bus size split of S26 vs S27 (fixed script) gives the missing-class gap per size range; descriptive.

## Result
Not run.
