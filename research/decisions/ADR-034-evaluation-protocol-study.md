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

## Result (s30e, 2026-11-11, all 9 steps completed; files in research/feasibility/eval_protocols/)
Fixed size-split script vs the evaluator, round 20 "all": S26 bus 0.617 vs 0.622, motorcycle 0.543 vs 0.541 — the
multi-label fix holds on the full val set (remaining gap = COCO greedy vs Ultralytics matching).

1. **P1 (4-class NMS): no effect.** |AP50(P1) − AP50(P0)| ≤ 0.0002 for every class on S26 and S29. Rule 1: P0 stays
   the only protocol; `max_det` shared by 80 classes is not limiting (closes V7).
2. **P2 (1280 px eval): rule not met.** Small-motorcycle AP50 +0.030 (S26, CI [−0.012, +0.082]) and +0.012 (S29,
   CI [−0.027, +0.065]): below 0.02 for S29 and neither CI excludes 0. Evaluating at 1280 helps small boxes (small
   bus +0.040 / +0.061, CIs exclude 0) but costs large ones (large bus −0.031 / −0.026, large truck −0.037 / −0.031,
   CIs exclude 0), so mAP50 falls (−0.005 / −0.008). Per the rule the next motorcycle step is the P2 head, not a
   1280 training run. Caveat recorded: the large-object loss is a train/test scale mismatch (trained at 960), so this
   result does not show that training at 1280 would fail.
3. **CBAM − S26, round 20, val-set CI:** motorcycle +0.007 [−0.011, +0.026], bus +0.001 [−0.004, +0.006], truck
   +0.0035 [+0.0006, +0.0064]. Only the truck difference excludes 0. The motorcycle gain (+0.016 on the mean of rounds
   10/20, +0.017 at round 20 with the Ultralytics evaluator) is not distinguishable from val-set sampling noise.
   ADR-032's "promising" stays descriptive; seeds 123 / 2024 are needed before any claim.
4. **Missing-class bus gap by size (S27 ideal − S26 missing, YOLO11m):** all +0.032 [+0.025, +0.039], small +0.020
   [−0.003, +0.042], medium +0.028 [+0.014, +0.044], large +0.015 [+0.006, +0.025]. The gap is present at every
   size and largest for medium buses: the missing-class effect is not a small-object effect. Truck gap +0.010
   [+0.006, +0.015]; motorcycle +0.006 [−0.015, +0.026] (motorcycle is on every client in both scenarios).
   These CIs cover val-set sampling only, not training seeds.
