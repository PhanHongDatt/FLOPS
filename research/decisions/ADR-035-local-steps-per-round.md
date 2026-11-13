# ADR-035: direction A — fewer local steps per round (1/4 epoch x 4x rounds)

- Status: accepted for an exploratory pair of runs (author: "đánh giá theo mức độ ưu tiên, tiến hành thực hiện",
  2026-11-12; priority 1 of the direction analysis)
- Related: ADR-022 (gap grows with severity; neither P4noKD nor FedProx narrows it), ADR-026 (centralized ceiling:
  no gap), ADR-030 (gap 0.038 YOLO11s / 0.036 YOLO11m), ADR-034 (gap at every object size, recall-driven)

## Hypothesis `[THESIS-HYPOTHESIS]`
The bus gap (ideal − missing) comes from the local steps a bus-free client takes between two aggregations
(≈ 375 at YOLO11m batch 8, ≈ 190 at YOLO11s batch 16). Aggregating k times as often with 1/k of an epoch per round
(same total steps, same images) shrinks it. In the limit of one step per round FedAvg is minibatch SGD on the union
of the data — the centralized setting, which has no gap here (YOLO11s central 0.589 ≥ FL missing 0.565; FL ideal
0.603) — so the gap must vanish for small enough steps; the run measures the size at k = 4.
`[LITERATURE]` (to verify before the registry entry): McMahan et al., AISTATS 2017 (FedAvg, local epochs E);
Li et al., ICLR 2020 ("On the Convergence of FedAvg on Non-IID Data": the bound grows with local steps).

## Decision `[ENGINEERING]`
`src/federated/local_fraction.py`: round r of a client trains on part (r − 1) mod 4 of a seeded permutation of its
images, redrawn every 4 rounds (each 4 consecutive rounds = exactly one epoch); RFS (t = 1.0) resamples that part
with frequencies from the part. Config `configs/experiments/convergence_5c_y11s_80r_q960.yaml`: P4t1, YOLO11s,
960 px, batch 16, `local_fraction: 0.25`, 80 rounds, cosine LR over 80 rounds, eval rounds 40 / 80 (epoch-
equivalents 10 / 20). Everything else as S21b / S28. Unit-tested (tests/test_local_fraction.py), incl. the client path.
Run on YOLO11s: the gap is the same as YOLO11m's and a run costs about half.

Runs: s31a = P4t1 @ s5moto (missing), s31b = P4t1 @ s5motopooled (ideal), seed 42.
Known confounds: the SGD optimizer (momentum) is rebuilt every round, 4x as often; FedAvg weights unchanged
(every client uses 1/4 of its images).

## Decision rule (fixed before the runs), mean of rounds 40 / 80 vs the k = 1 pair S21b / S28
1. Gap_bus(k=4) = AP50_bus(s31b) − AP50_bus(s31a) ≤ Gap_bus(k=1) − 0.015 = 0.038 − 0.015 = 0.023.
2. AP50 of every class at k = 4 (missing) not lower than S21b by more than 0.01.
3. Bootstrap (scripts/bootstrap_ci.py, round 80 vs round 20) is reported for the missing-scenario bus difference.
Pass → the curve k ∈ {1, 2, 4} and 3 seeds (own ADR). Fail → recorded as a negative result. Seed 42 only: descriptive.

## Cost `[RESOURCE-BUDGET]`
Per round about 1/4 of the training time plus a fixed per-client overhead (trainer, dataloader, teacher copy;
estimated ~30 s): ~6 h per run vs 4.6-6.1 h at k = 1, 2 runs in parallel on the two accounts. Uplink/downlink
volume x4 (simulation; reported as communication cost).

## Result
**Fail (rules 1 and 2) -> negative result, the k-curve and 3-seed follow-up are not run.** Seed 42, descriptive.
Both runs 80 of 80 rounds, 5 results / 0 failures each, artifact check passed; wall 5.4 h (S31a) and 6.1 h (S31b).

| mean r40/80 (k=4) vs r10/20 (k=1) | S21b | S31a | S28 | S31b | gap k=1 | gap k=4 |
|---|---:|---:|---:|---:|---:|---:|
| mAP50 | 0.6095 | 0.5990 | 0.6246 | 0.6096 | +0.015 | +0.011 |
| AP50 bus | 0.5650 | 0.5623 | 0.6034 | 0.5987 | **+0.038** | **+0.036** |
| AP50 truck | | | | | +0.020 | +0.005 |
| recall bus / TP bus | | | | | +0.085 / +263.5 | +0.038 / +104.5 |
| precision bus | | | | | -0.050 | +0.013 |

1. Gap_bus(k=4) = +0.036 > 0.023: **fail** (k=1 +0.038; per round +0.057 / +0.016 vs +0.054 / +0.022).
2. S31a vs S21b: motorcycle -0.022 (> 0.01): **fail**; truck -0.009, car -0.008, bus -0.003, mAP50 -0.011.
   The ideal run also loses at k=4 (S31b vs S28: mAP50 -0.015, truck -0.024, motorcycle -0.026, FP_truck +407).
3. Bootstrap not run (needs a GPU eval session with --dump); with rules 1-2 failed it cannot change the decision.

Observations (not causal claims): at k=4 the bus recall gap halves (TP gap 263.5 -> 104.5) but the ideal run
trades it for precision, so the AP50 gap is unchanged; the smaller truck gap comes from the ideal run getting
worse, not from S31a getting better. k=4 is still ~94 local steps per round, far from the one-step limit of the
hypothesis, so the hypothesis is not falsified, only shown to give no usable gain at affordable k; the 4x
optimizer-state resets (a new Ultralytics trainer every round; warmup is 0) are a plausible cause of the
across-the-board loss `[THESIS-HYPOTHESIS]`, untested.
