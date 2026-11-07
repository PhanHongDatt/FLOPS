# ADR-028: absent-class distillation on confusable anchors (docs/Solution.md, method B1)

- Status: proposed (implemented and unit-tested 2026-11-05; no run yet, GPU budget needs the author's approval)
- Related: ADR-012 / ADR-018 (not-true KD), ADR-014 / ADR-020 (D1 bus error decomposition), ADR-023 (s5moto,
  960 px), ADR-026 (centralized ceiling); verification note `research/notes/spec_verify_20261105.md`

## Context
FL with P4t1 is within 0.006 mAP50 of its centralized ceiling (ADR-026); what is left of the missing-class
cost is mostly bus recall (0.446 vs 0.509 at max-F1). D1 (ADR-020, P4noKD round 20) put the extra missed buses
on S1b5 vs pooled at confused car + truck +164, missed +140, low score −34: confusion is the largest single
part (≈ 61 % of the net extra FN, not the 74 % quoted in the spec), but it is only weakly dominant.

## Hypothesis `[THESIS-HYPOTHESIS]`
On a client with no box of class c, every truck/car anchor gets target 0 for c, so the c score is pushed down
hardest at the c / truck / car boundary. Replacing that 0 by the frozen COCO teacher's probability, on those
anchors only, keeps the boundary; background anchors keep target 0, so the negative signal against false c
detections stays (A2b removed it and FP bus rose ~3×, ADR-006).

Related work: Shmelkov et al., ICCV 2017 (`shmelkov2017`, distil old-class outputs when their labels are
absent) and FedNTD (`fedntd`). The restriction to confusable foreground anchors and the cap q_max are this
thesis's choices.

## Decision
Rule (spec §3.2), for each client k, c ∈ M_k (zero local boxes), anchor a:
`t[a,c] = λ · min(q[a,c], q_max) · s[a]` if `fg[a]` and `label[a] ∈ N(c)`, else unchanged; q = sigmoid of the
teacher logit, s[a] = task-aligned score. N(bus) = {truck, car}, N(truck) = {bus, car} (COCO ids 5 → 7, 2;
7 → 5, 2). λ = 1.0, q_max = 0.5, fixed here before any run.

Implementation `[YOLO-DOC]`, `[ENGINEERING]`: `src/preservation/absent_kd.py` (`AbsentKDBCE`), a wrapper of
`criterion.bce` installed by the existing teacher plugin (`NotTrueDistillation(absent=...)`, one teacher forward
for both). In 8.3.253 `target_scores_sum` and the box/DFL weights come from the assigner output before `bce`
runs, so only the classification term sees the new targets (V2). `M_k` comes from the client's partition
manifest and is never sent to the server. Config `configs/methods/absent_kd.yaml` (`enabled: false`).

Arms (`scripts/run_fl_experiment.py`; "B1" already names the ADR-015 arm):

| Arm | Spec name | Not-true KD | Change vs P4t1 |
|---|---|---|---|
| `P4t1AK` | B1_replace | off (β = 0) | + B1, − not-true KD |
| `P4t1AKadd` | B1_add | on (β = 1) | + B1 |
| `P4t1AKall` | ablation `scope: all_anchors` | off | B1 on every anchor |
| `P4t1AKq1` | ablation `q_max: 1.0` | off | no cap |
| `P4t1AKl05` | ablation `lam: 0.5` | off | half weight |

Spec point to settle: §3.2 requires a client with no absent class (C4) to train with exactly the baseline
loss. Only `P4t1AKadd` does; `P4t1AK` drops the not-true KD on every client, so on the ideal scenario it is
not the baseline and needs its own ideal run. `P4t1AKadd` on the ideal scenario has no absent class on any
client and is P4t1 itself, so the ideal-scenario baseline run serves both.

## Decision rule (spec §8, fixed before any run)
B1 passes only if all four hold, against P4t1 on the same partitions and seeds (42, 123, 2024), mean of
rounds 10/20:
1. Missing-class drop `AP50_bus(ideal) − AP50_bus(bus missing 3/5)`, paired by seed, smaller than P4t1's in
   3/3 seeds.
2. Real buses classified as truck + car (D1, conf 0.25, IoU 0.5) fewer than P4t1 (mean of 3 seeds).
3. Background bus FP (D1 `background`) at most +20 % vs P4t1.
4. mAP50 not lower than P4t1 by more than 0.01 in any seed.
Otherwise B1 is rejected and recorded as such. Seed-42-only runs are descriptive and cannot pass the rule.

## Cost `[RESOURCE-BUDGET]`
Kaggle T4 × 2, measured: P4t1 @ s5moto 960 px 6.1 h (S21b); P4noKD 960 px 3.6 h (S21a). B1 adds no extra
forward (it reuses the teacher pass), so a B1 arm ≈ 6.1 h.

| Block | Runs | GPU h |
|---|---:|---:|
| Spec stage 0: P4t1 + FedAvg, 3 scenarios × 3 seeds (S21b exists) | 17 | ≈ 81 |
| Spec stage 1: smoke + 3 ablations + main 3 scenarios × 3 seeds | 13 | ≈ 74 |
| Minimal descriptive check (seed 42): P4t1 @ ideal-allmoto, `P4t1AKadd` @ s5moto | 2 | ≈ 12 |

The weekly quota is 30 h per GPU account (two accounts), so the full spec is about three weeks of quota.

## Result
Smoke s24_g (2 rounds, seed 42, `P4t1AKadd` @ s5moto 960 px): completed, 5/5 clients aggregated each round,
losses finite. C4 (no absent class) round-1 train losses equal S21b P4t1 C4 round 1 to every printed digit,
so the baseline-loss rule holds on GPU; C0-C3 differ in cls loss only where a class is absent. Round-2
mAP50 0.536 is not comparable with 20-round runs. No decision; main runs need the author's approval.

Run s25_g (`P4t1AKadd` @ s5moto 960 px, 20 rounds, seed 42, paired with S21b), mean of rounds 10/20:
mAP50 0.6075 (−0.002), AP50 bus 0.5641 (−0.001), truck −0.002, car −0.000, motorcycle −0.005; bus recall
+0.020 and precision −0.027, bus TP +30.5 / FN −30.5, FP_bus +133.5 (+21.5 %), FP_truck +306.5 (+12.5 %).
Observation: B1 moves the bus/truck operating point (more detections, right and wrong) without changing AP.
Rule 4 holds for seed 42; rules 2-3 need D1 on both round-10/20 checkpoints; rule 1 needs the ideal runs.
Seed 42 only, descriptive. No improvement is visible that would justify the 3-seed runs.
