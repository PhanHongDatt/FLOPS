# ADR-033: main configuration = YOLO11m, 960 px, s5moto / ideal, 20 rounds

- Status: accepted (renumbered from ADR-031, which the MLOps ADR also took, 2026-11-11; author's choice 2026-11-10: the goal is the highest attainable AP, so YOLO11m)
- Related: ADR-023 (s5moto, 960 px), ADR-030 (YOLO11m: +0.035 AP50 over YOLO11s in both scenarios; same bus gap)

## Decision
Frozen for every main-matrix run; any change needs a new ADR and re-running the affected arms.
- Model YOLO11m from COCO weights, `configs/experiments/convergence_5c_y11m_20r_960.yaml` (960 px, batch 8,
  nbs 64, 5 clients, 20 rounds, 1 local epoch, eval rounds 10/20, metric = mean of rounds 10/20).
- Partitions: missing `s1b5_bus_3k_allmoto_seed42` (bus 0 on C0-C2, truck 0 on C3) and ideal
  `s1b5_pooled_3k_allmoto_seed42` (same images, every class on every client).
- Seeds 42, 123, 2024 (CLAUDE.md §14); the partition seed stays 42 (only the training seed changes).
- 20 rounds, not 40: on S9 (YOLO11s) rounds 20 → 40 added mAP50 +0.011 for twice the cost (S22).

## Matrix

| Arm | Role | missing | ideal | Have |
|---|---|---|---|---|
| A0c | FedAvg, COCO head (ADR-021 reference baseline) | 3 seeds | 3 seeds | none |
| P4t1 | current best (COCO head + not-true KD + cosine LR + RFS 1.0) | 3 seeds | 3 seeds | seed 42 both (S26, S27) |
| proposed | one method aimed at bus recall; chosen by its own ADR before its runs | 3 seeds | 3 seeds | none |

16 runs remain. Report per class: AP50, precision, recall, FP/FN, mean ± std over seeds;
gap_c = AP50_c(ideal) − AP50_c(missing) per arm; ΔAP vs A0c.

## Cost `[RESOURCE-BUDGET]`
P4t1 YOLO11m measured 10.0-10.1 h per run on T4 x 2 (S26/S27); A0c has no teacher forward, estimated ~6 h
(YOLO11s ratio P4noKD / P4t1 = 3.6 / 6.1, `[NEEDS-VERIFICATION]` on YOLO11m). Total ≈ 6 × 6 + 4 × 10 + 6 × 10
≈ 136 h ≈ 2.5 weeks on the two GPU accounts (~30 h/week each), one session per run, part 2 if cut at 11 h.

## Order
1. A0c seed 42, missing and ideal (one per account, in parallel): baseline and ΔAP of P4t1 at seed 42.
2. P4t1 and A0c seeds 123 / 2024.
3. Proposed method (ADR first), 3 seeds × 2 scenarios.

## Result, order 1 (A0c seed 42: S32a missing, S32b ideal), mean of rounds 10/20, descriptive
Both 20 of 20 rounds, 5 results / 0 failures each, artifact check passed; wall 5.7 h (S32a) and 4.8 h (S32b),
so the ~6 h estimate holds.

| AP50 | A0c missing | P4t1 missing (S26) | ΔP4t1 | A0c ideal | P4t1 ideal (S27) | ΔP4t1 |
|---|---:|---:|---:|---:|---:|---:|
| mAP50 | 0.5239 | 0.6422 | +0.118 | 0.5445 | 0.6559 | +0.111 |
| car | 0.7808 | 0.8211 | +0.040 | 0.7869 | 0.8224 | +0.036 |
| bus | 0.4793 | 0.6020 | +0.123 | 0.5055 | 0.6377 | +0.132 |
| truck | 0.5101 | 0.6224 | +0.112 | 0.5344 | 0.6333 | +0.099 |
| motorcycle | 0.3253 | 0.5234 | +0.198 | 0.3511 | 0.5301 | +0.179 |

gap_c (ideal − missing): A0c bus +0.026, truck +0.024, motorcycle +0.026, mAP50 +0.021;
P4t1 bus +0.036, truck +0.011, motorcycle +0.007, mAP50 +0.014.

Observations (seed 42, not causal claims):
- A0c on YOLO11m barely moves from the zero-shot COCO model (round 0 mAP50 0.486, bus 0.477, motorcycle 0.379):
  missing-scenario motorcycle falls below zero-shot (0.347 / 0.303 at rounds 10 / 20).
- Under A0c the missing scenario suppresses bus: recall 0.18 at precision 0.88 (ideal 0.44 / 0.64), TP_bus gap
  +506 vs +262 for P4t1. P4t1 roughly halves the bus recall gap; the bus AP50 gap is larger for P4t1 (+0.036 vs
  +0.026) only because both A0c AP50 values are low.
- ΔP4t1 is the joint effect of KD + cosine LR + RFS 1.0; it does not isolate KD (ADR-018 rule needs P3 vs A0c).

## Result, order 2 part 1 (seed 123: S35a P4t1 missing, S35b P4t1 ideal, S35c A0c missing), mean of rounds 10/20
All three 20 of 20 rounds, 5 results / 0 failures each, artifact check passed.

| AP50 | P4t1 missing s42 / s123 | P4t1 ideal s42 / s123 | A0c missing s42 / s123 |
|---|---|---|---|
| mAP50 | 0.6422 / 0.6429 | 0.6559 / 0.6562 | 0.5239 / 0.5313 |
| bus | 0.6020 / 0.6011 | 0.6377 / 0.6418 | 0.4793 / 0.4979 |
| truck | 0.6224 / 0.6224 | 0.6333 / 0.6319 | 0.5101 / 0.5231 |
| motorcycle | 0.5234 / 0.5265 | 0.5301 / 0.5283 | 0.3253 / 0.3227 |
| car | 0.8211 / 0.8214 | 0.8224 / 0.8227 | 0.7808 / 0.7816 |

P4t1 bus gap (ideal − missing): +0.036 (seed 42), +0.041 (seed 123); per round +0.041 / +0.030 and
+0.045 / +0.036; bus recall gap +0.078 / +0.108, TP_bus +262 / +299.5. P4t1 − A0c (missing): mAP50 +0.118 / +0.112,
bus +0.123 / +0.103. P4t1 seed-to-seed spread is ≤ 0.004 AP50 on every class. Still pending: A0c ideal seed 123,
all seed-2024 runs.

## Result, order 2 part 2 (A0c complete: S36a ideal s123, S36b missing s2024, S36c ideal s2024), mean of r10/20
All 20 of 20 rounds, 5 results / 0 failures each, artifact check passed.

| A0c, mean ± std over seeds 42 / 123 / 2024 | missing | ideal | gap (ideal − missing) |
|---|---|---|---|
| mAP50 | 0.527 ± 0.004 | 0.543 ± 0.003 | +0.016 ± 0.007 |
| bus | 0.492 ± 0.011 | 0.509 ± 0.004 | +0.017 ± 0.008 (per seed +0.026 / +0.010 / +0.015) |
| truck | 0.513 ± 0.010 | 0.532 ± 0.002 | +0.019 ± 0.010 |
| motorcycle | 0.321 ± 0.006 | 0.343 ± 0.012 | +0.022 ± 0.015 |
| bus recall | 0.182 ± 0.032 | 0.450 ± 0.011 | +0.268 ± 0.031 |

P4t1 (seeds 42 / 123, seed 2024 pending): bus gap +0.038 ± 0.004, recall gap +0.093 ± 0.021; truck +0.010,
motorcycle +0.004. Reading (descriptive): under A0c the missing scenario cuts bus recall far more (0.45 → 0.18) but
AP50 less, because both A0c AP50 values are low; P4t1 removes most of the truck / motorcycle gap and two thirds of
the bus recall gap, while its bus AP50 gap is about twice A0c's. Pending: P4t1 seed 2024 (missing and ideal).
