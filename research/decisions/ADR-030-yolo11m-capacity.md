# ADR-030: YOLO11m at 960 px on the missing-class scenario (model capacity)

- Status: accepted for one exploratory run (author's request 2026-11-07); seed 42 only
- Related: ADR-023 (s5moto, 960 px), ADR-026 (centralized ceiling of P4t1 / YOLO11s: bus 0.589, motorcycle 0.459),
  ADR-028 (B1 gave no AP change, s25_g)

## Context
P4t1 with YOLO11s is within 0.006 mAP50 of its centralized ceiling (ADR-026), and B1 changed no AP (s25_g).
Rare-class AP50 (bus 0.565, motorcycle 0.504) is far from the author's 0.70 target. The remaining question is
whether model capacity, not the federated setting, limits the rare classes.

## Hypothesis `[THESIS-HYPOTHESIS]`
A larger detector (YOLO11m, 20.2 M parameters vs 9.5 M for YOLO11s, both from COCO weights with the 80-class
head) raises bus and motorcycle AP50 under the same missing-class partition and FL method.

## Decision `[ENGINEERING]`
- Run P4t1 @ s5moto, 960 px, 20 rounds, seed 42 with `configs/experiments/convergence_5c_y11m_20r_960.yaml`;
  everything else as S21b. The not-true KD teacher is the COCO YOLO11m (P4t1 builds the teacher from the
  initial weights), so teacher strength changes together with the student; this is part of "the model".
- Batch 8 (16 for YOLO11s): VRAM. Ultralytics accumulates to nbs = 64, so effective batch and optimizer steps
  per round are the same; BatchNorm statistics use 8 images per step instead of 16 (confound, recorded).
- Run tag `m960` keeps the run directory apart from the YOLO11s runs.

## Comparison (descriptive, seed 42)
Against S21b (P4t1, YOLO11s, same partition, seed and rounds), mean of rounds 10/20: AP50, precision, recall per
class and mAP50. Any gain is attributed to model size + teacher + batch together; one seed cannot separate them
or exclude seed noise (±0.01-0.02 AP50).

## Cost `[RESOURCE-BUDGET]`
YOLO11s at 960: 4.6-6.1 h on T4 x 2. YOLO11m has ~3.2x the GFLOPs (68.0 vs 21.5, Ultralytics model card,
`[NEEDS-VERIFICATION]` exact figure), so an estimated 12-18 h: more than one 12 h Kaggle session. Part 1 stops
the arm at 11 h and keeps its checkpoints; part 2 restores the run and resumes (ADR-024/025 mechanism).

## Result
s26_g: one session, 20 of 20 rounds, no OOM at batch 8, wall 10.0 h on T4 x 2 (1.6x S21b, not 3x).
Mean of rounds 10/20 vs S21b (P4t1, YOLO11s): mAP50 0.6422 (+0.033), bus 0.6020 (+0.037), truck 0.6224 (+0.049),
car 0.8211 (+0.026), motorcycle 0.5234 (+0.019). Every class is higher at both round 10 and round 20; precision
and recall rise for every class except truck precision (+0.007) and motorcycle recall (+0.002); FP_truck +251.5.
The federated YOLO11m run is above the YOLO11s centralized ceiling (ADR-026: bus 0.589, motorcycle 0.459).
Observation, seed 42: the gains exceed the usual seed spread (±0.01-0.02) and are consistent across classes and
rounds, so capacity is a plausible limit; not attributable to size alone (teacher, batch) and not yet replicated.
Bus and motorcycle remain below the author's 0.70 target. Whether the missing-class gap (ideal − missing) shrinks
is unknown without an ideal-scenario YOLO11m run.

## Amendment 1 (2026-11-09): ideal scenario with YOLO11m (s27_g)
Question: how much AP50 does the missing-class partition cost YOLO11m? New partition
`s1b5_pooled_3k_allmoto_seed42` (the 15,000 s5moto images re-dealt evenly over the same 5 clients, every class on
every client; the notebook asserts identical images and no missing class); arm `P4t1@s5motopooled`, same YOLO11m
config, seed 42. Measure, mean of rounds 10/20: gap_c = AP50_c(ideal) − AP50_c(s5moto) per class, bus first.
Descriptive only. No YOLO11s run exists on this partition, so whether the gap is smaller than with YOLO11s
cannot be read from these runs; the older YOLO11s 640 px pooled-vs-S1b5 gap (different data and resolution) is
context, not a paired comparison. Estimated 10 h (as s26_g).

Result (s27_g, 10.1 h, 20 of 20 rounds, partition check passed): mean of rounds 10/20, ideal minus s5moto (S26):
bus AP50 0.6377 vs 0.6020, gap +0.036 (+0.041 at round 10, +0.030 at round 20); truck +0.011, motorcycle +0.007,
car +0.001, mAP50 +0.014. The bus gap comes from recall: +0.078 (TP +262, FN −262) against precision −0.045
(FP +122). Observation, seed 42: with YOLO11m the missing-class partition still costs bus about 0.036 AP50 and
0.08 recall, i.e. the bus deficit is a missed-detection problem rather than a false-alarm one; the larger model
did not remove the missing-class effect. Its size relative to YOLO11s is unknown (no YOLO11s run on this
partition).

## Amendment 2 (2026-11-09): YOLO11s on the ideal partition (s28)
Same arm and partition as s27_g with the YOLO11s config (`convergence_5c_y11s_20r_960.yaml`, batch 16), seed 42,
run on phdatt. Pairs with S21b to give the YOLO11s gap gap_c = AP50_c(ideal) − AP50_c(s5moto), compared with
the YOLO11m gap (bus +0.036). Descriptive, one seed: a difference in gaps below ~0.02 is within seed spread.
Estimated 6 h.

Result (s28, 5.8 h, 20 of 20 rounds). Model x scenario, seed 42, mean of rounds 10/20, AP50:

| | s5moto (missing) | ideal | gap |
|---|---:|---:|---:|
| YOLO11s bus | 0.5650 | 0.6034 | +0.038 |
| YOLO11m bus | 0.6020 | 0.6377 | +0.036 |
| YOLO11s truck | 0.5735 | 0.5939 | +0.020 |
| YOLO11m truck | 0.6224 | 0.6333 | +0.011 |
| YOLO11s mAP50 | 0.6095 | 0.6246 | +0.015 |
| YOLO11m mAP50 | 0.6422 | 0.6559 | +0.014 |

Observation: the bus gap is the same for both sizes (0.038 vs 0.036, and in both it is a recall loss of about
0.08 with ~262 extra missed buses). YOLO11m raises the level by ~0.035 in both scenarios but does not shrink the
missing-class effect on bus; YOLO11m under missing classes ≈ YOLO11s ideal (0.602 vs 0.603). The truck gap
halves (0.020 → 0.011) but one seed cannot separate 0.009 from seed spread. Per-round bus gaps vary (YOLO11s
0.054 / 0.022), so 3 seeds are needed before the gaps are reported as estimates.

