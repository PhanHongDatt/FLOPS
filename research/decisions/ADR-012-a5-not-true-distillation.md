# ADR-012: A5 — not-true distillation (FedNTD) adapted to the YOLOv8 head

**Date:** 2026-10-05
**Status:** accepted for an exploratory run (author approved running new directions before
G4/G5, labelled exploratory — CLAUDE.md §7)
**Scope:** `src/preservation/ntd_loss.py`, `NTDClient`, ablation preset `A5`, session s7b

## Context

s5/s6 (3 seeds, ADR-011): Missing-Class costs FedAvg −0.046 ± 0.008 AP50 bus vs the matched
control, mostly as missed buses (+487 FN). The proposed A4b is worse than FedAvg in all 3 seeds
(−0.014 ± 0.002) because removing the negative signal of bus-free clients raised FP bus ~3×
while recovering only ~72 FN. A direction is needed that keeps the negative signal but stops
local training from overwriting the global model's bus knowledge.

## Evidence

`[LITERATURE]` Lee, Jeong, Shin, Bae, Yun — "Preservation of the Global Knowledge by Not-True
Distillation in Federated Learning", NeurIPS 2022. Verified 2026-10-05 against the official
code (github.com/Lee-Gihun/FedNTD): `algorithms/fedntd/criterion.py`
`loss = CE + beta * tau^2 * KL(softmax_nt(z_g / tau) || log_softmax_nt(z_l / tau))`, not-true
positions from `refine_as_not_true`; defaults `config/fedntd.json`: tau = 1, beta = 1. The paper
frames local training as forgetting of out-of-distribution (locally absent) knowledge.

`[LITERATURE]` Shmelkov, Schmid, Alahari — "Incremental Learning of Object Detectors without
Catastrophic Forgetting", ICCV 2017: distilling a frozen detector's class responses preserves
classes absent from the new data (support for distillation in detection, not reused directly).

## Decision

`[THESIS-HYPOTHESIS]` adaptation (YOLOv8 scores are independent sigmoids, not one softmax):

- Teacher = frozen copy of the model received at the start of the round (the global model),
  evaluated on the same augmented batch.
- For every (anchor, class) entry whose task-aligned target is exactly 0 ("not true"; all
  classes of background anchors), add `beta * tau^2 * KL(Ber(s(z_g/tau)) || Ber(s(z/tau)))` to
  the BCE entry; same `sum / target_scores_sum` normalisation as the ordinary BCE.
- All clients use it (as in FedNTD); aggregation FedAvg; tau = 1, beta = 1 (official
  defaults, not tuned — §20).

Difference to A2b: the hard negative target stays; the teacher only adds a soft target where
the global model believes a class is present. On bus-free clients this protects bus scores on
bus-like regions instead of removing all "no bus here" pressure.

## Falsifiable prediction (pre-declared)

Against A0 on S1b, mean of rounds 20/25/30 (ADR-011 rule), seed 42:
- support: AP50 bus ≥ A0 + 0.010 (≈ 3× the seed std of A0, 0.0032) **and** FP bus not above
  A0 + 50 % **and** mAP50 not below A0 − 0.005;
- reject: AP50 bus ≤ A0 (no gain), or FP bus rises > 50 % (the A2b failure mode again).
- In between: inconclusive at one seed → needs seeds 123/2024 before any claim.

## Risks

- Teacher forward adds ~30 % compute per local step.
- Dense KD over 8,400 anchors × 4 classes is mostly background; beta = 1 may be weak for the
  rare positives-like regions. Not tuned in this run; a beta sweep would need its own record.
- A KL over a single Bernoulli loses FedNTD's "relative ranking among not-true classes";
  this is the adaptation being tested.

## Related
ADR-006 (A2b) · ADR-011 (protocol) · ADR-013 · ADR-014 · literature_registry: fedntd, shmelkov2017
