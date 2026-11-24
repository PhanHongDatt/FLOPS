# ADR-039: reference baselines on the YOLO11m main configuration

**Date:** 2026-11-19
**Status:** accepted (runs planned)
**Scope:** baselines (CLAUDE.md §11)

## Context

The supervisor accepted the YOLO11m 960 px configuration (ADR-033) and asked for more baselines next to
FedAvg (A0c) and Ours (P4t1). Only A0c and P4t1 exist on this configuration.

## Decision

Run on the missing-class partition `s1b5_bus_3k_allmoto_seed42` (s5moto), YOLO11m 960 px, 20 rounds,
1 epoch, eval rounds 10/20, the same evaluator and COCO-head init as A0c. Seed 42, one seed per baseline
(author's choice, 2026-11-19; raise to 3 seeds if the budget allows, CLAUDE.md §11).

| Session | Arm | What it is | Evidence |
|---|---|---|---|
| s38a | A1c | FedAvg weighted by per-class box counts | `[THESIS-HYPOTHESIS]` control, CLAUDE.md §11 |
| s38b | FedProxc | FedProx, mu = 0.01 (ADR-010) | `[LITERATURE]` Li et al., MLSys 2020 |
| s38c | A5c | original FedNTD: not-true KD from the received global model, beta = tau = 1 | `[LITERATURE]` Lee et al., NeurIPS 2022 |
| s38d | A0c@central | one client with the same 15,000 images (upper reference, ADR-026 protocol) | `[ENGINEERING]` |
| s38e/f | P4t1 seed 2024, missing / ideal | completes Ours to 3 seeds (ADR-033 order 3) | - |

A5c differs from Ours only in the teacher (received global model vs frozen COCO model) and in having no
cosine LR and no client RFS, so A5c vs P4t1 shows what the adaptation adds over the original method.

## Not run

- **FedNova:** every client has 3,000 images with the same batch and epochs, so tau_i is equal on all
  clients and FedNova's normalized average reduces exactly to FedAvg (`src/federated/strategies/fednova.py`,
  coef = p_i * tau_eff / tau_i = p_i). A run would repeat A0c.
- **SCAFFOLD:** disabled in `scripts/run_fl_experiment.py` (no faithful control-variate correction).

## Reading rule (fixed before the runs)

Mean of rounds 10/20, per class AP50, precision/recall, TP/FP/FN at 0.25, compared with A0c and P4t1 seed 42
on the same partition. One seed: descriptive only, no claim of a significant difference.

## Cost `[RESOURCE-BUDGET]`

Six sessions on T4 x2, one arm each, about 5.5 h per FedAvg-cost arm (A0c YOLO11m took 4.8-5.7 h); P4t1 arms take about 10 h (S26, S27, S35a/b 9.8-10.1 h), so s38e/f cost about 20 h, not 11 h (corrected 2026-11-24),
two sessions per account (phdatt, goshihayashi, alexnguyen273).

## Related ADRs

ADR-010 (FedProx), ADR-012 (NTD), ADR-021 (COCO-head baselines), ADR-026 (centralized), ADR-033 (main config).

## Result (s38a-d, seed 42, mean of rounds 10/20, descriptive)

| Arm | mAP50 | car | bus | truck | motorcycle | bus recall |
|---|---:|---:|---:|---:|---:|---:|
| A0c FedAvg (S32a) | 0.524 | 0.781 | 0.479 | 0.510 | 0.325 | 0.182 |
| A1c class-count FedAvg | 0.528 | 0.784 | 0.485 | 0.506 | 0.337 | 0.250 |
| FedProxc | 0.531 | 0.784 | 0.491 | 0.503 | 0.345 | 0.175 |
| A5c FedNTD | 0.547 | 0.785 | 0.496 | 0.504 | 0.402 | 0.355 |
| P4t1 Ours (S26) | 0.642 | 0.821 | 0.602 | 0.622 | 0.523 | 0.469 |
| A0c centralized | 0.459 | 0.764 | 0.403 | 0.451 | 0.217 | 0.445 |

- P4t1 is highest on every class; over the best baseline (A5c) +0.095 mAP50, bus +0.106, truck +0.118, moto +0.121.
- A1c and FedProxc stay within 0.02 of A0c on every class: class-count weighting does not explain the gain (§22.6).
- A5c helps motorcycle (+0.077 vs A0c), not bus or truck; the frozen COCO teacher + cosine LR + RFS of P4t1 add the rest.
- The centralized A0c falls below its own COCO round 0 (0.486; motorcycle 0.378 -> 0.217): 20 epochs of the plain
  recipe forget COCO knowledge. It is not a ceiling for P4t1. The fitting ceiling is P4t1 centralized (ADR-026,
  YOLO11s: federated within 0.006 mAP50); not run on YOLO11m. Recorded against the pre-registered "upper reference" role.
- s38e/f (P4t1 seed 2024) failed at `git clone` (exit 128, 45 s, no training) and were re-pushed 2026-11-20.

## Addendum 2026-11-20: P4t1 centralized ceiling (s38g)

Same protocol as ADR-026 on YOLO11m: P4t1 with one client holding the 15,000 images. Reading rule: ceiling minus
federated P4t1 seed 42 (S26) per class. Ablation (no KD / no cosine / no RFS) and FedNTD seeds 123/2024 are
deferred for GPU quota.
