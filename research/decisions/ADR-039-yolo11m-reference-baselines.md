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

Six sessions on T4 x2, one arm each, about 5.5 h each (A0c/P4t1 YOLO11m took 5.1-5.6 h): about 33 h,
two sessions per account (phdatt, goshihayashi, alexnguyen273).

## Related ADRs

ADR-010 (FedProx), ADR-012 (NTD), ADR-021 (COCO-head baselines), ADR-026 (centralized), ADR-033 (main config).
