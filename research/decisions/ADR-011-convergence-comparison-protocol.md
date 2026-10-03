# ADR-011: Convergence-length comparison on S1b-2k (baselines + proposed method)

**Date:** 2026-10-03
**Status:** accepted (requested by the author: run to convergence, 30–50 rounds, baselines and
the proposed method together; ~20 GPU-hours available this week)
**Scope:** experimental protocol for notebook 05 / sessions s5a, s5b

## Context

Session 2 (feasibility, S0, 5 rounds) did not converge (FedAvg mAP50 +0.030 in round 5) and its
FedAvg–centralized gap mixed federation cost with compute (5 vs 10 data passes). One full-data
FL round costs ~20 min on T4 x2, so 30–50 rounds × several arms does not fit the budget.

## Decision (identical for every arm — CLAUDE.md §20)

- **Partition:** S1b with `per_client: 2000` (plan.md §5.4's 4 × 2000 design):
  `s1b_bus_2k_seed42`; H1 control `s1_control_matched_2k_seed42` built from it.
- **Schedule:** 30 rounds × 1 local epoch, batch 16, seed 42 (`configs/experiments/convergence.yaml`).
- **Evaluation:** full global val every 5 rounds + round 0 + final round (`federated.eval_every`);
  AP at conf 0.001, FP/FN at 0.25 (ADR-009). Checkpoints every round.
- **Arms:** A0 FedAvg, FedProx (mu 0.01, ADR-010), A1 class-count FedAvg, A3 class-aware
  aggregation, A2b rho 0.25 (ADR-006), A4b = A2b + A3, and A0 on the matched control (H1).
  SCAFFOLD excluded (blocked, ADR-010); FedNova not run (budget).
- **Sessions:** s5a = A0, FedProx, A1, A3; s5b = A2b, A4b, A0@control. A failing arm is logged
  and the session continues; results are zipped (500-item limit).

## Status of the evidence this produces

Exploratory: G4/G5 are not passed (CLAUDE.md §7) and there is one seed (§14). It can show
whether the method changes missing-class behaviour on S1b and justify the 3-seed main runs;
it cannot establish the method.

## Budget estimate

~2.7 min/round (2 clients in parallel on 2 T4s, 2000 images each, plus trainer start-up) +
7 full-val evals × ~4.5 min ≈ 1.9 h per arm → ~13.5 GPU-h for 7 arms. Measure and record.

## Validation before launch

End-to-end CPU smoke of A4b on a tiny S1b partition (Ray + Flower + Ultralytics): rounds
complete, eval cadence applied, `aggregation_trace.yaml` shows per-class eligibility (bus
weighted only on the client that has bus). Local Windows-only failures came from concurrent
writes to Ultralytics' shared `labels/train.cache` (file locking); Linux renames atomically
and Kaggle s2 ran 4 clients with 0 failures.

## Related
ADR-006 · ADR-009 · ADR-010 · plan.md §5.3–§5.4, §7
