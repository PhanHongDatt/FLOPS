# ADR-014: Diagnostics D1 (error decomposition) and D3 (pooled re-split of S1b)

**Date:** 2026-10-05
**Status:** accepted
**Scope:** `src/evaluation/error_analysis.py`, `scripts/analyze_errors.py`,
`partition_pooled_iid`, `configs/partition/s1b_pooled_iid_2k_seed42.yaml`, session s7a

## Context

The H1 effect (A0 S1b vs matched control, −0.046 ± 0.008 AP50 bus, 3 seeds) mixes two causes:
bus absent on 2/4 clients **and** less bus data (control: 2,781 bus boxes vs 1,367, 2.03×).
Before investing in methods, it must be known (a) whether the distribution itself costs AP, and
(b) what kind of error the missing buses are.

## Decision

**D3 — pooled re-split** `[ENGINEERING]`: exactly the 8,000 S1b images, re-dealt evenly over the
same 4 clients, stratified by the set of classes each image holds (`S1-Pooled-IID`). Same images,
same box totals; only who holds them changes. Run A0 on it (arm `A0@pooled`), seed 42, ADR-011
protocol. This replaces the centralized upper bound (D4) of the earlier plan for this batch:
D4 needs one GPU for ~3 h (single 8,000-image client) and answers FL-vs-centralized, while D3
is the ceiling a Missing-Class method can aim for.

**D1 — error decomposition** `[ENGINEERING]`: for the round-30 global checkpoints of s5a/s5b/s6a/
s6b, classify every bus (and truck) GT as tp / confused_X / low_score / missed and every
confident bus prediction as on_X / loc / background (conf 0.001 predictions, split at 0.25,
IoU 0.5). Per-GT existence matching, so totals differ slightly from the confusion-matrix FP/FN.

## Pre-declared reading (rule: mean of rounds 20/25/30; seed-42 std of A0 ≈ 0.003)

- **D3:** `A0@pooled − A0@S1b` ≥ +0.010 AP50 bus → the Missing-Class distribution costs AP by
  itself; methods (A5/A6) are worth pursuing. |Δ| < 0.010 → the H1 effect is mostly the bus-data
  shortfall; report H1 as "less data", and a method that only re-weights cannot recover it.
  `A0@control − A0@pooled` measures the data-quantity part.
- **D1:** if S1b's extra FN (vs control) are mostly `low_score`, bus features survive and only
  scores are suppressed (calibration/score direction → A5/A6 plausible). Mostly `confused_truck`
  → class confusion (neighbouring-class direction). Mostly `missed` → representation loss.
  A4b's extra FP split by `on_truck` / `background` shows what the over-prediction hits.

## Related
ADR-009 (thresholds) · ADR-011 (protocol) · ADR-012 · ADR-013 · CLAUDE.md §10
