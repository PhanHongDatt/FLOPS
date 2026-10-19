# ADR-022: Missing-class severity sweep on the same 15,000 images, and FedProx on the IID re-split

**Date:** 2026-10-19
**Status:** accepted for feasibility runs (author: "tiến hành thực hiện cả 2 đợt cho tôi")
**Scope:** `partition_redeal` (scenario `S1-Redeal`), `configs/partition/s1b5_bus1_3k_moto_seed42.yaml`,
notebook 05 partition `s5bus1`, sessions s18a–s18c

## Context
ADR-021 established that P4noKD beats A0c, A1c and FedProxc in 3/3 seeds, but its gain is as large on the IID
re-split as on S1b5, so it does not close the missing-class gap. The quantity the paper needs is the gap

  gap(method, scenario) = AP50_bus(method, pooled) − AP50_bus(method, scenario), paired by seed,

i.e. the damage caused only by WHERE bus lives (same images, same box totals, same client sizes).
With bus absent on 3/5 clients the A0c gap is +0.031 / +0.036 (seeds 42 / 123); truck, absent on 1/5, shows
≈ 0. A dose–response over severity is a stronger test of H1 than one point, and is in scope (section 10,
"clients still containing the target class").

## Decision
1. **Severity level 4/5** — `s1b5_bus1_3k_moto_seed42`: the 15,000 images of S1b5 re-dealt over the same 5
   clients (3,000 each) so that bus lives on C4 only; truck stays absent on C3 exactly as in S1b5; car and
   motorcycle on every client. S1b5 (3/5) → bus1 (4/5) changes only how many clients hold bus.
   `[ENGINEERING]` `partition_redeal`: images grouped by (allowed clients, class signature), most constrained
   first, each group split over its allowed clients in proportion to client size; it raises before any
   training if the absent map cannot be satisfied (the notebook then skips only the arms that use it).
   Severity ladder: pooled 0/5, S1b5 3/5, bus1 4/5.
2. **Arms:** A0c and P4noKD on bus1, seeds 42 / 123 / 2024.
3. **FedProxc on pooled**, seeds 42 / 123 / 2024: the gap of the standard heterogeneity baseline, and
   whether its S1b5 deficit vs FedAvg is a general cost of the proximal term or specific to the missing class.
4. **Not run:** A1c on pooled — with every client holding every class its class-count weights are 0.190–0.209
   (bus) and 0.193–0.208 (truck) against 0.2 for FedAvg, so it equals FedAvg up to ±5 % of the class rows.

Same model, rounds, evaluator and rule as ADR-021 (mean of rounds 10 and 20).

## Pre-registered reading (fixed before any s18 run)
- **R1 (H1, dose–response):** A0c gap at 4/5 > A0c gap at 3/5 in all 3 seeds → the damage grows with the number of
  clients missing the class. Otherwise the effect is not shown to depend on severity.
- **R2 (does P4noKD mitigate at higher severity):** P4noKD gap at 4/5 < A0c gap at 4/5 in all 3 seeds → yes;
  otherwise no (reported as is).
- **R3 (FedProx):** FedProxc gap at 3/5 < A0c gap at 3/5 in all 3 seeds → FedProx reduces the missing-class damage;
  otherwise not. FedProxc − A0c on pooled is reported as its general cost or gain.
- Non-target classes (car, motorcycle, truck) at 4/5 vs 3/5 are reported for A0c and P4noKD; no threshold.
- All seeds are reported individually; n = 3, so no significance claim beyond "same sign in every seed".

## Cost
3 sessions × 3 arms (A0c@bus1, P4noKD@bus1, FedProxc@pooled), measured 2.0–2.5 h per arm → ~7.5 h per session,
under the 11 h session budget guard; ~22 GPU-h. goshihayashi s18a_g (seed 42); phdatt s18b (123), s18c (2024).
