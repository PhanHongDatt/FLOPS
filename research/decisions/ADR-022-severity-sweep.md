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

## Results (2026-10-19; s18a_g, s18b complete — s18c, seed 2024, still running)
Partition check (session log): bus1 class counts C0–C3 bus = 0, C4 bus = 2,628; truck 0 on C3 only; same 15,000
images. All 6 arms of seeds 42 / 123 completed 20 rounds. Mean of rounds 10/20, AP50, per seed (42 / 123):

| Method | scenario (clients without bus) | mAP50 | bus | car | truck | motorcycle |
|---|---|---|---|---|---|---|
| A0c | pooled (0/5) | 0.439 / 0.442 | 0.414 / 0.415 | 0.696 / 0.696 | 0.418 / 0.429 | 0.229 / 0.228 |
| A0c | S1b5 (3/5) | 0.434 / 0.429 | 0.382 / 0.378 | 0.694 / 0.698 | 0.437 / 0.417 | 0.224 / 0.223 |
| A0c | bus1 (4/5) | 0.425 / 0.422 | 0.373 / 0.361 | 0.693 / 0.695 | 0.414 / 0.406 | 0.219 / 0.227 |
| P4noKD | pooled (0/5) | 0.547 / 0.546 | 0.543 / 0.543 | 0.735 / 0.738 | 0.550 / 0.551 | 0.360 / 0.354 |
| P4noKD | S1b5 (3/5) | 0.529 / 0.534 | 0.487 / 0.502 | 0.735 / 0.735 | 0.531 / 0.531 | 0.365 / 0.367 |
| P4noKD | bus1 (4/5) | 0.520 / 0.523 | 0.464 / 0.474 | 0.733 / 0.734 | 0.510 / 0.524 | 0.372 / 0.362 |
| FedProxc | pooled (0/5) | 0.431 / 0.444 | 0.409 / 0.422 | 0.694 / 0.696 | 0.422 / 0.425 | 0.198 / 0.232 |
| FedProxc | S1b5 (3/5) | 0.428 / 0.417 | 0.367 / 0.338 | 0.694 / 0.694 | 0.427 / 0.432 | 0.226 / 0.204 |

Gap = AP50_bus(pooled) − AP50_bus(scenario), same method and seed (42 / 123):

| Method | gap at 3/5 | gap at 4/5 |
|---|---|---|
| A0c | +0.031 / +0.036 | +0.041 / +0.053 |
| P4noKD | +0.056 / +0.040 | +0.079 / +0.069 |
| FedProxc | +0.043 / +0.085 | not run |

- **R1, 2 of 2 seeds so far:** A0c gap 4/5 > 3/5 (+0.009, +0.017). The damage grows with the number of clients
  missing bus, but stays small (≤ 0.053 AP50, ≤ 13 % of the pooled bus AP).
- **R2 not met, 2 of 2 seeds:** P4noKD gap at 4/5 is larger than A0c's (+0.079 vs +0.041; +0.069 vs +0.053). In absolute
  terms P4noKD bus at 4/5 is still +0.091 / +0.113 above A0c at 4/5.
- **R3 not met, 2 of 2 seeds:** FedProxc gap at 3/5 is larger than A0c's (+0.043 vs +0.031; +0.085 vs +0.036).
  On pooled FedProxc ≈ A0c (mAP50 −0.008 / +0.002, bus −0.004 / +0.008), so its S1b5 deficit appears only when bus
  is missing on some clients.
- Non-target, 3/5 → 4/5: car and motorcycle move by ≤ 0.011 for both methods; truck drops for both
  (A0c −0.023 / −0.011, P4noKD −0.021 / −0.007). Truck is absent on C3 in both scenarios, but its other clients
  change in the re-deal (C4 holds 2,462 truck boxes), so this is reported, not interpreted.
- Seed 2024 (s18c) will be added when the session ends; R1–R3 require 3 of 3 seeds, so R1 stays provisional.

### Reading
Observation, not a causal claim: on the same images and box totals, moving bus from 2 clients to 1 lowers FedAvg's
bus AP by a further 0.01–0.02, so H1 holds as a small, severity-dependent effect. Neither the local recipe
(P4noKD) nor the proximal term (FedProx) narrows it; both widen it, P4noKD while still being better in absolute terms.
