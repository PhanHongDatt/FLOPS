# ADR-027: centralized ceiling with more bus and truck images

- Status: accepted (author request, 2026-11-03)
- Related: ADR-026 (ceiling on s5moto: bus 0.589, truck 0.569, motorcycle 0.459, all below the 0.70 target)

## Context
ADR-026 showed that the FL method is within 0.006 mAP50 of its centralized ceiling, and that the ceiling itself
is below AP50 0.70 on the rare classes. The author asked to try more data. Motorcycle has no images left
(s5moto already holds all 3,002 BDD100K train boxes); bus and truck do, because the 5 x 3,000 images are a
subset of the ~70,000 train images.

## Decision
- New partition `s1b5_bus_3k_rare_seed42`: same 5 x 3,000 images, seed, missing map (bus 0 on C0-C2, truck 0
  on C3) and motorcycle quota (460, every image) as s5moto; class quotas raised to bus 1,200 images per
  eligible client (C3, C4) and truck 900 per eligible client (C0, C1, C2, C4). Total image count and training
  cost are unchanged; only the share of rare-class images rises `[ENGINEERING]`.
- First run only the centralized ceiling on it (`s1b5_central_3k_rare_seed42`, 1 client), P4t1, 960 px,
  20 rounds, seed 42, same evaluator and config as ADR-026 (session s23b_g). The federated run on the same
  partition comes only if this ceiling moves materially.
- Reading, fixed before the run: mean of rounds 10/20 AP50 per class vs the ADR-026 ceiling. A rare class
  reaching 0.70 means data is the lever; a rise well short of 0.70 means data alone does not get there and the
  remaining lever is model size (own ADR). Car may drop because car-only images are fewer; it is reported.
- The actual box counts come from the manifest (a quota BDD100K cannot fill is logged). n = 1, descriptive.

## Result (s23b_g, completed 2026-11-03)
Quotas filled: bus 3,635 boxes (+37 % vs s5moto), truck 7,884 (+22 %), motorcycle 3,002, car 154,949.
Mean of rounds 10/20, same evaluator:

| AP50 | mAP50 | car | bus | truck | motorcycle |
|---|---:|---:|---:|---:|---:|
| ADR-026 ceiling (s5moto, 1 client) | 0.6037 | 0.7983 | 0.5891 | 0.5689 | 0.4585 |
| ADR-027 ceiling (more bus/truck, 1 client) | 0.6125 | 0.7974 | 0.6027 | 0.5865 | 0.4635 |
| difference | +0.009 | -0.001 | +0.014 | +0.018 | +0.005 |

Bus recall 0.509 -> 0.538 (FN 801 -> 753), truck precision 0.627 -> 0.648; car unchanged.

Reading (rule fixed above): more data helps bus and truck a little but leaves both far below 0.70
(0.603 / 0.587). Data alone does not reach the target; the remaining lever is model size (own ADR).
n = 1, descriptive; a difference of ~0.01-0.02 is within the range seen between seeds elsewhere (e.g. A0c bus
0.378-0.399 over three seeds), so it is not established as a real effect.
