# ADR-025: 9-client combined run extended to 40 rounds

- Status: accepted (author request, 2026-10-25)
- Related: ADR-023 (combined protocol), ADR-024 (9-client scenario, s21c result)

## Context
s21c (P4t1 @ S9-bus8, 960 px, 20 rounds) was still improving between rounds 10 and 20 (bus 0.507 -> 0.546,
motorcycle 0.483 -> 0.522). The author asked for the same scenario with about 40 rounds, reporting rounds 30 and 40
on the four classes.

## Decision
- Same partition, method (P4t1), image size, batch, seed 42 and evaluator as s21c; only `num_rounds: 40`
  (`configs/experiments/convergence_9c_y11s_40r_960.yaml`). The cosine LR now spans 40 rounds, so rounds 10/20 of
  this run are **not** the s21c rounds 10/20 and the two runs are separate experiments `[ENGINEERING]`.
- Global evaluation at rounds 10, 20, 30, 40 (curve); the reported reading is rounds 30 and 40 and their mean,
  as requested. This reading was fixed before the run.
- Cost: ~29 min/round measured in s21c -> ~19.5 h, so two sessions on goshihayashi: s22a_g (part 1, cut at 9.6 h)
  and s22b_g (part 2, mounts part 1's output, `restore_run` + `--resume` continue from the last checkpoint; round
  numbers, LR and seeds are absolute, `server.py` `round_offset`).
- Descriptive only: still no 9-client FedAvg reference (ADR-024).

## Result (s22a_g + s22b_g, completed 2026-10-27)
Part 2 restored part 1 and resumed from round 22; all 40 rounds aggregated 9 of 9 clients with 0 failures.
AP50 on the 4 classes (same evaluator as s21c):

| Round | mAP50 | car | bus | truck | motorcycle |
|---|---:|---:|---:|---:|---:|
| 10 | 0.5756 | 0.7884 | 0.4972 | 0.5535 | 0.4632 |
| 20 | 0.6086 | 0.8015 | 0.5274 | 0.5771 | 0.5285 |
| **30** | **0.6177** | **0.8071** | **0.5430** | **0.5887** | **0.5321** |
| **40** | **0.6195** | **0.8091** | **0.5502** | **0.5937** | **0.5250** |
| mean 30/40 | 0.6186 | 0.8081 | 0.5466 | 0.5912 | 0.5285 |

Observation: gains slow down after round 20 (mAP50 +0.033 from 10 to 20, +0.009 from 20 to 30, +0.002 from 30 to 40).
Bus, car and truck still rise slightly at round 40; motorcycle peaks at round 30. The mean of rounds 30 and 40 is
close to the s21c round-20 value (mAP50 0.6133, bus 0.5456). This is one seed with no 9-client FedAvg
reference, so it is descriptive only.
