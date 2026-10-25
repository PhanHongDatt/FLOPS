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
