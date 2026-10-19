# ADR-021: Reference baselines on the COCO-head setting (A1c, FedProxc, IID check)

**Date:** 2026-10-17
**Status:** accepted for feasibility runs (author: "tiến hành xây dựng các kịch bản baseline sau đó tiến hành chạy để so sánh")
**Scope:** presets `A1c`, `FedProxc`; `class_rows` / `num_rows` in `ClassAwareAggregation`; sessions s17a–s17e

## Context
Phase 2 (registry `EXP-2026-10-14-S16-*`): P4noKD (A0c + cosine round LR + RFS t=0.5) beats A0c (FedAvg) in
3/3 seeds on S1b5-3k, YOLO11s, 20 rounds. The constitution (sections 11, 22) does not allow calling it promising
until the gain is shown against the reference baselines on the same setting and IID behaviour is checked.
Every earlier baseline (A1, FedProx, ...) ran on the 4-class head with YOLOv8n, so it is not comparable.

## Decision
Same partition, model (YOLO11s, COCO 80-class head kept, ADR-018), rounds (20), local epochs, batch, image size,
evaluator and rule (mean of rounds 10 and 20) as A0c / P4noKD; seeds 42, 123, 2024.

| Arm | Server | Client | Answers |
|---|---|---|---|
| A1c | ClassCountFedAvg | none | does class-count weighting explain the gain? (section 22.6) |
| FedProxc | FedAvg | proximal term, mu = 0.01 (ADR-010) | standard non-IID baseline |
| A0c@s5pooled, P4noKD@s5pooled | FedAvg | none / P4noKD | IID check (section 22.4); seed 42 exists (s13c_g) |

- `[LITERATURE]` FedProx: Li et al., MLSys 2020 (ADR-010). Class-count weighting: the A1 control of ADR-007.
- `[ENGINEERING]` COCO head: the class conv has 80 rows. Class-count weighting acts on the target rows only
  (car 2, motorcycle 3, bus 5, truck 7); the other 76 rows get plain FedAvg, like every shared parameter.
- No tuning: mu and the RFS / LR settings stay at their earlier values.

### Not run, with reason (section 11 asks for this to be explicit)
- **SCAFFOLD:** disabled in the runner — the client applies no control-variate correction (see `_DISABLED_ALGORITHMS`).
  A faithful implementation needs its own ADR.
- **FedNova:** with 5 clients of 3,000 images and 1 local epoch every client does the same number of local steps;
  FedNova's normalised averaging then equals FedAvg (Wang et al., NeurIPS 2020), i.e. it would repeat A0c.
- **Centralized YOLO11s:** not in this batch; a later upper-bound reference if the author wants it.

## Pre-registered reading (fixed before any s17 run)
Metric = mean of rounds 10 and 20, AP50 bus and mAP50, per seed.
1. P4noKD beats A1c in all 3 seeds on both metrics → class-count weighting does not explain the gain.
2. P4noKD beats FedProxc in all 3 seeds on both metrics → the gain is not matched by the standard non-IID baseline.
3. IID: mean over 3 seeds of P4noKD@s5pooled mAP50 ≥ A0c@s5pooled mAP50 − 0.01 → IID not materially degraded.
Report every class for every arm; a failure of any rule is recorded as a negative result.

## Cost
5 sessions × 2 arms, ~2.1–2.5 h per arm + ~0.3 h setup → ~25 GPU-h on T4 ×2. goshihayashi s17a_g, s17b_g, s17e_g
(max 2 concurrent, s17e_g after one finishes); phdatt s17c, s17d.

## Results (2026-10-19; s17a–s17d, s17e_g still running)
All 8 arms completed. A1c trace: `class_count_weighted` on the 4 COCO rows of every level, every round; FedProxc ran
FedProxClient with mu = 0.01. Mean of rounds 10/20, S1b5-3k, mean ± std over seeds 42/123/2024:

| Arm | mAP50 | AP50 bus | car | truck | motorcycle | FN bus |
|---|---:|---:|---:|---:|---:|---:|
| A0c (FedAvg) | 0.434 ± 0.005 | 0.387 ± 0.011 | 0.697 | 0.429 | 0.223 | 1395 |
| A1c | 0.428 ± 0.009 | 0.358 ± 0.007 | 0.695 | 0.425 | 0.233 | 1311 |
| FedProxc | 0.424 ± 0.006 | 0.361 ± 0.021 | 0.693 | 0.430 | 0.213 | 1425 |
| P4noKD | 0.531 ± 0.002 | 0.492 ± 0.009 | 0.735 | 0.531 | 0.366 | 1224 |

- **Rule 1 met:** P4noKD − A1c, per seed: mAP50 +0.111 / +0.105 / +0.094, bus +0.136 / +0.137 / +0.127.
- **Rule 2 met:** P4noKD − FedProxc, per seed: mAP50 +0.101 / +0.117 / +0.102, bus +0.121 / +0.164 / +0.107.
  P4noKD is ahead on every class in every seed against both.
- Both baselines are slightly **below** FedAvg on bus in every seed (A1c −0.031 / −0.013 / −0.042;
  FedProxc −0.016 / −0.040 / −0.021): neither class-count weighting nor the proximal term helps here.
- **Rule 3 (IID), 2 of 3 seeds:** pooled P4noKD − A0c mAP50 +0.108 (42), +0.104 (123); bus +0.130 / +0.128.
  Pending seed 2024 (s17e_g).

### Reading
Observation, not a causal claim: P4noKD's gain on the IID re-split (~+0.106 mAP50, +0.129 bus) is as large as
on S1b5 (+0.097, +0.105). The missing-class gap (pooled − S1b5, AP50 bus) is 0.031 / 0.036 for A0c and
0.056 / 0.040 for P4noKD (seeds 42 / 123): it is **not reduced**. P4noKD is a better federated training recipe
for this detector in general, not a missing-class remedy — consistent with ADR-020 (P5f) and section 22
("shared representations dominate degradation").
