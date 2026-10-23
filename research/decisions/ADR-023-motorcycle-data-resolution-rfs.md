# ADR-023: Motorcycle — all motorcycle images, 960 px, KD + RFS t=1.0; one at a time, then combined

**Date:** 2026-10-23
**Status:** accepted for feasibility runs, seed 42 (author: "tiến hành thử nghiệm độc lập 3 phương pháp, sau đó
tiến hành thực hiện kết hợp lại")
**Scope:** `configs/partition/s1b5_bus_3k_allmoto_seed42.yaml`, `configs/experiments/convergence_5c_y11s_20r_960.yaml`,
arm `P4t1` (`scripts/run_fl_experiment.py`), notebook 05 (`s5moto`, `RUN_TAG`), sessions s19a_g / s19b / s19c_g

## Context
AP50 motorcycle stays low whatever the scenario (it is never a missing class, it is on every client):
A0c 0.218 / 0.223 / 0.223 and P4noKD 0.359 / 0.366 / 0.367 at 0/5, 3/5, 4/5 clients without bus.
It is a rare, small-object problem, not a missing-class one. Measured facts `[ĐO]`:
- BDD train holds 3,002 motorcycle boxes in 2,284 images; the 15,000-image S1b5 set already holds 1,963 (65 %)
  through the ADR-017 quota of 300 images per client. Using all of train would add only 1,039 boxes for
  4.6× the images (mostly car), so more images per round is not the lever.
- At 640 px a 1280×720 frame becomes 640×360: on the val set the median motorcycle side (√(w·h)) is 25.5 px,
  30 % are under 16 px and 60 % under 32 px (bus: 38.6 px, 17 %, 43 %).
- Ablation at seed 42 (AP50 motorcycle): A0c 0.223 → P3 (+KD) 0.276 → P3LR (+cosine LR) 0.342 → P4 (+RFS t=0.5)
  0.393 → P4noKD (−KD) 0.366. Dropping KD costs motorcycle 0.027, against 0.008 for bus.
- `[YOLO-DOC]` ultralytics 8.3.253 `data/augment.py` `CopyPaste` returns unchanged when the labels carry no
  segments, so copy-paste does nothing on BDD boxes; not used.

## Decision
Three ideas, each changing one thing against the reference **P4noKD @ S1b5, 640 px, seed 42**
(AP50 motorcycle 0.365, bus 0.487, mAP50 0.529; mean of rounds 10/20):

| Idea | Run | What changes | Session |
|---|---|---|---|
| 2 data | P4noKD @ `s5moto`, 640 | partition `s1b5_bus_3k_allmoto_seed42`: quota 460 → every motorcycle image | s19a_g |
| 4 method | P4t1 @ `s5`, 640 | KD back on and RFS threshold t 0.5 → 1.0 (P4 with t=1.0) | s19a_g |
| 3 resolution | P4noKD @ `s5`, 960 | `image_size` 960 for training and global evaluation | s19b_g |
| combined | P4t1 @ `s5moto`, 960 | all three ideas together | s19c |

- **Partition** `[ENGINEERING]`: same rule, seed, missing map and client size as S1b5; only the quota changes.
  Generated locally from `det_train.json` with the project partitioner (which reproduces the S1b5 manifest
  exactly): car 155,477, bus 2,657, truck 6,450, motorcycle 3,002 boxes (S1b5: 155,250 / 2,628 / 6,421 / 1,963);
  bus 0 on C0–C2, truck 0 on C3; motorcycle 581–625 per client. The bus supervision changes by +1 %.
- **Resolution and protocol** (`CLAUDE.md` §14): data and image size are protocol, not method, so the combined
  result is compared with A0c under the same protocol, not with the 640 px runs only. Global evaluation
  also runs at 960, so idea 3 measures training and evaluation resolution together.
- **P4t1** `[THESIS-HYPOTHESIS]`: Detectron2 RFS (`src/preservation/rfs.py`) with t = 1.0. On the all-motorcycle
  partition the motorcycle image frequency is ≈ 0.15, so its repeat factor is ≈ 2.6 (t = 0.5: ≈ 1.8).
  KD vs t is separated with the existing P4 run (KD, t = 0.5).
- **Run names**: 960 px runs carry `i960` after the arm (`P4noKDi960_feasibility_seed42_...`) so they never
  share a directory with the 640 px run of the same arm, seed and partition.

## Pre-registered reading (fixed before any s19 run)
- **Keep an idea for the combination** if, against the reference, ΔAP50 motorcycle ≥ +0.02 and neither
  bus nor mAP50 drops by more than 0.02. 0.02 is three times the largest seed spread of P4noKD motorcycle
  (std 0.007 at 4/5; 0.001 at 3/5). One seed only: "kept" means worth combining, not "proven".
- **Combined:** reported against the reference and against A0c under the same protocol (data + 960),
  for all four classes. If an idea is not kept, s19c_g is edited to leave it out before it is pushed
  and the edit is recorded here.
- Motorcycle has 452 val boxes in 334 images; any claim beyond feasibility needs seeds 123 and 2024
  (both the combined arm and its A0c baseline).
- Negative or mixed outcomes are recorded as they are.

## Amendment (2026-10-23, before any s19 run)
The author chose not to run the A0c baseline under the new protocol and to run the combined arm at the same time
as the single-idea arms ("thay vì chạy baseline mới thì hãy tiến hành chạy theo phương pháp đề xuất kết hợp với
các phương pháp để xem kết quả luôn"). Consequences, recorded rather than hidden:
- The combined arm holds **all three ideas** whatever the single-idea results; the keep rule above is now only a
  reading of which idea contributed, not a filter on the combination.
- The combined result is compared with P4noKD @ S1b5 640 (reference) and A0c @ S1b5 640. Because data and
  resolution also change, its gain over A0c mixes protocol and method (`CLAUDE.md` §14); it shows how far
  motorcycle can go, not that the method beats FedAvg under the same protocol. A0c @ `s5moto` 960 can be added
  later if that claim is needed.

## Cost (Tesla T4, estimates from measured 640 px times; 960 ≈ 2.25× compute `[NEEDS-VERIFICATION]`)
| Session | Arms | Estimate |
|---|---|---|
| s19a_g (goshihayashi) | P4noKD@s5moto 640, P4t1@s5 640 | ~2.3 h + ~3.4 h |
| s19b_g (goshihayashi) | P4noKD@s5 960 | ~5 h |
| s19c (phdatt) | P4t1@s5moto 960 | ~7.8 h |

≈ 18.5 GPU-h for seed 42. Risk: 960 px with the KD teacher may not fit batch 16 in 16 GB (no earlier run shows
the memory now, since the combined arm starts at the same time). A smaller batch would have to be used for every
arm compared at 960.

## Results (2026-10-23, seed 42)
**Valid (s19a_g, 5 of 5 clients every round)**, mean of rounds 10/20, against the reference P4noKD @ S1b5 640
(mAP50 0.529, bus 0.487, motorcycle 0.365):

| Run | mAP50 | bus | car | truck | motorcycle | Δ motorcycle | keep rule |
|---|---|---|---|---|---|---|---|
| idea 2: P4noKD @ s5moto 640 | 0.529 | 0.494 | 0.733 | 0.523 | 0.365 | +0.000 | not met |
| idea 4: P4t1 @ s5 640 | 0.545 | 0.501 | 0.732 | 0.526 | 0.423 | **+0.058** | **met** (bus +0.014, mAP50 +0.016) |

- Idea 2: 53 % more motorcycle boxes did not move motorcycle AP50 (0.3648 vs 0.3645). More motorcycle examples are
  not what limits it at 640 px.
- Idea 4 splits into its two parts with the existing P4 run (KD, t = 0.5, motorcycle 0.393): KD +0.028
  (P4 − P4noKD) and t = 1.0 +0.030 (P4t1 − P4). Both help; together they also lift bus and truck slightly.

**Invalid — failed-technical (s19b_g idea 3, s19c combined, s20a 9 clients):** CUDA out of memory at 960 px. Flower
aggregated only 0–2 of 5 clients (0–1 of 9 in s20a) in most rounds and did not stop, so these runs are not the
registered experiments; their round-10 numbers (bus 0.26–0.29) are not results. Cause: the Kaggle T4 x2 gives Ray
2 client actors, but `[YOLO-DOC]` Ultralytics 8.3.253 `select_device(0)` rewrites `CUDA_VISIBLE_DEVICES`, so both
actors trained on physical GPU 0 (each OOM message lists both actor pids on GPU 0) while GPU 1 idled; at 640 px two
fits fit in 15 GB, at 960 px they did not. Fix (2026-10-23): `resolve_device` keeps the GPU Ray assigned,
`release_training_memory` empties the cache after every local fit, and `_wrap_fail_on_client_failures` stops a run
at the first partial aggregation. Earlier 640 px runs are unaffected in content (two clients shared one GPU, every
client aggregated).
A scan of every downloaded log found one earlier partial round in a reported run: A1c seed 2024 (s17c), round 9
aggregated 4 of 5 clients (recorded in the registry; kept). Idea 3 and the combination remain untested.

**Rerun (2026-10-23, s21):** the three invalid runs are pushed again unchanged except for the fix: s21a_g (idea 3,
P4noKD @ s5 960), s21b_g (combined, P4t1 @ s5moto 960: amendment 1 keeps all three ideas, so the null result of
idea 2 does not remove it, which would be a post-hoc change) and s21c (ADR-024, 9 clients).
