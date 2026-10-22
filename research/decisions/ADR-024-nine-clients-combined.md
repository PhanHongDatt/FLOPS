# ADR-024: 9 clients, bus absent on 8/9, truck on 3/9, with the ADR-023 combined method

**Date:** 2026-10-23
**Status:** accepted for one feasibility run, seed 42 (author: "đẩy lên queue kịch bản 9 client trong đó có 8 client
thiếu lớp bus và mỗi client có 3000 ảnh, lớp truck thiếu 3/9 client, đồng thời sử dụng cách kết hợp cả 3 cách")
**Scope:** `configs/partition/s9_bus8_3k_moto_seed42.yaml`, `configs/experiments/convergence_9c_y11s_20r_960.yaml`,
`src/utils/kaggle_resume.py`, notebook 05 (`s9bus8`, run restore), sessions s20a / s20b_g

## Decision
- **Partition** `[ENGINEERING]` (same partitioner, seed 42, 3,000 images per client): C0–C7 hold no bus,
  C5–C7 also no truck, C8 holds all four classes; motorcycle quota 260 images per client (all clients may hold it).
  Generated locally from `det_train.json` (2026-10-23):

  | | car | bus | truck | motorcycle |
  |---|---|---|---|---|
  | total (27,000 images) | 278,715 | 3,908 | 11,456 | 2,789 |
  | C8 (the only bus client) | 30,564 | 3,908 | 1,684 | 339 |
  | C5–C7 | 30,667–31,018 | 0 | 0 | 292–306 |

  Side effects, reported rather than hidden: C8 is nearly all bus images (one bus client capped at 3,000 images);
  213 motorcycle boxes are lost because their images also hold bus and C8 is full; truck is uneven over
  C0–C4 (1,297–3,249) because truck images cannot go to C5–C7.
- **Method:** P4t1 (KD + cosine round LR + RFS t = 1.0) at 960 px for training and evaluation — the ADR-023
  combined arm. 20 rounds, 1 local epoch, eval at rounds 10 and 20, same val set (9,905 images).
- **Two sessions** `[ENGINEERING]`: 27,000 images are 1.8× the 5-client work, ≈ 41 min per round
  `[NEEDS-VERIFICATION]`, ≈ 14 h, above Kaggle's 12 h. Part 1 (s20a) is cut by the arm timeout at ~10.3 h
  after a round checkpoint; part 2 (s20b_g, goshihayashi) mounts part 1's output (uploaded as the private dataset goshihayashi/flops-s20a-output), `restore_run` copies the run directory back
  and `--resume` continues from the latest checkpoint. The cosine LR and the RFS draw depend on the absolute
  round, so the split does not change the schedule. Cost ≈ 10.5 GPU-h on phdatt (part 1) and ≈ 4 GPU-h on goshihayashi (part 2).

## What the run can and cannot show
- No 9-client reference exists (no A0c, no pooled 9-client re-split), so this run cannot measure a gap or a
  method gain. It is compared with the 5-client results only descriptively: more clients and more images
  change at the same time as severity (8/9 vs 3/5 or 4/5).
- Single seed; the ADR-023 risk (960 px + KD memory on a T4) also applies here.
