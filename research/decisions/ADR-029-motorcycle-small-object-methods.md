# ADR-029: motorcycle small-object methods M1-M3 (docs/Solution.md §4)

- Status: proposed (implemented and unit-tested 2026-11-05; smoke s24_g; main runs need the author's approval)
- Related: ADR-023 (s5moto: every motorcycle image, 960 px, RFS t = 1.0), ADR-026 (ceiling: motorcycle 0.459),
  ADR-028 (B1); verification note `research/notes/spec_verify_20261105.md` (V5, V6)

## Context
Motorcycle is on all 5 clients of s5moto (V5), so it is a rare and small class, not a missing one; any gain
here is reported apart from the missing-class results. All 3,002 BDD train boxes are already used
(ADR-023), and the centralized ceiling is 0.459 (ADR-026), so more data is not available; the remaining
levers are augmentation and label assignment.

## Decision (values fixed here, before any main run)
- **M1** `[ENGINEERING]`: Ultralytics `scale` 0.25 (default 0.5) every round; `mosaic` 0 in the last 5 of 20
  rounds. Ultralytics' `close_mosaic` counts local epochs (1 per round) and has no effect, so the client
  sets `mosaic` from the server round (`src/federated/client.py` `small_object_overrides`); the values used
  are reported per client and round as `aug_scale` / `aug_mosaic` fit metrics.
- **M2** `[LITERATURE]` Kisantal et al., arXiv:1902.07296; Ghiasi et al., CVPR 2021: each client pastes
  crops of its own motorcycle boxes (box + 2 px, feathered 3 px edge; BDD100K has no masks), p = 0.5 per
  image, 1-3 crops, original size, centre in the lower half, IoU ≤ 0.05 with every box, 20 tries
  (`src/data/moto_paste.py`). Pasted in `update_labels_info`, i.e. per image before mosaic (8.3.253);
  train set only; the crop bank never leaves the client. 20 annotated images per client in round 1 for a
  visual check.
- **M3** `[LITERATURE]` Wang et al., arXiv:2110.13389 (NWD): TAL overlap `0.5 · CIoU + 0.5 · NWD`
  (`src/model/nwd_assign.py`), for all classes as the spec writes it; C = mean sqrt(w·h) of the client's own
  motorcycle boxes at 960 px, computed each round from its labels and reported as `nwd_C`. Box/DFL losses
  unchanged. α = 1 reproduces the original assigner (unit test).
- Config `configs/methods/small_objects.yaml`, every block `enabled: false`; arms `P4t1M1`, `P4t1M2`,
  `P4t1M12`, `P4t1M123` = P4t1 + the named blocks.
- Literature entries for Kisantal 2019, Ghiasi 2021 and Wang 2021 still have to be added to
  `literature_registry.yaml` after checking the papers (spec §10); until then the tags above are the
  spec's citations, not yet verified here `[NEEDS-VERIFICATION]` for venue/details beyond the arXiv ids.

## Decision rule (spec §8, fixed before any run)
M1 + M2 pass only if, vs P4t1 on the same partitions and seeds (ideal and bus-missing 3/5; 42, 123, 2024),
mean of rounds 10/20: (1) AP50 motorcycle ≥ +0.010 in 3/3 seeds; (2) small-object motorcycle AP50 (GT area
< 32² px in the original image) higher on the 3-seed mean; (3) car, bus, truck AP50 each not lower by more
than 0.01. M3 passes on the same rule against P4t1 + M1 + M2. Rule (2) needs the size-split evaluation of
spec §6, which is not implemented yet; it must exist before the main runs.

## Smoke (s24_g)
2 rounds, 960 px, s5moto, seed 42, arms `P4t1AKadd` and `P4t1M123`
(`configs/experiments/smoke_5c_y11s_2r_960.yaml`). Correctness only: all 5 clients aggregate, the losses
are finite, `aug_*` / `paste_bank_size` / `nwd_C` are logged, the debug images look right. With 2 rounds
and K = 5, mosaic is off in both rounds (the switch itself is unit-tested). Not used for any decision.

## Result
Smoke s24_g (`P4t1M123`, 2 rounds, seed 42): completed, 5/5 clients each round, losses finite; scale 0.25 and
mosaic 0 confirmed in every client's args.yaml; bank 3,002 crops in total (every motorcycle box); `nwd_C`
46.5-53.6 px. Issues found before any main run:
1. `aug_*` fit metrics reach the server but no artifact stores them (logging gap).
2. Paste placement: "lower half" includes the ego-vehicle hood/dashboard and parked cars
   (`research/figures/debug_copy_paste/s24_paste_grid_C0_C3.jpg`); a placement bound below the hood line
   would need a new pre-registered value.
3. Descriptive only: round-2 motorcycle AP50 0.258 (round 0: 0.317), FP_motorcycle 528 vs 162 and FP_car
   2.1x the `P4t1AKadd` smoke arm. Mosaic was off in both rounds and M1-M3 share one arm, so nothing is
   attributed; a per-method run (M1, M2, M3 separately) is needed to locate the FP rise.
