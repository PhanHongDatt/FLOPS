# Verification of docs/Solution.md §1 (V1–V8) — 2026-11-05

Checked in code against the pinned versions (ultralytics 8.3.253, flwr 1.21.0, `.venv`). No GPU run.

| # | Finding | Evidence |
|---|---|---|
| V1 | The not-true distillation of P4t1 (ADR-018) acts on **every** (anchor, class) entry whose TAL target is exactly 0, i.e. all background anchors and every non-assigned class of foreground anchors. Teacher = the initial COCO YOLO11s (`teacher_from_init`: `run_dir/teacher_init.npz`, from `coco:yolo11s.pt`), frozen, `eval()`. It scores **the same augmented training batch** (`batch["img"]`) so it runs at the training resolution, 960 px in the s5moto runs. | `src/preservation/ntd_loss.py` `_NTDBCE.forward` (`not_true = target == 0`), `_TeacherCriterion.__call__`; `scripts/run_fl_experiment.py` preset `P4t1`, `teacher_from_init` block |
| V2 | `v8DetectionLoss.__call__`: `pred_scores` `[B, A, nc]` (logits, permuted), assigner returns `target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx`; the loss discards `target_labels` (`_`). `target_scores` `[B, A, nc]` = `one_hot(label) * normalised align metric`; `fg_mask` `[B, A]` bool. `target_scores_sum = max(target_scores.sum(), 1)` is computed **before** `self.bce(pred_scores, target_scores)`, and the box/DFL losses receive the unmodified `target_scores`. Consequence: a `criterion.bce` wrapper that rewrites the targets meets the spec's "sum before the edit" rule automatically and leaves box/DFL untouched; label = argmax of the target row, s[a] = its maximum. | `.venv/Lib/site-packages/ultralytics/utils/loss.py` lines 243–302 |
| V3 | `Detect.forward` in eval mode returns `(y, x)` with `x` the list of raw per-level maps (`y` only when `export`). The existing teacher path uses `out[1]` and never calls `teacher.train()`. | `ultralytics/nn/modules/head.py` lines 114–124; `ntd_loss.py` `_TeacherCriterion` |
| V4 | Teacher = `copy.deepcopy(student)` with the COCO weights loaded, so architecture, strides (8/16/32) and anchor order are identical; `head_scores` uses the same concat/split/permute as `v8DetectionLoss`, and a shape mismatch raises. | `ntd_loss.py` `head_scores`, `on_train_start`; `tests/test_ntd_loss.py::test_head_scores_layout_matches_v8_loss` |
| V5 | `s1b5_bus_3k_allmoto_seed42` (s5moto): 5 × 3,000 images; bus 0 on C0–C2, truck 0 on C3, C4 has every class. Motorcycle quota 460 images per client stops at 457/457/457/457/456, so **motorcycle is on every client** (3,002 boxes in total). Totals: car 155,477, bus 2,657, truck 6,450, motorcycle 3,002. `missing_classes`: C0–C2 `[bus]`, C3 `[truck]`, C4 `[]`. Motorcycle work (M1–M3) is therefore a rare/small-object improvement, not a missing-class method. | `configs/partition/s1b5_bus_3k_allmoto_seed42.yaml` header (counts measured from det_train.json, 2026-10-23); manifest built on Kaggle |
| V6 | The server sends a per-round fit config with `server_round` (`_advance_round`); the client builds its Ultralytics overrides in `_train_kwargs` (`extra_overrides` reaches `model.train`). A round-dependent `mosaic: 0.0` can be set on the client from `server_round` and `train_config` alone, with no new server field. `close_mosaic` is per epoch; with 1 local epoch per round it has no effect, as the spec says. | `src/federated/client.py` `_advance_round`, `_train_kwargs`; `src/model/yolo_wrapper.py` `train_one_round` |
| V7 | The evaluator calls `model.val` with no `classes` filter: NMS is class-aware over all 80 COCO classes; the val labels (COCO-id view) hold only the 4 target classes, so predictions of the other 76 classes match nothing and only enter their own (unreported) classes. mAP50 is recomputed as the mean of the 4 target AP50. Caveat: the shared `max_det=300` cap counts the other-class detections too. | `yolo_wrapper.py` `evaluate` |
| V8 | At 960 px on s5moto only **P4t1, seed 42, bus missing 3/5** exists (S21b, registry `EXP-2026-10-23-S21-P4t1-s5moto-y11s960-seed42`). Missing: FedAvg (A0c) in every scenario; P4t1 at seeds 123/2024; P4t1 and FedAvg on the ideal and bus-missing-4/5 scenarios. Those two scenarios also have no all-motorcycle partition yet (`s1b5_pooled_3k_moto` and `s1b5_bus1_3k_moto` re-deal the 300-quota s5 partition). | registry.yaml; `configs/partition/` |

## Correction to the spec
§3.1 says about 74 % of the extra missed buses come from class confusion. The recorded D1 result (ADR-020, P4noKD
round 20, 640 px) is: S1b5 − pooled excess = confused car + truck **+164**, missed **+140**, low score −34, i.e.
confusion ≈ 164 / 270 ≈ **61 %** of the net extra FN, and ADR-020 rates the classifier-confusion hypothesis as
only weakly supported (almost as many buses get no detection at all). ADR-028 uses the recorded numbers.

## Spec points that need a decision (recorded in ADR-028)
1. **B1_replace vs the "same loss as the baseline" rule.** §3.2 says a client with no absent class trains with
   exactly the baseline loss, but the baseline (P4t1) includes the not-true KD, and B1_replace switches it off on
   every client. Implemented as written: `P4t1AK` (replace) has no not-true KD anywhere, `P4t1AKadd` keeps it.
   Only `P4t1AKadd` satisfies the §3.2 rule for C4.
2. **Truck is also absent (C3).** With the spec's `confusable` map, B1 acts on C3 for truck (on bus/car anchors)
   as well as on C0–C2 for bus.
3. **Arm name.** "B1" already names the ADR-015 server-teacher arm; the spec's B1 is registered as `P4t1AK*`.
