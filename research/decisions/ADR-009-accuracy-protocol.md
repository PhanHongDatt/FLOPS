# ADR-009: Evaluation threshold, centralized schedule, batch size — accuracy protocol

**Date:** 2026-10-03
**Status:** accepted for items 1–4 (requested by the author: "optimise for high accuracy");
items 5–7 proposed, awaiting decision
**Scope:** evaluation protocol / training configuration (all arms and baselines alike)

## Context

Kaggle session 1 (ADR-008 fixes confirmed) exposed how the current settings limit or misreport
accuracy. Constraint from CLAUDE.md §20: every change below applies to **every** method and
baseline identically; none is tuned for the proposed method.

## Decision

1. **AP at the Ultralytics val threshold** `[YOLO-DOC]` — `evaluation.conf: 0.001` (was 0.25).
   `cfg/default.yaml`: *"confidence threshold; defaults: predict=0.25, val=0.001"*. At 0.25 the
   precision-recall curve is truncated, so AP is understated — most for low-confidence / rare
   classes such as **bus**, the primary target. With `conf == 0.001`, Ultralytics fills the
   confusion matrix at 0.25 by itself (`utils/metrics.py` `ConfusionMatrix.process_batch`:
   `conf = 0.25 if conf in {None, 0.001}`), so FP/FN keep the 0.25 operating point in the same
   single validation pass. New key `evaluation.operating_conf: 0.25` is used for per-class
   confidence statistics (F2/F3).
2. **Centralized G2 uses Ultralytics' schedule defaults** `[YOLO-DOC]` —
   `warmup_epochs 3.0`, `close_mosaic 10` (`cfg/default.yaml`). The 0 / 0 values were introduced
   for one-epoch FL rounds (warm-up would span the whole round) and are kept there.
3. **G2 seed is forwarded** `[ENGINEERING]` — `train_centralized.py` never passed `--seed` to
   training, so every G2 seed ran with Ultralytics' seed 0 (CLAUDE.md §14 violation).
4. **Feasibility batch 16** `[ENGINEERING]` — equals `base_config.yaml` (thesis value); the
   feasibility override of 8 had no recorded reason. `nbs 64` keeps the effective batch (and
   therefore the optimisation) the same; larger batches only raise throughput, i.e. more
   training per GPU-hour. One YOLOv8n client per T4 (`client_num_gpus: 1.0`) fits easily.

## Proposed, not implemented (need the author's decision)

5. **Transfer COCO's class rows** `[THESIS-HYPOTHESIS]` — COCO already contains car (2),
   motorcycle (3), bus (5), truck (7). Today the whole cv3 branch is re-initialised at nc=4
   (ADR-008), discarding that knowledge; keeping an 80-class head and remapping labels would
   start every client from a head that already detects the four classes — likely the largest
   single accuracy gain. It changes the class-head shape (80 rows, 80-channel cv3), hence F1,
   mask, class-aware aggregation, F2/F3 code, and the meaning of "missing" knowledge (a
   pre-trained bus row exists before FL). Requires its own ADR.
6. **More training per run** — FL feasibility = 5 rounds × 1 local epoch; main = 10 × 1.
   Accuracy is compute-bound; options: `local_epochs: 2`, more rounds. Experimental-design
   change (thesis Chapter 3) → needs approval and a compute estimate (session 2 measures it).
7. **Pre-resized dataset** `[ENGINEERING]` — images are 1280×720 and resized to 640 at load;
   with mosaic each sample decodes 4 images on 4 vCPUs. Storing a 640-px copy once as a Kaggle
   dataset could raise throughput without changing what the model sees at imgsz 640. Measure
   data-loading vs GPU time in session 2 before deciding.

## Consequences

- Reported AP values change meaning (now standard COCO-style); results produced before this
  ADR are not comparable — none exist yet (smoke only).
- Server eval at conf 0.001 keeps more boxes through NMS; expect a slower eval than the
  ~2 min/round measured in session 1 (record in session 2).
- F2/F3 decision rules use AP50 from this protocol.

## Related ADRs
ADR-001 · ADR-002-A2 · ADR-007 · ADR-008

## Addendum 2026-10-06 — three operating points in one validation pass

Checked against ultralytics 8.3.253 source after a reviewer noted that TP/FP/FN do not reproduce
the exported precision/recall:

- AP50 / mAP50: threshold-free (PR curve over predictions with conf ≥ 0.001), IoU 0.5 — **the metric
  every conclusion uses**.
- TP / FP / FN per class: `ConfusionMatrix.process_batch`, conf 0.25 (the val conf 0.001 is mapped to
  0.25), IoU 0.45, one-to-one matching.
- precision / recall per class: `ap_per_class`, at the single confidence that maximises the smoothed
  mean F1 over all classes, IoU 0.5.

P/R recomputed from TP/FP/FN therefore differ from the exported P/R by design (B2 v2, bus, round 30:
0.316/0.240 vs 0.467/0.276). Consistency checks that do hold on every evaluation: TP + FN = val GT
(car 102,506 · bus 1,597 · truck 4,245 · motorcycle 452) and mAP50 = mean of the four AP50.
Compare FN across classes only as FN/GT. Documented in docs/PHAN_TICH_NON_IID.md §2.6.
