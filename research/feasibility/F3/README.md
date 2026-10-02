# F3 — Matched Missing-Class Local Training

**Gate:** G5 (F3) · evidence for G4
**Status:** `code_ready` (2026-10-02) — tooling + CPU smoke done; no real run yet
**Prerequisites:** F1 runtime (done locally 2026-10-02, re-run on pinned Kaggle stack) · G2 checkpoint · S1b + S1-Control-Matched partitions

## Objective

Compare the local update of a client WITHOUT the target class against its matched
control, both trained from the same global checkpoint, with identical local-training
settings to FL clients (`src.utils.config.build_local_train_config`).

## Pair

| Side | Partition | Client | Target (bus) boxes |
|---|---|---|---|
| missing | `s1b_bus_seed42` | C0 | 0 (checked at run start) |
| control | `s1_control_matched_seed42` | C0 | > 0 (checked at run start) |

Same image count by construction; non-target box vector matched (see `match_report.yaml`,
copied into every run). `matched: false` → results are observations only (CLAUDE.md §10).
Within a seed both sides use the **same** training seed on purpose, so the data
difference is the only intended difference; seeds vary across repetitions.

## Start checkpoint `[YOLO-DOC]`

Must already detect the target class: G2 centralized `best.pt`, or an FL global after
warm-up rounds. **Raw `yolov8n.pt` is invalid**: at nc=4 all 36 cv3 keys are freshly
initialised (shape mismatch with COCO's 80 classes; ultralytics 8.3.253
`DetectionModel.load` → `intersect_dicts`). `summary.yaml: target_known_before` records this.

## Run

```bash
python scripts/run_f3.py --f3-config configs/feasibility/f3_matched_bus.yaml \
  --partition-dir <generate_partition --output-dir> --global-weights <G2 best.pt> \
  --eval-data-yaml <global data.yaml> --output-dir research/feasibility/F3/runs
```
Kaggle: `notebooks/03_missing_class_G4.py` Cell 6.

## Artifacts (per run)

`config.yaml` · `environment.json` · `match_report.yaml` · `metrics.csv`
(seed × side × metric: before/after/delta) · `update_analysis_seed<S>.yaml` · `summary.yaml`

Recorded per seed:
- **Module Δθ** (learned params only): backbone · neck · detect_cls · detect_reg · detect_dfl — L2, cosine missing-vs-control
- **`class_head`**: the 6 final-cv3 keys (780 elements)
- **Class rows**: per class L2, signed mean Δweight, signed mean Δbias, cosine vs control
- **`bn_stats`**: BN running mean/var, reported separately (data statistics, not Δθ)
- **Prediction**: AP50/AP/P/R/FP/FN per class before vs after

## Mechanism being tested `[THESIS-HYPOTHESIS]` grounded in `[YOLO-DOC]`

`v8DetectionLoss` uses `BCEWithLogitsLoss(reduction="none")` over `[B, anchors, nc]`
(ultralytics 8.3.253 `utils/loss.py`). With zero target boxes, every anchor's target
score for that class is 0, so the loss only pushes that class's logits down.
**Falsifiable prediction:** Δbias of the target row is more negative for `missing`
than for `control`, while non-target rows behave alike.

## Pre-registered decision rule — **DRAFT, needs author approval before the first real run**

Over the 3 seeds (sample std, ddof=1), with `gap = missing − control`:

| ID | Evidence | Pass condition | Field in `summary.yaml` |
|---|---|---|---|
| P1 | prediction | `mean(ΔAP50_bus gap) < −max(2·std, τ_AP)` | `prediction.AP50_bus.delta_gap` |
| P2 | parameter, target row | `mean(Δbias gap) < −2·std` **and** `mean(row-L2 ratio missing) > mean(row-L2 ratio control)` | `parameter.bias_delta_gap`, `parameter.target_row_l2_ratio_*` |
| P3 | specificity (report) | non-target `ΔAP50` gaps within `±max(2·std, τ_AP)` | `prediction.AP50_{car,truck,motorcycle}.delta_gap` |

- `τ_AP` = **[TO DECLARE]** absolute AP50 threshold (proposal: 0.01). Must be fixed before any real F3 output is inspected.
- **G4 parameter+prediction evidence** = P1 ∧ P2 (CLAUDE.md §8: L2 alone or AP alone is insufficient).
- P3 failing does not void P1/P2 but must be reported (shared-representation damage).
- `target_known_before: false` → run is **inconclusive**, not negative.
- Gate A/B/C is decided in ADR-003 together with F2, not from F3 alone.

## Smoke log

- 2026-10-02, CPU, synthetic 64-px images, 2 seeds: pipeline end-to-end OK. Smoke
  found two defects, both fixed: (1) nc=80/nc=4 crash in `build_model`, (2) client
  weights rounded to fp16 by Ultralytics' checkpoint reload — before the fix the
  class-row Δθ was identical across sides and seeds (pure rounding). Smoke numbers are
  NOT evidence (random images, untrained head).
