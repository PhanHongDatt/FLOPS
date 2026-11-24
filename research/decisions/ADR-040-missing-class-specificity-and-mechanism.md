# ADR-040: missing-class specificity and mechanism, reading rules fixed before the next runs

**Date:** 2026-11-24
**Status:** accepted (rules fixed; runs planned)
**Scope:** feasibility-gate / evaluation (CLAUDE.md §6 H1, §11, §20, §22)

## Context

Evidence that already exists (seen before this ADR, so it is context, not a test):

- On YOLO11s, KD adds little once cosine round LR and client RFS are in: P4 − P4noKD = +0.003 mAP50, +0.008 AP50 bus
  (ADR-019 addendum, rule fixed at commit `736e76d` before the result commit `1ba0853`).
- On YOLO11s, P4noKD does not shrink the bus gap. The ideal − missing gap is +0.056 for P4noKD and +0.031 for FedAvg
  (S13, commit `0349d76`): the gain is general, not missing-class specific.
- On YOLO11m, the AP50 bus gap is 0.017 for FedAvg (3 seeds; per-seed +0.026 / +0.010 / +0.015) and 0.038 for
  P4t1 (2 seeds). The relative gaps are 3.4 % and 5.9 %.
- On YOLO11s 640 px (`s1b5_bus_3k_moto`, RFS t = 0.5, seed 42, r20), D1 on P4noKD (S14) splits the loss of 269 bus TP (missing vs ideal) into confused car + truck +164,
  missed +140, low score −34. Loss is mainly confusion and misses, not low scores.
- On YOLO11m, P4t1 r20 seed 42 (s30e dumps): bus recall at precision ≥ 0.7 drops by 0.049, and recall at conf 0.25
  drops by 0.099. The best-F1 threshold moves 0.375 → 0.237, so about half of the operating-point loss is a score
  shift and about half is a ranking loss.

The baselines so far use the default FedAvg recipe. Ours (P4t1) also differs from them in cosine LR and RFS, so
"Ours beats FedAvg" does not isolate the teacher or KD.

## Decision

### 1. Primary metric

**Relative AP50 bus gap** `g = (AP_ideal − AP_missing) / AP_ideal`, computed per seed with the same seed in both
scenarios (`s1b5_pooled_3k_allmoto_seed42` vs `s1b5_bus_3k_allmoto_seed42`).

- Rounds: mean of rounds 10/20 (ADR-033 rule, unchanged).
- Round 20 alone is reported next to it in the main table, not in an appendix.

### 2. Comparisons

| Question | Method | Reference |
|---|---|---|
| Is the recipe missing-class specific? | FedAvg + cosine + RFS t = 1.0 (P4noKD, YOLO11m) | FedAvg (A0c) |
| Does KD add anything? | P4t1 | P4noKD |
| Does the frozen COCO teacher beat the received global teacher? | P4t1 | FedNTD + cosine + RFS |
| Is the default baseline weak? | FedAvg, lower lr0 | FedAvg (A0c) |

### 3. Reading zones (fixed now)

Let `d = mean(g_ref) − mean(g_method)` over seeds. Let `t95 = 2.776 × sqrt(s_ref²/n_ref + s_method²/n_method)`
(t with 4 df; Welch form).

| Zone | Condition | Wording allowed |
|---|---|---|
| Specific | `d > t95` **and** `g_method < g_ref` in every seed | "reduces the missing-class gap" |
| Contradicted | `mean(g_method) ≥ mean(g_ref)` | "not missing-class specific" |
| Inconclusive | otherwise | descriptive only; no claim either way |

- **Power limit, stated before the runs:** on `s5moto`, YOLO11m FedAvg has an absolute gap of 0.017 with a per-seed
  SD of about 0.008. With 3 seeds per arm, `t95` is about 0.018 absolute (about 3.5 % relative). A method that
  removes the whole gap only just reaches it.
- So on `s5moto`, "specific" is close to unreachable, and most outcomes will read "inconclusive" or "contradicted".
- **Replication scenario:** `s1b5_bus1` (bus on 1 of 5 clients; YOLO11s FedAvg gap 0.047, seed spread ≤ 0.004,
  ADR-022) is the replication scenario. Run it on YOLO11m only if the 4090 budget allows; until then, specificity
  claims are limited to the "contradicted" side.
- **Single-seed runs** (R1–R3 below) are read only for differences above 0.03 absolute (about 6 × the YOLO11s seed SD).
  Smaller differences are descriptive.

### 4. Mandatory secondary metrics (every run, both scenarios, rounds 10 and 20)

- **D1 bus table** (S14 form, conf 0.25, IoU 0.5): tp, confused_car, confused_truck, missed, low_score, fp. The
  missing − ideal excess is reported per category.
- **Truck FP split:** background, on a bus GT, on a car GT.
- **Bus recall at precision ≥ 0.7 and ≥ 0.8**, and the per-class best-F1 threshold, from the val prediction dumps.
- **Mean bus score** on bus GT boxes.

### 5. Mechanism reading (fixed now)

Let `L` be the bus TP lost from ideal to missing (D1, conf 0.25).

| Reading | Condition |
|---|---|
| Confusion and misses | confused_car + confused_truck + missed excess ≥ 2/3 of L |
| Score shift | low_score excess ≥ 1/2 of L, **or** the drop in recall at P ≥ 0.7 is < 1/3 of the drop in recall at conf 0.25 |
| Mixed | neither |

Read FedAvg YOLO11m first (3 seeds; a reading holds when 2 of 3 seeds agree). The YOLO11s P4noKD result (S14) is
reported as YOLO11s, not merged with it.

### 6. Framing switch (fixed now)

- If P4noKD (YOLO11m) is "contradicted" or "inconclusive", the thesis presents the recipe (COCO head + cosine LR + RFS
  ± KD) as a **general FL fine-tuning recipe**. H1 (the missing-class effect on FedAvg) is a characterised phenomenon.
- No sentence may say the recipe "solves" or "reduces" missing-class degradation.
- The "forgetting pretrained knowledge" framing may be used only with its own evidence. That means AP below round 0
  under the plain recipe (S38d: 0.459 < 0.486), and the matching numbers under the recipe, reported per class.

## Run order

| # | Run | Where | Estimate |
|---|---|---|---|
| D | Diagnostic: D1 + truck FP + PR dumps at r20 for A0c (3 seeds × 2 scenarios) and P4t1 (2 seeds × 2), and D1 + bus score for P4noKD YOLO11s (S12c, S13c; labelled YOLO11s) | Kaggle, one session | 1–2 h |
| R1 | P4noKD YOLO11m t = 1.0, both scenarios, seed 42 | 4090 | about 6–7 h each |
| R2 | FedNTD + cosine + RFS, missing, seed 42 | 4090 | about 10 h |
| R3 | FedAvg lr0 lower, missing, seed 42 | 4090 | about 5.5 h |
| R4 | P4t1 seed 2024, both scenarios (s38e/f) | 4090 | about 10 h each |

Deferred: s38g (P4t1 centralized, likely over the Kaggle session limit) and more FedNTD seeds. No new technique
(class-wise aggregation, rho loss) before R1–R3 are read.

## Evidence

- `[ENGINEERING]` zones and thresholds above are choices of this project, fixed before the runs. They are not a
  literature standard.
- `[LITERATURE]` FedNTD (Lee et al., NeurIPS 2022); RFS (Gupta et al., CVPR 2019).
- `[YOLO-DOC]` evaluator: Ultralytics 8.3.253 `val()`, conf 0.001, NMS IoU 0.7, multi-label, max_det 300 (ADR-034).

## Related ADRs

ADR-019 (P4noKD rule), ADR-022 (severity), ADR-026 (centralized), ADR-033 (main config), ADR-034 (evaluator),
ADR-037/038 (head-only fixes), ADR-039 (baselines).
