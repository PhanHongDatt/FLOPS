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
| D | Diagnostic (round 20 only): D1 + truck FP + PR dumps for A0c (3 seeds × 2 scenarios) and P4t1 (2 seeds × 2), and D1 + bus score for P4noKD YOLO11s (S12c, S13c; labelled YOLO11s) | Kaggle, one session | 1–2 h |
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

## Addendum 2026-11-24 (before any s40d result is read)

Changes sections 3 and 5 after review. s40d has not been pushed when this is committed.

**A1. Primary scenario for specificity.** Bus on 1 of 5 clients (`s1b5_bus1`) replaces `s5moto` as the primary
scenario for any specificity claim; `s5moto` becomes secondary.
- Already seen, YOLO11s 640, AP50 bus ideal − bus1 per seed: FedAvg 0.041 / 0.053, P4noKD 0.079 / 0.069
  (ADR-022 data). The section 3 statement "seed spread ≤ 0.004" referred to AP levels; the per-seed gap SD is
  about 0.009.
- YOLO11m has no bus1 partition yet. One with every motorcycle image (as `s5moto`) has to be built and checked
  before runs on it.

**A2. Threshold by formula, per scenario.** `t95 = t(0.975, df) × sqrt(s_ref²/n_ref + s_m²/n_m)`.
- `s` is the observed SD of per-seed gaps in that scenario.
- `df` is Welch-Satterthwaite.
- With `n_m = 1`, use `s_m = s_ref`.
- The 0.018 of section 3 is this formula for `s5moto` with 3 + 3 seeds, not a fixed constant.

**A3. Two references.**

| Reference | What it is | Used to judge |
|---|---|---|
| R0 | FedAvg default (A0c) | recipe vs default FedAvg |
| R1 | FedAvg + cosine + RFS t = 1.0 (P4noKD) | KD, frozen teacher, and any claim about Ours beyond the recipe |

**A4. Zones renamed and made symmetric.** `d = mean(g_ref) − mean(g_m)`.

| Zone | Condition |
|---|---|
| Specific | `d > t95` and `g_m < g_ref` in every seed |
| Opposite | `−d > t95` and `g_m > g_ref` in every seed |
| Not supported | otherwise. This replaces "contradicted"; it only means no evidence either way. |

**A5. Mechanism definitions (section 5 made exact).**

D1 for target bus: strong = conf ≥ 0.25, candidate = conf ≥ 0.001, match = IoU ≥ 0.5 (`decompose_image`). The
categories are assigned in this order:

| Category | Rule |
|---|---|
| tp | a strong bus box matches the GT |
| confused_X | else, the highest-scoring strong box of another class X matches |
| low_score | else, a bus candidate below 0.25 matches |
| missed | none of the above |

- **NMS mode.** The primary mode is multi-label NMS (`--multi-label`), which matches the global evaluator (ADR-034).
  Top-1 NMS is reported for comparison with S14. Top-1 NMS hides a bus score that sits under a higher class on the
  same box, so it moves GTs from low_score to confused_X. S14 was top-1, and its categories are read with this caveat.
- **Excess.** `e_k = count_missing − count_ideal` per category, and `L = tp_ideal − tp_missing`. A category's share
  is `max(e_k, 0) / Σ_j max(e_j, 0)`, over the GT categories only. Negative excesses are reported but not netted.

Reading:

| Reading | Condition |
|---|---|
| Confusion + misses | share(confused_car + confused_truck + missed) ≥ 2/3 |
| Score shift | share(low_score) ≥ 1/2, **or** the bus recall at precision ≥ 0.7 drops by < 0.02 while the recall at conf 0.25 drops by ≥ 0.05 |
| Mixed | both readings hold, or neither |

**A6. Single-seed rule.** The 0.03 threshold comes from the YOLO11s FedAvg per-seed gap SD of about 0.009. The SD of
a difference of two single-seed gaps is about 0.009 × √2 ≈ 0.013; 0.03 is about 2.3 × that. It is a heuristic,
not a test.

**A7. Fixed partition.** Every seed uses the same partition, so seed SDs measure training noise only, not partition
variability.

**A8. s40d reporting.** Every s40d number is labelled round 20 (not the primary mean of rounds 10/20). Results go
into the registry under S40D with the file they come from.

## Result s40d (round 20, read 2026-11-25 with `scripts/adr040_read_s40d.py`)

Sanity check: top-1 D1 for P4noKD YOLO11s s5 reproduces S14 exactly (tp 565, car 264, truck 179, low 154,
missed 435, fp 131).

**Mechanism reading by the A5 rule (FedAvg YOLO11m, multi-label, primary):** **confusion + misses**, 3 of 3 seeds.
- The share of confused + missed is 0.80 / 0.84 / 1.00; the share of low_score is 0.20 / 0.16 / 0.00.
- The confusion is mainly bus → car (+251 / +306 / +238).
- The score-shift alternative fails narrowly: bus recall at precision ≥ 0.7 drops by 0.040 / 0.028 / 0.041, against
  the < 0.02 the rule needs.
- YOLO11s FedAvg and P4noKD (s5 and bus1, seed 42) read the same way (share 0.79–1.00).
- Top-1 NMS moves GTs from low_score to missed, not to confused, so the S14 reading stands.

**Descriptive, not part of the rule.** The threshold-free bus metrics of FedAvg (dumps, ideal → missing):

| Seed | best-F1 threshold | best F1 | recall at best F1 | recall at conf 0.25 |
|---|---|---|---|---|
| 42 | 0.379 → 0.039 | 0.538 → 0.517 | 0.487 → 0.468 | 0.533 → 0.203 |
| 123 | 0.503 → 0.056 | 0.524 → 0.513 | 0.426 → 0.420 | 0.513 → 0.236 |
| 2024 | 0.395 → 0.022 | 0.527 → 0.513 | 0.463 → 0.496 | 0.517 → 0.217 |

- So most of the recall loss at conf 0.25 is recovered by a per-class threshold. The bus scores of FedAvg drop about
  10× while their ranking is mostly kept.
- At conf 0.25, D1 files such a GT under confused_car, because a car box on it scores ≥ 0.25 while its bus score is
  below 0.25. D1 at a fixed threshold cannot separate "ranked below car" from "lost".
- The rule's verdict stands as fixed. The calibration component is reported as a descriptive finding.
- Any criterion built on it is post hoc and must be labelled so.

**Ours (P4t1), 2 seeds:**
- Mixed by the rule: the low_score share is 0.54 at seed 42 and 0.28 at seed 123.
- From the earlier dump (seed 42), the best-F1 threshold moves 0.375 → 0.237 and best F1 0.631 → 0.600. Ours keeps
  the bus scores up (smaller score shift than FedAvg) but loses more ranking (F1 −0.031 against −0.011 to −0.021
  for FedAvg).

**Truck FP of Ours** (multi-label, as the evaluator):
- 2,305 / 2,137 in the missing scenario and 2,234 / 2,113 in the ideal one, so it is not a missing-class effect.
- 60 % of them sit on car GTs (1,390 / 1,243).
- Under top-1 NMS the truck FP of Ours falls to 1,355 / 1,260, below FedAvg ideal (1,556–1,659). Most of the excess
  is a second truck label on car boxes.
- FedAvg makes the opposite error: truck GT confused as car, 1,222–1,412 against 429–451 for Ours.
- The missing scenario adds about +240 truck FP on bus GTs for Ours (533 against 293); there is no consistent
  increase for FedAvg.
