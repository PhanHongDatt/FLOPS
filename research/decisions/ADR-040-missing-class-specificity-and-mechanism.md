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

## Addendum A9 2026-11-25: R1-R3 arms, bus1 partition for YOLO11m, per-class thresholds (before R1-R3 run)

**Arms** (`scripts/run_fl_experiment.py`, unit-tested to differ from their reference in one component):

| Run | Arm | Definition |
|---|---|---|
| R1 | `P4noKDt1` | P4t1 without KD (COCO head + cosine round lr + client RFS t = 1.0) |
| R2 | `A5cLRt1` | `P4noKDt1` with FedNTD (teacher = received global model) |
| R3 | `A0clr01` | A0c at lr0 × 0.1 (0.001). The factor is fixed here, before any run. |

**bus1 partition for YOLO11m.** `configs/partition/s1b5_bus1_3k_allmoto_seed42.yaml` re-deals the 15,000
`s5moto` images, with bus only on C4 and truck absent on C3. It is built in notebook 05 (scenario `s5motobus1`)
with assertions. Checked locally from `det_20/det_train.json`:
- Box totals equal the reference (car 155,477, bus 2,657, truck 6,450, motorcycle 3,002).
- Per client: C0–C2 bus 0; C3 bus 0 and truck 0; C4 bus 2,657, truck 2,415, motorcycle 828. The other clients hold
  truck 1,338–1,349 and motorcycle 478–648.
- C4 holds more truck than the others because bus images often hold trucks. This is reported, not corrected.
- The ideal pair is `s5motopooled`. A0c already has 3 seeds on it.

**Per-class thresholds (descriptive, round 20).** `scripts/adr040_class_thresholds.py` tunes the threshold on one
half of the val images and applies it to the other half (two-fold cross-fit). Bus F1, ideal / missing:

| Arm | Seed | Fixed conf 0.25 | Per-class threshold |
|---|---|---|---|
| FedAvg | 42 | 0.523 / 0.330 | 0.531 / 0.515 |
| FedAvg | 123 | 0.513 / 0.371 | 0.518 / 0.512 |
| FedAvg | 2024 | 0.516 / 0.349 | 0.520 / 0.510 |
| P4t1 | 42 | 0.611 / 0.599 | 0.629 / 0.591 |

- With per-class thresholds, the FedAvg bus F1 gap shrinks from 0.14–0.19 to 0.006–0.016.
- P4t1's gap is 0.038.
- P4t1 is still ahead of FedAvg on bus F1 in the missing scenario: +0.08 with per-class thresholds, against +0.25 at
  conf 0.25.
- From now on, operating-point comparisons report per-class cross-fit thresholds next to conf 0.25.

## Result R1 (s41a/b, 2026-11-26)

`P4noKDt1`, YOLO11m 960, seed 42, `s5moto` / `s5motopooled`. Runs took 7.3 h and 8.2 h (P4t1 about 10 h).
AP50, mean of rounds 10/20 (round 20 in brackets).

| Arm | Seed | bus missing | bus ideal | abs gap | rel gap | motorcycle missing / ideal | mAP50 missing / ideal |
|---|---|---|---|---|---|---|---|
| A0c | 42 / 123 / 2024 | 0.479 / 0.498 / 0.498 | 0.506 / 0.508 / 0.513 | 0.027 / 0.010 / 0.015 | 0.053 / 0.020 / 0.029 (r20 0.022 / 0.031 / 0.020) | 0.325 / 0.351 (s42) | 0.524 / 0.544 (s42) |
| P4noKDt1 | 42 | 0.611 (0.633) | 0.630 (0.649) | 0.019 | 0.030 (r20 0.025) | 0.483 / 0.469 | 0.639 / 0.641 |
| P4t1 | 42 / 123 | 0.602 / 0.601 | 0.638 / 0.642 | 0.036 / 0.041 | 0.056 / 0.064 (r20 0.046 / 0.055) | 0.523 / 0.530 (s42) | 0.642 / 0.656 (s42) |

**Reading (sections 3, A3, A4, A6).**
- R1 against R0 (A0c), bus gap: d = 0.017 − 0.019 = −0.002 absolute (relative +0.004). **Not supported** either way.
  The recipe without KD keeps the FedAvg-sized gap while lifting bus AP50 by about 0.13 in both scenarios.
- KD effect (P4t1 − P4noKDt1, seed 42, single-seed rule 0.03):
  - motorcycle +0.040 (missing) and +0.061 (ideal): above 0.03, in line with ADR-023 on YOLO11s.
  - bus −0.009 (missing), +0.008 (ideal); gap 0.036 against 0.019 (+0.017). Below 0.03: descriptive only.
  - truck −0.017 (missing), −0.007 (ideal); car equal; mAP50 +0.003 (missing), +0.015 (ideal).
- So, on this seed, the gain over FedAvg on bus comes from the recipe (COCO head + cosine + RFS t = 1.0). KD adds
  motorcycle AP, not missing-class bus AP.

**Descriptive, operating point at conf 0.25 (round 20, from round_metrics).** Bus recall, missing / ideal: A0c
0.205 / 0.463, P4noKDt1 0.413 / 0.575, P4t1 0.486 / 0.573. Without KD the bus recall drop is 0.162 against 0.087
with KD, while AP50 is equal. This points at KD keeping bus scores up (calibration), not ranking. It is not a reading:
D1 and the per-class cross-fit thresholds for R1 need its checkpoints (section 4, A9) and are still to run.

### R1 secondary metrics (s41e, round 20)

D1 bus, multi-label (top-1 in brackets), missing − ideal excess, seed 42:

| Arm | TP lost | confused car | confused truck | low score | missed | conf + miss share | A5 reading |
|---|---|---|---|---|---|---|---|
| A0c | 528 | 251 | 87 | 106 | 85 | 0.80 | confusion + misses |
| P4noKDt1 | 293 (285) | 122 | 59 | 79 | 33 | 0.73 (0.93) | confusion + misses |
| P4t1 | 158 | 19 | 39 | 85 | 15 | 0.46 | mixed |

- P4noKDt1 score-shift rule: R@P0.7 drop 0.573 → 0.531 = 0.042 (needs < 0.02). Not met.
- Bus best-F1 threshold, ideal → missing: A0c 0.38 → 0.04, P4noKDt1 0.41 → 0.12, P4t1 0.375 → 0.237.

Per-class cross-fit F1 (A9), ideal / missing:

| Arm | bus @0.25 | bus @class thr | motorcycle @class thr | truck @class thr |
|---|---|---|---|---|
| A0c | 0.523 / 0.330 | 0.531 / 0.515 | 0.449 / 0.428 | 0.548 / 0.522 |
| P4noKDt1 | 0.608 / 0.571 | 0.632 / 0.597 | 0.571 / 0.560 | 0.626 / 0.628 |
| P4t1 | 0.611 / 0.599 | 0.629 / 0.591 | 0.596 / 0.600 | 0.618 / 0.613 |

Truck FP (multi-label), missing / ideal: P4noKDt1 1,130 / 1,840 (on car GTs 449 / 905, on bus GTs 337 / 269);
P4t1 2,305 / 2,234 (on car 1,390 / 1,482, on bus 533 / 293).

Reading (single seed, descriptive except the A5 line):
- Without KD the bus loss reads like FedAvg (confusion + misses), at about half the size. KD moves it to "mixed".
- At per-class thresholds, bus F1 is the same with and without KD (0.597 / 0.591 missing; 0.632 / 0.629 ideal).
  KD's bus benefit at conf 0.25 (+0.028 missing) is a score-level (calibration) effect, not a ranking one.
- KD keeps a motorcycle gain at per-class thresholds (+0.025 to +0.040), as in AP50.
- Most truck FP on car GTs comes with KD (about 1,400 against 450–900 without it). Truck F1 is 0.005–0.015 lower
  with KD.

## Result R2, R3 (s41c, s41d, 2026-11-28)

Missing scenario (`s5moto`) only, YOLO11m 960, seed 42, AP50 mean of rounds 10/20. Runs took 9.04 h (R2) and 5.35 h (R3).

| Arm | mAP50 | car | bus | truck | motorcycle | bus recall / precision r20 | truck FP r20 |
|---|---|---|---|---|---|---|---|
| A0c (FedAvg) | 0.524 | 0.781 | 0.479 | 0.510 | 0.325 | 0.205 / 0.860 | 851 |
| R3 A0clr01 (FedAvg, lr0 0.001) | 0.604 | 0.793 | 0.569 | 0.579 | 0.475 | 0.389 / 0.778 | 1,109 |
| A5c (FedNTD) | 0.547 | 0.785 | 0.496 | 0.504 | 0.402 | 0.390 / 0.730 | 2,426 |
| R2 A5cLRt1 (FedNTD + cosine + RFS) | 0.633 | 0.822 | 0.588 | 0.618 | 0.502 | 0.497 / 0.765 | 2,613 |
| R1 P4noKDt1 | 0.639 | 0.821 | 0.611 | 0.639 | 0.483 | 0.413 / 0.844 | 1,358 |
| P4t1 (Ours) | 0.642 | 0.821 | 0.602 | 0.622 | 0.523 | 0.486 / 0.768 | 2,758 |

**Reading (section 2 questions, single-seed rule 0.03).**
- *Is the default baseline weak?* **Yes.** R3 − A0c: mAP50 +0.080, bus +0.090, truck +0.069, motorcycle +0.150, all
  above 0.03. lr0 0.01 is too high for fine-tuning from COCO. About two thirds of Ours' mAP50 lead over A0c (0.118)
  is reached by lowering lr0 alone. Ours − R3: mAP50 +0.038, bus +0.033, truck +0.043, motorcycle +0.048 (just above
  0.03).
- *Does the frozen COCO teacher beat the received global teacher?* **Not supported.** P4t1 − R2: mAP50 +0.009, bus
  +0.014, truck +0.004, motorcycle +0.021, all below 0.03.
- *Does FedNTD add to the recipe?* R2 − R1: mAP50 −0.006, bus −0.023, motorcycle +0.019. Below 0.03; descriptive.
- Descriptive: both KD arms (R2, P4t1) show higher bus recall at conf 0.25 than R1 (0.497 / 0.486 against 0.413) and
  about twice the truck FP (2,613 / 2,758 against 1,358). The calibration effect and the truck FP come with KD in
  general, not with the COCO teacher in particular.

**Consequences.**
- Baseline comparisons at the default lr0 (A0c, A1c, FedProx, A5c) overstate Ours' lead. Any comparison table must
  show R3 next to A0c, and claims about Ours over FedAvg use R3 as the FedAvg reference.
- R3 and R2 have no ideal-scenario pair, so their missing-class gaps are unknown.

## Addendum A10 2026-11-28: low-lr baselines (before s42a-d run)

R3 showed the default lr0 0.01 baseline is weak. The reference baselines are rerun at the same lr0 × 0.1:

| Session | Arm | Scenario | Seed | Purpose |
|---|---|---|---|---|
| s42a | `A0clr01` | `s5motopooled` | 42 | missing-class gap of FedAvg at low lr |
| s42b | `A1clr01` (class-count FedAvg, lr0 × 0.1) | `s5moto` | 42 | class-count control at low lr |
| s42c | `FedProxclr01` (FedProx μ 0.01, lr0 × 0.1) | `s5moto` | 42 | FedProx at low lr |
| s42d | `A0clr01` | `s5moto` | 123 | second seed for the Ours − R3 lead |

- The factor 0.1 is not tuned further; no other lr is tried for any arm, Ours included.
- Reading: single-seed rule 0.03 (A6). With s42d, Ours − A0clr01 on `s5moto` has 2 seeds per arm; the zone rules
  of A2/A4 apply with the Welch threshold.
- The low-lr FedAvg (A0clr01) replaces A0c as the FedAvg reference in comparison tables; A0c stays reported.
