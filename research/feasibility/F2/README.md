# F2 — Controlled Perturbation

**Gate:** G5 (F2)
**Status:** `code_ready` (2026-10-02) — tooling + CPU smoke done; no real run yet
**Prerequisites:** F1 runtime (done locally; re-run on Kaggle) · G2 checkpoint that detects all 4 classes

## Objective

Perturb only a candidate class-associated parameter group of a trained detector and
compare the effect on the target class with the effect on non-target classes and with a
norm-matched perturbation of shared parameters (CLAUDE.md §8 F2).

## Design `[ENGINEERING]`

All perturbations start from the same checkpoint and are evaluated on the same seeded
val subset (`val_subset`, default 2000 images).

| Kind | Parameters touched | Role |
|---|---|---|
| `row_noise` (rel, seed) | weight row *j* of `model.22.cv3.{0,1,2}.2.weight` (3×64 values); noise std = rel × RMS of the row | main probe; run for every row *j* → **effect matrix** E[j][i] = ΔAP50ᵢ |
| `row_bias_shift` (δ) | bias of class *j* at all 3 scales, −δ logits | **positive control**: lowers only class *j*'s logits, so the evaluator must show a diagonal-only effect |
| `shared_noise` (rel, seed) | shared `cv3.<s>.1.conv.weight` (all scales), total L2 = the bus-row noise L2 at the same rel/seed | **norm-matched control**: is the class row a more targeted lever than an equally large change to shared params? |

Biases are excluded from `row_noise`: a bias "row" is one scalar (≈ −8.5 after
`bias_init`), so relative noise would move the logit by several units and swamp the
weight-row effect (found in the CPU smoke run, 2026-10-02).

Recorded per perturbation: AP50 / AP / P / R / FP / FN per class (`evaluate`) and
per-class prediction count + mean confidence (`confidence_stats`, CLAUDE.md §13).

## Run

```bash
python scripts/run_f2.py --f2-config configs/feasibility/f2_perturb_bus.yaml \
  --weights <G2 best.pt> --eval-data-yaml <global data.yaml> \
  --output-dir research/feasibility/F2/runs
```
Kaggle: `notebooks/03_missing_class_G4.py` Cell 5b. Budget: 35 evaluations × (val + predict)
on the subset — `[RESOURCE-BUDGET-UNRESOLVED]` until G2 measures `t_eval`.

## Artifacts

`config.yaml` · `environment.json` · `val_subset_*.txt` · `metrics.csv` (spec × metric:
base/value/delta) · `perturbation_log.yaml` (spec + actual L2) · `summary.yaml`
(baseline + effect matrix, mean/std over seeds).

## Pre-registered decision rule — **DRAFT, needs author approval before the first real run**

E = `summary.yaml: effect_matrix`, metric ΔAP50, mean ± sample std over seeds; `τ_AP` shared with F3.

| ID | Check | Pass condition |
|---|---|---|
| F2-a | positive control (`row_bias_shift`, every row *j*) | `E[j][j] < −τ_AP` and `|E[j][i]| ≤ τ_AP` for all i ≠ j. **Fail ⇒ evaluator invalid, stop.** |
| F2-b | specificity (`row_noise`, row = bus, each rel) | `|E[bus][bus]| > 2·std` and `|E[bus][bus]| > max_{i≠bus} |E[bus][i]| + 2·std` |
| F2-c | norm-matched control | `|E_row_noise[bus][bus]| > |E_shared_noise[shared][bus]|` at the same rel |

Input to Gate A/B/C (decided in ADR-003 together with F3, not here):
- F2-b ∧ F2-c → supports **A** (the class row is an isolated, targeted lever)
- F2-b ∧ ¬F2-c → **B** (class rows are class-specific, but shared params carry as much class information)
- ¬F2-b → evidence toward **C** (no isolated class intervention)

`τ_AP` = **[TO DECLARE]** (proposal 0.01 AP50), fixed before any real F2 output is inspected.

## Smoke log

- 2026-10-02, CPU, synthetic 64-px images, untrained head (`yolov8n.pt` → nc=4): all 9
  smoke perturbations ran end-to-end and all artifacts were written. Baseline AP is 0,
  so the effects are not interpretable (the script warns about this). Not evidence.
