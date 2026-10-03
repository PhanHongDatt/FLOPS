# Baseline comparison — Centralized vs FedAvg vs FedProx

**Feasibility scale, single seed (42)**: a sanity comparison, not a reportable result (CLAUDE.md §14 requires 3 seeds). AP at conf 0.001, FP/FN at 0.25 (ADR-009).

## Final metrics

| metric | FedAvg | FedProx | G2-centralized |
|---|---|---|---|
| mAP50 | 0.3769 | 0.3756 | 0.4883 |
| mAP50-95 | 0.2438 | 0.2436 | 0.3232 |
| AP50_car | 0.6764 | 0.6784 | 0.7199 |
| AP50_bus | 0.3318 | 0.3316 | 0.4940 |
| AP50_truck | 0.4122 | 0.4094 | 0.5248 |
| AP50_motorcycle | 0.0871 | 0.0830 | 0.2145 |
| AP_car | 0.4072 | 0.4081 | 0.4421 |
| AP_bus | 0.2497 | 0.2518 | 0.3766 |
| AP_truck | 0.2828 | 0.2799 | 0.3713 |
| AP_motorcycle | 0.0355 | 0.0346 | 0.1030 |
| precision_car | 0.6767 | 0.6699 | 0.7198 |
| precision_bus | 0.4721 | 0.4728 | 0.5937 |
| precision_truck | 0.5015 | 0.5105 | 0.6074 |
| precision_motorcycle | 1.0000 | 1.0000 | 0.5414 |
| recall_car | 0.6411 | 0.6445 | 0.6724 |
| recall_bus | 0.3156 | 0.2987 | 0.4400 |
| recall_truck | 0.4035 | 0.3916 | 0.4813 |
| recall_motorcycle | 0.0000 | 0.0000 | 0.1814 |
| FN_car | 34021.0000 | 34667.0000 | 32804.0000 |
| FN_bus | 1290.0000 | 1339.0000 | 1010.0000 |
| FN_truck | 2795.0000 | 2867.0000 | 2497.0000 |
| FN_motorcycle | 452.0000 | 452.0000 | 371.0000 |
| FP_car | 30775.0000 | 28801.0000 | 21139.0000 |
| FP_bus | 818.0000 | 714.0000 | 529.0000 |
| FP_truck | 2054.0000 | 1814.0000 | 1425.0000 |
| FP_motorcycle | 0.0000 | 0.0000 | 59.0000 |

## Gap vs centralized (arm − G2)

| metric | FedAvg | FedProx |
|---|---|---|
| mAP50 | -0.1114 | -0.1127 |
| mAP50-95 | -0.0794 | -0.0796 |
| AP50_car | -0.0435 | -0.0415 |
| AP50_bus | -0.1621 | -0.1624 |
| AP50_truck | -0.1126 | -0.1154 |
| AP50_motorcycle | -0.1274 | -0.1315 |
| AP_car | -0.0348 | -0.0339 |
| AP_bus | -0.1269 | -0.1248 |
| AP_truck | -0.0885 | -0.0913 |
| AP_motorcycle | -0.0674 | -0.0684 |
| precision_car | -0.0431 | -0.0499 |
| precision_bus | -0.1216 | -0.1209 |
| precision_truck | -0.1059 | -0.0969 |
| precision_motorcycle | 0.4586 | 0.4586 |
| recall_car | -0.0314 | -0.0280 |
| recall_bus | -0.1244 | -0.1413 |
| recall_truck | -0.0777 | -0.0897 |
| recall_motorcycle | -0.1814 | -0.1814 |
| FN_car | 1217.0000 | 1863.0000 |
| FN_bus | 280.0000 | 329.0000 |
| FN_truck | 298.0000 | 370.0000 |
| FN_motorcycle | 81.0000 | 81.0000 |
| FP_car | 9636.0000 | 7662.0000 |
| FP_bus | 289.0000 | 185.0000 |
| FP_truck | 629.0000 | 389.0000 |
| FP_motorcycle | -59.0000 | -59.0000 |

## FedProx − FedAvg

| metric | Δ |
|---|---|
| mAP50 | -0.0013 |
| mAP50-95 | -0.0002 |
| AP50_car | 0.0020 |
| AP50_bus | -0.0003 |
| AP50_truck | -0.0029 |
| AP50_motorcycle | -0.0041 |
| AP_car | 0.0009 |
| AP_bus | 0.0021 |
| AP_truck | -0.0028 |
| AP_motorcycle | -0.0009 |
| precision_car | -0.0068 |
| precision_bus | 0.0007 |
| precision_truck | 0.0090 |
| precision_motorcycle | 0.0000 |
| recall_car | 0.0034 |
| recall_bus | -0.0169 |
| recall_truck | -0.0119 |
| recall_motorcycle | 0.0000 |
| FN_car | 646.0000 |
| FN_bus | 49.0000 |
| FN_truck | 72.0000 |
| FN_motorcycle | 0.0000 |
| FP_car | -1974.0000 |
| FP_bus | -104.0000 |
| FP_truck | -240.0000 |
| FP_motorcycle | 0.0000 |

## Convergence (mAP50 per round, server eval on global val)

| arm | best round | best | final round | final |
|---|---|---|---|---|
| FedAvg | 5 | 0.3769 | 5 | 0.3769 |
| FedProx | 5 | 0.3756 | 5 | 0.3756 |

## Wall-clock (run.log first → last timestamp)

| arm | minutes |
|---|---|
| FedAvg | 102.7333 |
| FedProx | 117.7500 |
| G2-centralized | 142.8500 |

## Per-round mAP50 / AP50 (server eval, global val)

| round | FedAvg mAP50 | FedAvg bus | FedAvg moto | FedProx mAP50 | FedProx bus | FedProx moto |
|---|---|---|---|---|---|---|
| 0 | 0.006 | 0.002 | 0.000 | 0.006 | 0.002 | 0.000 |
| 1 | 0.271 | 0.164 | 0.001 | 0.269 | 0.167 | 0.002 |
| 2 | 0.300 | 0.210 | 0.003 | 0.299 | 0.216 | 0.002 |
| 3 | 0.318 | 0.236 | 0.006 | 0.309 | 0.216 | 0.008 |
| 4 | 0.347 | 0.286 | 0.044 | 0.343 | 0.288 | 0.034 |
| 5 | 0.377 | 0.332 | 0.087 | 0.376 | 0.332 | 0.083 |

## Interpretation (observations vs. inferences, CLAUDE.md §24)

**Observed**
1. FedAvg reaches 77 % of centralized mAP50 (0.377 vs 0.488). The relative gap grows with class
   rarity: car −6 %, truck −21 %, bus −33 %, motorcycle −59 % (AP50).
2. Neither regime has converged: FedAvg mAP50 still gains +0.030 in round 5 (bus +0.046,
   motorcycle +0.043); G2 training losses still fall at epoch 10.
3. Motorcycle is learned only from round 4; at the 0.25 operating point it has no predictions
   (recall 0, precision reported as 1.0 — a degenerate value, not a strength).
4. FedProx (mu = 0.01) is within ±0.004 AP50 of FedAvg on every class — no measurable effect on
   IID S0 with one seed — and costs ~15 % more wall time (gradient hooks).

**Inferred, not established**
- The FedAvg–centralized gap is **not** a clean FL penalty: FL saw 5 passes over the data
  (5 rounds × 1 local epoch on ¼ each) vs 10 epochs for G2, and G2 used warm-up/close_mosaic.
  An equal-budget comparison (FL 10 rounds) is needed before attributing the gap to federation.
- Rare classes suffer most even without missing classes; any Missing-Class (H1) effect must be
  measured against this IID rare-class deficit, i.e. via the matched control, not vs. centralized.
- Motorcycle is too under-trained at this budget to carry class-level conclusions (consistent with
  plan decision D1: motorcycle is a secondary observation; bus is the target).
- FedProx's lack of effect is expected on IID data (it targets heterogeneity); its value must be
  judged on S1b, not S0.

**Budget measured (T4)**: G2 ≈ 13 min/epoch; FL ≈ 20 min/round incl. ~4–5 min server eval at conf
0.001; FedProx ≈ 23.5 min/round.
