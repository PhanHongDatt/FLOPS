# ADR-026: centralized ceiling for the best 5-client configuration

- Status: accepted (author request, 2026-11-01)
- Related: ADR-023 (P4t1 @ s5moto 960 px, S21b), ADR-025

## Context
The author wants AP50 >= 0.70 on the rare classes (bus, truck, motorcycle), better precision/recall and clearer
separation between scenarios. Best result so far, S21b (P4t1 @ s5moto, 960 px, seed 42, mean of rounds 10/20):
bus 0.565, truck 0.574, motorcycle 0.504. No upper bound exists for this data/model/resolution (the only
centralized run, G2, is YOLOv8n at 640 px: bus 0.494, truck 0.525). Motorcycle cannot get more data: s5moto
already holds all 3,002 motorcycle boxes of BDD100K train.

## Decision
- One run: P4t1 with **one client holding exactly the 15,000 s5moto images**
  (`configs/partition/s1b5_central_3k_allmoto_seed42.yaml`, a 1-client re-deal of the reference manifest),
  `configs/experiments/central_1c_y11s_20r_960.yaml`. Same model, weights, 960 px, batch, recipe (COCO head,
  KD from the frozen COCO model, cosine round LR, RFS t=1.0), 20 rounds, seed 42 and evaluator as S21b.
  Per round the single client sees the same images and optimizer steps as the five clients together `[ENGINEERING]`.
- Reading, fixed before the run: mean of rounds 10/20 per class (AP50, precision, recall), compared with S21b.
  - ceiling < 0.70 on a rare class: the 0.70 target is out of reach by changing the FL method alone; the next
    levers are data (more bus/truck images) or model size (needs its own ADR);
  - ceiling >= 0.70: the FL-vs-centralized gap is what the method still has to close.
- Session s23a_g (goshihayashi, T4 x2). One client uses one GPU, so the run is expected to take about 1.6x the
  S21b arm (~9-10 h); cut at 9.6 h, a part 2 resumes from the last checkpoint as in ADR-025.
- n = 1, descriptive. Precision/recall at another confidence threshold is an operating-point choice, reported
  separately and never as a model improvement.

## Result (s23a_g, completed 2026-11-02)
20 of 20 rounds, one session. Mean of rounds 10/20, same evaluator; precision/recall at the max-F1 confidence.

| | mAP50 | car | bus | truck | motorcycle |
|---|---:|---:|---:|---:|---:|
| S21b federated (5 clients, bus absent on 3) | 0.6095 | 0.7956 | 0.5650 | 0.5735 | 0.5039 |
| S23 centralized ceiling (1 client) | 0.6037 | 0.7983 | 0.5891 | 0.5689 | 0.4585 |
| ceiling − federated | -0.006 | +0.003 | +0.024 | -0.005 | -0.045 |

Bus is the only class the centralized run clearly does better: recall 0.446 -> 0.509, FN 1,031 -> 801
(precision 0.707 -> 0.661). Motorcycle is lower centralized (RFS t=1 computed on one pooled client repeats
motorcycle images less than on five clients; not tested separately).

Reading (rule fixed above): the ceiling is **below 0.70 on all three rare classes** (bus 0.589, truck 0.569,
motorcycle 0.459). With this data, model and resolution, AP50 0.70 cannot be reached by changing the FL method;
it needs more data (bus/truck images; motorcycle has none left in BDD100K train) or a larger model (own ADR).
The federated P4t1 is within 0.006 mAP50 of its centralized ceiling; the remaining Missing-Class cost is on bus
(-0.024 AP50, -0.063 recall). n = 1, descriptive.
