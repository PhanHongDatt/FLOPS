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
