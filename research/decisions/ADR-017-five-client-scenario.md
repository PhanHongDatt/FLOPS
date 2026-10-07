# ADR-017: Five-client Missing-Class scenario S1b5-3k with motorcycle oversampling

**Date:** 2026-10-08
**Status:** accepted (author: "5 client, mỗi client khoảng 3000 ảnh, tăng số bounding box motorcycle")
**Scope:** `configs/partition/s1b5_bus_3k_moto_seed42.yaml`, `s1b5_pooled_3k_moto_seed42.yaml`,
`configs/experiments/convergence_5c.yaml`, `class_quota` in `partition_missing_class`, sessions s9a/s9b

## Context

S1b-2k uses 8,000 images (11 % of BDD100K train), 4 clients, bus absent on 2/4. The author asked for a
larger, 5-client setting with more motorcycle boxes (motorcycle AP50 ≤ 0.05 in every trained arm; only
347 motorcycle boxes in S1b-2k).

## Decision

- **Partition** (seed 42, ordinary least-filled assignment of `partition_missing_class`):
  5 clients × 3,000 images = 15,000 (21.7 % of train); bus absent on C0, C1, C2 (3/5 = 60 % — one step more
  severe than S1b's 50 %); truck absent on C3; C4 holds all four classes.
- **Motorcycle oversampling** `[ENGINEERING]`: `class_quota: {motorcycle: 300}` — before the ordinary stream,
  each client that may hold motorcycle receives up to 300 images containing it; eligibility is unchanged,
  so a bus-free client never receives an image with a bus (tested). Simulated on the real labels (same code,
  same seed): motorcycle 662 → 1,963 boxes (≈ 390 per client); bus 2,628, truck 6,421, car 155,250;
  sizes 3,000 each. The whole train set has only 3,002 motorcycle boxes in 2,284 images, so the quota uses
  ~1,500 of them. This changes the motorcycle prior — reported as a sampling design, not a natural split.
- **Control for the missing-class effect:** the pooled re-split (D3, ADR-014) of exactly the same 15,000
  images over the same 5 clients.
- **Protocol:** as ADR-011 (YOLOv8n, 30 rounds × 1 local epoch, eval every 5 rounds, rule = mean of
  rounds 20/25/30), `num_clients: 5`. The runner now refuses a config whose `num_clients` differs from the
  partition. YOLOv8n because ADR-016 (s-tier) is still only proposed.

## Pre-declared reading (seed 42)

- `A0@s5pooled − A0@s5 ≥ +0.010` AP50 bus → the missing-class effect persists at 15,000 images and 5 clients.
  Compare its size with S1b-2k's D3 gap (+0.029): a larger gap is consistent with the higher severity
  (60 % vs 50 % of clients without bus); a smaller one with more data per bus-holding client.
- Motorcycle: report AP50 motorcycle of both arms against the S1b-2k values (≤ 0.010) — whether
  oversampling makes the class measurable at all. No threshold: descriptive only.

## Cost

Estimated ~1.6–1.9 h per arm on T4 ×2 (15,000 images per round, 5 clients in 3 GPU slots); two sessions in
parallel → ~3.5 GPU-h in total.

## Related
ADR-011 · ADR-014 · ADR-016 · CLAUDE.md §10 (severity variables)
