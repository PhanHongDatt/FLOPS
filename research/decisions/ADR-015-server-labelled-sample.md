# ADR-015: Server holds a labelled 1,000-image sample — teacher, pre-training, KD and preservation

**Date:** 2026-10-05
**Status:** accepted (the author changed the problem assumption: "server được cung cấp 1000 ảnh có
đủ 4 lớp, tách dữ liệu đó ra khỏi client"; exploratory, 1 seed — CLAUDE.md §7, §14)
**Scope:** `partition_server_sample`, `configs/partition/server_sample_1k_seed42.yaml`,
`scripts/train_server_teacher.py`, `TeacherKDClient`, `TeacherKDRhoClient`, server fine-tune in
`src/federated/server.py`, presets B1/B2/P1/P2, sessions s8t → s8a/s8b

## Context

s5–s7: every client-side loss change (A2b, A4b, A5, A6) stayed below FedAvg. D3 showed ~40 % of
the H1 loss comes from having less bus data, and D1 that missed buses are mostly classified as
truck/car. No loss re-weighting adds information; a labelled sample at the server does.

## Changed assumption

The server holds **1,000 labelled BDD100K train images** with all four classes, disjoint from
every client image (S1b, its matched control and the pooled re-split). Clients keep exactly the
S1b-2k partition, so all arms stay comparable with A0 seed 42 of s5a. This is a different FL
setting ("server-side proxy data") and is reported as such.

Sample: uniform random draw with seed 42 from the images no client holds (natural class mix, not
stratified toward bus); generation fails if any class is absent instead of redrawing.

## Evidence

- `[LITERATURE]` Zhao et al., "Federated Learning with Non-IID Data", arXiv:1806.00582 (abstract
  verified 2026-10-05): a small globally shared subset — 5 % of data — raised CIFAR-10 accuracy by
  30 % under non-IID.
- `[LITERATURE]` Nguyen et al., "Where to Begin? On the Impact of Pre-Training and Initialization in
  Federated Learning", ICLR 2023, arXiv:2206.15387 (abstract verified): starting FL from a model
  pre-trained on server proxy data reduces the effect of data heterogeneity.
- `[LITERATURE]` Lin et al., "Ensemble Distillation for Robust Model Fusion in Federated Learning",
  NeurIPS 2020, arXiv:2006.07242: server-side distillation with auxiliary data (motivation; not
  reproduced here).
- KD form: FedNTD not-true distillation as in A5 (ADR-012, verified against official code).

## Decision — arms (seed 42, ADR-011 protocol: S1b-2k, 30 rounds, eval every 5, rule 20/25/30)

| Arm | Start | Clients | Server | Isolates |
|---|---|---|---|---|
| **T** | COCO init | — | 50 epochs on the sample (G2 schedule), final EMA | value of the sample alone |
| **B1** | T | FedAvg | — | pre-training (Nguyen 2023) |
| **B2** | T | FedAvg | + 1 epoch on the sample after every aggregation | strongest direct use of the sample |
| **P1** | T | FedNTD KD from the **fixed T** (beta = tau = 1) | — | KD from a non-degraded teacher |
| **P2** | T | P1 + A2b rho = 0.25 on locally missing classes | — | the author's preservation on top |

`[THESIS-HYPOTHESIS]` P1/P2: a teacher trained with all four classes balanced keeps the
bus-vs-truck/car discrimination that bus-free clients erase (D1). Unlike A5, the teacher is fixed
and never degraded by federated training. Server fine-tune: same lr0/batch as clients, no warm-up,
in-process data loading in the Ray driver. All choices fixed before any result.

## Pre-declared reading (AP50 bus; seed-std of A0 ≈ 0.003 → threshold 0.010)

- B1 − A0 ≥ +0.010: pre-training on the sample helps by itself.
- B2 − B1 ≥ +0.010: repeated server fine-tuning adds to pre-training.
- **P1 is a KD contribution only if** P1 ≥ B1 + 0.010 (same start, isolates KD) **and**
  P1 ≥ B2 − 0.005 (not worse than simply fine-tuning on the sample). If B2 ≥ P1 + 0.010 the honest
  conclusion is "the server data helps, distillation adds nothing".
- P2 − P1 ≥ +0.010: preservation adds on top of KD.
- Guard for every claim: mAP50 not more than 0.005 below its reference arm.
- No FP threshold this time: s7 showed arms with higher AP (pooled, control) also have 4–9× FP bus,
  so FP alone is not a failure signal. FP/FN are still reported. (Decided before any s8 result.)

## Risks

- Gains may be "more data", which is exactly what B1/B2 control for; T alone shows the sample's own
  level.
- A weak T (1,000 images, ~13 % with bus) may pass its errors on through KD, as A5 did.
- Teacher is shared through the s8t output, so s8a/s8b start only after s8t finishes.

## Related
ADR-009 · ADR-011 · ADR-012 · ADR-014 · literature_registry: zhao2018noniid, nguyen2023where, feddf2020, fedntd
