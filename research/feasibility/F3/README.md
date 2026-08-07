# F3 — Matched Missing-Class Local Training

**Gate:** G5
**Status:** not started
**Prerequisite:** F1 completed

## Objective

Compare parameter updates from a client WITH vs WITHOUT the target class,
starting from the same global checkpoint.

## Delta to Track

Δθ = θ_local - θ_global

By module: backbone | neck | classification branch | regression branch | validated class-associated group

## Metrics

- L2/update norm per module
- cosine similarity / direction
- per-class AP change
- confidence change
- FP/FN change

## Claim Standard

A Missing-Class claim requires:
  parameter/update evidence + prediction-level evidence
  NOT L2 alone.

## Gate Decision Criteria

- A Supported: isolated class intervention is technically meaningful
- B Partial: use only demonstrably class-associated subset
- C Unsupported: stop parameter-masking path; review evidence-backed fallback

## Files

- `config.yaml`
- `update_analysis.yaml`
- `metrics.csv`
- `gate_decision.md` — record ADR reference
- `notes.md`
