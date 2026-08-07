# F2 — Controlled Perturbation

**Gate:** G5
**Status:** not started
**Prerequisite:** F1 completed, candidate class-associated parameter group identified

## Objective

Perturb only a candidate class-associated parameter group and measure class-level effects.

## Metrics to Record

- target-class AP
- non-target AP
- confidence statistics
- FP/FN per class
- prediction changes

## Decision Rule

Pre-register the decision rule BEFORE the final evaluation.
Compare target effects against non-target/control effects.
Do not invent a universal threshold.

## Files

- `config.yaml`
- `perturbation_log.yaml`
- `metrics.csv`
- `notes.md`
