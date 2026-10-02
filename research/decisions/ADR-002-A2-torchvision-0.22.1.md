# ADR-002-A2: torchvision 0.22.1 (pair for the pinned torch 2.7.1)

**Date:** 2026-10-02
**Status:** accepted — forced correction of an unsatisfiable pin; torch pin unchanged
**Scope:** yolo-version / environment (amends ADR-001 table row "torchvision", ADR-002 install step)

## Context

The first headless Kaggle run of notebook 01 (`phdatt/flops-s1-smoke-g1`, version 1, commit
`3238702`) stopped in Cell 1:

```
ERROR: Cannot install torch==2.7.1 and torchvision==0.22.0+cu128 because these package
versions have conflicting dependencies.
```

ADR-001 pinned `torch 2.7.1` together with `torchvision 0.22.0`; that combination cannot be
installed. It was never exercised before because no session had reached the install step.

## Evidence `[ENGINEERING]`

Wheel metadata from the official index `https://download.pytorch.org/whl/cu128`
(checked 2026-10-02, cp311 manylinux_2_28_x86_64):

| Wheel | `Requires-Dist` |
|---|---|
| `torchvision-0.22.0+cu128` | `torch (==2.7.0)` |
| `torchvision-0.22.1+cu128` | `torch (==2.7.1)` |

## Decision

Keep `torch==2.7.1` (ADR-001) and pin `torchvision==0.22.1`, the only torchvision release built
for it. The commented Docker pins move to `torchvision==0.22.1`, `torchaudio==2.7.1` for the same
reason. Updated: notebooks 01–04 install cells, `requirements.txt`, ADR-002 install step,
plan.md §3.

## Consequences

- No model code depends on torchvision APIs that changed between 0.22.0 and 0.22.1 (patch release).
- G1 is still open: `environment.lock` from the first successful Kaggle run is the canonical record.

## Related ADRs

ADR-001 (environment versions) · ADR-002 (Kaggle deployment) · ADR-002-A1 (mlflow/protobuf)
