# F1 — Parameter Mapping

**Gate:** G5
**Status:** not started

## Objective

Inspect pinned YOLOv8 source and runtime model to map parameters by module and class-specificity.

## Required Output Table

| Module | Parameter | Shape | Class-specific? | Shared? | Evidence | Mask candidate? |
|--------|-----------|-------|-----------------|---------|----------|-----------------|

## Modules to Inspect

- backbone
- neck
- Detect head
- classification branch
- regression branch
- class-related outputs
- shared parameters

## Files

- `config.yaml` — inspection config
- `parameter_map.yaml` — output mapping
- `notes.md` — observations and uncertainties
