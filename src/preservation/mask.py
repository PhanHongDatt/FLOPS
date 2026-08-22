"""H2 — Client-side knowledge preservation mask.

[THESIS-HYPOTHESIS] Reducing updates to validated class-associated parameters
(detect_cls final conv) for a locally missing class may preserve prior global
knowledge, following the spirit of selective parameter freezing.

Candidate factor rho (CLAUDE.md §6):
  rho=1.0  → normal update, no preservation
  rho=0.25 → partial preservation
  rho=0.0  → maximum preservation, keep global for missing class channels

Applies ONLY to: model.22.cv3.{scale}.2.weight / .2.bias
  — the FINAL 1×1 nn.Conv2d inside each cv3 branch where dim 0 = nc.
  — Confirmed from F1 source analysis: research/feasibility/F1/parameter_map.yaml

All other parameters (backbone, neck, cv2, dfl, intermediate cv3) are
returned unchanged from local training.

⚠️  RUNTIME VERIFICATION required before production use (parameter_map.yaml
    verification: runtime_confirmed=False as of F1). This module must NOT
    be used in G6+ evaluation until F1 runtime check passes.
"""
from __future__ import annotations

import re

import numpy as np

from src.data.bdd100k import TARGET_CLASSES

# Matches ONLY the final 1×1 conv in each cv3 branch:
#   model.22.cv3.<scale>.2.weight
#   model.22.cv3.<scale>.2.bias
# The <2> index is the nn.Conv2d layer; <0> and <1> are Conv modules (shared).
_FINAL_CV3_RE = re.compile(r"^model\.22\.cv3\.\d+\.2\.(weight|bias)$")

NUM_CLASSES: int = len(TARGET_CLASSES)          # 4
CLASS_INDEX: dict[str, int] = {c: i for i, c in enumerate(TARGET_CLASSES)}


def _is_maskable(name: str, shape: tuple[int, ...]) -> bool:
    """True only for final cv3 conv params where dim-0 == nc.

    Double-check shape to guard against unexpected model topology changes.
    """
    return bool(_FINAL_CV3_RE.match(name)) and len(shape) >= 1 and shape[0] == NUM_CLASSES


def apply_preservation_mask(
    local_params: list[np.ndarray],
    global_params: list[np.ndarray],
    param_names: list[str],
    missing_classes: list[str],
    rho: float,
) -> list[np.ndarray]:
    """Apply H2 preservation mask to local parameters before upload.

    For each maskable parameter (detect_cls final conv), for each missing class
    channel (dim 0), blend the local update toward global:

        masked[cls_idx] = global[cls_idx] + rho * (local[cls_idx] - global[cls_idx])
                        = (1 - rho) * global[cls_idx] + rho * local[cls_idx]

    rho=1.0 → identity (no effect)
    rho=0.0 → full preservation (keep global channel for missing class)
    rho=0.25 → partial preservation

    All other parameters returned unchanged.

    Args:
        local_params:    parameters after local training (ordered, matches param_names)
        global_params:   parameters received from server this round
        param_names:     ordered state_dict key list
        missing_classes: class names with zero positive boxes on this client
        rho:             preservation factor [THESIS-HYPOTHESIS]

    Returns:
        Masked parameter list (same structure/order as local_params).
    """
    if not missing_classes or rho == 1.0:
        return list(local_params)

    missing_indices = [CLASS_INDEX[c] for c in missing_classes if c in CLASS_INDEX]
    if not missing_indices:
        return list(local_params)

    masked: list[np.ndarray] = []
    for name, local, global_ in zip(param_names, local_params, global_params):
        if _is_maskable(name, local.shape):
            p = local.copy()
            for cls_idx in missing_indices:
                if cls_idx < p.shape[0]:
                    p[cls_idx] = global_[cls_idx] + rho * (local[cls_idx] - global_[cls_idx])
            masked.append(p)
        else:
            masked.append(local)

    return masked


def get_param_names(model) -> list[str]:
    """Return ordered state_dict key list from a YOLO model instance."""
    state = model.model.state_dict() if hasattr(model, "model") else model.state_dict()
    return list(state.keys())


def maskable_params(param_names: list[str], shapes: list[tuple[int, ...]]) -> list[str]:
    """Return names of parameters that will be masked by apply_preservation_mask."""
    return [n for n, s in zip(param_names, shapes) if _is_maskable(n, s)]
