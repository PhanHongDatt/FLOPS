"""YOLOv8n parameter mapping utility.

Used by F1 (Section 8, CLAUDE.md) to produce the parameter table for
research/feasibility/F1/parameter_map.yaml.

Verified module structure for ultralytics 8.3.253 (nano config):
  model.0 - model.9   : backbone (Conv, C2f, SPPF)
  model.10 - model.21 : neck (Upsample, Concat, C2f, Conv)
  model.22            : Detect head — contains cv2 (regression),
                                             cv3 (classification),
                                             dfl (Distribution Focal Loss)

Source:
- ultralytics/cfg/models/v8/yolov8.yaml (backbone + head config)
- ultralytics/nn/modules/head.py (Detect class definition)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParameterInfo:
    name: str            # full state_dict key (e.g., "model.22.cv3.0.weight")
    shape: tuple[int, ...]
    module_group: str    # backbone | neck | detect_cls | detect_reg | detect_dfl | shared
    class_specific: bool
    numel: int


BACKBONE_INDICES = tuple(range(0, 10))       # model.0 ... model.9
NECK_INDICES = tuple(range(10, 22))          # model.10 ... model.21
DETECT_INDEX = 22                             # model.22 (Detect head)

# Final 1x1 nn.Conv2d of each cv3 branch: the ONLY parameters with one slice
# per class (dim 0 == nc). Runtime-confirmed on ultralytics 8.3.253, nc=4:
# 3 scales x {weight (4, 64, 1, 1), bias (4,)} = 6 keys, 780 elements.
# cv3.<s>.0 / cv3.<s>.1 are Conv+BN blocks shared by all classes.
CLASS_HEAD_RE = re.compile(r"^model\.\d+\.cv3\.\d+\.2\.(weight|bias)$")
EXPECTED_CLASS_HEAD_KEYS = 6


def classify_parameter(name: str) -> str:
    """Return module_group for a parameter name.

    Rules follow the Detect class structure verified from
    ultralytics 8.3.253 head.py:
      cv2 → box regression (4 * reg_max channels output)
      cv3 → classification (nc channels output) — CLASS-SPECIFIC
      dfl → Distribution Focal Loss integral (fixed, not learned per class)
    """
    parts = name.split(".")
    if parts[0] != "model" or len(parts) < 2:
        return "unknown"

    try:
        idx = int(parts[1])
    except ValueError:
        return "unknown"

    if idx in BACKBONE_INDICES:
        return "backbone"
    if idx in NECK_INDICES:
        return "neck"
    if idx == DETECT_INDEX:
        if len(parts) >= 3:
            sub = parts[2]
            if sub == "cv3":
                return "detect_cls"     # class-specific
            if sub == "cv2":
                return "detect_reg"     # regression, shared across classes
            if sub == "dfl":
                return "detect_dfl"
        return "detect_other"
    return "unknown"


def is_class_specific(name: str) -> bool:
    """True only for the final cv3 conv (one row per class).

    Being in the classification branch is not enough: the earlier cv3 Conv+BN
    blocks feed every class logit and are shared. Treating the whole branch as
    class-specific made the "detect_cls" drift group 99.8% shared parameters
    (370,182 of 370,962 elements at nc=4).
    """
    return bool(CLASS_HEAD_RE.match(name))


def class_head_keys(
    names: list[str],
    shapes: list[tuple[int, ...]],
    nc: int,
) -> list[str]:
    """Names of the class-specific params, failing loudly unless exactly 6 match.

    A regex that matches nothing (index shift, version change, nc=80 model)
    would make every class-row analysis a silent no-op (plan.md R4).
    """
    keys = [
        n for n, s in zip(names, shapes)
        if is_class_specific(n) and len(s) >= 1 and s[0] == nc
    ]
    if len(keys) != EXPECTED_CLASS_HEAD_KEYS:
        raise ValueError(
            f"Expected exactly {EXPECTED_CLASS_HEAD_KEYS} class-head params "
            f"(3 scales x weight/bias, dim0 == {nc}), matched {len(keys)}: {keys}. "
            "Re-verify F1 (research/feasibility/F1/parameter_map.yaml)."
        )
    return keys


def model_state_dict(model: Any) -> dict[str, Any]:
    """state_dict with "model.<idx>." keys for an ultralytics YOLO or a DetectionModel.

    ``YOLO.model`` is the DetectionModel; ``DetectionModel.model`` is the layer
    Sequential. Calling ``.model.state_dict()`` on a DetectionModel therefore
    drops the "model." prefix and every key falls out of the parameter map.
    """
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "model"):
        return inner.state_dict()
    return model.state_dict()


def build_parameter_map(model: Any) -> list[ParameterInfo]:
    """Introspect a YOLO model and return per-parameter classification.

    model: an Ultralytics YOLO instance (from build_model()) or a DetectionModel.
    """
    info: list[ParameterInfo] = []
    state = model_state_dict(model)
    for name, tensor in state.items():
        shape = tuple(tensor.shape)
        group = classify_parameter(name)
        info.append(ParameterInfo(
            name=name,
            shape=shape,
            module_group=group,
            class_specific=is_class_specific(name),
            numel=int(tensor.numel()),
        ))
    return info


def summarize(info: list[ParameterInfo]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for p in info:
        g = p.module_group
        if g not in summary:
            summary[g] = {"params": 0, "numel": 0}
        summary[g]["params"] += 1
        summary[g]["numel"] += p.numel
    return summary
