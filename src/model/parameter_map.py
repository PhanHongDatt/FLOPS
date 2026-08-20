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
    """Return True only for parameters that are structurally class-specific.

    Per Detect definition: cv3.*.weight/bias contain class predictions with
    output shape (nc, ...). All other params are shared.
    """
    return classify_parameter(name) == "detect_cls"


def build_parameter_map(model: Any) -> list[ParameterInfo]:
    """Introspect a YOLO model and return per-parameter classification.

    model: an Ultralytics YOLO instance (from build_model()).
    """
    info: list[ParameterInfo] = []
    state = model.model.state_dict() if hasattr(model, "model") else model.state_dict()
    for name, tensor in state.items():
        shape = tuple(tensor.shape)
        group = classify_parameter(name)
        info.append(ParameterInfo(
            name=name,
            shape=shape,
            module_group=group,
            class_specific=(group == "detect_cls"),
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
