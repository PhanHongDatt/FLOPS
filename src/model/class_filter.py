"""Evaluate the COCO-head detector on the four target classes only (ADR-034, evaluation protocol P1).

[YOLO-DOC] ultralytics 8.3.253: ``Detect`` in eval mode returns ``(y, x)`` with ``y = [B, 4 + nc, anchors]``
(boxes, then sigmoid class scores); NMS keeps up to ``max_det`` = 300 boxes per image over ALL classes. With the
80-class COCO head, boxes of the 76 non-target classes compete for those 300 slots. A forward hook sets the
non-target scores to 0, so they fall below conf = 0.001 before NMS; target scores are untouched. The trained
weights are not changed — this is an evaluation setting, applied identically to every run it is used for.
"""
from __future__ import annotations

from typing import Any

import torch


def _hook(keep: torch.Tensor):
    def hook(module: torch.nn.Module, inputs: Any, output: Any) -> Any:
        if module.training:
            return output
        y, rest = (output[0], output[1:]) if isinstance(output, tuple) else (output, None)
        mask = torch.zeros(y.shape[1] - 4, dtype=torch.bool, device=y.device)
        mask[keep.to(y.device)] = True
        y = torch.cat([y[:, :4], y[:, 4:] * mask.to(y.dtype)[None, :, None]], 1)
        return (y, *rest) if rest is not None else y
    return hook


def keep_only_classes(model: Any, class_ids: list[int]) -> torch.utils.hooks.RemovableHandle:
    """Register the hook on the Detect layer of a YOLO object (or DetectionModel); returns the handle."""
    net = getattr(model, "model", model)
    net = getattr(net, "model", net)                  # YOLO -> DetectionModel -> Sequential
    detect = net[-1]
    return detect.register_forward_hook(_hook(torch.tensor(sorted(class_ids))))
