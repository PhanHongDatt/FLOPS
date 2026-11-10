"""Architecture variants of the COCO YOLO11 detector (ADR-032).

``cbam`` — a gated CBAM block on each of the three neck outputs that feed Detect (P3, P4, P5):

    y = x + gamma * (CBAM(x) - x),   gamma a learnable scalar per level, initialised to 0.

[LITERATURE] CBAM: Woo et al., ECCV 2018 (channel then spatial attention); the module is Ultralytics' own
``CBAM`` [YOLO-DOC 8.3.253 ``ultralytics.nn.modules.conv.CBAM``]. The zero-initialised gate is an
[ENGINEERING] choice: at initialisation the variant computes exactly the COCO model, so the pretrained head,
the not-true KD teacher (built from the initial weights) and round-0 metrics are unchanged; the blocks only
act as far as training opens the gate.

Layout `[YOLO-DOC]` (cfg/models/11/yolo11.yaml): layers 0-22 unchanged, three new layers 23-25 (one
GatedCBAM per Detect input 16 / 19 / 22), Detect moves from index 23 to 26 with inputs [23, 24, 25]. Only the
Detect keys are renamed (``model.23.`` -> ``model.26.``), so every COCO weight is loaded; the new layers are
the only parameters without pretrained values. Ultralytics' trainer rebuilds the model from ``model.yaml``;
``parse_model`` resolves module names in ``ultralytics.nn.tasks``, where ``register`` puts ``GatedCBAM``.
"""
from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn

ARCH_SEP = "+"
ARCHS = ("cbam",)


class GatedCBAM(nn.Module):
    def __init__(self, c1: int, kernel_size: int = 7) -> None:
        super().__init__()
        from ultralytics.nn.modules import CBAM
        self.cbam = CBAM(c1, kernel_size)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * (self.cbam(x) - x)


def register() -> None:
    """Make ``GatedCBAM`` resolvable by Ultralytics' ``parse_model`` (needed in every process that builds)."""
    import ultralytics.nn.tasks as tasks
    tasks.GatedCBAM = GatedCBAM


def split_arch(weights: str) -> tuple[str, str | None]:
    """``"coco:yolo11m.pt+cbam"`` -> ``("coco:yolo11m.pt", "cbam")``; no suffix -> arch ``None``."""
    base, sep, arch = weights.partition(ARCH_SEP)
    if not sep:
        return weights, None
    if arch not in ARCHS:
        raise ValueError(f"unknown architecture variant {arch!r} in {weights!r}; known: {ARCHS}")
    return base, arch


def cbam_yaml(base_yaml: dict[str, Any], channels: list[int]) -> dict[str, Any]:
    """Copy of a YOLO11 model yaml with a GatedCBAM on each Detect input; ``channels`` = their widths."""
    y = copy.deepcopy(base_yaml)
    head = y["head"]
    f, n, m, args = head[-1]
    if m != "Detect" or len(f) != 3:
        raise ValueError(f"expected a 3-level Detect as the last layer, got {head[-1]}")
    det_idx = len(y["backbone"]) + len(head) - 1
    y["head"] = head[:-1] + [[src, 1, "GatedCBAM", [c]] for src, c in zip(f, channels)] \
        + [[[det_idx, det_idx + 1, det_idx + 2], n, m, args]]
    return y


def build_cbam(base: nn.Module) -> nn.Module:
    """DetectionModel = ``base`` (a loaded YOLO11 DetectionModel) + GatedCBAM on the Detect inputs.

    Raises unless every ``base`` tensor lands in the new model with the same shape."""
    from ultralytics.nn.tasks import DetectionModel
    register()
    det = base.model[-1]
    det_idx = len(base.model) - 1
    channels = [int(seq[0].conv.in_channels) for seq in det.cv2]
    cfg = cbam_yaml(base.yaml, channels)
    new = DetectionModel(cfg, ch=int(base.yaml.get("ch", 3)), nc=int(det.nc), verbose=False)
    shift = len(channels)
    old_prefix, new_prefix = f"model.{det_idx}.", f"model.{det_idx + shift}."
    src = {(new_prefix + k[len(old_prefix):]) if k.startswith(old_prefix) else k: v
           for k, v in base.state_dict().items()}
    target = new.state_dict()
    bad = [k for k, v in src.items() if k not in target or target[k].shape != v.shape]
    if bad:
        raise ValueError(f"CBAM variant: {len(bad)} COCO tensors do not map, e.g. {bad[:3]}")
    missing = [k for k in target if k not in src]
    if any(".cbam." not in k and not k.endswith(".gamma") for k in missing):
        raise ValueError(f"CBAM variant: unexpected new tensors {missing[:3]}")
    new.load_state_dict({**target, **src}, strict=True)
    new.names = base.names
    for attr in ("args", "pt_path"):
        if hasattr(base, attr):
            setattr(new, attr, getattr(base, attr))
    return new
