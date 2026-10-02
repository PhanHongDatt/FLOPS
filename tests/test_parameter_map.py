"""Tests for parameter mapping (F1 support) — Section 18, CLAUDE.md."""
from __future__ import annotations

import pytest

from src.model.parameter_map import (
    BACKBONE_INDICES,
    DETECT_INDEX,
    NECK_INDICES,
    build_parameter_map,
    class_head_keys,
    classify_parameter,
    is_class_specific,
)


@pytest.mark.parametrize("name,expected", [
    ("model.0.conv.weight", "backbone"),
    ("model.9.cv1.conv.weight", "backbone"),
    ("model.10.weight", "neck"),
    ("model.21.cv3.bn.bias", "neck"),
    ("model.22.cv2.0.conv.weight", "detect_reg"),
    ("model.22.cv3.0.conv.weight", "detect_cls"),
    ("model.22.dfl.conv.weight", "detect_dfl"),
    ("model.22.stride", "detect_other"),
    ("not_a_model.weight", "unknown"),
])
def test_classify_parameter(name, expected):
    assert classify_parameter(name) == expected


def test_backbone_indices_contiguous():
    assert BACKBONE_INDICES == tuple(range(0, 10))


def test_neck_indices_contiguous():
    assert NECK_INDICES == tuple(range(10, 22))


def test_detect_index_is_22():
    assert DETECT_INDEX == 22


# Key names below are the runtime state_dict keys of YOLOv8n at nc=4,
# ultralytics 8.3.253 (F1 runtime check, 2026-10-02). The previous version of
# this test used "model.22.cv3.2.conv.weight", a key that does not exist.
@pytest.mark.parametrize("name", [
    "model.22.cv3.0.2.weight", "model.22.cv3.0.2.bias",
    "model.22.cv3.1.2.weight", "model.22.cv3.1.2.bias",
    "model.22.cv3.2.2.weight", "model.22.cv3.2.2.bias",
])
def test_final_cv3_conv_is_class_specific(name):
    assert is_class_specific(name) is True


@pytest.mark.parametrize("name", [
    # cv3.<s>.0 / cv3.<s>.1 are Conv+BN blocks shared by every class:
    # they are in the classification branch but are NOT class-specific.
    "model.22.cv3.0.0.conv.weight",
    "model.22.cv3.1.1.bn.weight",
    "model.22.cv3.2.1.bn.running_mean",
    "model.22.cv2.2.2.weight",
    "model.22.dfl.conv.weight",
    "model.0.conv.weight",
])
def test_shared_parameters_are_not_class_specific(name):
    assert is_class_specific(name) is False


def test_build_parameter_map_accepts_detection_model_and_wrapper():
    """A bare DetectionModel yields keys without the "model." prefix.

    Before the fix every key was then classified "unknown" and the cls-head
    regex matched zero keys, silently.
    """
    import torch.nn as nn

    class FakeDetectionModel(nn.Module):  # mirrors DetectionModel.model = Sequential
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Sequential(*[nn.Identity() for _ in range(22)], nn.Conv2d(2, 4, 1))

    class FakeWrapper:  # mirrors ultralytics.YOLO: .model is the DetectionModel
        def __init__(self, dm: nn.Module) -> None:
            self.model = dm

    dm = FakeDetectionModel()
    from_dm = [p.name for p in build_parameter_map(dm)]
    from_wrapper = [p.name for p in build_parameter_map(FakeWrapper(dm))]
    assert from_dm == from_wrapper == ["model.22.weight", "model.22.bias"]


def test_class_head_keys_requires_exactly_six():
    names = [f"model.22.cv3.{s}.2.{k}" for s in range(3) for k in ("weight", "bias")]
    shapes = [(4, 64, 1, 1), (4,)] * 3
    assert class_head_keys(names, shapes, nc=4) == names
    with pytest.raises(ValueError, match="exactly 6"):
        class_head_keys(names[:4], shapes[:4], nc=4)
    with pytest.raises(ValueError, match="exactly 6"):
        class_head_keys(names, [(80, 64, 1, 1), (80,)] * 3, nc=4)
