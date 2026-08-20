"""Tests for parameter mapping (F1 support) — Section 18, CLAUDE.md."""
from __future__ import annotations

import pytest

from src.model.parameter_map import (
    BACKBONE_INDICES,
    DETECT_INDEX,
    NECK_INDICES,
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


def test_only_cv3_is_class_specific():
    assert is_class_specific("model.22.cv3.2.conv.weight") is True
    assert is_class_specific("model.22.cv2.2.conv.weight") is False
    assert is_class_specific("model.22.dfl.conv.weight") is False
    assert is_class_specific("model.0.conv.weight") is False
