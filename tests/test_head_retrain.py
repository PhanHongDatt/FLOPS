"""ADR-038 classification-branch retrain helpers."""
from __future__ import annotations

import numpy as np
import pytest

from src.model.head_retrain import cls_branch_freeze_list, merge_cls_branch


def test_freeze_list_leaves_only_cv3():
    fl = cls_branch_freeze_list(24)
    assert fl[:2] == ["0", "1"] and "22" in fl and "23.cv2" in fl and "23.dfl" in fl
    names = [f"model.{x}." for x in fl]
    assert not any(n in "model.23.cv3.0.2.weight" for n in names)
    assert any(n in "model.23.cv2.0.0.conv.weight" for n in names)
    assert any(n in "model.22.cv1.conv.weight" for n in names)
    assert not any(n in "model.23.cv3.1.0.0.bn.weight" for n in names)


def test_merge_keeps_cv3_restores_frozen_and_rejects_real_moves():
    keys = ["model.0.conv.weight", "model.23.cv3.0.2.weight", "model.23.cv2.0.2.bias"]
    b = [np.full(2, 3.0, np.float32), np.zeros(2, np.float32), np.ones(2, np.float32)]
    rounding = [b[0] + np.float32(2e-7), np.ones(2, np.float32), b[2]]
    merged, changed = merge_cls_branch(keys, b, rounding)
    assert changed == ["model.23.cv3.0.2.weight"]
    assert np.array_equal(merged[0], b[0]) and np.array_equal(merged[1], np.ones(2))
    with pytest.raises(ValueError):
        merge_cls_branch(keys, b, [b[0], b[1], b[2] + 0.01])
    with pytest.raises(ValueError):
        merge_cls_branch(keys, b, b[:2])


def test_real_model_freeze_names():
    from src.model.yolo_wrapper import build_model
    net = build_model("coco:yolo11s.pt").model
    fl = [f"model.{x}." for x in cls_branch_freeze_list(len(net.model))]
    trainable = [k for k, _ in net.named_parameters() if not any(f in k for f in fl + [".dfl"])]
    assert trainable and all(".cv3." in k for k in trainable)
    assert any(k.endswith("cv3.0.2.weight") for k in trainable)
