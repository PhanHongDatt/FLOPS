"""ADR-037 class-row transplant."""
from __future__ import annotations

import numpy as np
import pytest

from src.model.row_transplant import final_cls_conv_keys, transplant_row

KEYS = ["model.0.conv.weight", "model.23.cv3.0.2.weight", "model.23.cv3.0.2.bias", "model.23.cv3.0.1.conv.weight",
        "model.23.cv3.1.2.weight", "model.23.cv3.1.2.bias", "model.23.cv2.0.2.weight"]


def _params(fill: float) -> list[np.ndarray]:
    shapes = [(4, 3), (80, 8, 1, 1), (80,), (8, 8), (80, 8, 1, 1), (80,), (64, 8, 1, 1)]
    return [np.full(s, fill, dtype=np.float32) for s in shapes]


def test_only_the_class_row_of_the_final_cls_conv_moves():
    rec, don = _params(0.0), _params(1.0)
    out = transplant_row(KEYS, rec, don, class_id=5)
    for k, o in zip(KEYS, out):
        if k in final_cls_conv_keys(KEYS):
            assert o[5].min() == o[5].max() == 1.0
            assert np.delete(o, 5, axis=0).max() == 0.0
        else:
            assert o.max() == 0.0
    assert all(r.max() == 0.0 for r in rec)          # inputs are not mutated


def test_keys_and_shape_checks():
    assert final_cls_conv_keys(KEYS) == ["model.23.cv3.0.2.weight", "model.23.cv3.0.2.bias",
                                         "model.23.cv3.1.2.weight", "model.23.cv3.1.2.bias"]
    with pytest.raises(ValueError):
        transplant_row(KEYS, _params(0.0)[:-1], _params(1.0), 5)
    bad = _params(1.0)
    bad[2] = np.zeros(4, dtype=np.float32)
    with pytest.raises(ValueError):
        transplant_row(KEYS, _params(0.0), bad, 5)
    with pytest.raises(ValueError):
        final_cls_conv_keys(["model.0.conv.weight"])


def test_real_model_keys():
    from src.model.yolo_wrapper import build_model
    keys = list(build_model("coco:yolo11s.pt").model.state_dict())
    assert len(final_cls_conv_keys(keys)) == 6
