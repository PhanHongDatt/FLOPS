"""Tests for H2 / A2a parameter-level preservation (CLAUDE.md §18 Preservation).

Required cases: rho=1 equals a normal update; rho=0 preserves ONLY the validated
protected group; shared parameters unaffected unless explicitly intended.

Note on scope: these criteria apply to A2a (parameter-level). They deliberately
do NOT apply to A2b (loss-level), which changes shared parameters by design —
see research/plan/plan.md §6.1. A2b will need its own criteria.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.data.bdd100k import TARGET_CLASSES
from src.preservation.mask import (
    apply_preservation_mask,
    maskable_params,
)

NC = len(TARGET_CLASSES)
CAR, BUS, TRUCK, MOTO = TARGET_CLASSES
BUS_IDX = TARGET_CLASSES.index(BUS)

PARAM_NAMES = [
    "model.0.conv.weight",          # backbone — shared
    "model.22.cv2.0.2.weight",      # box regression — shared across classes
    "model.22.cv3.0.0.conv.weight",  # intermediate cv3 conv — shared
    "model.22.cv3.0.2.weight",      # class head (nc, 1, 1, 1)
    "model.22.cv3.0.2.bias",        # class head (nc,)
]


def _params(fill: float) -> list[np.ndarray]:
    return [
        np.full((2, 2), fill, dtype=np.float32),
        np.full((4, 2), fill, dtype=np.float32),
        np.full((3, 3), fill, dtype=np.float32),
        np.full((NC, 1, 1, 1), fill, dtype=np.float32),
        np.full((NC,), fill, dtype=np.float32),
    ]


def test_maskable_params_only_final_cv3():
    shapes = [p.shape for p in _params(0.0)]
    assert maskable_params(PARAM_NAMES, shapes) == [
        "model.22.cv3.0.2.weight",
        "model.22.cv3.0.2.bias",
    ]


def test_rho_one_is_identity():
    local, global_ = _params(5.0), _params(1.0)
    out = apply_preservation_mask(local, global_, PARAM_NAMES, [BUS], rho=1.0)
    for o, l in zip(out, local):
        assert o == pytest.approx(l)


def test_no_missing_classes_is_identity():
    local, global_ = _params(5.0), _params(1.0)
    out = apply_preservation_mask(local, global_, PARAM_NAMES, [], rho=0.0)
    for o, l in zip(out, local):
        assert o == pytest.approx(l)


def test_rho_zero_restores_global_only_for_missing_class_rows():
    local, global_ = _params(5.0), _params(1.0)
    out = apply_preservation_mask(local, global_, PARAM_NAMES, [BUS], rho=0.0)

    # Protected rows reverted to global...
    assert out[3][BUS_IDX] == pytest.approx(global_[3][BUS_IDX])
    assert out[4][BUS_IDX] == pytest.approx(global_[4][BUS_IDX])
    # ...other class rows in the SAME tensor keep the local update.
    for idx, name in enumerate(TARGET_CLASSES):
        if idx == BUS_IDX:
            continue
        assert out[3][idx] == pytest.approx(local[3][idx]), name
        assert out[4][idx] == pytest.approx(local[4][idx]), name


def test_rho_zero_leaves_shared_params_untouched():
    local, global_ = _params(5.0), _params(1.0)
    out = apply_preservation_mask(local, global_, PARAM_NAMES, [BUS], rho=0.0)
    for i in (0, 1, 2):   # backbone, cv2 regression, intermediate cv3
        assert out[i] == pytest.approx(local[i]), PARAM_NAMES[i]


def test_rho_partial_blends_towards_global():
    local, global_ = _params(5.0), _params(1.0)
    out = apply_preservation_mask(local, global_, PARAM_NAMES, [BUS], rho=0.25)
    # 1 + 0.25*(5-1) = 2.0
    assert out[4][BUS_IDX] == pytest.approx(2.0)


def test_multiple_missing_classes():
    local, global_ = _params(5.0), _params(1.0)
    out = apply_preservation_mask(local, global_, PARAM_NAMES, [BUS, MOTO], rho=0.0)
    for cls in (BUS, MOTO):
        assert out[4][TARGET_CLASSES.index(cls)] == pytest.approx(1.0)
    for cls in (CAR, TRUCK):
        assert out[4][TARGET_CLASSES.index(cls)] == pytest.approx(5.0)


def test_unknown_class_name_is_ignored():
    local, global_ = _params(5.0), _params(1.0)
    out = apply_preservation_mask(local, global_, PARAM_NAMES, ["bicycle"], rho=0.0)
    for o, l in zip(out, local):
        assert o == pytest.approx(l)


def test_input_arrays_are_not_mutated():
    local, global_ = _params(5.0), _params(1.0)
    local_copy = [p.copy() for p in local]
    apply_preservation_mask(local, global_, PARAM_NAMES, [BUS], rho=0.0)
    for p, q in zip(local, local_copy):
        assert p == pytest.approx(q)
