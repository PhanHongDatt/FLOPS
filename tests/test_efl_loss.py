"""A6 / A6c — Equalized Focal Loss and plain focal loss on the YOLOv8 head (ADR-013)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.preservation.efl_loss import FocalClassLoss, _FocalBCE
from src.preservation.loss_plugins import build_loss_plugin


def _ultralytics_focal(pred, label, gamma=2.0, alpha=0.25):
    """ultralytics 8.3.253 utils/loss.py FocalLoss without its final reduction."""
    loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
    p = pred.sigmoid()
    p_t = label * p + (1 - label) * (1 - p)
    return loss * (1.0 - p_t) ** gamma * (label * alpha + (1 - label) * (1 - alpha))


def test_focal_matches_ultralytics_form_with_soft_targets():
    z = torch.randn(2, 5, 4)
    t = torch.rand(2, 5, 4) * (torch.rand(2, 5, 4) > 0.7)
    loss = _FocalBCE(4, 2.0, 0.25, 4.0, equalize=False)(z, t)
    assert torch.allclose(loss, _ultralytics_focal(z, t), atol=1e-6)


def test_efl_starts_as_focal():
    z, t = torch.randn(1, 6, 4), torch.zeros(1, 6, 4)
    efl = _FocalBCE(4, 2.0, 0.25, 4.0, equalize=True)
    assert torch.allclose(efl(z, t), _ultralytics_focal(z, t), atol=1e-6)


def test_class_without_positives_gets_max_focusing_and_weight():
    efl = _FocalBCE(4, 2.0, 0.25, 4.0, equalize=True)
    z = torch.zeros(1, 10, 4, requires_grad=True)
    t = torch.zeros(1, 10, 4)
    t[0, :5, 0] = 1.0                                  # car positives only; bus never positive
    efl(z, t).sum().backward()
    assert efl.pos_neg[1] == 0.0 and efl.pos_neg[0] > 0
    g = efl.class_gamma()
    assert g[1] == pytest.approx(6.0)                  # gamma_b + s * (1 - 0)
    assert g[0] < 6.0
    # a confident wrong bus score keeps a large loss; an easy negative almost none
    z2 = torch.tensor([[[0.0, 3.0, 0.0, 0.0], [0.0, -3.0, 0.0, 0.0]]])
    loss = efl(z2, torch.zeros_like(z2))
    assert loss[0, 0, 1] > 100 * loss[0, 1, 1]


def test_state_roundtrip_and_plugin_restores_bce(tmp_path):
    path = tmp_path / "C0" / "efl_state.json"
    crit = SimpleNamespace(bce=torch.nn.BCEWithLogitsLoss(reduction="none"), nc=4)
    model = SimpleNamespace(criterion=crit, init_criterion=lambda: crit)
    trainer = SimpleNamespace(model=model)
    hook = build_loss_plugin({"kind": "efl", "state_path": str(path)})
    hook.on_train_start(trainer)
    z = torch.zeros(1, 4, 4, requires_grad=True)
    t = torch.zeros(1, 4, 4)
    t[0, 0, 0] = 1.0
    crit.bce(z, t).sum().backward()
    hook.on_train_end(trainer)
    assert isinstance(crit.bce, torch.nn.BCEWithLogitsLoss)
    saved = json.loads(path.read_text())
    assert saved["pos_grad"][0] > 0 and saved["pos_grad"][1] == 0
    hook2 = build_loss_plugin({"kind": "efl", "state_path": str(path)})   # next round
    hook2.on_train_start(trainer)
    assert crit.bce.pos_grad[0] == pytest.approx(saved["pos_grad"][0])


def test_non_finite_gradients_are_not_accumulated():
    efl = _FocalBCE(2, 2.0, 0.25, 4.0, equalize=True)
    efl._collect(torch.tensor([[float("inf"), 1.0]]), torch.tensor([[1.0, 0.0]]))
    assert efl.pos_grad.sum() == 0


def test_factory_kinds():
    assert build_loss_plugin({"kind": "focal"}).equalize is False
    assert build_loss_plugin({"kind": "efl"}).equalize is True
    assert build_loss_plugin({"kind": "ntd"}).beta == 1.0
    with pytest.raises(ValueError):
        build_loss_plugin({"kind": "seesaw"})
    with pytest.raises(ValueError):
        FocalClassLoss(equalize=True, alpha=2.0)
