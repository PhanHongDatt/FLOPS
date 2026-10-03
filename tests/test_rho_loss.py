"""A2b — loss-level preservation: BCE of locally-missing classes scaled by rho.

[YOLO-DOC] ultralytics 8.3.253 v8DetectionLoss: ``self.bce = BCEWithLogitsLoss(reduction="none")``
applied to pred_scores [B, anchors, nc]; DetectionModel.loss creates ``self.criterion``
lazily via ``init_criterion()``.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.preservation.rho_loss import RhoClassLoss, class_loss_weights

CLASSES = ("car", "bus", "truck", "motorcycle")


def test_weights_vector():
    assert class_loss_weights(CLASSES, ["bus"], 0.25) == [1.0, 0.25, 1.0, 1.0]
    assert class_loss_weights(CLASSES, [], 0.0) == [1.0, 1.0, 1.0, 1.0]
    with pytest.raises(ValueError):
        class_loss_weights(CLASSES, ["bus"], 1.5)
    with pytest.raises(ValueError, match="person"):
        class_loss_weights(CLASSES, ["person"], 0.5)


def _trainer():
    criterion = SimpleNamespace(bce=torch.nn.BCEWithLogitsLoss(reduction="none"))
    model = SimpleNamespace(criterion=criterion, init_criterion=lambda: criterion)
    return SimpleNamespace(model=model)


def test_bce_of_missing_class_is_scaled():
    trainer = _trainer()
    hook = RhoClassLoss([1.0, 0.0, 1.0, 1.0])
    hook.on_train_start(trainer)
    logits = torch.zeros(2, 3, 4, requires_grad=True)
    target = torch.zeros(2, 3, 4)
    loss = trainer.model.criterion.bce(logits, target)
    assert loss.shape == (2, 3, 4)
    assert torch.all(loss[..., 1] == 0)                 # bus channel removed (rho = 0)
    assert torch.all(loss[..., 0] > 0)
    loss.sum().backward()
    assert torch.all(logits.grad[..., 1] == 0)          # no "there is no bus" pressure


def test_criterion_created_lazily_and_restored():
    criterion = SimpleNamespace(bce=torch.nn.BCEWithLogitsLoss(reduction="none"))
    model = SimpleNamespace(criterion=None, init_criterion=lambda: criterion)
    trainer = SimpleNamespace(model=model)
    hook = RhoClassLoss([1.0, 0.5, 1.0, 1.0])
    hook.on_train_start(trainer)
    assert trainer.model.criterion is criterion
    hook.on_train_end(trainer)
    assert isinstance(criterion.bce, torch.nn.BCEWithLogitsLoss)   # original restored


def test_all_ones_is_inactive():
    assert RhoClassLoss([1.0, 1.0, 1.0, 1.0]).active is False
