"""A5 — not-true distillation adapted to the YOLOv8 sigmoid head (ADR-012)."""
from __future__ import annotations

import pytest
import torch

from src.preservation.ntd_loss import NotTrueDistillation, _NTDBCE, _TeacherCriterion, bernoulli_kl, head_scores


def test_bernoulli_kl_zero_iff_same_and_positive_otherwise():
    z = torch.tensor([-3.0, 0.0, 2.0])
    assert torch.allclose(bernoulli_kl(z, z, 1.0), torch.zeros(3), atol=1e-6)
    assert torch.all(bernoulli_kl(z, z + 1.0, 1.0) > 0)
    # tau^2 scaling of the temperature-softened KL
    assert torch.allclose(bernoulli_kl(z, z + 1, 2.0), 4 * bernoulli_kl(z / 2, (z + 1) / 2, 1.0), atol=1e-6)


def test_kd_only_on_not_true_entries_and_pulls_toward_teacher():
    bce = _NTDBCE(torch.nn.BCEWithLogitsLoss(reduction="none"), beta=1.0, tau=1.0)
    logits = torch.zeros(1, 2, 4, requires_grad=True)
    target = torch.zeros(1, 2, 4)
    target[0, 0, 0] = 0.8                                  # anchor 0 is a car (true class)
    bce.teacher_logits = torch.full((1, 2, 4), 2.0)       # teacher: confident on everything
    plain = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    loss = bce(logits, target)
    assert torch.allclose(loss[0, 0, 0], plain[0, 0, 0])             # true entry: no KD
    assert torch.all(loss[0, 1] > plain[0, 1])                       # not-true entries: KD added
    loss.sum().backward()
    # target 0 alone pushes the logit down (grad +0.5); the confident teacher pulls it up
    # harder (grad -(s(2) - s(0)) = -0.38), so the net gradient shrinks but stays positive
    assert 0 < logits.grad[0, 1, 1] < 0.5


def test_no_teacher_means_plain_bce():
    bce = _NTDBCE(torch.nn.BCEWithLogitsLoss(reduction="none"), 1.0, 1.0)
    z, t = torch.randn(2, 3, 4), torch.zeros(2, 3, 4)
    assert torch.allclose(bce(z, t), torch.nn.functional.binary_cross_entropy_with_logits(z, t, reduction="none"))


def test_head_scores_layout_matches_v8_loss():
    nc, reg_max = 4, 16
    no = nc + 4 * reg_max
    feats = [torch.randn(2, no, 8, 8), torch.randn(2, no, 4, 4)]
    s = head_scores(feats, no, reg_max, nc)
    assert s.shape == (2, 64 + 16, nc)
    assert torch.equal(s[1, 0], feats[0][1, 4 * reg_max:, 0, 0])
    assert torch.equal(s[1, 64], feats[1][1, 4 * reg_max:, 0, 0])


class _Head(torch.nn.Module):
    """Tiny 'detector': one feature map; eval mode returns (y, [x]) like Detect."""

    def __init__(self, nc=4, reg_max=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, nc + 4 * reg_max, 1)

    def forward(self, img):
        x = self.conv(img)
        return x if self.training else (None, [x])


class _Crit:
    def __init__(self):
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.nc, self.reg_max, self.no = 4, 1, 8
        self.seen = None

    def __call__(self, preds, batch):
        scores = head_scores([preds], self.no, self.reg_max, self.nc)
        target = torch.zeros_like(scores)
        out = self.bce(scores, target)
        teacher = getattr(self.bce, "teacher_logits", None)
        self.seen = None if teacher is None else teacher.clone()
        return out.sum()


def test_callbacks_snapshot_teacher_and_restore():
    head = _Head()
    crit = _Crit()
    head.criterion, head.init_criterion = crit, lambda: crit
    trainer = type("T", (), {"model": head})()
    hook = NotTrueDistillation(beta=1.0, tau=1.0)
    hook.on_train_start(trainer)
    assert isinstance(head.criterion, _TeacherCriterion)
    with torch.no_grad():                                  # local training moves the student...
        head.conv.weight.add_(1.0)
    img = torch.randn(2, 3, 4, 4)
    head.criterion(head(img), {"img": img})
    teacher = head.criterion.teacher
    assert not any(p.requires_grad for p in teacher.parameters())
    assert not torch.equal(teacher.conv.weight, head.conv.weight)   # ...the teacher stays the global model
    assert crit.seen is not None and crit.seen.shape == (2, 16, 4)
    assert crit.bce.teacher_logits is None                 # cleared after each batch
    hook.on_train_end(trainer)
    assert head.criterion is crit and isinstance(crit.bce, torch.nn.BCEWithLogitsLoss)


def test_validation():
    with pytest.raises(ValueError):
        NotTrueDistillation(beta=-1)
    with pytest.raises(ValueError):
        NotTrueDistillation(tau=0)
    assert NotTrueDistillation(beta=0).active is False
