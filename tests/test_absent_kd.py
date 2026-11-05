"""B1 — absent-class distillation on confusable foreground anchors (ADR-028, docs/Solution.md §3.5)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.preservation.absent_kd import AbsentKDBCE, absent_class_targets, load_absent_kd_config
from src.preservation.ntd_loss import NotTrueDistillation, _NTDBCE, _TeacherCriterion

CAR, MOTO, BUS, TRUCK = 2, 3, 5, 7
CONF = {BUS: (TRUCK, CAR), TRUCK: (BUS, CAR)}


def _targets():
    """[1, 5, 8]: anchor 0 truck, 1 car, 2 motorcycle, 3 bus, 4 background."""
    t = torch.zeros(1, 5, 8)
    t[0, 0, TRUCK], t[0, 1, CAR], t[0, 2, MOTO], t[0, 3, BUS] = 0.6, 0.9, 0.4, 0.7
    return t


def test_no_missing_class_returns_identical_targets():                      # test 1
    t = _targets()
    out = absent_class_targets(t, torch.rand(1, 5, 8), [], CONF, 1.0, 0.5)
    assert torch.equal(out, t)


def test_only_confusable_foreground_entries_of_missing_class_change():       # tests 2-4
    t, q = _targets(), torch.full((1, 5, 8), 0.8)
    out = absent_class_targets(t, q, [BUS], CONF, 1.0, 0.5)
    changed = (out != t).nonzero().tolist()
    assert changed == [[0, 0, BUS], [0, 1, BUS]]                 # truck and car anchors, bus column only
    assert out[0, 0, TRUCK] == t[0, 0, TRUCK] and out[0, 1, CAR] == t[0, 1, CAR]   # assigned class kept
    assert torch.isclose(out[0, 0, BUS], torch.tensor(0.5 * 0.6))                 # lam * min(q, q_max) * s
    assert torch.isclose(out[0, 1, BUS], torch.tensor(0.5 * 0.9))
    assert out[0, 4].sum() == 0                                   # background untouched
    new = out != t
    assert torch.all(out[new] <= 1.0 * 0.5 * t.amax(-1, keepdim=True).expand_as(t)[new] + 1e-7)
    assert torch.equal(t, _targets())                             # input not modified


def test_all_anchors_scope_reaches_background():
    t, q = _targets(), torch.full((1, 5, 8), 0.3)
    out = absent_class_targets(t, q, [BUS], CONF, 0.5, 0.5, scope="all_anchors")
    assert torch.isclose(out[0, 4, BUS], torch.tensor(0.15))
    assert torch.isclose(out[0, 2, BUS], torch.tensor(0.15 * 0.4))
    with pytest.raises(ValueError):
        absent_class_targets(t, q, [BUS], CONF, 1.0, 0.5, scope="everywhere")


# ── real ultralytics 8.3.253 criterion (YOLO11n from yaml, 80 classes, CPU, no download) ──
def _model():
    from ultralytics.nn.tasks import DetectionModel
    torch.manual_seed(0)
    m = DetectionModel("yolo11n.yaml", nc=80, verbose=False)
    m.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    return m


def _batch():
    return {"img": torch.rand(2, 3, 64, 64),
            "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
            "cls": torch.tensor([[float(TRUCK)], [float(CAR)], [float(CAR)]]),
            "bboxes": torch.tensor([[0.3, 0.3, 0.4, 0.4], [0.7, 0.7, 0.3, 0.3], [0.5, 0.5, 0.5, 0.5]])}


def _loss(model, batch):
    model.train()
    return model.criterion(model(batch["img"]), batch)


def test_disabled_equals_baseline_and_normaliser_is_unchanged():              # tests 5-6
    model = _model()
    model.criterion = model.init_criterion()
    batch = _batch()
    base_total, base_items = _loss(model, batch)
    trainer = SimpleNamespace(model=model)
    assert NotTrueDistillation(beta=0.0, absent=None).active is False       # disabled: nothing installed
    assert NotTrueDistillation(beta=0.0, absent={"missing": []}).active is False

    sums = []
    plain = model.criterion.bce

    class Spy(torch.nn.Module):                                             # targets seen by the cls term
        def forward(self, logits, target):
            sums.append(float(target.sum()))
            return plain(logits, target)

    spy = Spy()

    on = NotTrueDistillation(beta=0.0, absent={"missing": [BUS], "confusable": CONF, "lam": 1.0, "q_max": 0.5})
    on.on_train_start(trainer)
    assert isinstance(model.criterion, _TeacherCriterion) and isinstance(model.criterion.bce, AbsentKDBCE)
    model.criterion.bce.inner = spy
    _, items = _loss(model, batch)
    model.criterion.bce.missing = []                                        # client without absent class
    total_none, _ = _loss(model, batch)
    on.on_train_end(trainer)
    assert model.criterion.bce is plain
    assert sums[0] > sums[1]                       # B1 added soft bus targets on the truck/car anchors
    # box and DFL terms identical: their weights and target_scores_sum come from the unmodified targets
    assert torch.allclose(items[0], base_items[0], atol=1e-6) and torch.allclose(items[2], base_items[2], atol=1e-6)
    assert items[1] != base_items[1]
    assert torch.allclose(total_none, base_total, atol=1e-6)


def test_absent_kd_raises_cls_loss_target_and_teacher_is_frozen():          # test 7
    model = _model()
    model.criterion = model.init_criterion()
    trainer = SimpleNamespace(model=model)
    hook = NotTrueDistillation(beta=1.0, absent={"missing": [BUS], "confusable": CONF, "lam": 1.0, "q_max": 0.5})
    hook.on_train_start(trainer)
    crit = model.criterion
    assert isinstance(crit.bce, AbsentKDBCE) and isinstance(crit.bce.inner, _NTDBCE)   # B1_add chain
    teacher = crit.teacher
    before = {k: v.clone() for k, v in teacher.state_dict().items()}
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    total, _ = _loss(model, _batch())
    total.sum().backward()
    opt.step()
    after = teacher.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)       # weights and BN statistics
    assert not teacher.training and not any(p.requires_grad for p in teacher.parameters())
    assert crit.bce.teacher_logits is None and crit.bce.inner.teacher_logits is None
    hook.on_train_end(trainer)


def test_config_file_defaults_and_overrides():
    cfg = load_absent_kd_config()
    assert cfg["enabled"] is False and cfg["lam"] == 1.0 and cfg["q_max"] == 0.5
    assert cfg["confusable"] == {BUS: [TRUCK, CAR], TRUCK: [BUS, CAR]} and cfg["scope"] == "confusable"
    assert load_absent_kd_config({"enabled": True, "q_max": 1.0})["q_max"] == 1.0
    with pytest.raises(ValueError):
        load_absent_kd_config({"scope": "x"})
    with pytest.raises(ValueError):
        AbsentKDBCE(torch.nn.BCEWithLogitsLoss(reduction="none"), [BUS], CONF, 1.0, 0.0)


def test_client_spec_maps_missing_classes_to_coco_ids():
    from src.federated.client_variants import AbsentKDClient
    c = object.__new__(AbsentKDClient)
    c.train_config = {"teacher_params": "t.npz", "kd_beta": 0.0,
                      "absent_kd": load_absent_kd_config({"enabled": True})}
    c.missing_classes = ["bus"]
    spec = c._loss_spec()
    assert spec["kind"] == "ntd" and spec["beta"] == 0.0 and spec["teacher_path"] == "t.npz"
    assert spec["absent"]["missing"] == [BUS] and spec["absent"]["q_max"] == 0.5
    c.missing_classes = []                                    # C4: no absent class -> no B1, plain P4t1 loss
    assert "absent" not in c._loss_spec()
    c.train_config = {**c.train_config, "absent_kd": load_absent_kd_config()}   # enabled: false
    with pytest.raises(ValueError):
        c._loss_spec()
