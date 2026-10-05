"""ADR-015 — server-held labelled sample: selection, fixed-teacher KD, server fine-tune."""
from __future__ import annotations

import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.data.bdd100k import TARGET_CLASSES
from src.data.partitioner import partition_server_sample
from src.preservation.ntd_loss import NotTrueDistillation, _NTDBCE, load_ordered_state
from src.preservation.rho_loss import _WeightedBCE
from src.utils.kaggle_paths import find_output_file

CAR, BUS, TRUCK, MOTO = TARGET_CLASSES
IDX = {c: i for i, c in enumerate(TARGET_CLASSES)}


def _labels(tmp_path, spec):
    d = tmp_path / "labels"
    d.mkdir()
    for name, classes in spec.items():
        (d / (Path(name).stem + ".txt")).write_text("\n".join(f"{IDX[c]} 0.5 0.5 0.1 0.1" for c in classes))
    return sorted(spec), d


def test_server_sample_is_disjoint_seeded_and_covers_all_classes(tmp_path):
    spec = {f"img{i:03d}.jpg": [CAR] + ([BUS] if i % 5 == 0 else []) + ([TRUCK] if i % 7 == 0 else [])
            + ([MOTO] if i % 11 == 0 else []) for i in range(300)}
    names, labels = _labels(tmp_path, spec)
    clients = set(names[:100])
    m = partition_server_sample(names, labels, clients, n_images=120, seed=42, partition_id="srv")
    sample = m.client_assignments["S"]
    assert len(sample) == 120 and not set(sample) & clients
    assert all(m.class_counts["S"][c] > 0 for c in TARGET_CLASSES)
    again = partition_server_sample(names, labels, clients, n_images=120, seed=42, partition_id="srv")
    assert again.client_assignments == m.client_assignments
    other = partition_server_sample(names, labels, clients, n_images=120, seed=43, partition_id="srv")
    assert other.client_assignments != m.client_assignments


def test_server_sample_without_a_class_fails_loudly(tmp_path):
    names, labels = _labels(tmp_path, {f"img{i}.jpg": [CAR] for i in range(50)})
    with pytest.raises(ValueError, match="lacks classes"):
        partition_server_sample(names, labels, set(), n_images=20, seed=1, partition_id="srv")
    with pytest.raises(ValueError, match="outside the clients"):
        partition_server_sample(names, labels, set(names), n_images=20, seed=1, partition_id="srv")


def _save_state(module, path):
    np.savez(path, *[v.detach().cpu().numpy() for v in module.state_dict().values()])


def test_load_ordered_state_roundtrip_and_shape_check(tmp_path):
    src, dst = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
    _save_state(src, tmp_path / "t.npz")
    load_ordered_state(dst, tmp_path / "t.npz")
    assert torch.equal(src.weight, dst.weight)
    with pytest.raises(ValueError):
        load_ordered_state(torch.nn.Linear(4, 2), tmp_path / "t.npz")


class _Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1)

    def forward(self, img):
        x = self.conv(img)
        return x if self.training else (None, [x])


def test_fixed_teacher_is_loaded_and_rho_weights_sit_under_the_kd(tmp_path):
    fixed = _Head()
    torch.nn.init.constant_(fixed.conv.weight, 0.5)
    _save_state(fixed, tmp_path / "teacher.npz")
    student = _Head()
    crit = SimpleNamespace(bce=torch.nn.BCEWithLogitsLoss(reduction="none"), nc=4, reg_max=1, no=8)
    student.criterion, student.init_criterion = crit, lambda: crit
    hook = NotTrueDistillation(teacher_path=tmp_path / "teacher.npz", class_weights=[1, 0.25, 1, 1])
    hook.on_train_start(SimpleNamespace(model=student))
    teacher = student.criterion.teacher
    assert torch.equal(teacher.conv.weight, fixed.conv.weight)          # the FIXED teacher, not the student
    assert not torch.equal(teacher.conv.weight, student.conv.weight)
    assert isinstance(crit.bce, _NTDBCE) and isinstance(crit.bce.base, _WeightedBCE)
    hook.on_train_end(None)
    assert isinstance(crit.bce, torch.nn.BCEWithLogitsLoss) and student.criterion is crit


def test_find_output_file_in_session_zip(tmp_path):
    src = tmp_path / "out"
    f = src / "artifacts" / "runs" / "T-teacher_feasibility_seed42_x" / "checkpoint" / "teacher.npz"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x")
    (tmp_path / "input" / "s8t").mkdir(parents=True)
    with zipfile.ZipFile(tmp_path / "input" / "s8t" / "flops_results.zip", "w") as z:
        z.write(f, f.relative_to(src).as_posix())
    got = find_output_file("artifacts/runs/T-teacher_*/checkpoint/teacher.npz",
                           input_root=tmp_path / "input", extract_to=tmp_path / "x")
    assert got is not None and got.read_bytes() == b"x"
    assert find_output_file("nothing/*.npz", input_root=tmp_path / "input", extract_to=tmp_path / "x") is None


def test_server_finetune_wraps_aggregation(tmp_path, monkeypatch):
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

    import src.model.yolo_wrapper as yw
    from src.federated.server import _wrap_server_finetune

    calls = []
    model = SimpleNamespace(params=None)
    monkeypatch.setattr(yw, "build_model", lambda w: model)
    monkeypatch.setattr(yw, "set_parameters", lambda m, p: setattr(m, "params", [a.copy() for a in p]))
    monkeypatch.setattr(yw, "get_parameters", lambda m: m.params)

    def fake_train(model, **kw):
        calls.append(kw)
        model.params = [a + 1 for a in model.params]

    monkeypatch.setattr(yw, "train_one_round", fake_train)
    strategy = SimpleNamespace(aggregate_fit=lambda r, res, fail: (ndarrays_to_parameters([np.zeros(2)]), {}))
    _wrap_server_finetune(strategy, {"data_yaml": "s.yaml", "batch_size": 16, "image_size": 640, "lr0": 0.01,
                                     "seed": 42}, tmp_path, round_offset=4, weights="w")
    params, metrics = strategy.aggregate_fit(3, [], [])
    assert parameters_to_ndarrays(params)[0].tolist() == [1.0, 1.0]     # fine-tuned params are returned
    assert metrics["server_finetune"] == 1
    assert calls[0]["name"] == "round_7" and calls[0]["epochs"] == 1 and calls[0]["workers"] == 0
