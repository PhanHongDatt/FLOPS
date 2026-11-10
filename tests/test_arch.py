"""ADR-032 architecture variant: gated CBAM on the Detect inputs of the COCO YOLO11 model."""
from __future__ import annotations

import pytest
import torch

from src.model.arch import cbam_yaml, split_arch


def test_split_arch():
    assert split_arch("coco:yolo11m.pt+cbam") == ("coco:yolo11m.pt", "cbam")
    assert split_arch("coco:yolo11m.pt") == ("coco:yolo11m.pt", None)
    with pytest.raises(ValueError):
        split_arch("coco:yolo11m.pt+p2")


def test_cbam_yaml_layout():
    base = {"backbone": [[-1, 1, "Conv", [64]]] * 11,
            "head": [[-1, 1, "C3k2", [256]]] * 12 + [[[16, 19, 22], 1, "Detect", ["nc"]]]}
    y = cbam_yaml(base, [128, 256, 512])
    assert y["head"][-4:-1] == [[16, 1, "GatedCBAM", [128]], [19, 1, "GatedCBAM", [256]],
                                [22, 1, "GatedCBAM", [512]]]
    assert y["head"][-1] == [[23, 24, 25], 1, "Detect", ["nc"]]
    assert base["head"][-1][0] == [16, 19, 22]                     # input untouched


@pytest.fixture(scope="module")
def models():
    from src.model.yolo_wrapper import build_model
    return build_model("coco:yolo11n.pt"), build_model("coco:yolo11n.pt+cbam")


def test_cbam_variant_equals_coco_model_at_init(models):
    base, var = models
    x = torch.rand(1, 3, 160, 160)
    base.model.eval(), var.model.eval()
    with torch.no_grad():
        a, b = base.model(x)[0], var.model(x)[0]
    assert torch.equal(a, b)
    nb = sum(p.numel() for p in base.model.parameters())
    nv = sum(p.numel() for p in var.model.parameters())
    assert 0 < nv - nb < 0.05 * nb
    assert var.model.model[-1].nc == 80 and list(var.model.model[-1].f) == [23, 24, 25]


def test_trainer_rebuild_keeps_variant_and_weights(models):
    """Ultralytics' trainer: DetectionModel(model.yaml).load(model) — must give the same network."""
    from ultralytics.nn.tasks import DetectionModel
    _, var = models
    rebuilt = DetectionModel(var.model.yaml, nc=80, verbose=False)
    rebuilt.load(var.model, verbose=False)
    a, b = var.model.state_dict(), rebuilt.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_gate_gets_gradient(models):
    _, var = models
    var.model.train()
    out = var.model(torch.rand(1, 3, 160, 160))
    sum(o.float().pow(2).mean() for o in out).backward()
    gates = [m.gamma for m in var.model.model if type(m).__name__ == "GatedCBAM"]
    assert len(gates) == 3 and all(g.grad is not None and g.grad.abs().item() > 0 for g in gates)
    var.model.zero_grad()


@pytest.mark.slow
def test_variant_survives_a_real_training_round(tmp_path):
    """Ultralytics rebuilds the model from model.yaml inside train(); the variant and its keys must survive,
    the gates must move off 0, and the trained parameters must load into a fresh variant (server model)."""
    import numpy as np
    import yaml
    from PIL import Image

    from src.data.coco_view import coco_names
    from src.model.yolo_wrapper import build_model, get_parameters, set_parameters, train_one_round
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
        for i in range(4):
            Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(tmp_path / "images" / split / f"{i}.jpg")
            (tmp_path / "labels" / split / f"{i}.txt").write_text(f"{(2, 3, 5, 7)[i]} 0.5 0.5 0.4 0.4\n")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({"path": str(tmp_path.resolve()), "train": "images/train",
                                         "val": "images/val", "nc": 80, "names": coco_names()}))
    client = build_model("coco:yolo11n.pt+cbam")
    keys = list(client.model.state_dict().keys())
    train_one_round(client, data_yaml, epochs=1, batch=2, img_size=64, lr0=0.01,
                    device="cpu", project=tmp_path / "runs", name="c0", workers=0, nbs=2)   # 1 step per batch
    assert list(client.model.state_dict().keys()) == keys
    gates = [float(m.gamma) for m in client.model.model if type(m).__name__ == "GatedCBAM"]
    assert len(gates) == 3 and any(g != 0.0 for g in gates)
    set_parameters(build_model("coco:yolo11n.pt+cbam"), get_parameters(client))
