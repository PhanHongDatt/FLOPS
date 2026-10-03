"""build_model must return a 4-class detector — regression test for the nc=80/nc=4 crash.

Before the fix, ``build_model("yolov8n.pt")`` returned the 80-class COCO model.
Clients become nc=4 after their first ``model.train(data=<4-class yaml>)``, but
the server's evaluate model stayed nc=80, so ``set_parameters`` raised
"size mismatch for model.22.cv3.0.0.conv.weight" at the first post-round eval
(reproduced on CPU, 2026-10-02). The whole cv3 branch differs, not only the
class rows: Detect uses c3 = max(ch[0], min(nc, 100)) hidden channels.

These tests need the real Ultralytics package and are skipped without it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ultralytics")

import torch  # noqa: E402

from src.data.bdd100k import TARGET_CLASSES
from src.model.yolo_wrapper import (
    build_model,
    get_parameters,
    set_parameters,
    train_one_round,
)

# yolov8n.yaml builds without any download; its weights are random, which is
# exactly why build_model must seed construction for the determinism test.
SOURCE = "yolov8n.yaml"
NC = len(TARGET_CLASSES)


def _state(model) -> dict[str, np.ndarray]:
    return {k: v.cpu().numpy() for k, v in model.model.state_dict().items()}


def test_model_has_target_class_count():
    model = build_model(SOURCE)
    assert model.model.model[-1].nc == NC
    state = _state(model)
    for scale in range(3):
        assert state[f"model.22.cv3.{scale}.2.weight"].shape[0] == NC
        assert state[f"model.22.cv3.{scale}.2.bias"].shape == (NC,)


def test_names_are_target_classes():
    model = build_model(SOURCE)
    assert [model.model.names[i] for i in range(NC)] == list(TARGET_CLASSES)


def test_construction_is_deterministic():
    """Every Ray actor builds its own model; the re-initialised head must agree."""
    a, b = _state(build_model(SOURCE)), _state(build_model(SOURCE))
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


def test_init_seed_changes_head_only_when_source_is_fixed():
    a, b = _state(build_model(SOURCE, init_seed=0)), _state(build_model(SOURCE, init_seed=1))
    assert any(not np.array_equal(a[k], b[k]) for k in a)


def test_shape_matched_weights_are_transferred(monkeypatch):
    """Backbone/neck/cv2 come from the source checkpoint (Ultralytics intersect_dicts)."""
    import torch
    from ultralytics import YOLO
    with torch.random.fork_rng():
        torch.manual_seed(0)
        source = YOLO(SOURCE)
    src_state = {k: v.cpu().numpy() for k, v in source.model.state_dict().items()}
    built = _state(build_model(SOURCE, init_seed=0))
    for key in ("model.0.conv.weight", "model.15.cv1.conv.weight", "model.22.cv2.0.2.weight"):
        np.testing.assert_array_equal(built[key], src_state[key], err_msg=key)


def test_server_model_accepts_client_parameters():
    """The exact crash: server-built model must load a client's parameters."""
    server = build_model(SOURCE)
    client = build_model(SOURCE)
    set_parameters(server, get_parameters(client))


def _tiny_dataset(root: Path) -> Path:
    import yaml
    from PIL import Image

    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(4):
            img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(img).save(root / "images" / split / f"{i}.jpg")
            (root / "labels" / split / f"{i}.txt").write_text(f"{i % NC} 0.5 0.5 0.4 0.4\n")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(root.resolve()), "train": "images/train", "val": "images/val",
        "nc": NC, "names": list(TARGET_CLASSES),
    }))
    return data_yaml


@pytest.mark.slow
def test_trained_client_parameters_load_into_fresh_server_model(tmp_path):
    data_yaml = _tiny_dataset(tmp_path / "ds")
    client = build_model(SOURCE)
    keys_before = list(client.model.state_dict().keys())
    train_one_round(
        client, data_yaml, epochs=1, batch=2, img_size=64, lr0=0.01,
        device="cpu", project=tmp_path / "runs", name="c0", workers=0,
    )
    assert list(client.model.state_dict().keys()) == keys_before
    set_parameters(build_model(SOURCE), get_parameters(client))


@pytest.mark.slow
def test_trained_parameters_are_fp32_ema_not_fp16_checkpoint(tmp_path):
    """Ultralytics reloads best.pt after train(); that checkpoint stores the EMA
    in fp16 (trainer.py save_model: ``deepcopy(ema).half()``). Every client
    update was therefore rounded to fp16 each round, and in F3 the class-row
    Δθ of a short local run was pure rounding noise (identical for both sides
    of the matched pair and for every seed — observed 2026-10-02).
    """
    data_yaml = _tiny_dataset(tmp_path / "ds")
    model = build_model(SOURCE)
    before = {k: v.clone() for k, v in model.model.state_dict().items()}
    # nbs == batch → accumulate = 1, so the optimizer actually steps on 2 batches
    train_one_round(
        model, data_yaml, epochs=1, batch=2, img_size=64, lr0=0.01, device="cpu",
        project=tmp_path / "runs", name="c0", workers=0, nbs=2,
    )
    after = model.model.state_dict()
    ema = model.trainer.ema.ema.state_dict()
    for k, v in after.items():
        assert v.dtype == ema[k].dtype and (v == ema[k]).all(), k
    w = after["model.0.conv.weight"]
    assert not (w == before["model.0.conv.weight"]).all(), "optimizer never stepped"
    assert not (w == w.half().float()).all(), "weights are fp16-rounded"


@pytest.mark.slow
def test_evaluate_does_not_fuse_the_callers_model(tmp_path):
    """Ultralytics val() fuses Conv+BN in place (355 → 127 state_dict entries).
    The server reuses one evaluate model across rounds, so the next
    set_parameters crashed with a parameter-count mismatch."""
    from src.model.yolo_wrapper import evaluate

    data_yaml = _tiny_dataset(tmp_path / "ds")
    model = build_model(SOURCE)
    params = get_parameters(model)
    evaluate(model, data_yaml, img_size=64, conf=0.25, iou=0.7, device="cpu")
    assert len(model.model.state_dict()) == len(params)
    set_parameters(model, params)


@pytest.mark.slow
def test_weights_survive_in_place_half_of_ema_during_final_val(tmp_path):
    """On CUDA with AMP the validator runs ``trainer.ema.ema.half()`` in place on
    the final epoch (engine/validator.py), then ``.float()`` — fp16-rounding the
    EMA before train() returns. CPU never does this, so simulate it right after
    the final validation and require that the returned weights are unrounded."""
    data_yaml = _tiny_dataset(tmp_path / "ds")
    model = build_model(SOURCE)

    def amp_like_round_trip(trainer) -> None:
        trainer.ema.ema.half().float()

    model.add_callback("on_fit_epoch_end", amp_like_round_trip)
    train_one_round(
        model, data_yaml, epochs=1, batch=2, img_size=64, lr0=0.01, device="cpu",
        project=tmp_path / "runs", name="c0", workers=0, nbs=2,
    )
    w = model.model.state_dict()["model.0.conv.weight"]
    assert not (w == w.half().float()).all(), "weights are fp16-rounded"


@pytest.mark.slow
def test_local_training_validates_on_stub_not_global_val(tmp_path):
    data_yaml = _tiny_dataset(tmp_path / "ds")
    model = build_model(SOURCE)
    train_one_round(
        model, data_yaml, epochs=1, batch=2, img_size=64, lr0=0.01, device="cpu",
        project=tmp_path / "runs", name="c0", workers=0, val_stub_images=2,
    )
    val = Path(model.trainer.data["val"])
    assert val.name == "val_stub.txt"
    assert len(val.read_text().split()) == 2


@pytest.mark.slow
def test_confidence_stats_follow_weight_changes(tmp_path):
    """YOLO.predict caches its predictor with the model it first saw; without a
    reset, stats after a weight change would still describe the old weights."""
    from src.model.yolo_wrapper import confidence_stats

    data_yaml = _tiny_dataset(tmp_path / "ds")
    images = sorted(str(p) for p in (tmp_path / "ds" / "images" / "val").iterdir())
    model = build_model(SOURCE)
    # make every class fire: raise all class-head biases
    params = get_parameters(model)
    names = list(model.model.state_dict().keys())
    hot = [p + 20.0 if n.endswith(".2.bias") and ".cv3." in n else p for n, p in zip(names, params)]
    set_parameters(model, hot)
    stats_hot = confidence_stats(model, images, img_size=64, conf=0.25, iou=0.7, device="cpu")
    assert set(stats_hot) == {f"{s}_{c}" for c in TARGET_CLASSES for s in ("n_pred", "mean_conf")}
    assert sum(stats_hot[f"n_pred_{c}"] for c in TARGET_CLASSES) > 0

    cold = [p - 40.0 if n.endswith(".2.bias") and ".cv3." in n else p for n, p in zip(names, hot)]
    set_parameters(model, cold)
    stats_cold = confidence_stats(model, images, img_size=64, conf=0.25, iou=0.7, device="cpu")
    assert sum(stats_cold[f"n_pred_{c}"] for c in TARGET_CLASSES) == 0
    assert len(model.model.state_dict()) == len(names)  # not fused in place


@pytest.mark.slow
def test_fedprox_term_changes_the_update_in_real_training(tmp_path):
    """Same seed/data: mu > 0 must pull weights toward the start point (smaller drift)."""
    data_yaml = _tiny_dataset(tmp_path / "ds")
    start = {k: v.clone() for k, v in build_model(SOURCE).model.state_dict().items()}

    def drift(mu: float, name: str) -> float:
        model = build_model(SOURCE)
        train_one_round(model, data_yaml, epochs=1, batch=2, img_size=64, lr0=0.01,
                        device="cpu", project=tmp_path / "runs", name=name, workers=0,
                        nbs=2, proximal_mu=mu)
        sd = model.model.state_dict()
        return sum(float((sd[k].float() - start[k].float()).pow(2).sum())
                   for k in sd if k.endswith("conv.weight")) ** 0.5

    d0, d_big = drift(0.0, "mu0"), drift(50.0, "mu50")
    assert d0 > 0
    assert d_big < d0          # the proximal term really acts on the trainer's params


@pytest.mark.slow
def test_rho_zero_freezes_the_missing_class_row_in_real_training(tmp_path):
    """A2b: with rho = 0 for bus, the bus class-head bias gets no gradient (biases
    have no weight decay), so it is bit-identical after training; car's moves."""
    data_yaml = _tiny_dataset(tmp_path / "ds")
    model = build_model(SOURCE)
    keys = [f"model.22.cv3.{s}.2.bias" for s in range(3)]
    before = {k: model.model.state_dict()[k].clone() for k in keys}
    train_one_round(model, data_yaml, epochs=1, batch=2, img_size=64, lr0=0.01, device="cpu",
                    project=tmp_path / "runs", name="rho0", workers=0, nbs=2,
                    cls_loss_weights=[1.0, 0.0, 1.0, 1.0])
    after = model.model.state_dict()
    bus, car = TARGET_CLASSES.index("bus"), TARGET_CLASSES.index("car")
    for k in keys:
        assert torch.equal(after[k][bus], before[k][bus]), k
    assert any(not torch.equal(after[k][car], before[k][car]) for k in keys)


def test_ultralytics_mlflow_autologging_is_disabled():
    """Ultralytics' own MLflow callback logged best.pt + last.pt for every client of every
    round (s5b: 714 .pt files, 46k files, 4.2 GB in mlruns). Results of record are our CSVs."""
    from ultralytics.utils import SETTINGS

    build_model(SOURCE)
    assert SETTINGS["mlflow"] is False
