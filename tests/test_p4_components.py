"""ADR-019 — P4 components: cosine round learning rate, client-side repeat-factor sampling,
and the hardened Ultralytics label cache."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from src.federated.client import cosine_round_lr
from src.preservation.rfs import repeat_factors, resample, rfs_data_yaml


def test_cosine_round_lr_endpoints_and_monotone():
    lrs = [cosine_round_lr(r, 20, 0.01, 0.0005) for r in range(1, 21)]
    assert lrs[0] == pytest.approx(0.01) and lrs[-1] == pytest.approx(0.0005)
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))
    assert cosine_round_lr(10, 1, 0.01, 0.0005) == 0.01


def test_repeat_factors_match_detectron2_formula():
    # 10 images: class 0 in all, class 1 in 1 image (f = 0.1), class 2 in 5 (f = 0.5); t = 0.3
    cls = [{0}] * 4 + [{0, 2}] * 5 + [{0, 1}]
    r = repeat_factors(cls, 0.3)
    assert r[0] == 1.0                                         # f = 1 -> max(1, sqrt(0.3)) = 1
    assert r[4] == pytest.approx(1.0)                          # sqrt(0.3 / 0.5) < 1 -> 1
    assert r[-1] == pytest.approx(math.sqrt(0.3 / 0.1))        # rarest class decides
    assert repeat_factors([set(), {0}], 0.5)[0] == 1.0        # no labels -> 1
    with pytest.raises(ValueError):
        repeat_factors(cls, 0.0)


def test_resample_is_seeded_and_unbiased_on_average():
    imgs = [f"i{k}" for k in range(1000)]
    a = resample(imgs, [1.7] * 1000, seed=3)
    assert a == resample(imgs, [1.7] * 1000, seed=3)
    assert 1600 < len(a) < 1800                                  # E = 1.7 per image
    assert resample(["x"], [1.0], seed=0) == ["x"]


def test_rfs_yaml_repeats_rare_images_only(tmp_path):
    root = tmp_path / "yolo"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    names = []
    for i in range(20):
        n = f"im{i:02d}.jpg"
        (root / "images" / "train" / n).write_bytes(b"x")
        cls = "3" if i == 0 else "2"                               # one rare image
        (root / "labels" / "train" / f"im{i:02d}.txt").write_text(f"{cls} 0.5 0.5 0.1 0.1")
        names.append(str(root / "images" / "train" / n))
    lst = tmp_path / "C0_train.txt"
    lst.write_text("\n".join(names) + "\n")
    dy = tmp_path / "data_C0.yaml"
    dy.write_text(yaml.safe_dump({"path": str(root), "train": str(lst), "val": "images/val", "nc": 80}))
    out = yaml.safe_load(rfs_data_yaml(dy, tmp_path / "rfs", t=0.3, seed=1).read_text())
    picked = [ln for ln in Path(out["train"]).read_text().splitlines() if ln]
    # rare image: f = 0.05 -> r = sqrt(6) ~ 2.45 -> 2 or 3 copies; common: f = 0.95 -> r = 1
    assert picked.count(names[0]) in (2, 3)
    assert all(picked.count(n) == 1 for n in names[1:])
    assert out["val"] == "images/val" and out["nc"] == 80


def test_label_cache_read_errors_become_rebuilds(tmp_path):
    import ultralytics.data.dataset as ds

    from src.model.yolo_wrapper import _harden_label_cache

    bad = tmp_path / "train.cache"
    bad.write_bytes(b"=corrupt")
    _harden_label_cache()
    _harden_label_cache()                                          # idempotent
    with pytest.raises(FileNotFoundError):
        ds.load_dataset_cache_file(bad)
    with pytest.raises(FileNotFoundError):
        ds.load_dataset_cache_file(tmp_path / "missing.cache")


def test_p4_presets():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.run_fl_experiment import _ABLATIONS, LR_MIN_FACTOR
    assert _ABLATIONS["P3LR"]["lr_cosine"] and "rfs_t" not in _ABLATIONS["P3LR"]
    assert _ABLATIONS["P4"]["rfs_t"] == 0.5 and _ABLATIONS["P4"]["lr_cosine"]
    assert LR_MIN_FACTOR == 0.05
