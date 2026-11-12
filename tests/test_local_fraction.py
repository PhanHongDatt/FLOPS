"""Direction A (ADR-035): a fraction of a local epoch per round."""
from __future__ import annotations

import pytest
import yaml

from src.federated.local_fraction import fraction_data_yaml, parts_for, round_part

IMAGES = [f"/d/images/train/{i:04d}.jpg" for i in range(1003)]


def test_parts_for_accepts_unit_fractions_only():
    assert parts_for(1.0) == 1 and parts_for(0.5) == 2 and parts_for(0.25) == 4
    for bad in (0.0, 1.5, 0.3, -0.25):
        with pytest.raises(ValueError):
            parts_for(bad)


def test_full_fraction_is_identity():
    assert round_part(IMAGES, 1.0, 7, 42, "C0") == IMAGES


def test_k_rounds_cover_every_image_exactly_once():
    for start in (1, 5, 9):                                       # passes 0, 1, 2
        parts = [round_part(IMAGES, 0.25, r, 42, "C0") for r in range(start, start + 4)]
        flat = [im for p in parts for im in p]
        assert sorted(flat) == sorted(IMAGES)                     # disjoint and complete
        assert max(map(len, parts)) - min(map(len, parts)) <= 1   # balanced


def test_deterministic_and_reshuffled_per_pass_and_client():
    a = round_part(IMAGES, 0.25, 1, 42, "C0")
    assert a == round_part(IMAGES, 0.25, 1, 42, "C0")
    assert a == round_part(list(reversed(IMAGES)), 0.25, 1, 42, "C0")   # input order does not matter
    assert a != round_part(IMAGES, 0.25, 5, 42, "C0")             # next pass, new permutation
    assert a != round_part(IMAGES, 0.25, 1, 42, "C1")
    assert a != round_part(IMAGES, 0.25, 1, 123, "C0")
    with pytest.raises(ValueError):
        round_part(IMAGES, 0.25, 0, 42, "C0")


def test_fraction_data_yaml_writes_the_part(tmp_path):
    lst = tmp_path / "train.txt"
    lst.write_text("\n".join(IMAGES) + "\n", encoding="utf-8")
    src = tmp_path / "data_C0.yaml"
    src.write_text(yaml.safe_dump({"path": str(tmp_path), "train": "train.txt", "val": "v.txt", "nc": 4}))
    out = fraction_data_yaml(src, tmp_path / "r3", 0.25, 3, 42, "C0")
    data = yaml.safe_load(out.read_text())
    got = [ln for ln in open(data["train"], encoding="utf-8").read().splitlines() if ln]
    assert got == round_part(IMAGES, 0.25, 3, 42, "C0") and data["val"] == "v.txt" and data["nc"] == 4
    assert lst.read_text(encoding="utf-8").count("\n") == len(IMAGES)    # source untouched


def test_client_trains_on_the_round_part_before_rfs(tmp_path):
    """YOLOFlowerClient._train_kwargs: the part is taken first, RFS resamples the part (ADR-035)."""
    from src.federated.client import YOLOFlowerClient
    imgs = []
    for i in range(40):
        (tmp_path / "images").mkdir(exist_ok=True)
        (tmp_path / "labels").mkdir(exist_ok=True)
        img = tmp_path / "images" / f"{i:03d}.jpg"
        imgs.append(str(img))
        (tmp_path / "labels" / f"{i:03d}.txt").write_text(("5" if i < 4 else "2") + " 0.5 0.5 0.2 0.2\n")
    (tmp_path / "train.txt").write_text("\n".join(imgs) + "\n", encoding="utf-8")
    src = tmp_path / "data_C0.yaml"
    src.write_text(yaml.safe_dump({"path": str(tmp_path), "train": "train.txt", "val": "train.txt", "nc": 80}))
    c = object.__new__(YOLOFlowerClient)
    c.train_config = {"lr0": 0.01, "local_fraction": 0.25, "rfs_t": 1.0, "local_epochs": 1, "batch_size": 8,
                      "image_size": 64, "device": "cpu"}
    c._round, c.base_seed, c.partition_id, c.client_id = 6, 42, 0, "C0"
    c.run_dir, c.data_yaml, c.model = tmp_path / "run", src, None
    kw = c._train_kwargs()
    used = [ln for ln in open(yaml.safe_load(open(kw["data_yaml"]))["train"], encoding="utf-8").read().splitlines() if ln]
    part = round_part(imgs, 0.25, 6, 42, "C0")
    assert set(used) == set(part) and len(used) >= len(part)            # RFS only repeats images of the part
