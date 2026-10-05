"""Dataset discovery must work for both Kaggle mount layouts."""
from __future__ import annotations

import pytest

from src.utils.kaggle_paths import find_bdd100k_root


def _make(root, rel: str):
    folder = root / rel
    (folder / "labels" / "det_20").mkdir(parents=True)
    (folder / "labels" / "det_20" / "det_train.json").write_text("[]")
    return folder


def test_web_editor_layout(tmp_path):
    expected = _make(tmp_path, "datasets/phdatt/bdd100k-flops/bdd100k_kaggle")
    assert find_bdd100k_root(tmp_path) == expected


def test_api_push_layout(tmp_path):
    expected = _make(tmp_path, "bdd100k-flops/bdd100k_kaggle")
    assert find_bdd100k_root(tmp_path) == expected


def test_folder_without_labels_is_ignored(tmp_path):
    (tmp_path / "other" / "bdd100k_kaggle").mkdir(parents=True)
    expected = _make(tmp_path, "bdd100k-flops/bdd100k_kaggle")
    assert find_bdd100k_root(tmp_path) == expected


def test_missing_dataset_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="bdd100k-flops"):
        find_bdd100k_root(tmp_path)


def _g2_zip(path, rel="artifacts/runs/G2-centralized_feasibility_seed42_centralized/checkpoint/train/weights/best.pt"):
    import zipfile
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(rel, b"weights")
    return path


def test_find_g2_weights_inside_attached_results_zip(tmp_path):
    from src.utils.kaggle_paths import find_g2_weights
    _g2_zip(tmp_path / "input" / "flops-s2-baseline-g2-g3" / "flops_results.zip")
    found = find_g2_weights(input_root=tmp_path / "input", search_roots=[], extract_to=tmp_path / "x")
    assert found is not None and found.read_bytes() == b"weights"
    assert found.name == "best.pt"


def test_find_g2_weights_prefers_live_run(tmp_path):
    from src.utils.kaggle_paths import find_g2_weights
    live = tmp_path / "runs" / "G2-centralized_feasibility_seed42_centralized" / "checkpoint" / "train" / "weights"
    live.mkdir(parents=True)
    (live / "best.pt").write_bytes(b"live")
    _g2_zip(tmp_path / "input" / "k" / "flops_results.zip")
    found = find_g2_weights(input_root=tmp_path / "input", search_roots=[tmp_path / "runs"],
                            extract_to=tmp_path / "x")
    assert found.read_bytes() == b"live"


def test_find_g2_weights_none(tmp_path):
    from src.utils.kaggle_paths import find_g2_weights
    assert find_g2_weights(input_root=tmp_path / "nope", search_roots=[], extract_to=tmp_path / "x") is None


def test_public_mirror_is_shimmed_into_the_expected_layout(tmp_path):
    import os

    import pytest

    from src.utils.kaggle_paths import shim_public_bdd100k

    try:
        os.symlink(tmp_path, tmp_path / "probe", target_is_directory=True)
    except OSError:
        pytest.skip("symlinks need extra privileges on this Windows account (Kaggle runs Linux)")

    mirror = tmp_path / "input" / "solesensei_bdd100k"
    (mirror / "bdd100k" / "bdd100k" / "images" / "100k" / "train").mkdir(parents=True)
    (mirror / "bdd100k" / "bdd100k" / "images" / "100k" / "val").mkdir(parents=True)
    labels = mirror / "bdd100k_labels_release" / "bdd100k" / "labels"
    labels.mkdir(parents=True)
    for split in ("train", "val"):
        (labels / f"bdd100k_labels_images_{split}.json").write_text("[]")
    root = shim_public_bdd100k(tmp_path / "input", shim_root=tmp_path / "shim")
    assert (root / "labels" / "det_20" / "det_train.json").read_text() == "[]"
    assert (root / "images" / "100k" / "train").is_dir()


def test_public_mirror_is_flattened_and_must_be_complete(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    import pytest

    from src.utils.kaggle_paths import shim_public_bdd100k

    mirror = tmp_path / "input" / "solesensei_bdd100k"
    imgs = mirror / "bdd100k" / "bdd100k" / "images" / "100k"
    for sub, name in (("train", "a.jpg"), ("train/trainA", "b.jpg"), ("train/testB", "c.jpg"), ("val", "v.jpg")):
        (imgs / sub).mkdir(parents=True, exist_ok=True)
        (imgs / sub / name).write_bytes(b"x")
    labels = mirror / "bdd100k_labels_release" / "bdd100k" / "labels"
    labels.mkdir(parents=True)
    (labels / "bdd100k_labels_images_train.json").write_text(json.dumps([{"name": n} for n in "a.jpg b.jpg c.jpg".split()]))
    (labels / "bdd100k_labels_images_val.json").write_text(json.dumps([{"name": "v.jpg"}]))
    links = {}

    def fake_symlink(self, target, target_is_directory=False):
        links[self.relative_to(self.parents[len(self.relative_to(tmp_path).parts) - 2])] = target
        self.write_bytes(b"")                       # so exists() is true afterwards

    monkeypatch.setattr(Path, "symlink_to", fake_symlink)
    root = shim_public_bdd100k(tmp_path / "input", shim_root=tmp_path / "shim")
    k = Path("bdd100k_kaggle")
    assert root == tmp_path / "shim" / k
    assert links[k / "images" / "100k" / "train" / "b.jpg"] == imgs / "train" / "trainA" / "b.jpg"
    assert links[k / "images" / "100k" / "train" / "c.jpg"] == imgs / "train" / "testB" / "c.jpg"
    assert links[k / "labels" / "det_20" / "det_val.json"] == labels / "bdd100k_labels_images_val.json"

    (labels / "bdd100k_labels_images_train.json").write_text(json.dumps([{"name": "zzz.jpg"}]))
    with pytest.raises(FileNotFoundError, match="lacks 1 of 1"):
        shim_public_bdd100k(tmp_path / "input", shim_root=tmp_path / "shim2")
    assert shim_public_bdd100k(tmp_path / "nothing", shim_root=tmp_path / "s3") is None
