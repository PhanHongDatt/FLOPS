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
