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
