"""Tests for per-round checkpointing and resume (plan.md §13 K4/K5).

Resume is what makes a Tier-1 run survive Kaggle's 12 h session cap, so its
mechanics are tested without needing a GPU.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.experiments.checkpoint import (
    find_latest_checkpoint,
    list_checkpoints,
    load_global_checkpoint,
    prune_checkpoints,
    prune_client_round_weights,
    save_global_checkpoint,
)
from src.experiments.runner import default_run_id


def _params(n_arrays: int = 3, fill: float = 1.0) -> list[np.ndarray]:
    return [np.full((2, 2), fill + i, dtype=np.float32) for i in range(n_arrays)]


def test_save_and_load_roundtrip(tmp_path):
    params = _params()
    path = save_global_checkpoint(tmp_path, 3, params)
    assert path.name == "global_round_003.npz"
    loaded = load_global_checkpoint(path)
    assert len(loaded) == len(params)
    for a, b in zip(loaded, params):
        assert a == pytest.approx(b)


def test_load_preserves_order_beyond_ten_arrays(tmp_path):
    """np.savez names entries arr_0..arr_N; lexical sort would put arr_10 before
    arr_2 and silently permute the whole state_dict."""
    params = [np.full((1,), float(i), dtype=np.float32) for i in range(12)]
    path = save_global_checkpoint(tmp_path, 1, params)
    loaded = load_global_checkpoint(path)
    assert [float(a[0]) for a in loaded] == [float(i) for i in range(12)]


def test_find_latest_checkpoint_picks_highest_round(tmp_path):
    for r in (1, 2, 10, 9):
        save_global_checkpoint(tmp_path, r, _params(fill=float(r)))
    found = find_latest_checkpoint(tmp_path)
    assert found is not None
    round_idx, path = found
    assert round_idx == 10
    assert path.name == "global_round_010.npz"


def test_find_latest_checkpoint_on_empty_dir(tmp_path):
    assert find_latest_checkpoint(tmp_path) is None
    assert find_latest_checkpoint(tmp_path / "missing") is None


def test_final_params_file_is_not_treated_as_a_round_checkpoint(tmp_path):
    save_global_checkpoint(tmp_path, 1, _params())
    np.savez(tmp_path / "final_params.npz", *_params())
    assert [r for r, _ in list_checkpoints(tmp_path)] == [1]


def test_prune_keeps_last_n(tmp_path):
    for r in range(1, 6):
        save_global_checkpoint(tmp_path, r, _params())
    removed = prune_checkpoints(tmp_path, keep_last=2)
    assert len(removed) == 3
    assert [r for r, _ in list_checkpoints(tmp_path)] == [4, 5]


def test_prune_disabled_keeps_everything(tmp_path):
    for r in range(1, 4):
        save_global_checkpoint(tmp_path, r, _params())
    assert prune_checkpoints(tmp_path, keep_last=0) == []
    assert len(list_checkpoints(tmp_path)) == 3


def test_prune_never_deletes_final_params(tmp_path):
    for r in range(1, 4):
        save_global_checkpoint(tmp_path, r, _params())
    final = tmp_path / "final_params.npz"
    np.savez(final, *_params())
    prune_checkpoints(tmp_path, keep_last=1)
    assert final.exists()


def test_prune_client_round_weights_only_targets_that_round(tmp_path):
    run_dir = tmp_path / "run"
    for cid in ("C0", "C1"):
        for rnd in (1, 2):
            w = run_dir / "clients" / cid / f"round_{rnd}" / "weights"
            w.mkdir(parents=True)
            (w / "last.pt").write_bytes(b"x")
            (w / "best.pt").write_bytes(b"x")
            (w.parent / "results.csv").write_text("keep me", encoding="utf-8")

    removed = prune_client_round_weights(run_dir, 1)
    assert len(removed) == 4                      # 2 clients x 2 files
    assert not list((run_dir / "clients" / "C0" / "round_1" / "weights").glob("*.pt"))
    assert list((run_dir / "clients" / "C0" / "round_2" / "weights").glob("*.pt"))
    # Metrics next to the deleted weights survive.
    assert (run_dir / "clients" / "C0" / "round_1" / "results.csv").exists()


def test_prune_client_round_weights_on_missing_dirs_is_noop(tmp_path):
    assert prune_client_round_weights(tmp_path, 1) == []


def test_default_run_id_is_deterministic_and_has_no_timestamp():
    a = default_run_id("A3", "feasibility", 42, "s1_mc_seed42")
    b = default_run_id("A3", "feasibility", 42, "s1_mc_seed42")
    assert a == b == "A3_feasibility_seed42_s1_mc_seed42"


def test_default_run_id_sanitises_path_separators():
    rid = default_run_id("A3", "main", 42, "scen/a\\b")
    assert "/" not in rid and "\\" not in rid
