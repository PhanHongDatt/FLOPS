"""Per-round global checkpointing and resume (research/plan/plan.md §13 K4/K5).

Kaggle sessions are capped at 12 h and can be interrupted at any point, so a run
that can only be executed start-to-finish is not usable for Tier 1. This module
keeps the mechanics numpy-only so it is unit-testable on a machine with no GPU,
no torch and no ultralytics.

Layout inside a run directory::

    <run_dir>/checkpoint/global_round_001.npz
    <run_dir>/checkpoint/global_round_002.npz
    ...
    <run_dir>/checkpoint/final_params.npz      # written at the last round

Each ``global_round_NNN.npz`` holds the aggregated global parameters AFTER that
round, in state_dict order (``arr_0``, ``arr_1``, ...).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

CHECKPOINT_PATTERN = re.compile(r"^global_round_(\d+)\.npz$")
CHECKPOINT_TEMPLATE = "global_round_{round:03d}.npz"


def save_global_checkpoint(
    ckpt_dir: Path,
    round_idx: int,
    parameters: list[np.ndarray],
) -> Path:
    """Write the global parameters for ``round_idx`` and return the path."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / CHECKPOINT_TEMPLATE.format(round=int(round_idx))
    # The temp name must itself end in ".npz": np.savez appends ".npz" to any
    # path that does not, which would leave the real temp file somewhere else
    # and make the rename below fail.
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, *parameters)
    tmp.replace(path)          # atomic-ish: never leave a half-written ckpt
    return path


def list_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    """Return [(round, path), ...] sorted ascending by round."""
    if not ckpt_dir.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for p in ckpt_dir.iterdir():
        m = CHECKPOINT_PATTERN.match(p.name)
        if m and p.is_file():
            found.append((int(m.group(1)), p))
    return sorted(found, key=lambda t: t[0])


def find_latest_checkpoint(ckpt_dir: Path) -> tuple[int, Path] | None:
    """Highest completed round and its checkpoint, or None if there is none."""
    found = list_checkpoints(ckpt_dir)
    return found[-1] if found else None


def load_global_checkpoint(path: Path) -> list[np.ndarray]:
    """Load parameters back in their saved order."""
    with np.load(path) as data:
        # np.savez names entries arr_0..arr_N; sort numerically, not lexically,
        # otherwise arr_10 would come before arr_2 and the whole state_dict
        # would be silently permuted.
        keys = sorted(data.files, key=lambda k: int(k.split("_")[-1]))
        return [data[k] for k in keys]


def prune_checkpoints(ckpt_dir: Path, keep_last: int) -> list[Path]:
    """Delete all but the ``keep_last`` most recent round checkpoints.

    ``final_params.npz`` is never touched (it does not match the pattern).
    ``keep_last <= 0`` disables pruning. Returns the deleted paths.
    """
    if keep_last is None or keep_last <= 0:
        return []
    found = list_checkpoints(ckpt_dir)
    removed: list[Path] = []
    for _, path in found[:-keep_last] if len(found) > keep_last else []:
        path.unlink(missing_ok=True)
        removed.append(path)
    return removed


def prune_client_round_weights(run_dir: Path, round_idx: int) -> list[Path]:
    """Delete a round's per-client Ultralytics weight files after aggregation.

    Ultralytics writes ``last.pt`` + ``best.pt`` per client per round under
    ``<run_dir>/clients/<cid>/round_<n>/weights/``. At 4 clients x 10 rounds that
    is roughly 0.5 GB per run, against a 20 GB ``/kaggle/working`` budget
    (plan.md §13 K5). The aggregated global checkpoint is what the experiment
    needs; these per-round client weights are not reused after aggregation.
    Logs and metrics under the same directories are left in place.
    """
    removed: list[Path] = []
    clients_dir = run_dir / "clients"
    if not clients_dir.is_dir():
        return removed
    for client_dir in clients_dir.iterdir():
        weights_dir = client_dir / f"round_{int(round_idx)}" / "weights"
        if not weights_dir.is_dir():
            continue
        for pt in weights_dir.glob("*.pt"):
            pt.unlink(missing_ok=True)
            removed.append(pt)
    return removed
