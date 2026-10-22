"""Continue a run that a previous Kaggle session cut at its time budget (ADR-024).

The earlier session's output is attached as a kernel source and mounted under /kaggle/input, either as
``flops_results.zip`` (``kaggle_finalize``) or already unpacked. ``restore_run`` copies that run directory
back into the local runs directory, so ``run_fl_experiment.py --resume`` starts from its latest checkpoint.

Standard library only.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ARCHIVE = "flops_results.zip"


def _checkpoints(names: list[str], prefix: str) -> list[str]:
    return [n for n in names if n.startswith(prefix + "checkpoint/global_round_") and n.endswith(".npz")]


def restore_run(run_id: str, runs_dir: Path, input_root: Path = Path("/kaggle/input")) -> Path | None:
    """Copy ``run_id`` from an earlier session's output into ``runs_dir``.

    Does nothing (returns None) when the run already exists locally or no source holds a checkpoint of it.
    Among several sources the one with the most checkpointed rounds wins, so a later part is preferred."""
    dest = Path(runs_dir) / run_id
    if dest.exists():
        return None
    best: tuple[int, str, Path] | None = None
    prefix = f"artifacts/runs/{run_id}/"
    for d in sorted(Path(input_root).glob(f"**/artifacts/runs/{run_id}")):
        n = max((int(p.stem.split("_")[-1]) for p in d.glob("checkpoint/global_round_*.npz")), default=0)
        if n and (best is None or n > best[0]):
            best = (n, "dir", d)
    for z in sorted(Path(input_root).glob(f"**/{ARCHIVE}")):
        with zipfile.ZipFile(z) as zf:
            ck = _checkpoints(zf.namelist(), prefix)
        n = max((int(Path(c).stem.split("_")[-1]) for c in ck), default=0)
        if n and (best is None or n > best[0]):
            best = (n, "zip", z)
    if best is None:
        return None
    _, kind, src = best
    if kind == "dir":
        shutil.copytree(src, dest)
    else:
        with zipfile.ZipFile(src) as zf:
            for name in zf.namelist():
                if name.startswith(prefix) and not name.endswith("/"):
                    out = dest / name[len(prefix):]
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(zf.read(name))
    return dest
