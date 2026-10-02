"""Locate the BDD100K Kaggle dataset regardless of how it was mounted.

The web editor mounts private datasets at /kaggle/input/datasets/<owner>/<slug>/,
while notebooks pushed with ``kaggle kernels push`` may see /kaggle/input/<slug>/.
Notebooks call :func:`find_bdd100k_root` instead of hard-coding either layout.
"""
from __future__ import annotations

from pathlib import Path

DATASET_FOLDER = "bdd100k_kaggle"
_MARKER = Path("labels") / "det_20" / "det_train.json"


def find_bdd100k_root(input_root: Path = Path("/kaggle/input")) -> Path:
    """Return the ``bdd100k_kaggle`` folder that holds ``labels/det_20/det_train.json``.

    Raises FileNotFoundError listing what was searched when nothing matches.
    """
    patterns = (
        f"datasets/*/*/{DATASET_FOLDER}",   # web editor: datasets/<owner>/<slug>/
        f"*/{DATASET_FOLDER}",              # API / classic: <slug>/
        DATASET_FOLDER,
    )
    for pattern in patterns:
        for candidate in sorted(input_root.glob(pattern)):
            if (candidate / _MARKER).exists():
                return candidate
    raise FileNotFoundError(
        f"No {DATASET_FOLDER}/{_MARKER.as_posix()} under {input_root} "
        f"(searched {', '.join(patterns)}). Attach the bdd100k-flops dataset."
    )
