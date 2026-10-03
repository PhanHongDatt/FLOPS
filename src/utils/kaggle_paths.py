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


_G2_REL = "G2-centralized_*/checkpoint/train/weights/best.pt"


def find_g2_weights(
    input_root: Path = Path("/kaggle/input"),
    search_roots: list[Path] | None = None,
    extract_to: Path = Path("/kaggle/tmp/inputs"),
) -> Path | None:
    """Newest G2 checkpoint: a live run under ``search_roots`` first, then attached outputs.

    Outputs of earlier sessions (``kernel_sources``) arrive as ``flops_results.zip``
    (src/utils/kaggle_finalize.py); the G2 run is extracted from it on demand.
    """
    import zipfile

    for root in search_roots or []:
        hits = sorted(Path(root).glob(_G2_REL), key=lambda p: p.stat().st_mtime)
        if hits:
            return hits[-1]
    if not Path(input_root).exists():
        return None
    loose = sorted(Path(input_root).glob(f"**/{_G2_REL}"))
    if loose:
        return loose[-1]
    for k, archive in enumerate(sorted(Path(input_root).glob("**/flops_results.zip"))):
        with zipfile.ZipFile(archive) as z:
            members = [n for n in z.namelist()
                       if n.startswith("artifacts/runs/G2-centralized_")
                       and n.endswith("/checkpoint/train/weights/best.pt")]
            if members:
                dest = Path(extract_to) / f"zip{k}"
                z.extract(members[-1], dest)
                return dest / members[-1]
    return None
