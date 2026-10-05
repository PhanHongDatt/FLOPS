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
    shim = shim_public_bdd100k(input_root)      # public mirror attached instead (other accounts)
    if shim is not None and (shim / _MARKER).exists():
        return shim
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


def find_output_file(
    pattern: str,
    input_root: Path = Path("/kaggle/input"),
    extract_to: Path = Path("/kaggle/tmp/inputs"),
) -> Path | None:
    """A file from an earlier session's output (``kernel_sources``), loose or inside
    ``flops_results.zip``. ``pattern`` is an fnmatch pattern on the path relative to the
    output root, e.g. ``artifacts/runs/T-teacher_*/checkpoint/teacher.npz``. Newest name wins.
    """
    import fnmatch
    import zipfile

    if not Path(input_root).exists():
        return None
    loose = sorted(Path(input_root).glob(f"**/{pattern}"))
    if loose:
        return loose[-1]
    for k, archive in enumerate(sorted(Path(input_root).glob("**/flops_results.zip"))):
        with zipfile.ZipFile(archive) as z:
            members = sorted(n for n in z.namelist() if fnmatch.fnmatch(n, pattern))
            if members:
                dest = Path(extract_to) / f"out{k}"
                z.extract(members[-1], dest)
                return dest / members[-1]
    return None


# Public mirror solesensei/solesensei_bdd100k: same images and, for car/bus/truck/motor,
# box-for-box the same train/val labels as det_20 (checked 2026-10-05 on all 69,863 train and
# 10,000 val images), but laid out as BDD100K's 2018 release.
_PUBLIC_IMAGES = Path("bdd100k") / "bdd100k" / "images" / "100k"
_PUBLIC_LABELS = Path("bdd100k_labels_release") / "bdd100k" / "labels"


def shim_public_bdd100k(input_root: Path = Path("/kaggle/input"),
                        shim_root: Path = Path("/kaggle/tmp/bdd100k_shim")) -> Path | None:
    """Expose the public mirror in the ``bdd100k_kaggle`` layout with symlinks; None if absent.

    The mirror scatters each split over sub-folders (``train/``, ``train/trainA`` … ``testB``),
    so every split is flattened into one folder of per-image links. Every image named in the
    split's label file must be found: prepare_bdd100k.py skips missing images silently, which
    would change the partitions without any error.
    """
    import json

    found = [b for pat in ("*/", "datasets/*/*/") for b in sorted(Path(input_root).glob(pat + _PUBLIC_LABELS.as_posix()))]
    for base in found:
        mirror = base.parents[2]
        images = mirror / _PUBLIC_IMAGES
        if not (images / "train").is_dir():
            continue
        root = Path(shim_root) / DATASET_FOLDER
        (root / "labels" / "det_20").mkdir(parents=True, exist_ok=True)
        for split in ("train", "val"):
            ann = base / f"bdd100k_labels_images_{split}.json"
            link = root / "labels" / "det_20" / f"det_{split}.json"
            if not link.exists():
                link.symlink_to(ann)
            out = root / "images" / "100k" / split
            out.mkdir(parents=True, exist_ok=True)
            by_name: dict[str, Path] = {}
            for img in sorted((images / split).rglob("*.jpg")):
                by_name.setdefault(img.name, img)
            for name, img in by_name.items():
                dst = out / name
                if not dst.exists():
                    dst.symlink_to(img)
            names = [f["name"] for f in json.loads(ann.read_text(encoding="utf-8"))]
            missing = [n for n in names if n not in by_name]
            if missing:
                raise FileNotFoundError(
                    f"public mirror lacks {len(missing)} of {len(names)} labelled {split} images "
                    f"(e.g. {missing[:3]}); the partitions would silently differ — not using it")
        return root
    return None
