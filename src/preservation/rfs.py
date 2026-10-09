"""Repeat-factor sampling (RFS) inside each client (ADR-019, component ③ of P4).

[LITERATURE] LVIS (Gupta, Dollár, Girshick); formula as in the official Detectron2 implementation
(`detectron2/data/samplers/distributed_sampler.py`, `repeat_factors_from_category_frequency`, checked
2026-10-09):

    f(c) = fraction of the images (of this client) that contain class c
    r(c) = max(1, sqrt(t / f(c)))
    r(I) = max over the classes in image I of r(c)          (1 for an image without labels)

Each image then appears floor(r(I)) times plus one more with probability r(I) - floor(r(I)) (stochastic
rounding, seeded per client and round, as Detectron2's RepeatFactorTrainingSampler does per epoch).

[ENGINEERING] Implemented as a rewritten image list: ultralytics 8.3.253 keeps duplicate entries
(`get_img_files` sorts without de-duplicating; the label cache is a list built pair by pair), so a
repeated line is a repeated training sample. Loss and labels are untouched — unlike A2b/A6, the negative
signal of every image is kept; rare images are simply seen more often. Frequencies are computed from the
client's OWN images only: nothing is shared.
"""
from __future__ import annotations

import math
import os
import random
from collections.abc import Sequence
from pathlib import Path

import yaml


def _label_path(img: str) -> Path:
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    img = img.replace("/", os.sep)
    return Path(sb.join(img.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt")


def image_classes(img: str) -> set[int]:
    lf = _label_path(img)
    if not lf.exists():
        return set()
    return {int(ln.split()[0]) for ln in lf.read_text(encoding="utf-8").splitlines() if ln.strip()}


def repeat_factors(classes_per_image: Sequence[set[int]], t: float) -> list[float]:
    """Detectron2's r(I) for each image, from per-image class sets."""
    if t <= 0:
        raise ValueError(f"repeat threshold t must be > 0, got {t}")
    n = len(classes_per_image)
    freq: dict[int, float] = {}
    for cls in classes_per_image:
        for c in cls:
            freq[c] = freq.get(c, 0) + 1
    rep = {c: max(1.0, math.sqrt(t / (k / n))) for c, k in freq.items()}
    return [max((rep[c] for c in cls), default=1.0) for cls in classes_per_image]


def resample(images: Sequence[str], factors: Sequence[float], seed: int) -> list[str]:
    rng = random.Random(seed)
    out: list[str] = []
    for img, r in zip(images, factors):
        k = int(math.floor(r))
        k += 1 if rng.random() < r - k else 0
        out.extend([img] * k)
    return out


def rfs_data_yaml(data_yaml: Path, out_dir: Path, t: float, seed: int) -> Path:
    """Copy of ``data_yaml`` whose ``train`` image list is repeat-factor resampled."""
    data_yaml = Path(data_yaml)
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(data.get("path") or data_yaml.parent)
    train = Path(data["train"])
    train = train if train.is_absolute() else root / train
    if not train.is_file():
        raise ValueError(f"{data_yaml}: RFS needs an image-list train entry, got {train}")
    images = [ln.strip() for ln in train.read_text(encoding="utf-8").splitlines() if ln.strip()]
    factors = repeat_factors([image_classes(im) for im in images], t)
    picked = resample(images, factors, seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lst = out_dir / f"{data_yaml.stem}_rfs.txt"
    lst.write_text("\n".join(picked) + "\n", encoding="utf-8")
    out = out_dir / f"{data_yaml.stem}_rfs.yaml"
    out.write_text(yaml.safe_dump({**data, "train": str(lst)}), encoding="utf-8")
    return out
