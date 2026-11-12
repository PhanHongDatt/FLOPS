"""Direction A (ADR-035): a fraction of a local epoch per FL round.

[THESIS-HYPOTHESIS] The missing-class gap comes from the many local steps a client without bus takes between two
aggregations (one epoch ≈ 375 steps at batch 8). Training on 1/k of the client's images per round and running k
times as many rounds keeps the total number of local steps (and of images seen) the same while aggregating k times
as often. In the limit of one step per round FedAvg is minibatch SGD on the union of the clients' data, i.e. the
centralized setting, which has no missing-class gap in this project (ADR-026 vs S21b) — so the gap must shrink as k
grows; the experiment measures how fast.

[ENGINEERING] Round r of client c trains on part (r - 1) mod k of a permutation of its images that is redrawn
every k rounds (seeded by base seed, client and pass index): every k consecutive rounds cover each image exactly
once. Repeat-factor sampling, when on, is applied to that part afterwards, with frequencies from the part.
"""
from __future__ import annotations

import random
from pathlib import Path

import yaml


def parts_for(fraction: float) -> int:
    """k = 1 / fraction; only exact unit fractions (1, 1/2, 1/4, ...) are allowed, so k rounds = one epoch."""
    if not 0 < fraction <= 1:
        raise ValueError(f"local_fraction must be in (0, 1], got {fraction}")
    k = round(1 / fraction)
    if abs(k * fraction - 1) > 1e-9:
        raise ValueError(f"local_fraction must be 1/k for an integer k, got {fraction}")
    return k


def round_part(images: list[str], fraction: float, server_round: int, seed: int, client_id: str) -> list[str]:
    """The images client ``client_id`` trains on in ``server_round`` (1-based)."""
    k = parts_for(fraction)
    if k == 1:
        return list(images)
    if server_round < 1:
        raise ValueError(f"server_round is 1-based, got {server_round}")
    pass_index, part = divmod(server_round - 1, k)
    order = sorted(images)
    random.Random(f"{seed}|{client_id}|{pass_index}").shuffle(order)
    return order[part::k]


def fraction_data_yaml(data_yaml: Path, out_dir: Path, fraction: float, server_round: int, seed: int,
                       client_id: str) -> Path:
    """Copy of ``data_yaml`` whose ``train`` list is this round's part (the input file is not modified)."""
    data_yaml = Path(data_yaml)
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(data.get("path") or data_yaml.parent)
    train = Path(data["train"])
    train = train if train.is_absolute() else root / train
    if not train.is_file():
        raise ValueError(f"{data_yaml}: local_fraction needs an image-list train entry, got {train}")
    images = [ln.strip() for ln in train.read_text(encoding="utf-8").splitlines() if ln.strip()]
    picked = round_part(images, fraction, server_round, seed, client_id)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lst = out_dir / f"{data_yaml.stem}_part.txt"
    lst.write_text("\n".join(picked) + "\n", encoding="utf-8")
    out = out_dir / f"{data_yaml.stem}_part.yaml"
    out.write_text(yaml.safe_dump({**data, "path": str(root), "train": str(lst)}), encoding="utf-8")
    return out
