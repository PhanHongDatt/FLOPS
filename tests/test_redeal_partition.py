"""ADR-022 — re-deal the SAME images of a partition under a different absent-class map (severity sweep)."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.data.bdd100k import TARGET_CLASSES
from src.data.partitioner import partition_missing_class, partition_redeal

CAR, BUS, TRUCK, MOTO = TARGET_CLASSES
CLS_IDX = {c: i for i, c in enumerate(TARGET_CLASSES)}
ABSENT = {BUS: ["C0", "C1", "C2"], TRUCK: ["C2"]}          # bus only on C3


def _dataset(tmp_path: Path, n_bus: int = 80) -> tuple[list[str], Path]:
    label_dir = tmp_path / "labels" / "train"
    label_dir.mkdir(parents=True)
    spec = {}
    for i in range(300):
        spec[f"plain_{i:04d}.jpg"] = {CAR: 4}
    for i in range(n_bus):
        spec[f"bus_{i:04d}.jpg"] = {CAR: 2, BUS: 2}
    for i in range(60):
        spec[f"truck_{i:04d}.jpg"] = {CAR: 1, TRUCK: 1}
    for i in range(40):
        spec[f"moto_{i:04d}.jpg"] = {CAR: 1, MOTO: 1}
    for name, counts in spec.items():
        lines = [f"{CLS_IDX[c]} 0.5 0.5 0.1 0.1" for c, n in counts.items() for _ in range(n)]
        (label_dir / (Path(name).stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
    return sorted(spec), label_dir


def _reference(names, label_dir):
    return partition_missing_class(
        image_names=names, label_dir=label_dir, num_clients=4,
        missing_map={"C0": [BUS], "C1": [BUS], "C2": [TRUCK], "C3": []},
        seed=42, partition_id="s1b", per_client=100,
    )


def _imgs(m):
    return sorted(n for v in m.client_assignments.values() for n in v)


def test_same_images_same_totals_same_sizes_and_exact_absent_map(tmp_path):
    names, label_dir = _dataset(tmp_path)
    ref = _reference(names, label_dir)
    out = partition_redeal(ref, label_dir, ABSENT, seed=42, partition_id="bus1")
    assert _imgs(out) == _imgs(ref)
    for cls in TARGET_CLASSES:
        assert sum(c[cls] for c in out.class_counts.values()) == sum(c[cls] for c in ref.class_counts.values())
    assert sorted(len(v) for v in out.client_assignments.values()) == \
        sorted(len(v) for v in ref.client_assignments.values())
    assert out.missing_classes == {"C0": [BUS], "C1": [BUS], "C2": [BUS, TRUCK], "C3": []}
    assert out.scenario == "S1-Redeal"


def test_unconstrained_class_is_spread_over_every_client(tmp_path):
    names, label_dir = _dataset(tmp_path)
    out = partition_redeal(_reference(names, label_dir), label_dir, ABSENT, seed=42, partition_id="bus1")
    moto = [out.class_counts[c][MOTO] for c in sorted(out.client_assignments)]
    assert min(moto) > 0 and max(moto) - min(moto) <= 3


def test_deterministic(tmp_path):
    names, label_dir = _dataset(tmp_path)
    ref = _reference(names, label_dir)
    a = partition_redeal(ref, label_dir, ABSENT, seed=7, partition_id="p")
    b = partition_redeal(ref, label_dir, ABSENT, seed=7, partition_id="p")
    assert a.client_assignments == b.client_assignments


def test_infeasible_absent_map_fails_before_any_training(tmp_path):
    names, label_dir = _dataset(tmp_path, n_bus=80)
    ref = _reference(names, label_dir)
    # every client but C3 refuses bus AND C3 refuses truck: C3 cannot take all bus images + is too small
    with pytest.raises(ValueError, match="cannot place"):
        partition_redeal(ref, label_dir, {BUS: ["C0", "C1", "C2"], TRUCK: ["C0", "C1", "C2", "C3"]},
                         seed=42, partition_id="x")
