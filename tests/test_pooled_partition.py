"""D3 — pooled re-split of a Missing-Class partition (ADR-014). Pure stdlib."""
from __future__ import annotations

from pathlib import Path

from src.data.bdd100k import TARGET_CLASSES
from src.data.partitioner import partition_missing_class, partition_pooled_iid

CAR, BUS, TRUCK, MOTO = TARGET_CLASSES
CLS_IDX = {c: i for i, c in enumerate(TARGET_CLASSES)}


def _dataset(tmp_path: Path) -> tuple[list[str], Path]:
    label_dir = tmp_path / "labels" / "train"
    label_dir.mkdir(parents=True)
    spec = {}
    for i in range(300):
        spec[f"plain_{i:04d}.jpg"] = {CAR: 4}
    for i in range(80):
        spec[f"bus_{i:04d}.jpg"] = {CAR: 2, BUS: 2}
    for i in range(60):
        spec[f"truck_{i:04d}.jpg"] = {CAR: 1, TRUCK: 1}
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


def test_same_images_same_totals_evenly_spread(tmp_path):
    names, label_dir = _dataset(tmp_path)
    ref = _reference(names, label_dir)
    assert ref.class_counts["C0"][BUS] == 0 and ref.class_counts["C1"][BUS] == 0
    pooled = partition_pooled_iid(ref, label_dir, seed=42, partition_id="pooled")

    ref_imgs = sorted(n for v in ref.client_assignments.values() for n in v)
    pooled_imgs = sorted(n for v in pooled.client_assignments.values() for n in v)
    assert pooled_imgs == ref_imgs                                   # identical data
    for cls in TARGET_CLASSES:                                       # identical totals
        assert sum(c[cls] for c in pooled.class_counts.values()) == \
            sum(c[cls] for c in ref.class_counts.values())
    sizes = [len(v) for v in pooled.client_assignments.values()]
    assert max(sizes) - min(sizes) <= 1
    bus = [pooled.class_counts[c][BUS] for c in sorted(pooled.client_assignments)]
    assert min(bus) > 0 and max(bus) - min(bus) <= 2 * 2              # within two bus images
    assert all(BUS not in missing for missing in pooled.missing_classes.values())
    assert pooled.scenario == "S1-Pooled-IID"


def test_deterministic(tmp_path):
    names, label_dir = _dataset(tmp_path)
    ref = _reference(names, label_dir)
    a = partition_pooled_iid(ref, label_dir, seed=7, partition_id="p")
    b = partition_pooled_iid(ref, label_dir, seed=7, partition_id="p")
    c = partition_pooled_iid(ref, label_dir, seed=8, partition_id="p")
    assert a.client_assignments == b.client_assignments
    assert a.client_assignments != c.client_assignments
