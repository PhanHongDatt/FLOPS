"""Tests for the balanced missing-class split and the matched-pair control.

Both are prerequisites for comparison C1 / gate G4: without balanced client sizes
the S1 scenario confounds "missing class" with "less data", and without a real
matched control there is nothing to attribute the difference against
(CLAUDE.md §10, research/plan/plan.md §5.3–§5.4).

Pure stdlib — no GPU, torch or ultralytics.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.data.bdd100k import TARGET_CLASSES
from src.data.partitioner import (
    build_attribute_index,
    load_label_index,
    partition_matched_control,
    partition_missing_class,
    save_match_report,
)

CAR, BUS, TRUCK, MOTO = TARGET_CLASSES
CLS_IDX = {c: i for i, c in enumerate(TARGET_CLASSES)}


def _write_dataset(tmp_path: Path, spec: dict[str, dict[str, int]]) -> tuple[list[str], Path]:
    """spec: {image_name: {class: box_count}} -> (image_names, label_dir)."""
    label_dir = tmp_path / "labels" / "train"
    label_dir.mkdir(parents=True, exist_ok=True)
    for name, counts in spec.items():
        lines = []
        for cls, n in counts.items():
            lines += [f"{CLS_IDX[cls]} 0.5 0.5 0.1 0.1"] * n
        (label_dir / (Path(name).stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
    return sorted(spec), label_dir


def _uniform_spec(n_plain: int, n_bus: int) -> dict[str, dict[str, int]]:
    """n_plain car-only images + n_bus images holding car and bus."""
    spec: dict[str, dict[str, int]] = {}
    for i in range(n_plain):
        spec[f"plain_{i:04d}.jpg"] = {CAR: 4}
    for i in range(n_bus):
        spec[f"bus_{i:04d}.jpg"] = {CAR: 3, BUS: 1}
    return spec


# ── balanced assignment (§5.4) ────────────────────────────────────────────
def test_missing_class_balances_client_sizes(tmp_path):
    """The old first-eligible rule let C0 absorb nearly everything it was eligible
    for; sizes must now stay within a couple of images."""
    names, label_dir = _write_dataset(tmp_path, _uniform_spec(400, 200))
    m = partition_missing_class(
        image_names=names, label_dir=label_dir, num_clients=4,
        missing_map={"C0": [BUS], "C1": [BUS], "C2": [], "C3": []},
        seed=42, partition_id="p",
    )
    sizes = [len(v) for v in m.client_assignments.values()]
    assert sum(sizes) == len(names)
    assert max(sizes) - min(sizes) <= 1, sizes


def test_missing_class_still_guarantees_zero_boxes(tmp_path):
    names, label_dir = _write_dataset(tmp_path, _uniform_spec(400, 200))
    m = partition_missing_class(
        image_names=names, label_dir=label_dir, num_clients=4,
        missing_map={"C0": [BUS], "C1": [BUS], "C2": [], "C3": []},
        seed=42, partition_id="p",
    )
    for cid in ("C0", "C1"):
        assert m.class_counts[cid][BUS] == 0
        assert BUS in m.missing_classes[cid]
    assert m.class_counts["C2"][BUS] + m.class_counts["C3"][BUS] == 200


def test_per_client_cap_is_respected(tmp_path):
    names, label_dir = _write_dataset(tmp_path, _uniform_spec(400, 200))
    m = partition_missing_class(
        image_names=names, label_dir=label_dir, num_clients=4,
        missing_map={"C0": [BUS], "C1": [], "C2": [], "C3": []},
        seed=42, partition_id="p", per_client=50,
    )
    sizes = [len(v) for v in m.client_assignments.values()]
    assert all(s <= 50 for s in sizes), sizes
    assert sum(sizes) == 200


def test_missing_class_is_deterministic(tmp_path):
    names, label_dir = _write_dataset(tmp_path, _uniform_spec(200, 100))
    kwargs = dict(
        image_names=names, label_dir=label_dir, num_clients=4,
        missing_map={"C0": [BUS], "C1": [], "C2": [], "C3": []},
        seed=7, partition_id="p",
    )
    a = partition_missing_class(**kwargs)
    b = partition_missing_class(**kwargs)
    assert a.client_assignments == b.client_assignments


def test_no_image_assigned_twice(tmp_path):
    names, label_dir = _write_dataset(tmp_path, _uniform_spec(300, 150))
    m = partition_missing_class(
        image_names=names, label_dir=label_dir, num_clients=4,
        missing_map={"C0": [BUS], "C1": [TRUCK], "C2": [], "C3": []},
        seed=42, partition_id="p",
    )
    allocated = [img for imgs in m.client_assignments.values() for img in imgs]
    assert len(allocated) == len(set(allocated))


def test_images_every_client_excludes_are_dropped(tmp_path):
    names, label_dir = _write_dataset(tmp_path, _uniform_spec(100, 40))
    m = partition_missing_class(
        image_names=names, label_dir=label_dir, num_clients=2,
        missing_map={"C0": [BUS], "C1": [BUS]},
        seed=42, partition_id="p",
    )
    assert sum(len(v) for v in m.client_assignments.values()) == 100
    for cid in ("C0", "C1"):
        assert m.class_counts[cid][BUS] == 0


# ── label index ───────────────────────────────────────────────────────────
def test_load_label_index_counts_boxes_per_image(tmp_path):
    names, label_dir = _write_dataset(tmp_path, {
        "a.jpg": {CAR: 2, BUS: 3},
        "b.jpg": {TRUCK: 1},
    })
    index = load_label_index(names, label_dir)
    assert index["a.jpg"][CAR] == 2 and index["a.jpg"][BUS] == 3
    assert index["b.jpg"][TRUCK] == 1 and index["b.jpg"][CAR] == 0


def test_load_label_index_handles_missing_label_file(tmp_path):
    _, label_dir = _write_dataset(tmp_path, {"a.jpg": {CAR: 1}})
    index = load_label_index(["a.jpg", "ghost.jpg"], label_dir)
    assert index["ghost.jpg"] == {c: 0 for c in TARGET_CLASSES}


# ── matched control (§5.3) ────────────────────────────────────────────────
def _s1_reference(tmp_path, n_plain=600, n_bus=300):
    names, label_dir = _write_dataset(tmp_path, _uniform_spec(n_plain, n_bus))
    ref = partition_missing_class(
        image_names=names, label_dir=label_dir, num_clients=4,
        missing_map={"C0": [BUS], "C1": [BUS], "C2": [], "C3": []},
        seed=42, partition_id="s1b", per_client=100,
    )
    return names, label_dir, ref


def test_matched_control_keeps_image_count_identical(tmp_path):
    names, label_dir, ref = _s1_reference(tmp_path)
    ctrl, report = partition_matched_control(
        image_names=names, label_dir=label_dir, reference=ref,
        target_class=BUS, seed=42, partition_id="ctrl",
    )
    for cid in ref.client_assignments:
        assert len(ctrl.client_assignments[cid]) == len(ref.client_assignments[cid])
        assert report.per_client[cid]["images"] == report.per_client[cid]["reference_images"]


def test_matched_control_gives_the_target_class_to_every_client(tmp_path):
    names, label_dir, ref = _s1_reference(tmp_path)
    ctrl, _ = partition_matched_control(
        image_names=names, label_dir=label_dir, reference=ref,
        target_class=BUS, seed=42, partition_id="ctrl",
    )
    # The whole point: the vacant clients of S1 now HOLD the target class.
    for cid in ("C0", "C1"):
        assert ref.class_counts[cid][BUS] == 0
        assert ctrl.class_counts[cid][BUS] > 0
        assert BUS not in ctrl.missing_classes[cid]


def test_matched_control_does_not_overlap_the_reference_donors(tmp_path):
    names, label_dir, ref = _s1_reference(tmp_path)
    ctrl, _ = partition_matched_control(
        image_names=names, label_dir=label_dir, reference=ref,
        target_class=BUS, seed=42, partition_id="ctrl",
    )
    ref_all = {i for v in ref.client_assignments.values() for i in v}
    swapped_in = {
        i for cid, v in ctrl.client_assignments.items()
        for i in v if i not in set(ref.client_assignments[cid])
    }
    # Images swapped in must come from OUTSIDE the reference partition, so the
    # control never reuses the very images it is being compared against.
    assert swapped_in and not (swapped_in & ref_all)


def test_matched_control_clients_are_disjoint(tmp_path):
    names, label_dir, ref = _s1_reference(tmp_path)
    ctrl, _ = partition_matched_control(
        image_names=names, label_dir=label_dir, reference=ref,
        target_class=BUS, seed=42, partition_id="ctrl",
    )
    allocated = [i for v in ctrl.client_assignments.values() for i in v]
    assert len(allocated) == len(set(allocated))


def test_matched_control_reports_match_status_per_column(tmp_path):
    names, label_dir, ref = _s1_reference(tmp_path)
    _, report = partition_matched_control(
        image_names=names, label_dir=label_dir, reference=ref,
        target_class=BUS, seed=42, partition_id="ctrl",
    )
    for cid, info in report.per_client.items():
        assert info["checks"]["image_count"] is True
        assert "nuisance_boxes" in info["checks"]
        assert f"boxes_{CAR}" in info["checks"]
        # The target class is excluded from the match checks on purpose: it is the
        # variable under study, not a nuisance variable to equalise.
        assert f"boxes_{BUS}" not in info["checks"]
        # The GRAND total is reported but NOT gated — it must differ by roughly the
        # target boxes the control gained, so gating it would be wrong.
        assert "total_boxes" not in info["checks"]
        assert info["total_boxes"] >= info["nuisance_boxes"]


def test_matched_control_grand_total_differs_by_the_target_boxes(tmp_path):
    names, label_dir, ref = _s1_reference(tmp_path)
    _, report = partition_matched_control(
        image_names=names, label_dir=label_dir, reference=ref,
        target_class=BUS, seed=42, partition_id="ctrl",
    )
    for info in report.per_client.values():
        gained_target = info["target_boxes"] - info["reference_target_boxes"]
        nuisance_drift = info["nuisance_boxes"] - info["reference_nuisance_boxes"]
        total_drift = info["total_boxes"] - info["reference_total_boxes"]
        assert total_drift == gained_target + nuisance_drift


def test_matched_control_is_deterministic(tmp_path):
    names, label_dir, ref = _s1_reference(tmp_path)
    kwargs = dict(image_names=names, label_dir=label_dir, reference=ref,
                  target_class=BUS, seed=42, partition_id="ctrl")
    a, _ = partition_matched_control(**kwargs)
    b, _ = partition_matched_control(**kwargs)
    assert a.client_assignments == b.client_assignments


def test_matched_control_reports_failure_when_donor_pool_is_exhausted(tmp_path):
    """Honesty requirement: a control that could not reach the target level says so
    instead of silently being published as matched (§10)."""
    names, label_dir = _write_dataset(tmp_path, _uniform_spec(400, 20))
    ref = partition_missing_class(
        image_names=names, label_dir=label_dir, num_clients=4,
        missing_map={"C0": [BUS], "C1": [BUS], "C2": [], "C3": []},
        seed=42, partition_id="s1b", per_client=100,
    )
    _, report = partition_matched_control(
        image_names=names, label_dir=label_dir, reference=ref,
        target_class=BUS, seed=42, partition_id="ctrl",
    )
    assert report.matched is False
    assert any("donor pool" in r for r in report.unmatched_reasons)


def test_matched_control_rejects_unknown_target_class(tmp_path):
    names, label_dir, ref = _s1_reference(tmp_path)
    with pytest.raises(ValueError, match="target_class"):
        partition_matched_control(
            image_names=names, label_dir=label_dir, reference=ref,
            target_class="bicycle", seed=42, partition_id="ctrl",
        )


def test_attribute_matching_prefers_same_attribute_value(tmp_path):
    names, label_dir, ref = _s1_reference(tmp_path)
    # plain_* are daytime, bus_* alternate day/night: attribute matching should keep
    # the per-client attribute distribution close.
    attrs = {n: ("daytime" if n.startswith("plain") else
                 ("daytime" if int(n.split("_")[1].split(".")[0]) % 2 == 0 else "night"))
             for n in names}
    _, report = partition_matched_control(
        image_names=names, label_dir=label_dir, reference=ref,
        target_class=BUS, seed=42, partition_id="ctrl",
        attribute_index=attrs, attribute_key="timeofday",
    )
    for info in report.per_client.values():
        assert info["attribute_max_shift"] is not None
        assert "attr_timeofday" in info["checks"]


def test_without_attribute_index_the_column_is_absent_not_claimed(tmp_path):
    names, label_dir, ref = _s1_reference(tmp_path)
    _, report = partition_matched_control(
        image_names=names, label_dir=label_dir, reference=ref,
        target_class=BUS, seed=42, partition_id="ctrl",
    )
    assert report.attribute_key is None
    for info in report.per_client.values():
        assert info["attribute_max_shift"] is None
        assert not any(k.startswith("attr_") for k in info["checks"])


def test_save_match_report_roundtrip(tmp_path):
    names, label_dir, ref = _s1_reference(tmp_path)
    _, report = partition_matched_control(
        image_names=names, label_dir=label_dir, reference=ref,
        target_class=BUS, seed=42, partition_id="ctrl",
    )
    out = tmp_path / "out" / "match_report.yaml"
    save_match_report(report, out)
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["target_class"] == BUS
    assert data["tolerance"] == pytest.approx(0.05)
    assert set(data["per_client"]) == set(ref.client_assignments)


def test_build_attribute_index_reads_frame_attributes(tmp_path):
    import json
    ann = tmp_path / "det.json"
    ann.write_text(json.dumps([
        {"name": "a.jpg", "attributes": {"timeofday": "night", "weather": "rainy"}},
        {"name": "b.jpg", "attributes": {"timeofday": "daytime"}},
        {"name": "c.jpg"},
    ]), encoding="utf-8")
    idx = build_attribute_index(ann, key="timeofday")
    assert idx == {"a.jpg": "night", "b.jpg": "daytime"}
    assert build_attribute_index(ann, key="weather") == {"a.jpg": "rainy"}


# ── rare-class quota (ADR-017) ──────────────────────────────────────────
def test_class_quota_oversamples_without_breaking_missing_classes(tmp_path):
    spec = {}
    for i in range(600):
        spec[f"plain_{i:04d}.jpg"] = {CAR: 3}
    for i in range(80):
        spec[f"bus_{i:04d}.jpg"] = {CAR: 2, BUS: 1}
    for i in range(60):
        spec[f"moto_{i:04d}.jpg"] = {CAR: 1, MOTO: 1}
    for i in range(20):
        spec[f"motobus_{i:04d}.jpg"] = {BUS: 1, MOTO: 1}
    names, label_dir = _write_dataset(tmp_path, spec)
    mm = {"C0": [BUS], "C1": [BUS], "C2": []}
    plain = partition_missing_class(names, label_dir, 3, mm, seed=1, partition_id="p", per_client=100)
    boosted = partition_missing_class(names, label_dir, 3, mm, seed=1, partition_id="p", per_client=100,
                                      class_quota={MOTO: 20})
    for cid in ("C0", "C1"):
        assert boosted.class_counts[cid][BUS] == 0                     # contract still holds
        assert boosted.class_counts[cid][MOTO] >= 20
    assert all(len(v) == 100 for v in boosted.client_assignments.values())
    assert sum(c[MOTO] for c in boosted.class_counts.values()) > sum(c[MOTO] for c in plain.class_counts.values())
    with pytest.raises(ValueError):
        partition_missing_class(names, label_dir, 3, mm, seed=1, partition_id="p", class_quota={"person": 5})
