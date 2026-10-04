from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
import bisect
import statistics
from typing import Any, Iterable

import yaml

from src.data.bdd100k import TARGET_CLASSES, load_bdd100k_annotations, filter_target_classes

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PartitionManifest:
    partition_id: str
    seed: int
    scenario: str          # S0 | S1 | S1-Control
    num_clients: int
    client_assignments: dict[str, list[str]]   # client_id -> [image_name, ...]
    class_counts: dict[str, dict[str, int]]    # client_id -> {class: count}
    missing_classes: dict[str, list[str]]      # client_id -> [missing class, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition_id": self.partition_id,
            "seed": self.seed,
            "scenario": self.scenario,
            "num_clients": self.num_clients,
            "client_assignments": self.client_assignments,
            "class_counts": self.class_counts,
            "missing_classes": self.missing_classes,
        }


def _count_classes(
    image_names: list[str],
    label_dir: Path,
) -> dict[str, int]:
    counts: dict[str, int] = {c: 0 for c in TARGET_CLASSES}
    for name in image_names:
        lf = label_dir / (Path(name).stem + ".txt")
        if not lf.exists():
            continue
        for line in lf.read_text(encoding="utf-8").strip().splitlines():
            parts = line.split()
            if parts:
                idx = int(parts[0])
                if idx < len(TARGET_CLASSES):
                    counts[TARGET_CLASSES[idx]] += 1
    return counts


def load_label_index(
    image_names: list[str],
    label_dir: Path,
) -> dict[str, dict[str, int]]:
    """Per-image class box counts, read in ONE pass over the label files.

    BDD100K train has ~70k label files; the previous flow read them once while
    assigning images and then again to count per-client classes. One pass matters
    on Kaggle, where the label directory lives on the slower working volume.

    Returns ``{image_name: {class_name: box_count}}``; images with no label file
    map to all-zero counts so callers never have to special-case them.
    """
    index: dict[str, dict[str, int]] = {}
    for name in image_names:
        counts = {c: 0 for c in TARGET_CLASSES}
        lf = label_dir / (Path(name).stem + ".txt")
        if lf.exists():
            for line in lf.read_text(encoding="utf-8").strip().splitlines():
                parts = line.split()
                if parts:
                    try:
                        idx = int(parts[0])
                    except ValueError:
                        continue
                    if 0 <= idx < len(TARGET_CLASSES):
                        counts[TARGET_CLASSES[idx]] += 1
        index[name] = counts
    return index


def _sum_counts(
    image_names: Iterable[str],
    index: dict[str, dict[str, int]],
) -> dict[str, int]:
    total = {c: 0 for c in TARGET_CLASSES}
    for name in image_names:
        for cls, n in index.get(name, {}).items():
            total[cls] += n
    return total


def _present_classes(name: str, index: dict[str, dict[str, int]]) -> set[str]:
    return {c for c, n in index.get(name, {}).items() if n > 0}


def partition_iid(
    image_names: list[str],
    label_dir: Path,
    num_clients: int,
    seed: int,
    partition_id: str,
    per_client: int | None = None,
) -> PartitionManifest:
    """Split images approximately evenly across clients (S0).

    ``per_client`` keeps only the first N images of each client's split (smoke
    runs); the split itself is unchanged, so a capped partition is a prefix of
    the uncapped one.
    """
    rng = random.Random(seed)
    shuffled = list(image_names)
    rng.shuffle(shuffled)

    splits: dict[str, list[str]] = {}
    chunk = len(shuffled) // num_clients
    for i in range(num_clients):
        client_id = f"C{i}"
        start = i * chunk
        end = start + chunk if i < num_clients - 1 else len(shuffled)
        splits[client_id] = shuffled[start:end][:per_client] if per_client else shuffled[start:end]

    counts = {cid: _count_classes(imgs, label_dir) for cid, imgs in splits.items()}
    missing = {cid: [c for c in TARGET_CLASSES if counts[cid][c] == 0] for cid in splits}

    return PartitionManifest(
        partition_id=partition_id,
        seed=seed,
        scenario="S0",
        num_clients=num_clients,
        client_assignments=splits,
        class_counts=counts,
        missing_classes=missing,
    )


def partition_missing_class(
    image_names: list[str],
    label_dir: Path,
    num_clients: int,
    missing_map: dict[str, list[str]],  # client_id -> [classes to exclude]
    seed: int,
    partition_id: str,
    scenario: str = "S1",
    per_client: int | None = None,
    label_index: dict[str, dict[str, int]] | None = None,
) -> PartitionManifest:
    """Create a Missing-Class Non-IID partition (S1 or S1-Control).

    ``missing_map`` defines which classes each client must have zero positives
    for. An image is eligible for a client only if it contains NONE of that
    client's excluded classes, so an excluded class ends up with zero images AND
    zero boxes — a true missing class, not a fake background (deleting the labels
    instead would be the partially-labelled problem, a different hypothesis).

    Assignment uses the **least-filled eligible client**, not the first eligible
    one. First-eligible made C0 absorb nearly everything it was eligible for, so
    client sizes diverged wildly and the scenario confounded "missing class" with
    "less data" — exactly what CLAUDE.md §10 requires S1-Control to rule out. The
    least-filled rule keeps client sizes within a few images of each other.

    Args:
        per_client: optional cap on images per client. When the least-filled
            eligible client is already full, every eligible client is full, so the
            image is dropped (and counted).
        label_index: optional precomputed output of ``load_label_index`` to avoid
            re-reading ~70k label files.

    Side effect to report, not hide: because images holding a rare class can only
    go to clients that do not exclude it, those clients receive that class at a
    higher rate than its dataset-wide base rate. Always report the ACTUAL
    per-client class distribution from the manifest.
    """
    rng = random.Random(seed)
    shuffled = list(image_names)
    rng.shuffle(shuffled)

    index = label_index if label_index is not None else load_label_index(shuffled, label_dir)

    client_ids = [f"C{i}" for i in range(num_clients)]
    splits: dict[str, list[str]] = {cid: [] for cid in client_ids}
    excluded_by_client = {cid: set(missing_map.get(cid, [])) for cid in client_ids}

    dropped_ineligible = 0
    dropped_full = 0
    for img in shuffled:
        img_classes = _present_classes(img, index)
        eligible = [
            cid for cid in client_ids
            if not (img_classes & excluded_by_client[cid])
        ]
        if not eligible:
            # Image contains ONLY classes that every client excludes. Assigning it
            # anyway would falsify the missing-class contract (CLAUDE.md §10, §22),
            # so drop it and report the count.
            dropped_ineligible += 1
            continue
        # Least-filled eligible client; ties broken by client order for determinism.
        target = min(eligible, key=lambda cid: (len(splits[cid]), client_ids.index(cid)))
        if per_client is not None and len(splits[target]) >= per_client:
            dropped_full += 1
            continue
        splits[target].append(img)

    if dropped_ineligible:
        logger.warning(
            "partition_missing_class(%s): dropped %d/%d images with no eligible "
            "client (every client excludes a class present in the image). "
            "Reduce missing_map overlap if this shrinks the training set unacceptably.",
            partition_id, dropped_ineligible, len(shuffled),
        )
    if dropped_full:
        logger.info(
            "partition_missing_class(%s): %d image(s) unused because every eligible "
            "client reached per_client=%s.",
            partition_id, dropped_full, per_client,
        )

    # Rebalance pass: the greedy stream can still leave a couple of images of
    # drift (an image eligible only for the clients that are currently fuller).
    # Client size is the one confound S1-Control exists to rule out, so close the
    # gap exactly whenever a legal move exists. Deterministic: always move the
    # first eligible image of the largest client to the smallest.
    for _ in range(len(shuffled)):
        order = sorted(client_ids, key=lambda cid: (len(splits[cid]), cid))
        lo, hi = order[0], order[-1]
        if len(splits[hi]) - len(splits[lo]) <= 1:
            break
        moved = False
        for img in splits[hi]:
            if not (_present_classes(img, index) & excluded_by_client[lo]):
                splits[hi].remove(img)
                splits[lo].append(img)
                moved = True
                break
        if not moved:
            # No image of the largest client is eligible for the smallest one;
            # the remaining imbalance is forced by the missing_map, not by the
            # assignment order. Report it rather than distorting the scenario.
            logger.info(
                "partition_missing_class(%s): residual size gap %d is forced by "
                "missing_map eligibility, not by assignment order.",
                partition_id, len(splits[hi]) - len(splits[lo]),
            )
            break

    sizes = {cid: len(imgs) for cid, imgs in splits.items()}
    if sizes and max(sizes.values()) > 0:
        spread = (max(sizes.values()) - min(sizes.values())) / max(sizes.values())
        logger.info(
            "partition_missing_class(%s): client sizes %s (spread %.2f%%)",
            partition_id, sizes, spread * 100,
        )

    counts = {cid: _sum_counts(imgs, index) for cid, imgs in splits.items()}
    missing = {cid: [c for c in TARGET_CLASSES if counts[cid][c] == 0] for cid in splits}

    return PartitionManifest(
        partition_id=partition_id,
        seed=seed,
        scenario=scenario,
        num_clients=num_clients,
        client_assignments=splits,
        class_counts=counts,
        missing_classes=missing,
    )


def save_partition_manifest(manifest: PartitionManifest, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.dump(manifest.to_dict(), f, default_flow_style=False, allow_unicode=True)


def load_partition_manifest(path: Path) -> PartitionManifest:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return PartitionManifest(**data)


# ─────────────────────────────────────────────────────────────────────────────
# Matched-pair control (CLAUDE.md §10, research/plan/plan.md §5.3)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MatchReport:
    """What the matched control actually achieved, column by column.

    The point of this object is that a failed match is REPORTED rather than
    assumed away. configs/partition/s1_control.yaml previously claimed matching
    that its construction (a plain IID split) did not provide, which would let a
    data-size or composition effect be reported as a missing-class effect.
    """

    target_class: str
    tolerance: float
    matched: bool
    attribute_key: str | None
    per_client: dict[str, dict[str, Any]]
    unmatched_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_class": self.target_class,
            "tolerance": self.tolerance,
            "matched": self.matched,
            "attribute_key": self.attribute_key,
            "per_client": self.per_client,
            "unmatched_reasons": self.unmatched_reasons,
        }


def build_attribute_index(
    ann_json: Path,
    key: str = "timeofday",
) -> dict[str, str]:
    """Map image name to one BDD100K frame attribute, e.g. timeofday or weather.

    Matching the control on such an attribute removes an obvious confound: a
    control drawn mostly from daytime frames is not a control for a night-heavy
    group. Optional — when no index is supplied the matcher records that this
    column was NOT matched instead of pretending it was.
    """
    from src.data.bdd100k import load_bdd100k_annotations

    out: dict[str, str] = {}
    for frame in load_bdd100k_annotations(ann_json):
        name = frame.get("name")
        if not name:
            continue
        value = (frame.get("attributes") or {}).get(key)
        if value is not None:
            out[str(name)] = str(value)
    return out


def _within(a: float, b: float, tolerance: float) -> bool:
    """Relative difference within tolerance; two zeros count as matched."""
    scale = max(abs(a), abs(b))
    if scale == 0:
        return True
    return abs(a - b) / scale <= tolerance


def partition_matched_control(
    image_names: list[str],
    label_dir: Path,
    reference: PartitionManifest,
    target_class: str,
    seed: int,
    partition_id: str,
    tolerance: float = 0.05,
    attribute_index: dict[str, str] | None = None,
    attribute_key: str | None = None,
    label_index: dict[str, dict[str, int]] | None = None,
    candidate_window: int = 64,
) -> tuple[PartitionManifest, MatchReport]:
    """Build the S1-Control matched pair for an S1 reference partition.

    Goal (CLAUDE.md §10): two groups as similar as feasible in image count, total
    bbox count, recording source and non-target class frequencies, differing
    primarily in whether the target class is PRESENT.

    Construction (plan.md §5.3): start from the reference client's own image set,
    then swap a minimal number of images holding no target boxes for images that do,
    choosing each replacement to have a similar total box count (and the same
    attribute value when an attribute index is given). The level aimed for is the
    median target-box count of the reference clients that already hold the class, so
    the control looks like a normal client rather than a target-saturated one.

    Starting from the reference set — rather than resampling from scratch — is what
    keeps the image count identical by construction and the non-target profile
    close, leaving the target class as the main difference.

    Returns:
        (manifest, report). The report says which columns matched within
        ``tolerance``; ``report.matched`` is False when any did not. A False report
        does not raise — the caller decides — but it must be published alongside any
        result, since §10 forbids attributing degradation to Missing-Class while
        major confounds remain.
    """
    if target_class not in TARGET_CLASSES:
        raise ValueError(f"target_class {target_class!r} not in {TARGET_CLASSES}")

    rng = random.Random(seed)
    index = label_index if label_index is not None else load_label_index(image_names, label_dir)

    ref_clients = sorted(reference.client_assignments)
    ref_target = {
        cid: reference.class_counts.get(cid, {}).get(target_class, 0)
        for cid in ref_clients
    }
    donors = [n for n in ref_target.values() if n > 0]
    # Clients that already hold the class define what "normal" looks like. With no
    # such client (e.g. S1d, where the class is absent everywhere) there is nothing
    # to match to, so fall back to the dataset-wide rate and say so.
    if donors:
        goal_target = int(statistics.median(donors))
    else:
        total = sum(c.get(target_class, 0) for c in index.values())
        per_image = total / max(len(index), 1)
        mean_size = statistics.mean(
            len(v) for v in reference.client_assignments.values()
        ) if reference.client_assignments else 0
        goal_target = int(per_image * mean_size)
        logger.warning(
            "partition_matched_control(%s): no reference client holds %s, so the "
            "target level is estimated from the dataset-wide rate. Treat this "
            "control as weaker evidence.",
            partition_id, target_class,
        )

    used: set[str] = set()
    for cid in ref_clients:
        used.update(reference.client_assignments[cid])

    # Donor pool: images holding the target class that the reference partition did
    # not use at all, so the control never overlaps the group it is matched to.
    pool = [
        n for n in image_names
        if n not in used and index.get(n, {}).get(target_class, 0) > 0
    ]
    rng.shuffle(pool)

    def total_boxes(name: str) -> int:
        return sum(index.get(name, {}).values())

    splits: dict[str, list[str]] = {}
    per_client_report: dict[str, dict[str, Any]] = {}
    reasons: list[str] = []
    donor_cursor = 0

    for cid in ref_clients:
        control = list(reference.client_assignments[cid])
        ref_counts = _sum_counts(control, index)
        current_target = ref_counts.get(target_class, 0)

        # Only images with ZERO target boxes may be removed; removing one that
        # holds the target would fight the goal.
        removable = sorted(
            (n for n in control if index.get(n, {}).get(target_class, 0) == 0),
            key=lambda n: (total_boxes(n), n),
        )
        removable_boxes = [total_boxes(n) for n in removable]
        swaps = 0

        # Non-target classes are the nuisance variables to hold constant. Choosing
        # a replacement purely by TOTAL box count was not enough: donor images
        # carry truck/motorcycle at the dataset rate, so hundreds of swaps drifted
        # those columns past tolerance even while the total stayed put. The cost
        # below is the running drift of the non-target VECTOR, normalised per class
        # so a rare class is not swamped by `car`.
        nuisance = [c for c in TARGET_CLASSES if c != target_class]
        scale = {c: max(ref_counts.get(c, 0), 1) for c in nuisance}
        drift = {c: 0 for c in nuisance}

        def drift_cost(remove_counts: dict[str, int], add_counts: dict[str, int]) -> float:
            return sum(
                abs(drift[c] - remove_counts.get(c, 0) + add_counts.get(c, 0)) / scale[c]
                for c in nuisance
            )

        while current_target < goal_target and donor_cursor < len(pool) and removable:
            donor = pool[donor_cursor]
            donor_cursor += 1
            donor_counts = index.get(donor, {})
            donor_boxes = total_boxes(donor)
            donor_attr = attribute_index.get(donor) if attribute_index else None

            # Evaluate only the candidates whose total box count is closest to the
            # donor's: a bounded window keeps this O(window) per swap instead of
            # O(len(removable)), which matters at ~4000 images per client.
            pivot = bisect.bisect_left(removable_boxes, donor_boxes)
            lo = max(0, pivot - candidate_window)
            hi = min(len(removable), pivot + candidate_window)

            best_pos: int | None = None
            best_cost: tuple[float, int, str] | None = None
            for pos in range(lo, hi):
                cand = removable[pos]
                cand_counts = index.get(cand, {})
                attr_penalty = (
                    0 if attribute_index is None
                    or attribute_index.get(cand) == donor_attr
                    else 1
                )
                cost = (drift_cost(cand_counts, donor_counts), attr_penalty, cand)
                if best_cost is None or cost < best_cost:
                    best_pos, best_cost = pos, cost
            if best_pos is None:
                break

            removed = removable.pop(best_pos)
            removable_boxes.pop(best_pos)
            removed_counts = index.get(removed, {})
            for c in nuisance:
                drift[c] += donor_counts.get(c, 0) - removed_counts.get(c, 0)

            control.remove(removed)
            control.append(donor)
            used.add(donor)
            current_target += donor_counts.get(target_class, 0)
            swaps += 1

        control.sort()
        splits[cid] = control
        new_counts = _sum_counts(control, index)

        # The GRAND total cannot match by definition: the control holds target
        # boxes the reference group does not. The meaningful invariants are the
        # image count and the NON-TARGET box profile — the target class is the
        # variable under study, everything else is a nuisance variable to hold
        # constant. The grand total is reported for transparency, not gated.
        ref_nuisance = sum(ref_counts.get(c, 0) for c in nuisance)
        new_nuisance = sum(new_counts.get(c, 0) for c in nuisance)
        checks: dict[str, bool] = {
            "image_count": len(control) == len(reference.client_assignments[cid]),
            "nuisance_boxes": _within(new_nuisance, ref_nuisance, tolerance),
        }
        for cls in nuisance:
            checks[f"boxes_{cls}"] = _within(
                new_counts.get(cls, 0), ref_counts.get(cls, 0), tolerance
            )

        attr_shift: float | None = None
        if attribute_index is not None:
            def _dist(names: list[str]) -> dict[str, float]:
                vals = [attribute_index.get(n, "unknown") for n in names]
                if not vals:
                    return {}
                return {v: vals.count(v) / len(vals) for v in set(vals)}

            d_new = _dist(control)
            d_ref = _dist(reference.client_assignments[cid])
            keys = set(d_new) | set(d_ref)
            attr_shift = max(
                (abs(d_new.get(k, 0.0) - d_ref.get(k, 0.0)) for k in keys), default=0.0
            )
            # plan.md §5.3 allows 5 percentage points on the attribute distribution.
            checks[f"attr_{attribute_key or 'attribute'}"] = attr_shift <= 0.05

        client_matched = all(checks.values())
        if not client_matched:
            failed = [k for k, ok in checks.items() if not ok]
            reasons.append(f"{cid}: " + ", ".join(failed))
        if current_target < goal_target:
            reasons.append(
                f"{cid}: target {target_class} boxes {current_target} < goal "
                f"{goal_target} (donor pool or removable images exhausted)"
            )

        per_client_report[cid] = {
            "images": len(control),
            "reference_images": len(reference.client_assignments[cid]),
            "swaps": swaps,
            "target_boxes": current_target,
            "reference_target_boxes": ref_target[cid],
            "goal_target_boxes": goal_target,
            "total_boxes": sum(new_counts.values()),
            "reference_total_boxes": sum(ref_counts.values()),
            "nuisance_boxes": new_nuisance,
            "reference_nuisance_boxes": ref_nuisance,
            "boxes": new_counts,
            "reference_boxes": ref_counts,
            "attribute_max_shift": attr_shift,
            "checks": checks,
            "matched": client_matched,
        }

    counts = {cid: _sum_counts(imgs, index) for cid, imgs in splits.items()}
    missing = {
        cid: [c for c in TARGET_CLASSES if counts[cid][c] == 0] for cid in splits
    }

    manifest = PartitionManifest(
        partition_id=partition_id,
        seed=seed,
        scenario="S1-Control",
        num_clients=len(splits),
        client_assignments=splits,
        class_counts=counts,
        missing_classes=missing,
    )
    report = MatchReport(
        target_class=target_class,
        tolerance=tolerance,
        matched=all(v["matched"] for v in per_client_report.values()) and not reasons,
        attribute_key=attribute_key,
        per_client=per_client_report,
        unmatched_reasons=reasons,
    )
    if not report.matched:
        logger.warning(
            "partition_matched_control(%s): control is NOT fully matched: %s. "
            "Publish this report with any S1-vs-control result; CLAUDE.md §10 "
            "forbids attributing degradation to Missing-Class while confounds remain.",
            partition_id, "; ".join(reasons),
        )
    return manifest, report


def save_match_report(report: MatchReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.dump(report.to_dict(), f, default_flow_style=False, allow_unicode=True)


# ─────────────────────────────────────────────────────────────────────────────
# Pooled re-split (diagnostic D3, ADR-014)
# ─────────────────────────────────────────────────────────────────────────────

def partition_pooled_iid(
    reference: PartitionManifest,
    label_dir: Path,
    seed: int,
    partition_id: str,
) -> PartitionManifest:
    """Re-deal EXACTLY the reference partition's images evenly over the same clients.

    The pooled data are identical to the reference (same images, same box totals of
    every class), so a reference-vs-pooled difference can only come from WHO holds
    the images — the Missing-Class distribution — not from how much bus data
    exists. That is the confound the matched control could not remove (it has
    2.03x the bus boxes, ADR-014).

    Stratified by class-presence signature: images are grouped by the set of
    classes they contain, each group is shuffled with ``seed``, and the groups are
    dealt round-robin with one running pointer, so every client receives the same
    image count (±1) and near-equal shares of each signature.
    """
    clients = sorted(reference.client_assignments)
    pool = sorted({n for imgs in reference.client_assignments.values() for n in imgs})
    index = load_label_index(pool, label_dir)
    strata: dict[tuple[str, ...], list[str]] = {}
    for name in pool:
        key = tuple(c for c in TARGET_CLASSES if index[name][c] > 0)
        strata.setdefault(key, []).append(name)

    rng = random.Random(seed)
    splits: dict[str, list[str]] = {cid: [] for cid in clients}
    k = 0
    for key in sorted(strata):
        group = list(strata[key])
        rng.shuffle(group)
        for name in group:
            splits[clients[k % len(clients)]].append(name)
            k += 1

    counts = {cid: _sum_counts(imgs, index) for cid, imgs in splits.items()}
    missing = {cid: [c for c in TARGET_CLASSES if counts[cid][c] == 0] for cid in clients}
    return PartitionManifest(
        partition_id=partition_id,
        seed=seed,
        scenario="S1-Pooled-IID",
        num_clients=len(clients),
        client_assignments=splits,
        class_counts=counts,
        missing_classes=missing,
    )
