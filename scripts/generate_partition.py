"""Generate partition manifest + per-client YOLO data.yaml from a partition config.

Bridges the gap between:
  - configs/partition/*.yaml  (partition CONFIG: scenario/seed/num_clients/missing_map)
  - runtime artifacts required by scripts/run_fl_experiment.py:
      * PartitionManifest YAML (loaded by src.data.partitioner.load_partition_manifest)
      * Per-client data_C{i}.yaml (YOLO data.yaml pointing to per-client image list)

Output layout:
  <output_dir>/<partition_id>/
    manifest.yaml
    C0_train.txt, C1_train.txt, ...     # newline-separated absolute image paths
    data_C0.yaml, data_C1.yaml, ...     # YOLO data.yaml per client

Val set is shared globally (CLAUDE.md §14: same evaluator across methods).

Usage:
  python scripts/generate_partition.py \\
    --partition-config configs/partition/s0_iid.yaml \\
    --yolo-root data/bdd100k_yolo \\
    --output-dir data/partitions
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from src.data.bdd100k import TARGET_CLASSES
from src.data.partitioner import (
    MatchReport,
    PartitionManifest,
    build_attribute_index,
    load_partition_manifest,
    partition_iid,
    partition_matched_control,
    partition_pooled_iid,
    partition_server_sample,
    partition_missing_class,
    save_match_report,
    save_partition_manifest,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _list_train_images(yolo_root: Path) -> list[str]:
    train_dir = yolo_root / "images" / "train"
    if not train_dir.exists():
        raise FileNotFoundError(
            f"YOLO train images dir not found: {train_dir}. "
            f"Run scripts/prepare_bdd100k.py first."
        )
    images = sorted(p.name for p in train_dir.glob("*.jpg"))
    if not images:
        raise RuntimeError(f"No .jpg images found in {train_dir}")
    return images


def _dispatch_partition(
    config: dict[str, Any],
    images: list[str],
    label_dir: Path,
) -> PartitionManifest:
    scenario = config["scenario"]
    seed = int(config["seed"])
    num_clients = int(config["num_clients"])
    partition_id = config["partition_id"]

    if scenario == "S0":
        return partition_iid(
            image_names=images,
            label_dir=label_dir,
            num_clients=num_clients,
            seed=seed,
            partition_id=partition_id,
            per_client=config.get("per_client"),
        )

    if scenario in ("S1", "S1-Control"):
        missing_map = config.get("missing_map") or {}
        return partition_missing_class(
            image_names=images,
            label_dir=label_dir,
            num_clients=num_clients,
            missing_map=missing_map,
            seed=seed,
            partition_id=partition_id,
            scenario=scenario,
            per_client=config.get("per_client"),
        )

    raise ValueError(
        f"Unknown scenario '{scenario}'. Expected one of: "
        "S0, S1, S1-Control, S1-Control-Matched."
    )


def _dispatch_matched_control(
    config: dict[str, Any],
    images: list[str],
    label_dir: Path,
    config_dir: Path,
    output_dir: Path,
) -> tuple[PartitionManifest, MatchReport]:
    """Build the matched-pair control for an existing S1 partition (§5.3, §10).

    Needs `reference_manifest` (path to the S1 manifest it must match) and
    `target_class`. `attribute_json` + `match_attribute` are optional: with them the
    control is also matched on a BDD100K frame attribute such as timeofday;
    without them the match report records that column as not matched rather than
    implying it was.
    """
    ref_rel = config.get("reference_manifest")
    if not ref_rel:
        raise ValueError(
            "scenario S1-Control-Matched requires 'reference_manifest' — the S1 "
            "partition manifest this control must be matched against."
        )
    ref_path = Path(ref_rel)
    for candidate in (ref_path, config_dir / ref_rel, output_dir / ref_rel):
        if candidate.exists():
            ref_path = candidate
            break
    else:
        raise FileNotFoundError(
            f"reference_manifest not found: {ref_rel}. Generate the S1 partition first."
        )

    reference = load_partition_manifest(ref_path)
    logger.info(
        "Matching control against %s (scenario=%s, clients=%d)",
        ref_path, reference.scenario, reference.num_clients,
    )

    attribute_index = None
    attribute_key = config.get("match_attribute")
    ann_json = config.get("attribute_json")
    if ann_json and attribute_key:
        ann_path = Path(ann_json)
        if ann_path.exists():
            attribute_index = build_attribute_index(ann_path, key=str(attribute_key))
            logger.info(
                "Loaded attribute index %r for %d images", attribute_key, len(attribute_index)
            )
        else:
            logger.warning(
                "attribute_json %s not found — control will NOT be matched on %r "
                "and the match report will say so.", ann_path, attribute_key,
            )
            attribute_key = None
    elif attribute_key:
        logger.warning(
            "match_attribute=%r given without attribute_json — skipping attribute "
            "matching and recording it as unmatched.", attribute_key,
        )
        attribute_key = None

    return partition_matched_control(
        image_names=images,
        label_dir=label_dir,
        reference=reference,
        target_class=str(config["target_class"]),
        seed=int(config["seed"]),
        partition_id=str(config["partition_id"]),
        tolerance=float(config.get("tolerance", 0.05)),
        attribute_index=attribute_index,
        attribute_key=attribute_key,
    )


def _dispatch_pooled(config: dict[str, Any], label_dir: Path, output_dir: Path) -> PartitionManifest:
    """D3 (ADR-014): the reference partition's own images, re-dealt evenly."""
    ref = config.get("reference_manifest")
    if not ref:
        raise ValueError("scenario S1-Pooled-IID requires 'reference_manifest'")
    ref_path = Path(ref) if Path(ref).is_absolute() else output_dir / ref
    if not ref_path.exists():
        raise FileNotFoundError(f"reference manifest {ref_path} not found — generate it first")
    return partition_pooled_iid(
        reference=load_partition_manifest(ref_path),
        label_dir=label_dir,
        seed=int(config["seed"]),
        partition_id=str(config["partition_id"]),
    )


def _dispatch_server_sample(config: dict[str, Any], images: list[str], label_dir: Path,
                            output_dir: Path) -> PartitionManifest:
    """ADR-015: labelled sample held by the server, disjoint from every listed client partition."""
    exclude: set[str] = set()
    for ref in config.get("exclude_manifests") or []:
        ref_path = Path(ref) if Path(ref).is_absolute() else output_dir / ref
        if not ref_path.exists():
            raise FileNotFoundError(f"exclude manifest {ref_path} not found — generate it first")
        for imgs in load_partition_manifest(ref_path).client_assignments.values():
            exclude.update(imgs)
    return partition_server_sample(
        image_names=images, label_dir=label_dir, exclude=exclude,
        n_images=int(config["n_images"]), seed=int(config["seed"]),
        partition_id=str(config["partition_id"]),
        min_boxes_per_class=int(config.get("min_boxes_per_class", 1)),
    )


def _isolated_yolo_root(manifest: PartitionManifest, yolo_root: Path, out_dir: Path) -> Path:
    """A private YOLO tree for the manifest's images: linked images, COPIED label files (so the
    label folder — and Ultralytics' cache next to it — is its own), shared val linked as is."""
    import os
    import shutil

    root = out_dir / "yolo"
    img_dir, lbl_dir = root / "images" / "train", root / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for imgs in manifest.client_assignments.values():
        for name in imgs:
            src_img = (yolo_root / "images" / "train" / name).resolve()
            dst_img = img_dir / name
            if not dst_img.exists():
                try:
                    os.symlink(src_img, dst_img)
                except OSError:
                    shutil.copy2(src_img, dst_img)
            src_lbl = yolo_root / "labels" / "train" / (Path(name).stem + ".txt")
            if src_lbl.exists():
                shutil.copy2(src_lbl, lbl_dir / src_lbl.name)
    for sub in ("images", "labels"):
        dst = root / sub / "val"
        if not dst.exists():
            try:
                os.symlink((yolo_root / sub / "val").resolve(), dst, target_is_directory=True)
            except OSError:
                shutil.copytree(yolo_root / sub / "val", dst)
    return root


def _write_client_image_list(
    client_id: str,
    image_names: list[str],
    yolo_root: Path,
    out_dir: Path,
) -> Path:
    train_img_dir = (yolo_root / "images" / "train").resolve()
    out_path = out_dir / f"{client_id}_train.txt"
    lines = [str(train_img_dir / name) for name in image_names]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _write_client_data_yaml(
    client_id: str,
    train_list_path: Path,
    yolo_root: Path,
    out_dir: Path,
) -> Path:
    yolo_root_abs = yolo_root.resolve()
    data = {
        "path": str(yolo_root_abs),
        "train": str(train_list_path.resolve()),
        "val": "images/val",  # shared global val (CLAUDE.md §14)
        "nc": len(TARGET_CLASSES),
        "names": list(TARGET_CLASSES),
    }
    out_path = out_dir / f"data_{client_id}.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False)
    return out_path


def generate_partition_artifacts(
    partition_config_path: Path,
    yolo_root: Path,
    output_dir: Path,
) -> tuple[Path, dict[str, Path]]:
    """Produce manifest.yaml and per-client data_Ci.yaml files.

    Returns (manifest_path, {client_id: data_yaml_path}).
    """
    with partition_config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    partition_id = config["partition_id"]
    out_dir = output_dir / partition_id
    out_dir.mkdir(parents=True, exist_ok=True)

    label_dir = yolo_root / "labels" / "train"
    if not label_dir.exists():
        raise FileNotFoundError(f"YOLO train labels dir not found: {label_dir}")

    images = _list_train_images(yolo_root)
    logger.info("Loaded %d train images from %s", len(images), yolo_root)

    if config["scenario"] == "S1-Control-Matched":
        manifest, report = _dispatch_matched_control(
            config, images, label_dir, partition_config_path.parent, output_dir
        )
        save_match_report(report, out_dir / "match_report.yaml")
        logger.info("Wrote match report: %s (matched=%s)",
                    out_dir / "match_report.yaml", report.matched)
        if not report.matched:
            logger.warning(
                "Control for %s is NOT fully matched. Any S1-vs-control result must "
                "be published together with match_report.yaml (CLAUDE.md §10).",
                config["partition_id"],
            )
    elif config["scenario"] == "S1-Pooled-IID":
        manifest = _dispatch_pooled(config, label_dir, output_dir)
    elif config["scenario"] == "Server-Sample":
        manifest = _dispatch_server_sample(config, images, label_dir, output_dir)
    else:
        manifest = _dispatch_partition(config, images, label_dir)

    manifest_path = out_dir / "manifest.yaml"
    save_partition_manifest(manifest, manifest_path)
    logger.info("Wrote manifest: %s", manifest_path)

    data_yaml_map: dict[str, Path] = {}
    # The server sample gets its own YOLO tree (ADR-015): Ultralytics keys its label cache on
    # the label FOLDER (labels/train.cache), so a server fine-tune sharing labels/train with
    # the clients overwrote their cache and concurrent client rewrites corrupted it
    # (s8c_g B2 v1: every client failed from round 2 with "UnpicklingError: invalid load key").
    src_root = _isolated_yolo_root(manifest, yolo_root, out_dir) if config["scenario"] == "Server-Sample" else yolo_root
    for client_id, client_images in manifest.client_assignments.items():
        train_list = _write_client_image_list(
            client_id, client_images, src_root, out_dir
        )
        data_yaml = _write_client_data_yaml(client_id, train_list, src_root, out_dir)
        data_yaml_map[client_id] = data_yaml
        logger.info(
            "Client %s: %d images | class_counts=%s | missing=%s",
            client_id,
            len(client_images),
            manifest.class_counts.get(client_id, {}),
            manifest.missing_classes.get(client_id, []),
        )

    return manifest_path, data_yaml_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--partition-config",
        type=Path,
        required=True,
        help="Path to partition config YAML (e.g. configs/partition/s0_iid.yaml)",
    )
    ap.add_argument(
        "--yolo-root",
        type=Path,
        required=True,
        help="YOLO-formatted BDD100K root produced by prepare_bdd100k.py",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/partitions"),
        help="Directory to write per-partition artifacts (default: data/partitions)",
    )
    args = ap.parse_args()

    manifest_path, data_yaml_map = generate_partition_artifacts(
        partition_config_path=args.partition_config,
        yolo_root=args.yolo_root,
        output_dir=args.output_dir,
    )
    logger.info(
        "Done. Manifest: %s | data.yaml files: %d",
        manifest_path,
        len(data_yaml_map),
    )


if __name__ == "__main__":
    main()
