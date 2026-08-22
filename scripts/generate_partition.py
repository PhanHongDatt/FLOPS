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
    PartitionManifest,
    partition_iid,
    partition_missing_class,
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
        )

    raise ValueError(
        f"Unknown scenario '{scenario}'. Expected one of: S0, S1, S1-Control."
    )


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

    manifest = _dispatch_partition(config, images, label_dir)
    manifest_path = out_dir / "manifest.yaml"
    save_partition_manifest(manifest, manifest_path)
    logger.info("Wrote manifest: %s", manifest_path)

    data_yaml_map: dict[str, Path] = {}
    for client_id, client_images in manifest.client_assignments.items():
        train_list = _write_client_image_list(
            client_id, client_images, yolo_root, out_dir
        )
        data_yaml = _write_client_data_yaml(client_id, train_list, yolo_root, out_dir)
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
