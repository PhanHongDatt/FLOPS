"""Model package + evaluation protocol of a finished run (ADR-031). [ENGINEERING]

A packaged model is ``final_params.npz`` (ordered ``state_dict`` arrays, the checkpoint format of
src/experiments/checkpoint.py) plus ``model_meta.json``. The card carries the run identity, the
summary metrics and the evaluation protocol; the registry promotion rule compares protocols, so
two models are only ranked when they were scored the same way.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from src.data.bdd100k import TARGET_CLASSES

EVAL_PROTOCOL_FILE = "eval_protocol.json"
MODEL_META_FILE = "model_meta.json"
FINAL_PARAMS = Path("checkpoint") / "final_params.npz"
PACKAGE_FORMAT = "flops-npz-state-dict/v1"

# Keys that must match between two models before one may replace the other (registry policy
# default; configs/mlops/registry.yaml). ``model_weights`` is recorded but not compared: it says
# how to load the package, not how it was evaluated.
PROTOCOL_KEYS: tuple[str, ...] = ("image_size", "eval_conf", "eval_iou", "eval_set_id", "class_names")
_IMG_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm"}


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _val_image_names(data_yaml: Path) -> list[str]:
    data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8")) or {}
    if "val" not in data:
        raise ValueError(f"{data_yaml} has no 'val' entry")
    root = Path(data.get("path") or Path(data_yaml).parent)
    entry = Path(str(data["val"]))
    entry = entry if entry.is_absolute() else root / entry
    if entry.is_dir():
        return sorted(p.name for p in entry.iterdir() if p.suffix.lower() in _IMG_SUFFIXES)
    lines = [ln.strip() for ln in entry.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return sorted(Path(ln).name for ln in lines)


def eval_set_fingerprint(data_yaml: Path) -> dict[str, Any]:
    """Identity of the evaluation image set: count + hash of the sorted image file names.

    Names, not paths: the same BDD100K val set mounted at different roots (Kaggle, local, the
    COCO-id view of ADR-018) has the same fingerprint.
    """
    names = _val_image_names(data_yaml)
    digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()[:16]
    return {"eval_set_id": f"{len(names)}-{digest}", "eval_images": len(names)}


def write_eval_protocol(run_dir: Path, *, model_weights: str, image_size: int, eval_conf: float,
                        eval_iou: float, eval_data_yaml: Path | None,
                        class_names: tuple[str, ...] = TARGET_CLASSES) -> dict[str, Any]:
    """Write ``eval_protocol.json`` with the EFFECTIVE weights (after any ``coco:`` rewrite)."""
    protocol: dict[str, Any] = {
        "model_weights": str(model_weights),
        "image_size": int(image_size),
        "eval_conf": float(eval_conf),
        "eval_iou": float(eval_iou),
        "class_names": list(class_names),
    }
    if eval_data_yaml is not None and Path(eval_data_yaml).exists():
        try:
            protocol.update(eval_set_fingerprint(Path(eval_data_yaml)))
        except (OSError, ValueError, yaml.YAMLError) as exc:
            # bookkeeping must not end a run; without eval_set_id the model is simply not promotable
            import logging
            logging.getLogger(__name__).warning("eval set fingerprint of %s failed: %r", eval_data_yaml, exc)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / EVAL_PROTOCOL_FILE).write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return protocol


def read_summary_metrics(path: Path) -> dict[str, float]:
    """``metrics.csv`` (columns metric,value) -> {metric: value} for numeric values."""
    out: dict[str, float] = {}
    if not Path(path).is_file():
        return out
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                out[str(row["metric"])] = float(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.is_file() else {}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def build_model_card(run_dir: Path, eval_data_yaml: Path | None = None,
                     weights_override: str | None = None) -> dict[str, Any]:
    """Collect everything the registry needs from a finished run directory.

    Raises FileNotFoundError when the run has no ``checkpoint/final_params.npz`` (an unfinished
    run cannot be registered). ``eval_data_yaml`` / ``weights_override`` fill the protocol of
    runs made before ``eval_protocol.json`` existed (backfill of downloaded Kaggle runs).
    """
    run_dir = Path(run_dir)
    params = run_dir / FINAL_PARAMS
    if not params.is_file():
        raise FileNotFoundError(f"{params} not found — only finished runs can be registered")
    config = _read_yaml(run_dir / "config.yaml")
    env = _read_json(run_dir / "environment.json")
    protocol = _read_json(run_dir / EVAL_PROTOCOL_FILE)
    protocol_source = EVAL_PROTOCOL_FILE if protocol else "config.yaml"
    if not protocol:
        model = config.get("model", {})
        evaluation = config.get("evaluation", {})
        protocol = {k: v for k, v in {
            "model_weights": model.get("weights"),
            "image_size": model.get("image_size"),
            "eval_conf": evaluation.get("conf"),
            "eval_iou": evaluation.get("iou"),
            "class_names": list(config.get("classes") or TARGET_CLASSES),
        }.items() if v is not None}
    if weights_override:
        protocol["model_weights"] = weights_override
        protocol_source += "+cli"
    if eval_data_yaml is not None:
        protocol.update(eval_set_fingerprint(Path(eval_data_yaml)))

    import numpy as np
    with np.load(params) as data:
        num_arrays = len(data.files)

    return {
        "format": PACKAGE_FORMAT,
        "run_name": run_dir.name,
        "exp_id": env.get("exp_id"),
        "run_class": env.get("run_class") or config.get("run_class"),
        "seed": env.get("seed"),
        "partition_id": env.get("partition_id"),
        "git_commit": env.get("git_commit"),
        "metrics": read_summary_metrics(run_dir / "metrics.csv"),
        "protocol": protocol,
        "protocol_source": protocol_source,
        "params_file": str(params),
        "params_sha256": _sha256(params),
        "num_arrays": num_arrays,
    }


def write_model_meta(run_dir: Path, card: dict[str, Any]) -> Path:
    meta = {k: v for k, v in card.items() if k != "params_file"}
    path = Path(run_dir) / MODEL_META_FILE
    path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return path


def load_packaged_model(package_dir: Path, verify: bool = True) -> Any:
    """Rebuild the YOLO model of a package directory (``final_params.npz`` + ``model_meta.json``).

    Needs ultralytics. ``set_parameters`` checks the array count strictly, so a package built for
    a different architecture/head fails loudly instead of loading a partial model.
    """
    from src.experiments.checkpoint import load_global_checkpoint
    from src.model.yolo_wrapper import build_model, set_parameters

    package_dir = Path(package_dir)
    meta = _read_json(package_dir / MODEL_META_FILE)
    if meta.get("format") != PACKAGE_FORMAT:
        raise ValueError(f"{package_dir}: unknown package format {meta.get('format')!r}")
    params = package_dir / "final_params.npz"
    if verify and _sha256(params) != meta.get("params_sha256"):
        raise ValueError(f"{params}: sha256 does not match model_meta.json")
    model = build_model(str(meta["protocol"]["model_weights"]))
    set_parameters(model, load_global_checkpoint(params))
    return model
