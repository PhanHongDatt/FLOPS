from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO

from src.data.bdd100k import TARGET_CLASSES


def build_model(weights: str = "yolov8n.pt") -> YOLO:
    return YOLO(weights)


def get_parameters(model: YOLO) -> list[np.ndarray]:
    return [
        val.cpu().numpy()
        for val in model.model.state_dict().values()
    ]


def set_parameters(model: YOLO, parameters: list[np.ndarray]) -> None:
    state_dict = model.model.state_dict()
    new_state = {
        k: torch.tensor(v, dtype=state_dict[k].dtype)
        for k, v in zip(state_dict.keys(), parameters)
    }
    model.model.load_state_dict(new_state, strict=True)


def train_one_round(
    model: YOLO,
    data_yaml: Path,
    epochs: int,
    batch: int,
    img_size: int,
    lr0: float,
    device: int | str,
    project: Path,
    name: str,
) -> dict[str, float]:
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=img_size,
        optimizer="SGD",
        lr0=lr0,
        device=device,
        project=str(project),
        name=name,
        exist_ok=True,
        verbose=False,
    )
    # Extract last epoch metrics
    metrics: dict[str, float] = {}
    if results and hasattr(results, "results_dict"):
        metrics = {k: float(v) for k, v in results.results_dict.items()}
    return metrics


def evaluate(
    model: YOLO,
    data_yaml: Path,
    img_size: int,
    conf: float,
    iou: float,
    device: int | str,
) -> dict[str, float]:
    results = model.val(
        data=str(data_yaml),
        imgsz=img_size,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )
    metrics: dict[str, float] = {}
    if results:
        metrics["mAP50"] = float(results.box.map50)
        metrics["mAP50-95"] = float(results.box.map)
        for i, cls_name in enumerate(TARGET_CLASSES):
            if i < len(results.box.ap_class_index):
                metrics[f"AP50_{cls_name}"] = float(results.box.ap50[i])
                metrics[f"AP_{cls_name}"] = float(results.box.ap[i])
    return metrics
