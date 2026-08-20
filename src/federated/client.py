from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import flwr as fl
from flwr.common import NDArrays, Scalar

from src.model.yolo_wrapper import (
    build_model,
    evaluate,
    get_parameters,
    set_parameters,
    train_one_round,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class YOLOFlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: str,
        partition_id: int,
        data_yaml: Path,
        weights: str,
        train_config: dict[str, Any],
        run_dir: Path,
    ) -> None:
        self.client_id = client_id
        self.partition_id = partition_id
        self.data_yaml = data_yaml
        self.train_config = train_config
        self.run_dir = run_dir
        self.model = build_model(weights)
        self._round = 0

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        return get_parameters(self.model)

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        self._round += 1
        set_parameters(self.model, parameters)

        metrics = train_one_round(
            model=self.model,
            data_yaml=self.data_yaml,
            epochs=self.train_config["local_epochs"],
            batch=self.train_config["batch_size"],
            img_size=self.train_config["image_size"],
            lr0=self.train_config["lr0"],
            device=self.train_config.get("device", 0),
            project=self.run_dir / "clients" / self.client_id,
            name=f"round_{self._round}",
        )

        logger.info(
            "Client %s round %d fit complete — metrics: %s",
            self.client_id, self._round, metrics,
        )

        num_examples = self.train_config.get("num_train_examples", 1)
        return get_parameters(self.model), num_examples, metrics

    def evaluate(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        set_parameters(self.model, parameters)

        metrics = evaluate(
            model=self.model,
            data_yaml=self.data_yaml,
            img_size=self.train_config["image_size"],
            conf=self.train_config["conf"],
            iou=self.train_config["iou"],
            device=self.train_config.get("device", 0),
        )

        loss = 1.0 - metrics.get("mAP50", 0.0)
        num_examples = self.train_config.get("num_val_examples", 1)
        return loss, num_examples, metrics


def make_client_fn(
    partition_manifests: dict[str, Path],
    data_yaml_map: dict[str, Path],
    weights: str,
    train_config: dict[str, Any],
    run_dir: Path,
):
    def client_fn(cid: str) -> fl.client.NumPyClient:
        return YOLOFlowerClient(
            client_id=cid,
            partition_id=int(cid),
            data_yaml=data_yaml_map[cid],
            weights=weights,
            train_config=train_config,
            run_dir=run_dir,
        )
    return client_fn
