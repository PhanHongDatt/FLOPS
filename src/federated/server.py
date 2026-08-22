from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import flwr as fl
from flwr.common import NDArrays, Scalar
from flwr.server import ServerConfig

from src.federated.strategies.fedavg import build_fedavg
from src.utils.logger import get_logger, log_metrics, setup_mlflow, end_run

logger = get_logger(__name__)

def _load_strategy_builder(algorithm: str, param_names: list[str] | None = None):
    from src.federated.strategies import fedavg, fedprox, scaffold, fednova
    from src.federated.strategies.class_aware_agg import build_class_aware_agg

    builders = {
        "FedAvg": fedavg.build_fedavg,
        "FedProx": fedprox.build_fedprox,
        "SCAFFOLD": lambda **kw: scaffold.Scaffold(**kw),
        "FedNova": lambda **kw: fednova.FedNova(**kw),
        "ClassAwareAgg": lambda **kw: build_class_aware_agg(
            param_names=param_names or [],
            **kw,
        ),
    }
    if algorithm not in builders:
        raise ValueError(f"Unknown FL algorithm: {algorithm}")
    return builders[algorithm]


class MetricsCallback:
    """Log per-round global metrics to MLflow."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self._rows: list[dict[str, Any]] = []

    def on_round_end(self, server_round: int, metrics: dict[str, float]) -> None:
        row = {"round": server_round, **metrics}
        self._rows.append(row)
        log_metrics(metrics, step=server_round)
        logger.info("Round %d metrics: %s", server_round, metrics)

    def save(self) -> None:
        import csv
        out = self.run_dir / "round_metrics.csv"
        if not self._rows:
            return
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._rows[0].keys())
            writer.writeheader()
            writer.writerows(self._rows)


def _build_centralized_evaluate_fn(
    global_data_yaml: Path,
    eval_config: dict[str, Any],
    run_dir: Path,
    num_rounds: int,
    weights: str = "yolov8n.pt",
) -> Callable[[int, NDArrays, dict[str, Scalar]], Optional[tuple[float, dict[str, Scalar]]]]:
    """Build a Flower centralized evaluate_fn that runs per-class eval on the
    global val set after each round and persists per_class_metrics.csv +
    final_params.npz on the final round.

    Rationale: Flower's default fit-metrics aggregation drops non-scalar per-class
    entries. Centralized eval via evaluate_fn preserves them (CLAUDE.md §13, §21).
    """
    from src.model.yolo_wrapper import build_model, evaluate, set_parameters
    from src.evaluation.metrics import (
        flatten_per_class_metrics,
        save_per_class_metrics,
    )

    # Lazy-init to avoid loading YOLO in server process before first round
    _holder: dict[str, Any] = {"model": None}

    def evaluate_fn(
        server_round: int,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        if _holder["model"] is None:
            _holder["model"] = build_model(weights)
        model = _holder["model"]
        set_parameters(model, parameters)

        metrics = evaluate(
            model=model,
            data_yaml=global_data_yaml,
            img_size=eval_config["image_size"],
            conf=eval_config["conf"],
            iou=eval_config["iou"],
            device=eval_config.get("device", 0),
        )
        logger.info("Round %d centralized eval: %s", server_round, metrics)

        if server_round == num_rounds:
            save_per_class_metrics(
                flatten_per_class_metrics(metrics),
                run_dir / "per_class_metrics.csv",
            )
            ckpt_dir = run_dir / "checkpoint"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            np.savez(ckpt_dir / "final_params.npz", *parameters)
            logger.info("Saved final per-class metrics + params to %s", run_dir)

        loss = 1.0 - float(metrics.get("mAP50", 0.0))
        return loss, {k: float(v) for k, v in metrics.items()}

    return evaluate_fn


def run_fl_server(
    algorithm: str,
    fl_config: dict[str, Any],
    client_fn,
    run_dir: Path,
    mlflow_experiment: str,
    mlflow_uri: str = "http://mlflow:5000",
    server_address: str = "0.0.0.0:8080",
    param_names: list[str] | None = None,
    global_data_yaml: Path | None = None,
    eval_config: dict[str, Any] | None = None,
) -> None:
    run_id = setup_mlflow(mlflow_experiment, mlflow_uri, run_name=algorithm)
    logger.info("MLflow run_id: %s", run_id)

    strategy_builder = _load_strategy_builder(algorithm, param_names=param_names)

    strategy_kwargs: dict[str, Any] = dict(
        num_rounds=fl_config["num_rounds"],
        fraction_fit=fl_config["fraction_fit"],
        fraction_evaluate=fl_config["fraction_evaluate"],
        min_fit_clients=fl_config["num_clients"],
        min_evaluate_clients=fl_config["num_clients"],
        min_available_clients=fl_config["num_clients"],
    )

    if global_data_yaml is not None and eval_config is not None:
        strategy_kwargs["evaluate_fn"] = _build_centralized_evaluate_fn(
            global_data_yaml=global_data_yaml,
            eval_config=eval_config,
            run_dir=run_dir,
            num_rounds=fl_config["num_rounds"],
        )
    else:
        logger.warning(
            "run_fl_server called without global_data_yaml/eval_config — "
            "per_class_metrics.csv and final_params.npz WILL NOT be produced. "
            "This violates CLAUDE.md §21 artifact contract."
        )

    strategy = strategy_builder(**strategy_kwargs)

    callback = MetricsCallback(run_dir)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=fl_config["num_clients"],
        config=ServerConfig(num_rounds=fl_config["num_rounds"]),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.25},
    )

    callback.save()
    end_run()
    logger.info("FL run complete. Artifacts in %s", run_dir)
