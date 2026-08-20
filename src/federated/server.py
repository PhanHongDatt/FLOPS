from __future__ import annotations

from pathlib import Path
from typing import Any

import flwr as fl
from flwr.server import ServerConfig

from src.federated.strategies.fedavg import build_fedavg
from src.utils.logger import get_logger, log_metrics, setup_mlflow, end_run

logger = get_logger(__name__)

def _load_strategy_builder(algorithm: str):
    from src.federated.strategies import fedavg, fedprox, scaffold, fednova
    builders = {
        "FedAvg": fedavg.build_fedavg,
        "FedProx": fedprox.build_fedprox,
        "SCAFFOLD": lambda **kw: scaffold.Scaffold(**kw),
        "FedNova": lambda **kw: fednova.FedNova(**kw),
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


def run_fl_server(
    algorithm: str,
    fl_config: dict[str, Any],
    client_fn,
    run_dir: Path,
    mlflow_experiment: str,
    mlflow_uri: str = "http://mlflow:5000",
    server_address: str = "0.0.0.0:8080",
) -> None:
    run_id = setup_mlflow(mlflow_experiment, mlflow_uri, run_name=algorithm)
    logger.info("MLflow run_id: %s", run_id)

    strategy_builder = _load_strategy_builder(algorithm)
    strategy = strategy_builder(
        num_rounds=fl_config["num_rounds"],
        fraction_fit=fl_config["fraction_fit"],
        fraction_evaluate=fl_config["fraction_evaluate"],
        min_fit_clients=fl_config["num_clients"],
        min_evaluate_clients=fl_config["num_clients"],
        min_available_clients=fl_config["num_clients"],
    )

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
