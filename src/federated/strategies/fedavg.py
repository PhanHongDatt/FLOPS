"""FedAvg strategy wrapper.

[LITERATURE] McMahan et al., 2017, AISTATS.
Role: reproduction — used as primary baseline.
See research/evidence/literature_registry.yaml entry: fedavg.
"""
from __future__ import annotations

from typing import Any

import flwr as fl
from flwr.server.strategy import FedAvg as _FedAvg


def build_fedavg(
    num_rounds: int,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 4,
    min_evaluate_clients: int = 4,
    min_available_clients: int = 4,
    **kwargs: Any,
) -> fl.server.strategy.Strategy:
    return _FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        **kwargs,
    )
