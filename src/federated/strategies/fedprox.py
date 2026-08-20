"""FedProx strategy wrapper.

[LITERATURE] Li et al., 2020, MLSys — arXiv:1812.06127
Role: reproduction — baseline for heterogeneous FL.

Verified: FedProx is built-in to Flower 1.21.0
(framework/py/flwr/server/strategy/fedprox.py, subclass of FedAvg).
Server-side is identical to FedAvg; the proximal term is added client-side
to the loss function: L' = L + (mu/2) * ||w - w_t||^2
"""
from __future__ import annotations

from typing import Any

import flwr as fl
from flwr.server.strategy import FedProx as _FedProx


def build_fedprox(
    proximal_mu: float = 0.01,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 4,
    min_evaluate_clients: int = 4,
    min_available_clients: int = 4,
    **kwargs: Any,
) -> fl.server.strategy.Strategy:
    return _FedProx(
        proximal_mu=proximal_mu,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        **kwargs,
    )
