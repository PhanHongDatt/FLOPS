"""FedNova strategy — custom implementation.

[LITERATURE] Wang et al., 2020, NeurIPS — arXiv:2007.07481
Role: reproduction — baseline for objective inconsistency correction.

Formula (from paper Section 3, verified against JYWa/FedNova reference impl):
  Client i:
    - runs tau_i local SGD steps
    - accumulates normalized update d_i (magnitude-normalized gradient direction)
    - returns (d_i, tau_i, n_i)
  Server aggregation:
    tau_eff = sum_i(p_i * tau_i)   where p_i = n_i / sum(n)
    x_new   = x - tau_eff * sum_i( p_i * (d_i / tau_i) )

  Equivalent formulation using cumulative gradient cum_grad = tau_i * eta_l * d_i:
    x_new = x - eta_g * sum_i( (p_i * tau_eff / tau_i) * (cum_grad_i / eta_l) / tau_eff )
          = x - sum_i( p_i * (tau_eff / tau_i) * cum_grad_i )

Reference: https://github.com/JYWa/FedNova/blob/master/distoptim/FedNova.py

⚠️  This strategy REQUIRES a FedNova-aware client that:
  1. Tracks tau_i (local step count) during training.
  2. Returns (normalized_update, tau_i) via fit result.
See src/federated/client_variants.py — FedNovaClient.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import flwr as fl
from flwr.common import (
    FitIns, FitRes,
    NDArrays, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FedNova(FedAvg):
    """FedNova: normalized averaging for heterogeneous local step counts."""

    def __init__(
        self,
        eta_global: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.eta_global = eta_global
        self._prev_params: NDArrays | None = None

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        self._prev_params = parameters_to_ndarrays(parameters)
        config = {"server_round": server_round}
        fit_ins = FitIns(parameters, config)

        clients = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_available_clients,
        )
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list,
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        if not results or self._prev_params is None:
            return None, {}

        # Extract updates: each client returns y_i (updated params) + tau_i + n_i
        client_data: list[tuple[NDArrays, float, float]] = []
        for _, res in results:
            y_i = parameters_to_ndarrays(res.parameters)
            tau_i = float(res.metrics.get("tau_i", 1)) if res.metrics else 1.0
            n_i = float(res.num_examples)
            client_data.append((y_i, tau_i, n_i))

        total_n = sum(n for _, _, n in client_data) or 1.0
        # tau_eff = sum(p_i * tau_i)
        tau_eff = sum((n / total_n) * tau for _, tau, n in client_data)

        # Compute normalized aggregated delta:
        # x_new = x - sum( p_i * (tau_eff / tau_i) * (x - y_i) )
        aggregated: NDArrays | None = None
        for y_i, tau_i, n_i in client_data:
            p_i = n_i / total_n
            coef = p_i * (tau_eff / max(tau_i, 1e-8))
            delta_i = [xp - y for xp, y in zip(self._prev_params, y_i)]
            if aggregated is None:
                aggregated = [coef * d for d in delta_i]
            else:
                for j, d in enumerate(delta_i):
                    aggregated[j] = aggregated[j] + coef * d

        # x_new = x - eta_g * aggregated_delta
        new_params = [
            xp - self.eta_global * a
            for xp, a in zip(self._prev_params, aggregated)
        ] if aggregated is not None else self._prev_params

        return (
            ndarrays_to_parameters(new_params),
            {"tau_eff": float(tau_eff), "n_selected": len(results)},
        )
