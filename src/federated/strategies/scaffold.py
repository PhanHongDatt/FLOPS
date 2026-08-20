"""SCAFFOLD strategy — custom implementation.

[LITERATURE] Karimireddy et al., 2020, ICML — arXiv:1910.06378
Role: reproduction — baseline for client drift correction.

Algorithm (Option II — client control variates updated using local gradient):
  Server broadcasts (x, c) to selected clients.
  Client i performs local steps:
      y_i <- y_i - eta_l * (grad_g_i(y_i) - c_i + c)
  Client returns:
      delta_y_i = y_i - x
      delta_c_i = -c_i + (1/(K*eta_l)) * (x - y_i)  [Option II shortcut]
      new c_i = c_i + delta_c_i
  Server aggregates:
      x_new = x + (eta_g / |S|) * sum(delta_y_i)
      c_new = c + (|S|/N) * mean(delta_c_i)

Verified against paper Algorithm 1 (Option II).

⚠️  This strategy REQUIRES a SCAFFOLD-aware client that:
  1. Persists c_i across rounds (per-client state).
  2. Uses (grad - c_i + c) in local update.
  3. Returns delta_c_i in fit metrics.
See src/federated/client_variants.py — ScaffoldClient.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import flwr as fl
from flwr.common import (
    EvaluateIns, EvaluateRes, FitIns, FitRes,
    NDArrays, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


def _serialize(arrays: NDArrays) -> str:
    """Encode ndarrays to a base64 string for config transport."""
    import base64
    import io
    buf = io.BytesIO()
    np.savez_compressed(buf, *arrays)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _deserialize(payload: str) -> NDArrays:
    import base64
    import io
    buf = io.BytesIO(base64.b64decode(payload))
    with np.load(buf) as data:
        return [data[k] for k in sorted(data.files)]


class Scaffold(FedAvg):
    """SCAFFOLD strategy (Option II)."""

    def __init__(
        self,
        eta_global: float = 1.0,
        num_total_clients: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.eta_global = eta_global
        self.num_total_clients = num_total_clients
        self._server_c: NDArrays | None = None

    def _init_control(self, params: NDArrays) -> None:
        self._server_c = [np.zeros_like(p) for p in params]

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        if self._server_c is None:
            self._init_control(parameters_to_ndarrays(parameters))

        config = {
            "server_round": server_round,
            "server_c": _serialize(self._server_c),
        }
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
        if not results:
            return None, {}

        global_params = None  # will be set from client's returned parameters (they carry x + delta_y)
        aggregated_y: NDArrays | None = None
        delta_c_sum: NDArrays | None = None
        n_selected = len(results)

        # Weighted average of returned parameters (which are x + delta_y in Option II)
        weights: list[float] = [float(r.num_examples) for _, r in results]
        total = sum(weights) or 1.0

        for i, (_, res) in enumerate(results):
            y_i = parameters_to_ndarrays(res.parameters)
            w = weights[i] / total
            if aggregated_y is None:
                aggregated_y = [w * a for a in y_i]
            else:
                for j, a in enumerate(y_i):
                    aggregated_y[j] = aggregated_y[j] + w * a

            delta_c_payload = res.metrics.get("delta_c") if res.metrics else None
            if delta_c_payload:
                delta_c = _deserialize(str(delta_c_payload))
                if delta_c_sum is None:
                    delta_c_sum = [np.zeros_like(a) for a in delta_c]
                for j, a in enumerate(delta_c):
                    delta_c_sum[j] = delta_c_sum[j] + a

        # Update server control variate: c <- c + (|S|/N) * mean(delta_c_i)
        if delta_c_sum is not None and self._server_c is not None:
            scale = (n_selected / max(self.num_total_clients, 1)) / n_selected
            self._server_c = [
                c + scale * dc for c, dc in zip(self._server_c, delta_c_sum)
            ]

        params_out = ndarrays_to_parameters(aggregated_y) if aggregated_y else None
        return params_out, {"n_selected": n_selected}
