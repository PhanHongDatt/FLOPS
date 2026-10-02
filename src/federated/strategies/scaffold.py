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
        # Previous global x — required for the Algorithm 1 server step
        # x_new = x + eta_g * mean_i(delta_y_i). Without it the previous
        # implementation silently degenerated into weighted FedAvg, i.e. it
        # labelled FedAvg results as SCAFFOLD (CLAUDE.md §22).
        self._prev_params: NDArrays | None = None

    def _init_control(self, params: NDArrays) -> None:
        self._server_c = [np.zeros_like(p) for p in params]

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        self._prev_params = parameters_to_ndarrays(parameters)
        if self._server_c is None:
            self._init_control(self._prev_params)

        # Merge the server's on_fit_config_fn first so the ABSOLUTE round number
        # survives a resumed run; server_c is this strategy's own addition.
        config: dict = {"server_round": server_round}
        if self.on_fit_config_fn is not None:
            config.update(self.on_fit_config_fn(server_round))
        config["server_c"] = _serialize(self._server_c)
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

        x = self._prev_params
        delta_y_sum: NDArrays | None = None
        delta_c_sum: NDArrays | None = None
        n_selected = len(results)

        # Algorithm 1 (Option II) server step — UNWEIGHTED mean of client
        # deltas, scaled by the global step size:
        #     x_new = x + eta_g * (1 / |S|) * sum_i (y_i - x)
        # Note this is deliberately NOT num_examples-weighted: SCAFFOLD's
        # correction lives in the control variates, not in the server weights.
        for _, res in results:
            y_i = parameters_to_ndarrays(res.parameters)
            delta_y = [y - xp for y, xp in zip(y_i, x)]
            if delta_y_sum is None:
                delta_y_sum = [np.array(d, copy=True) for d in delta_y]
            else:
                for j, d in enumerate(delta_y):
                    delta_y_sum[j] = delta_y_sum[j] + d

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

        if delta_y_sum is None:
            return None, {}

        scale_y = self.eta_global / max(n_selected, 1)
        new_params = [xp + scale_y * d for xp, d in zip(x, delta_y_sum)]
        return (
            ndarrays_to_parameters(new_params),
            {"n_selected": n_selected, "eta_global": float(self.eta_global)},
        )


def build_scaffold(
    num_rounds: int,
    eta_global: float = 1.0,
    num_total_clients: int = 4,
    **kwargs: Any,
) -> Scaffold:
    """Factory with the uniform builder signature used by server._load_strategy_builder.

    `num_rounds` is consumed here and NOT forwarded: flwr's FedAvg.__init__ has
    no `num_rounds` parameter and no **kwargs, so forwarding it raised
    TypeError before the strategy could be constructed.
    """
    return Scaffold(
        eta_global=eta_global,
        num_total_clients=num_total_clients,
        **kwargs,
    )
