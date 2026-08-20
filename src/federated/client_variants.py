"""Strategy-specific client variants.

FedAvg / FedProx use the base YOLOFlowerClient.
SCAFFOLD and FedNova need extra client-side state and metrics reporting.

FedProx: proximal term added inside local training loop.
        (Note: with ultralytics wrapper we cannot easily inject the proximal
         term without patching the trainer; see FedProxClient below for
         a placeholder that documents the requirement.)
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
from flwr.common import NDArrays, Scalar

from src.federated.client import YOLOFlowerClient
from src.model.yolo_wrapper import get_parameters, set_parameters, train_one_round


def _serialize(arrays: NDArrays) -> str:
    buf = io.BytesIO()
    np.savez_compressed(buf, *arrays)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _deserialize(payload: str) -> NDArrays:
    buf = io.BytesIO(base64.b64decode(payload))
    with np.load(buf) as data:
        return [data[k] for k in sorted(data.files)]


class ScaffoldClient(YOLOFlowerClient):
    """SCAFFOLD-aware client with per-client control variate c_i (Option II)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._c_i: NDArrays | None = None       # per-client control variate
        self._last_x: NDArrays | None = None    # global params received this round

    def _init_c_i(self, params: NDArrays) -> None:
        self._c_i = [np.zeros_like(p) for p in params]

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        # NOTE: Ultralytics trainer does not natively accept a gradient-shifting
        # hook. Full SCAFFOLD requires patching the optimizer to add
        # (c - c_i) to the gradient at every step. For now we run standard
        # local training and compute Option II's control variate update from
        # the pre/post model difference, which is the algebraic shortcut
        # given in the paper (Algorithm 1, Option II).
        if self._c_i is None:
            self._init_c_i(parameters)
        self._last_x = [np.array(p, copy=True) for p in parameters]

        # Run local training identical to FedAvg
        set_parameters(self.model, parameters)
        train_metrics = train_one_round(
            model=self.model,
            data_yaml=self.data_yaml,
            epochs=self.train_config["local_epochs"],
            batch=self.train_config["batch_size"],
            img_size=self.train_config["image_size"],
            lr0=self.train_config["lr0"],
            device=self.train_config.get("device", 0),
            project=self.run_dir / "clients" / self.client_id,
            name=f"round_{self._round + 1}",
        )
        self._round += 1
        y_i = get_parameters(self.model)

        # Option II delta_c_i shortcut:
        #   delta_c_i = -c_i + (1 / (K * eta_l)) * (x - y_i)
        K = int(self.train_config["local_epochs"])
        eta_l = float(self.train_config["lr0"])
        scale = 1.0 / max(K * eta_l, 1e-8)
        delta_c = [
            (-c) + scale * (x - y)
            for c, x, y in zip(self._c_i, self._last_x, y_i)
        ]
        # Update stored c_i
        self._c_i = [c + dc for c, dc in zip(self._c_i, delta_c)]

        metrics: dict[str, Scalar] = {
            **{k: float(v) for k, v in train_metrics.items()},
            "delta_c": _serialize(delta_c),
        }
        num_examples = int(self.train_config.get("num_train_examples", 1))
        return y_i, num_examples, metrics


class FedNovaClient(YOLOFlowerClient):
    """FedNova-aware client that reports tau_i (local step count)."""

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        set_parameters(self.model, parameters)

        local_epochs = int(self.train_config["local_epochs"])
        batch_size = int(self.train_config["batch_size"])
        num_train = int(self.train_config.get("num_train_examples", 1))

        train_metrics = train_one_round(
            model=self.model,
            data_yaml=self.data_yaml,
            epochs=local_epochs,
            batch=batch_size,
            img_size=self.train_config["image_size"],
            lr0=self.train_config["lr0"],
            device=self.train_config.get("device", 0),
            project=self.run_dir / "clients" / self.client_id,
            name=f"round_{self._round + 1}",
        )
        self._round += 1

        # tau_i = number of local SGD steps = local_epochs * ceil(N_i / B)
        steps_per_epoch = max(1, (num_train + batch_size - 1) // batch_size)
        tau_i = local_epochs * steps_per_epoch

        metrics: dict[str, Scalar] = {
            **{k: float(v) for k, v in train_metrics.items()},
            "tau_i": int(tau_i),
        }
        return get_parameters(self.model), num_train, metrics


class FedProxClient(YOLOFlowerClient):
    """FedProx-aware client.

    ⚠️  The proximal term (mu/2)||w - w_t||^2 must be added to the loss
    at every local SGD step. Ultralytics' trainer does not expose a hook
    for this without patching. Two options:
      (a) Monkey-patch the trainer's loss function to add proximal term.
      (b) Use custom training loop bypassing Ultralytics' trainer.

    Until one is chosen and validated, this client falls back to FedAvg
    behaviour, which means [NEEDS-VERIFICATION] before FedProx results
    can be reported as scientifically valid.
    """

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        # TODO: inject proximal term into ultralytics trainer.
        # See v8_yolo_prox integration ADR (to be written).
        return super().fit(parameters, config)
