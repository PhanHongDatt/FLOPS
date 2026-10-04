"""Strategy-specific client variants.

FedAvg / FedProx use the base YOLOFlowerClient.
SCAFFOLD and FedNova need extra client-side state and metrics reporting.
PreservationClient (H2): applies detection-cls preservation mask before upload.

FedProx: proximal term added inside local training loop.
        (Note: with ultralytics wrapper we cannot easily inject the proximal
         term without patching the trainer; see FedProxClient below for
         a placeholder that documents the requirement.)
"""
from __future__ import annotations

import base64
import io
import json
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
        self._advance_round(config)
        set_parameters(self.model, parameters)
        train_metrics = train_one_round(model=self.model, **self._train_kwargs())
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

        metrics = self._fit_metrics(train_metrics)
        metrics["delta_c"] = _serialize(delta_c)
        return y_i, self.num_train_examples, metrics


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
        num_train = int(self.num_train_examples)

        self._advance_round(config)
        train_metrics = train_one_round(model=self.model, **self._train_kwargs())

        # tau_i = number of local SGD steps = local_epochs * ceil(N_i / B)
        steps_per_epoch = max(1, (num_train + batch_size - 1) // batch_size)
        tau_i = local_epochs * steps_per_epoch

        metrics = self._fit_metrics(train_metrics)
        metrics["tau_i"] = int(tau_i)
        return get_parameters(self.model), num_train, metrics


class FedProxClient(YOLOFlowerClient):
    """FedProx client: local objective F_k(w) + (mu/2)||w - w^t||^2.

    [LITERATURE] Li et al., 2020 (MLSys). ``proximal_mu`` arrives in the fit
    config from Flower's FedProx strategy (configure_fit). The proximal gradient
    is added through per-parameter hooks on the Ultralytics trainer's model
    (src/federated/proximal.py). Previously this class fell back to FedAvg.
    """

    def _train_kwargs(self) -> dict[str, Any]:
        return {**super()._train_kwargs(), "proximal_mu": self._proximal_mu}

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        if "proximal_mu" not in config:
            raise ValueError("FedProxClient needs 'proximal_mu' in the fit config (FedProx strategy)")
        self._proximal_mu = float(config["proximal_mu"])
        params, n, metrics = super().fit(parameters, config)
        metrics["proximal_mu"] = self._proximal_mu
        return params, n, metrics


class PreservationClient(YOLOFlowerClient):
    """H2 — Client-side knowledge preservation client.

    After local training, applies the preservation mask to detect_cls final-conv
    parameters for missing classes before returning parameters to the server.

    Also reports 'class_counts_json' in fit metrics so the server-side
    ClassAwareAggregation (H3) can determine eligible clients per class.

    This is ablation **A2a** (parameter-level). It reads ``missing_classes``,
    ``rho`` and ``class_counts`` from PER-CLIENT constructor arguments, not from
    the shared ``train_config`` dict — that dict is one shared object and could
    never hold per-client values.

    ⚠️  Known identity (research/plan/plan.md §6.3): the rows A2a protects are
    exactly the rows ClassAwareAggregation already excludes for this client, so
    ``A4a (A2a + A3) === A3``. A4a is therefore a numerical identity test, not a
    separate experimental arm.

    GATE CONSTRAINT (CLAUDE.md §7): Not scientifically valid until G5 passes
    and F1 runtime parameter map is verified (runtime_confirmed in parameter_map.yaml).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._global_params: NDArrays | None = None
        self._param_names: list[str] | None = None

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        from src.preservation.mask import apply_preservation_mask, get_param_names

        # Snapshot global params before local training
        self._global_params = [np.array(p, copy=True) for p in parameters]
        if self._param_names is None:
            self._param_names = get_param_names(self.model)

        set_parameters(self.model, parameters)

        self._advance_round(config)
        train_metrics = train_one_round(model=self.model, **self._train_kwargs())

        local_params = get_parameters(self.model)

        masked_params = apply_preservation_mask(
            local_params=local_params,
            global_params=self._global_params,
            param_names=self._param_names,
            missing_classes=self.missing_classes,
            rho=self.rho,
        )

        metrics = self._fit_metrics(train_metrics)
        metrics["rho"] = float(self.rho)
        metrics["mechanism"] = "preservation_param"
        metrics["missing_classes"] = json.dumps(self.missing_classes)
        return masked_params, self.num_train_examples, metrics


class LossPreservationClient(YOLOFlowerClient):
    """A2b — loss-level preservation (ADR-006).

    The BCE term of each locally-missing class is multiplied by ``rho`` during local
    training (src/preservation/rho_loss.py). ``missing_classes`` and ``rho`` are
    per-client constructor arguments (ADR-007). rho = 1 is ordinary FedAvg training.
    Also reports ``class_counts_json`` (base class) so A4b's server can use it.
    """

    def _train_kwargs(self) -> dict[str, Any]:
        from src.data.bdd100k import TARGET_CLASSES
        from src.preservation.rho_loss import class_loss_weights
        weights = class_loss_weights(TARGET_CLASSES, self.missing_classes, self.rho)
        return {**super()._train_kwargs(), "cls_loss_weights": weights}

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        params, n, metrics = super().fit(parameters, config)
        metrics["rho"] = float(self.rho)
        metrics["mechanism"] = "preservation_loss"
        metrics["missing_classes"] = json.dumps(self.missing_classes)
        return params, n, metrics


class LossVariantClient(YOLOFlowerClient):
    """A5 / A6 / A6c — classification-loss variants (ADR-012, ADR-013).

    Every client trains with the variant (as in FedNTD / EFL); only the loss changes,
    aggregation stays FedAvg. Subclasses fix ``LOSS_SPEC``. Reports
    ``class_counts_json`` through the base class like every client.
    """

    LOSS_SPEC: dict[str, Any] = {}

    def _loss_spec(self) -> dict[str, Any]:
        return dict(self.LOSS_SPEC)

    def _train_kwargs(self) -> dict[str, Any]:
        return {**super()._train_kwargs(), "cls_loss": self._loss_spec()}

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        params, n, metrics = super().fit(parameters, config)
        metrics["mechanism"] = str(self.LOSS_SPEC["kind"])
        return params, n, metrics


class NTDClient(LossVariantClient):
    """A5: not-true distillation from the received global model; official defaults tau = beta = 1."""

    LOSS_SPEC = {"kind": "ntd", "beta": 1.0, "tau": 1.0}


class EFLClient(LossVariantClient):
    """A6: Equalized Focal Loss; gradient statistics kept per client across rounds."""

    LOSS_SPEC = {"kind": "efl", "gamma": 2.0, "alpha": 0.25, "scale": 4.0}

    def _loss_spec(self) -> dict[str, Any]:
        return {**super()._loss_spec(),
                "state_path": str(Path(self.run_dir) / "clients" / self.client_id / "efl_state.json")}


class FocalClient(LossVariantClient):
    """A6c: plain focal loss (EFL with equal factors) — the control that isolates equalization."""

    LOSS_SPEC = {"kind": "focal", "gamma": 2.0, "alpha": 0.25}
