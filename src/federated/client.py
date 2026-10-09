from __future__ import annotations

import json
import math
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


def derive_local_seed(base_seed: int, server_round: int, partition_id: int) -> int:
    """Deterministic per-(experiment, round, client) training seed.

    Ultralytics re-seeds inside its trainer from ``args.seed``, and Flower
    simulation clients run in separate Ray actors, so seeding the launcher
    process has no effect on local training. Without this the experiment seeds
    (42/123/2024) never reached local training and all three "seeds" shared
    identical data order and augmentation — making multi-seed std meaningless
    (CLAUDE.md §14).

    The mapping is injective for base_seed < 2^20, round < 1000, client < 100.
    """
    return ((int(base_seed) * 1000) + int(server_round)) * 100 + int(partition_id)


def cosine_round_lr(server_round: int, num_rounds: int, lr0: float, lr_min: float) -> float:
    """Learning rate of round ``server_round`` (1-based): cosine from ``lr0`` (round 1) to ``lr_min``
    (last round). [LITERATURE] Li et al., "On the Convergence of FedAvg on Non-IID Data"
    (arXiv:1907.02189): on non-IID data FedAvg needs a decaying learning rate. The cosine shape is
    [ENGINEERING] (ADR-019)."""
    if num_rounds <= 1:
        return lr0
    x = min(max(server_round - 1, 0), num_rounds - 1) / (num_rounds - 1)
    return lr_min + 0.5 * (lr0 - lr_min) * (1.0 + math.cos(math.pi * x))


class YOLOFlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: str,
        partition_id: int,
        data_yaml: Path,
        weights: str,
        train_config: dict[str, Any],
        run_dir: Path,
        num_train_examples: int = 1,
        num_val_examples: int = 1,
        class_counts: dict[str, int] | None = None,
        missing_classes: list[str] | None = None,
        rho: float = 1.0,
        base_seed: int = 0,
    ) -> None:
        # num_train_examples / num_val_examples MUST be the client's actual
        # dataset sizes. FedAvg / FedNova aggregation weight = num_examples;
        # using a shared default of 1 (previous behaviour when scripts didn't
        # populate it) degenerates weighted-average into unweighted mean and
        # invalidates CLAUDE.md §12 aggregation contract.
        #
        # class_counts / missing_classes / rho are PER-CLIENT and must not be
        # read out of the shared train_config dict: that dict is one object
        # shared by every client, so it cannot hold per-client values.
        self.client_id = client_id
        self.partition_id = partition_id
        self.data_yaml = data_yaml
        self.train_config = train_config
        self.run_dir = run_dir
        self.model = build_model(weights)
        self._round = 0
        self.num_train_examples = int(num_train_examples)
        self.num_val_examples = int(num_val_examples)
        self.class_counts = dict(class_counts or {})
        self.missing_classes = list(missing_classes or [])
        self.rho = float(rho)
        self.base_seed = int(base_seed)

    # ── shared helpers for subclasses ─────────────────────────────────────
    def _train_kwargs(self) -> dict[str, Any]:
        cfg = self.train_config
        seed = derive_local_seed(self.base_seed, self._round, self.partition_id)
        lr0 = float(cfg["lr0"])
        sched = cfg.get("lr_schedule")
        if sched:                                          # ADR-019 component ②
            lr0 = cosine_round_lr(self._round, int(sched["rounds"]), lr0, float(sched["lr_min"]))
        data_yaml = self.data_yaml
        if cfg.get("rfs_t"):                               # ADR-019 component ③
            from src.preservation.rfs import rfs_data_yaml
            data_yaml = rfs_data_yaml(data_yaml, self.run_dir / "clients" / self.client_id / f"rfs_round_{self._round}",
                                      float(cfg["rfs_t"]), seed)
        return dict(
            data_yaml=data_yaml,
            epochs=cfg["local_epochs"],
            batch=cfg["batch_size"],
            img_size=cfg["image_size"],
            lr0=lr0,
            device=cfg.get("device", 0),
            project=self.run_dir / "clients" / self.client_id,
            name=f"round_{self._round}",
            seed=seed,
            workers=int(cfg.get("workers", 2)),
            warmup_epochs=float(cfg.get("warmup_epochs", 0.0)),
            close_mosaic=int(cfg.get("close_mosaic", 0)),
            run_val=bool(cfg.get("client_run_val", False)),
            deterministic=bool(cfg.get("deterministic", True)),
            nbs=cfg.get("nbs"),
        )

    def _fit_metrics(self, train_metrics: dict[str, float]) -> dict[str, Scalar]:
        """Metrics every client reports, regardless of algorithm.

        ``class_counts_json`` lives here — not only in PreservationClient —
        because ClassAwareAggregation needs it for per-class eligibility even
        when the client applies no preservation at all (ablation A3). Without
        it, A3 silently degraded to FedAvg while the trace claimed class-count
        weighting.
        """
        metrics: dict[str, Scalar] = {k: float(v) for k, v in train_metrics.items()}
        metrics["class_counts_json"] = json.dumps(self.class_counts)
        metrics["client_id"] = self.client_id
        metrics["n_train"] = int(self.num_train_examples)
        return metrics

    # ── Flower API ────────────────────────────────────────────────────────
    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        return get_parameters(self.model)

    def _advance_round(self, config: dict[str, Scalar]) -> int:
        """Set the current round from the server's ABSOLUTE round number.

        Falls back to a local counter only if the server sent none. The absolute
        number matters on resume: a client's own counter restarts at 1, which
        would replay earlier rounds' training seeds and overwrite their output
        directories.
        """
        server_round = config.get("server_round") if config else None
        self._round = int(server_round) if server_round is not None else self._round + 1
        return self._round

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        self._advance_round(config)
        set_parameters(self.model, parameters)

        train_metrics = train_one_round(model=self.model, **self._train_kwargs())

        logger.info(
            "Client %s round %d fit complete - metrics: %s",
            self.client_id, self._round, train_metrics,
        )

        return (
            get_parameters(self.model),
            self.num_train_examples,
            self._fit_metrics(train_metrics),
        )

    def evaluate(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        """Client-side evaluation.

        NOTE: this is NOT the reporting path. Reported metrics come from the
        server's centralized ``evaluate_fn`` so that every method is scored by
        the same evaluator on the same val set (CLAUDE.md §14). Because every
        per-client data yaml points ``val:`` at the shared global val set,
        running this on all clients repeats the same full validation N times per
        round; set ``fraction_evaluate: 0.0`` to skip it.
        """
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
        return loss, self.num_val_examples, metrics
