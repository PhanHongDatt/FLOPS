from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import flwr as fl
from flwr.common import NDArrays, Scalar
from flwr.server import ServerConfig

from src.federated.strategies.fedavg import build_fedavg
from src.utils.logger import get_logger, log_metrics, setup_mlflow, end_run

logger = get_logger(__name__)

def _load_strategy_builder(
    algorithm: str,
    param_names: list[str] | None = None,
    trace_dir: Path | None = None,
    strategy_options: dict[str, Any] | None = None,
):
    """Return a builder with the uniform signature ``build(num_rounds=..., **kw)``.

    Every builder must CONSUME ``num_rounds`` rather than forward it: flwr's
    ``FedAvg.__init__`` has no ``num_rounds`` parameter and no ``**kwargs``, so
    the previous ``lambda **kw: Scaffold(**kw)`` form raised TypeError before the
    strategy could even be constructed — SCAFFOLD, FedNova and ClassAwareAgg were
    all unreachable.
    """
    from src.federated.strategies import fedavg, fedprox, scaffold, fednova
    from src.federated.strategies.class_aware_agg import (
        build_class_aware_agg,
        build_class_count_fedavg,
    )

    opts = dict(strategy_options or {})

    def _class_aware(**kw: Any):
        return build_class_aware_agg(
            param_names=param_names or [], trace_dir=trace_dir, **opts, **kw
        )

    def _class_count(**kw: Any):
        return build_class_count_fedavg(
            param_names=param_names or [], trace_dir=trace_dir, **opts, **kw
        )

    builders = {
        "FedAvg": fedavg.build_fedavg,
        "FedProx": fedprox.build_fedprox,
        "SCAFFOLD": scaffold.build_scaffold,
        "FedNova": fednova.build_fednova,
        "ClassAwareAgg": _class_aware,          # A3 / A4b server side
        "ClassCountFedAvg": _class_count,       # A1 control
    }
    if algorithm not in builders:
        raise ValueError(
            f"Unknown FL algorithm: {algorithm}. Known: {sorted(builders)}"
        )
    return builders[algorithm]


def _append_round_row(csv_path: Path, row: dict[str, Any]) -> None:
    """Append one round's metrics to round_metrics.csv incrementally.

    Rebuild header on every call so late-appearing keys (e.g. AP50_bus that
    only exists after class detected) are captured without losing prior rows.
    """
    existing_rows: list[dict[str, Any]] = []
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)

    existing_rows.append({k: v for k, v in row.items()})
    all_keys: set[str] = set()
    for r in existing_rows:
        all_keys.update(r.keys())
    fieldnames = ["round"] + sorted(k for k in all_keys if k != "round")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)


def _build_centralized_evaluate_fn(
    global_data_yaml: Path,
    eval_config: dict[str, Any],
    run_dir: Path,
    num_rounds: int,
    weights: str = "yolov8n.pt",
    round_offset: int = 0,
    keep_last_checkpoints: int = 2,
    prune_client_weights: bool = True,
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
        save_summary_metrics,
    )
    from src.experiments.checkpoint import (
        prune_checkpoints,
        prune_client_round_weights,
        save_global_checkpoint,
    )

    # Lazy-init to avoid loading YOLO in server process before first round
    _holder: dict[str, Any] = {"model": None}

    def evaluate_fn(
        server_round: int,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        # Flower always counts rounds from 1 within a single start_simulation
        # call, so on resume we offset to keep ABSOLUTE round numbers in the
        # CSVs, MLflow steps and checkpoint filenames.
        abs_round = round_offset + server_round

        # Flower also calls evaluate_fn once with server_round == 0 to score the
        # INITIAL parameters, before any training. That is not a completed round:
        # checkpointing it would create global_round_000.npz, which
        # find_latest_checkpoint would then treat as progress — so a run that
        # failed inside round 1 could only be retried with --resume, and a
        # resumed run would re-append a duplicate row for its offset round.
        is_initial_eval = server_round == 0

        if not is_initial_eval:
            # Checkpoint BEFORE evaluating: if evaluation OOMs or the Kaggle
            # session dies mid-eval, the aggregated parameters for this round are
            # already on disk and the run resumes from here instead of redoing it.
            ckpt_dir = run_dir / "checkpoint"
            save_global_checkpoint(ckpt_dir, abs_round, list(parameters))
            prune_checkpoints(ckpt_dir, keep_last_checkpoints)
            if prune_client_weights:
                prune_client_round_weights(run_dir, abs_round)

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
        logger.info("Round %d (absolute) centralized eval: %s", abs_round, metrics)

        # Append per-round metrics incrementally — replaces the previously
        # dead MetricsCallback whose on_round_end was never invoked by Flower.
        # §21 requires round_metrics.csv.
        # On a resumed run the initial evaluation re-scores the checkpoint we
        # loaded, i.e. a round already present in the CSV — skip it rather than
        # writing a duplicate row. On a fresh run round 0 is genuinely useful:
        # it is the global model's per-class AP before any federated training.
        if not (is_initial_eval and round_offset > 0):
            _append_round_row(
                run_dir / "round_metrics.csv",
                {"round": abs_round, **{k: float(v) for k, v in metrics.items()}},
            )
            try:
                log_metrics({k: float(v) for k, v in metrics.items()}, step=abs_round)
            except Exception as exc:  # MLflow optional / offline
                logger.debug("MLflow log_metrics failed at round %d: %s", abs_round, exc)

        if not is_initial_eval and abs_round >= num_rounds:
            save_per_class_metrics(
                flatten_per_class_metrics(metrics),
                run_dir / "per_class_metrics.csv",
            )
            # §21 also requires metrics.csv (summary of final-round scalars).
            # Skip non-scalar values (defensive; evaluate() returns floats today).
            summary = {
                k: float(v) for k, v in metrics.items()
                if isinstance(v, (int, float))
            }
            save_summary_metrics(summary, run_dir / "metrics.csv")
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
    strategy_options: dict[str, Any] | None = None,
    resume: bool = False,
) -> None:
    from flwr.common import ndarrays_to_parameters

    from src.experiments.checkpoint import find_latest_checkpoint, load_global_checkpoint

    run_id = setup_mlflow(mlflow_experiment, mlflow_uri, run_name=algorithm)
    logger.info("MLflow run_id: %s", run_id)

    total_rounds = int(fl_config["num_rounds"])

    # ── Resume (plan.md §13 K4) ───────────────────────────────────────────
    round_offset = 0
    initial_params: NDArrays | None = None
    latest = find_latest_checkpoint(run_dir / "checkpoint")
    if latest is not None:
        done_round, ckpt_path = latest
        if not resume:
            raise FileExistsError(
                f"Run directory already holds checkpoints up to round {done_round} "
                f"({ckpt_path.name}). Pass --resume to continue this run, or use a "
                "different --run-id to start a fresh one. Refusing to silently "
                "restart from round 1 and overwrite existing round metrics."
            )
        if done_round >= total_rounds:
            logger.info(
                "Run already completed %d/%d rounds — nothing to do.",
                done_round, total_rounds,
            )
            end_run()
            return
        round_offset = done_round
        initial_params = load_global_checkpoint(ckpt_path)
        logger.info(
            "Resuming from round %d (%s); %d round(s) remaining.",
            done_round, ckpt_path.name, total_rounds - done_round,
        )
        if algorithm in ("SCAFFOLD", "FedNova"):
            logger.warning(
                "Resume for %s is APPROXIMATE: server-side state (SCAFFOLD "
                "control variate c / FedNova prev-params) is not checkpointed "
                "and resets to zero. Report this if a resumed %s run is used.",
                algorithm, algorithm,
            )
    elif resume:
        logger.info("--resume given but no checkpoint found; starting from round 1.")

    remaining_rounds = total_rounds - round_offset

    strategy_builder = _load_strategy_builder(
        algorithm,
        param_names=param_names,
        trace_dir=run_dir,                       # §12/§21 aggregation_trace.yaml
        strategy_options=strategy_options,
    )

    # Clients must learn the ABSOLUTE round number. Their own counter restarts at
    # 1 inside a resumed start_simulation call, which would replay the seeds of
    # rounds 1..k and overwrite those rounds' client output directories — making
    # a resumed run differ from an uninterrupted one.
    def _fit_config(server_round: int) -> dict[str, Scalar]:
        return {"server_round": round_offset + server_round}

    strategy_kwargs: dict[str, Any] = dict(
        num_rounds=total_rounds,
        fraction_fit=fl_config["fraction_fit"],
        fraction_evaluate=fl_config["fraction_evaluate"],
        min_fit_clients=fl_config["num_clients"],
        min_evaluate_clients=fl_config["num_clients"],
        min_available_clients=fl_config["num_clients"],
        on_fit_config_fn=_fit_config,
    )
    if initial_params is not None:
        strategy_kwargs["initial_parameters"] = ndarrays_to_parameters(initial_params)

    if global_data_yaml is not None and eval_config is not None:
        strategy_kwargs["evaluate_fn"] = _build_centralized_evaluate_fn(
            global_data_yaml=global_data_yaml,
            eval_config=eval_config,
            run_dir=run_dir,
            num_rounds=total_rounds,
            weights=str(eval_config.get("weights", "yolov8n.pt")),
            round_offset=round_offset,
            keep_last_checkpoints=int(fl_config.get("keep_last_checkpoints", 2)),
            prune_client_weights=bool(fl_config.get("prune_client_weights", True)),
        )
    else:
        logger.warning(
            "run_fl_server called without global_data_yaml/eval_config — "
            "per_class_metrics.csv and final_params.npz WILL NOT be produced. "
            "This violates CLAUDE.md §21 artifact contract."
        )

    strategy = strategy_builder(**strategy_kwargs)

    # Default num_gpus=1.0 serialises clients onto the GPU. The previous default
    # of 0.25 let Ray place 4 YOLOv8n trainings on one device concurrently:
    # certain OOM on a 4 GB laptop GPU and borderline on a 16 GB T4/P100
    # (plan.md §13 K3). num_cpus=1 x 4 clients also starves the Ray driver on a
    # ~4 vCPU Kaggle notebook (K1), so both are config-driven now.
    client_resources = {
        "num_cpus": int(fl_config.get("client_num_cpus", 1)),
        "num_gpus": float(fl_config.get("client_num_gpus", 1.0)),
    }
    logger.info(
        "Ray client_resources=%s; running rounds %d..%d of %d",
        client_resources, round_offset + 1, total_rounds, total_rounds,
    )
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=fl_config["num_clients"],
        config=ServerConfig(num_rounds=remaining_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )

    end_run()
    logger.info("FL run complete. Artifacts in %s", run_dir)
