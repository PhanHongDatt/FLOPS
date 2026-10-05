"""Run federated experiment — G3+ baseline and ablation runner.

Executes a Flower simulation with the chosen server strategy and client
mechanism on a prepared partition of BDD100K. Records to MLflow + local
artifacts.

The experiment matrix has TWO independent axes (research/plan/plan.md §7.1):

    server strategy   : FedAvg | FedProx | SCAFFOLD | FedNova
                        ClassCountFedAvg (A1) | ClassAwareAgg (A3/A4)
    client mechanism  : none | preservation_param (A2a) | preservation_loss (A2b)
                        ntd (A5) | efl (A6) | focal (A6c) | kd_teacher (P1) | kd_teacher_rho (P2)

``--ablation`` is a preset that fills both plus ``--rho``; explicit flags win.

Usage:
  # A0 baseline
  python scripts/run_fl_experiment.py \\
    --partition artifacts/partitions/s1_mc/manifest.yaml \\
    --data-yaml-dir artifacts/partitions/s1_mc \\
    --algorithm FedAvg --run-class smoke --seed 42

  # A3 server-only
  python scripts/run_fl_experiment.py ... --ablation A3 --run-class feasibility

  # resume an interrupted run (same --run-id / same default id)
  python scripts/run_fl_experiment.py ... --ablation A3 --resume
"""
from __future__ import annotations

import argparse
import faulthandler
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.partitioner import load_partition_manifest
from src.experiments.runner import prepare_run, finalize_run
from src.federated.client import YOLOFlowerClient
from src.federated.client_variants import (
    EFLClient,
    FedNovaClient,
    FedProxClient,
    FocalClient,
    LossPreservationClient,
    NTDClient,
    PreservationClient,
    ScaffoldClient,
    TeacherKDClient,
    TeacherKDRhoClient,
)
from src.federated.server import run_fl_server
from src.utils.config import build_local_train_config, load_experiment_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Server strategies ─────────────────────────────────────────────────────
_STRATEGIES = (
    "FedAvg",
    "FedProx",
    "SCAFFOLD",
    "FedNova",
    "ClassCountFedAvg",   # A1 — class-count weighting, NO no-contributor rule
    "ClassAwareAgg",      # A3/A4 — class-count weighting + no-contributor rule
)

# Strategies that need the ordered state_dict key list + aggregation trace.
_CLASS_AWARE_STRATEGIES = ("ClassAwareAgg", "ClassCountFedAvg")

# ── Client mechanisms ─────────────────────────────────────────────────────
_MECHANISMS = ("none", "preservation_param", "preservation_loss", "ntd", "efl", "focal",
               "kd_teacher", "kd_teacher_rho")

# Mechanisms parameterised by rho (H2). The loss variants carry their own
# literature defaults (ADR-012/013) and take no rho.
_RHO_MECHANISMS = ("preservation_param", "preservation_loss", "kd_teacher_rho")

_MECHANISM_CLIENTS = {
    "preservation_param": PreservationClient,     # A2a
    "preservation_loss": LossPreservationClient,  # A2b (gated on F1 + ADR-006)
    "ntd": NTDClient,                             # A5  (ADR-012)
    "efl": EFLClient,                             # A6  (ADR-013)
    "focal": FocalClient,                         # A6c (ADR-013)
    "kd_teacher": TeacherKDClient,                # P1  (ADR-015)
    "kd_teacher_rho": TeacherKDRhoClient,         # P2  (ADR-015)
}

# Strategy-specific clients used when mechanism == "none".
_STRATEGY_CLIENTS = {
    "FedProx": FedProxClient,      # proximal term via gradient hooks (src/federated/proximal.py)
    "SCAFFOLD": ScaffoldClient,
    "FedNova": FedNovaClient,
}

# ── Ablation presets (plan.md §7.1) ───────────────────────────────────────
_ABLATIONS: dict[str, dict[str, object]] = {
    "A0":  {"algorithm": "FedAvg",           "mechanism": "none"},
    "A1":  {"algorithm": "ClassCountFedAvg", "mechanism": "none"},
    "A2a": {"algorithm": "FedAvg",           "mechanism": "preservation_param"},
    "A2b": {"algorithm": "FedAvg",           "mechanism": "preservation_loss"},
    "A3":  {"algorithm": "ClassAwareAgg",    "mechanism": "none"},
    # A4a is a numerical IDENTITY test against A3, not a separate arm (§6.3).
    "A4a": {"algorithm": "ClassAwareAgg",    "mechanism": "preservation_param"},
    "A4b": {"algorithm": "ClassAwareAgg",    "mechanism": "preservation_loss"},
    # Directions after the s5/s6 negative result (ADR-012, ADR-013); exploratory.
    "A5":  {"algorithm": "FedAvg",           "mechanism": "ntd"},
    "A6":  {"algorithm": "FedAvg",           "mechanism": "efl"},
    "A6c": {"algorithm": "FedAvg",           "mechanism": "focal"},
    # Server holds a labelled 1,000-image sample (ADR-015). Every arm starts from the
    # teacher trained on it (--teacher-params); B2 also fine-tunes on it after each round.
    "B1":  {"algorithm": "FedAvg", "mechanism": "none",           "teacher_init": True},
    "B2":  {"algorithm": "FedAvg", "mechanism": "none",           "teacher_init": True, "server_ft": True},
    "P1":  {"algorithm": "FedAvg", "mechanism": "kd_teacher",     "teacher_init": True},
    "P2":  {"algorithm": "FedAvg", "mechanism": "kd_teacher_rho", "teacher_init": True},
}

# Algorithms recognised by argparse but blocked at runtime with a scientific
# reason. Prevents silently producing results that violate CLAUDE.md §22
# (labeling one algorithm's results as another).
_DISABLED_ALGORITHMS: dict[str, str] = {
    # FedProx re-enabled 2026-10-03: proximal term implemented (ADR-010).
    "SCAFFOLD": (
        "SCAFFOLD is disabled: ScaffoldClient applies no (c - c_i) gradient correction "
        "during local training, computes delta_c_i with -c_i instead of -c (paper: "
        "c_i+ = c_i - c + (x - y_i)/(K eta_l)), uses K = local epochs instead of local "
        "optimizer steps, and loses c_i between rounds because Flower recreates clients. "
        "Ultralytics' SGD momentum / weight decay / gradient accumulation also change the "
        "Option II scaling. Results would be FedAvg-like runs labelled SCAFFOLD "
        "(CLAUDE.md §22). Needs an ADR + faithful implementation before re-enabling."
    ),
}


def _set_seed(seed: int) -> None:
    """Seed the launcher process.

    NOTE: Flower simulation clients run in separate Ray actors, so this does NOT
    reach local training. Per-client/per-round training seeds are derived in
    ``src.federated.client.derive_local_seed`` and passed to the Ultralytics
    trainer explicitly.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_arms(args: argparse.Namespace) -> tuple[str, str, float]:
    """Resolve (algorithm, mechanism, rho) from --ablation + explicit flags."""
    algorithm = args.algorithm
    mechanism = args.client_mechanism

    if args.ablation:
        preset = _ABLATIONS[args.ablation]
        if algorithm is None:
            algorithm = str(preset["algorithm"])
        elif algorithm != preset["algorithm"]:
            logger.warning(
                "--algorithm %s overrides ablation %s preset (%s).",
                algorithm, args.ablation, preset["algorithm"],
            )
        if mechanism is None:
            mechanism = str(preset["mechanism"])
        elif mechanism != preset["mechanism"]:
            logger.warning(
                "--client-mechanism %s overrides ablation %s preset (%s).",
                mechanism, args.ablation, preset["mechanism"],
            )

    if algorithm is None:
        raise SystemExit("Specify --algorithm or --ablation.")
    if mechanism is None:
        mechanism = "none"

    rho = args.rho
    if mechanism in _RHO_MECHANISMS and rho is None:
        raise SystemExit(
            f"--rho is required for --client-mechanism {mechanism}. There is no "
            "literature-supported default (plan.md §7.3): rho is a "
            "[THESIS-HYPOTHESIS] value selected by the pre-registered sweep over "
            "{0, 0.25, 1}. Pass it explicitly so the run is self-documenting."
        )
    if mechanism not in _RHO_MECHANISMS:
        if rho is not None and mechanism != "none":
            raise SystemExit(f"--rho does not apply to --client-mechanism {mechanism}.")
        rho = 1.0

    if mechanism != "none" and algorithm in ("SCAFFOLD", "FedNova"):
        raise SystemExit(
            f"Combination {algorithm} + {mechanism} is not supported: those "
            "clients override fit() with their own update rule, so the "
            "preservation step would not be applied. Not in the plan's matrix."
        )
    return algorithm, mechanism, float(rho)


def exp_id_for_arm(arm: str, mechanism: str, rho: float) -> str:
    """Experiment id used to build the run directory name.

    Shared with scripts/sweep_experiments.py so the sweep can predict a cell's
    run directory and decide skip / resume without re-running it.
    """
    return f"{arm}_rho{rho:g}" if mechanism in _RHO_MECHANISMS else arm


def mechanism_for_arm(arm: str) -> str:
    """Client mechanism of an ablation preset ('none' for unknown arms)."""
    preset = _ABLATIONS.get(arm)
    return str(preset["mechanism"]) if preset else "none"


def _pick_client_class(algorithm: str, mechanism: str):
    if mechanism != "none":
        return _MECHANISM_CLIENTS[mechanism]
    return _STRATEGY_CLIENTS.get(algorithm, YOLOFlowerClient)


def _build_client_fn(
    client_cls,
    partition_manifest,
    data_yaml_map,
    weights: str,
    train_config: dict,
    run_dir: Path,
    num_val_examples: int,
    rho: float,
    base_seed: int,
):
    # Flower's start_simulation passes cid as a numeric string "0"..."N-1",
    # not the manifest keys "C0"..."CN-1". Sort keys deterministically and
    # map by index. Without this adapter, data_yaml_map[cid] raises KeyError
    # on round 0 and simulation never starts.
    client_keys = sorted(partition_manifest.client_assignments.keys())

    def client_fn(cid: str):
        idx = int(cid)
        if idx < 0 or idx >= len(client_keys):
            raise ValueError(
                f"Flower client id {cid!r} is out of range for "
                f"{len(client_keys)} partition clients {client_keys}"
            )
        client_key = client_keys[idx]
        n_train = len(partition_manifest.client_assignments[client_key])
        # Per-client values come from the manifest, NOT from the shared
        # train_config dict (which is one object for all clients and therefore
        # cannot hold per-client class counts).
        return client_cls(
            client_id=client_key,
            partition_id=idx,
            data_yaml=data_yaml_map[client_key],
            weights=weights,
            train_config=train_config,
            run_dir=run_dir,
            num_train_examples=n_train,
            num_val_examples=num_val_examples,
            class_counts=dict(partition_manifest.class_counts.get(client_key, {})),
            missing_classes=list(partition_manifest.missing_classes.get(client_key, [])),
            rho=rho,
            base_seed=base_seed,
        )
    return client_fn


def _count_val_examples(global_data_yaml: Path | None) -> int:
    """Count val images from the global data.yaml — one shared val set per §14."""
    if global_data_yaml is None or not global_data_yaml.exists():
        return 1
    with global_data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    root = data.get("path")
    val_rel = data.get("val")
    if not root or not val_rel:
        return 1
    val_dir = Path(root) / val_rel
    if not val_dir.is_dir():
        return 1
    return sum(1 for _ in val_dir.glob("*.jpg")) or 1


def _resolve_param_names(weights: str) -> list[str]:
    """Ordered state_dict keys, required by class-aware aggregation.

    Built once in the launcher process. Needs ultralytics installed — which is
    exactly where class-aware aggregation is going to run anyway.
    """
    from src.model.yolo_wrapper import build_model
    model = build_model(weights)
    return list(model.model.state_dict().keys())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", type=Path, required=True,
                    help="Path to partition manifest YAML")
    ap.add_argument("--algorithm", choices=_STRATEGIES, default=None,
                    help="Server aggregation strategy. Optional if --ablation is given.")
    ap.add_argument("--ablation", choices=sorted(_ABLATIONS), default=None,
                    help="Preset for (algorithm, client mechanism) per plan.md §7.1")
    ap.add_argument("--client-mechanism", choices=_MECHANISMS, default=None,
                    help="Client-side mechanism (H2). Default: none")
    ap.add_argument("--rho", type=float, default=None,
                    help="Preservation factor; required when a client mechanism is set")
    ap.add_argument("--teacher-params", type=Path, default=None,
                    help="Teacher .npz trained on the server sample (ADR-015): initial global "
                         "parameters of B1/B2/P1/P2 and the fixed KD teacher of P1/P2")
    ap.add_argument("--server-data-yaml", type=Path, default=None,
                    help="data.yaml of the server's labelled sample; B2 fine-tunes on it each round")
    ap.add_argument("--tau-elig", type=int, default=1,
                    help="Eligibility threshold for class-aware aggregation "
                         "(plan.md §5.1: primary 1, sensitivity variant 50)")
    ap.add_argument("--data-yaml-dir", type=Path, required=True,
                    help="Directory containing per-client data_C{i}.yaml files")
    ap.add_argument("--exp-config", type=Path, default=None)
    ap.add_argument("--run-class", choices=["smoke", "feasibility", "main"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-id", default=None,
                    help="Explicit run directory name. Default is deterministic "
                         "from (ablation, rho, run-class, seed, partition).")
    ap.add_argument("--resume", action="store_true",
                    help="Continue an interrupted run from its last round checkpoint")
    ap.add_argument("--fraction-evaluate", type=float, default=None,
                    help="Override federated.fraction_evaluate. 0.0 skips client-side "
                         "evaluation; the reporting path is the server's centralized "
                         "evaluate_fn either way (CLAUDE.md §14).")
    ap.add_argument("--mlflow-uri", default="http://mlflow:5000")
    ap.add_argument("--mlflow-experiment", default="G3-federated")
    ap.add_argument(
        "--global-data-yaml",
        type=Path,
        default=None,
        help="Global YOLO data.yaml for centralized post-round eval. "
             "Required to produce per_class_metrics.csv (CLAUDE.md §13, §21).",
    )
    args = ap.parse_args()

    # Hang diagnostics: dump every thread's stack to stderr periodically, so a
    # stuck run (Kaggle s1 v4: silent >90 min after round 1) shows where it is.
    faulthandler.dump_traceback_later(int(os.environ.get("FLOPS_HANG_DUMP_SECS", "1800")), repeat=True)

    algorithm, mechanism, rho = _resolve_arms(args)
    preset = _ABLATIONS.get(args.ablation or "", {})
    if preset.get("teacher_init") and (args.teacher_params is None or not args.teacher_params.exists()):
        raise SystemExit(f"--ablation {args.ablation} needs an existing --teacher-params (ADR-015)")
    if preset.get("server_ft") and (args.server_data_yaml is None or not args.server_data_yaml.exists()):
        raise SystemExit(f"--ablation {args.ablation} needs an existing --server-data-yaml (ADR-015)")

    if algorithm in _DISABLED_ALGORITHMS:
        raise NotImplementedError(_DISABLED_ALGORITHMS[algorithm])

    _set_seed(args.seed)

    if args.exp_config is None:
        args.exp_config = Path(__file__).parents[1] / "configs" / "experiments" / f"{args.run_class}.yaml"
    config = load_experiment_config(args.exp_config)

    manifest = load_partition_manifest(args.partition)
    data_yaml_map = {
        cid: args.data_yaml_dir / f"data_{cid}.yaml"
        for cid in manifest.client_assignments
    }
    for cid, p in data_yaml_map.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing per-client data yaml: {p}")

    # Arm name == run directory prefix. A bare strategy keeps its own name so the
    # sweep can predict the directory (see scripts/sweep_experiments.py).
    if args.ablation:
        arm_id = args.ablation
    elif mechanism == "none":
        arm_id = algorithm
    else:
        arm_id = f"{algorithm}-{mechanism}"
    exp_id = exp_id_for_arm(arm_id, mechanism, rho)

    run_dir, _ = prepare_run(
        exp_id=exp_id,
        config_path=args.exp_config,
        run_class=args.run_class,
        seed=args.seed,
        partition_id=manifest.partition_id,
        run_id=args.run_id,
    )

    # Save partition manifest into run dir per Section 21
    with (run_dir / "partition_manifest.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(manifest.to_dict(), f, default_flow_style=False, allow_unicode=True)

    fed = config["federated"]
    train_config = build_local_train_config(config)
    if mechanism in ("kd_teacher", "kd_teacher_rho"):
        if args.teacher_params is None:
            raise SystemExit(f"--client-mechanism {mechanism} needs --teacher-params")
        train_config["teacher_params"] = str(args.teacher_params.resolve())

    client_cls = _pick_client_class(algorithm, mechanism)
    num_val_examples = _count_val_examples(args.global_data_yaml)
    logger.info(
        "Arm=%s algorithm=%s mechanism=%s rho=%g tau_elig=%d client=%s",
        arm_id, algorithm, mechanism, rho, args.tau_elig, client_cls.__name__,
    )
    logger.info("Global val set: %d images (shared per §14)", num_val_examples)

    client_fn = _build_client_fn(
        client_cls, manifest, data_yaml_map,
        weights=config["model"]["weights"],
        train_config=train_config,
        run_dir=run_dir,
        num_val_examples=num_val_examples,
        rho=rho,
        base_seed=args.seed,
    )

    fraction_evaluate = (
        args.fraction_evaluate
        if args.fraction_evaluate is not None
        else fed["fraction_evaluate"]
    )

    fl_config = {
        "num_rounds": fed["num_rounds"],
        "num_clients": fed["num_clients"],
        "fraction_fit": fed["fraction_fit"],
        "fraction_evaluate": fraction_evaluate,
        "client_num_cpus": fed.get("client_num_cpus", 1),
        "client_num_gpus": fed.get("client_num_gpus", 1.0),
        "keep_last_checkpoints": fed.get("keep_last_checkpoints", 2),
        "prune_client_weights": fed.get("prune_client_weights", True),
        "proximal_mu": fed.get("proximal_mu", 0.01),    # FedProx only (ADR-010)
    }

    eval_config = {
        "image_size": config["model"]["image_size"],
        "conf": config["evaluation"]["conf"],
        "iou": config["evaluation"]["iou"],
        "device": config["train"]["device"],
        "weights": config["model"]["weights"],
        "workers": config["evaluation"].get("workers", 0),
        "eval_every": fed.get("eval_every", 1),   # ADR-011
    }

    param_names: list[str] | None = None
    strategy_options: dict[str, object] | None = None
    if algorithm in _CLASS_AWARE_STRATEGIES:
        param_names = _resolve_param_names(config["model"]["weights"])
        strategy_options = {"eligibility_threshold": int(args.tau_elig)}

    run_fl_server(
        algorithm=algorithm,
        fl_config=fl_config,
        client_fn=client_fn,
        run_dir=run_dir,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_uri=args.mlflow_uri,
        param_names=param_names,
        global_data_yaml=args.global_data_yaml,
        eval_config=eval_config if args.global_data_yaml else None,
        strategy_options=strategy_options,
        resume=args.resume,
        init_params_path=args.teacher_params if preset.get("teacher_init") else None,
        server_finetune=({"data_yaml": str(args.server_data_yaml), "epochs": 1, "seed": args.seed,
                          "batch_size": train_config["batch_size"], "image_size": train_config["image_size"],
                          "lr0": train_config["lr0"], "device": train_config["device"],
                          "nbs": train_config.get("nbs")}
                         if preset.get("server_ft") else None),
    )

    finalize_run(run_dir, algorithm=algorithm)
    logger.info("FL run complete: arm=%s / seed=%d / %s", arm_id, args.seed, run_dir.name)


if __name__ == "__main__":
    main()
