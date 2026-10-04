"""Run a declared comparison sweep: every (scenario × arm × seed) cell, in order.

This is the answer to "can the baselines be run together with the method": they
are run by the same sweep, on the same partition, with the same seeds and the
same evaluator — which is what makes the comparison valid (CLAUDE.md §14) —
but each cell is still its own process and its own run directory.

Cells run SEQUENTIALLY on purpose. With `client_num_gpus: 1.0` even the clients
inside one round are serialised; running two arms at once would simply OOM
(plan.md §13 K3). Sequential + resumable is what survives a 12 h Kaggle session.

Usage:
  python scripts/sweep_experiments.py --sweep configs/sweeps/c1_h1_missing_class.yaml \\
      --base-dir /kaggle/working --global-data-yaml /kaggle/working/data/bdd100k_yolo/data.yaml

  python scripts/sweep_experiments.py --sweep ... --dry-run     # print the plan only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_fl_experiment import (  # noqa: E402
    _ABLATIONS,
    _STRATEGIES,
    _RHO_MECHANISMS,
    exp_id_for_arm,
    mechanism_for_arm,
)
from src.experiments.checkpoint import find_latest_checkpoint  # noqa: E402
from src.experiments.runner import default_run_id  # noqa: E402
from src.utils.artifacts import ARTIFACTS_ROOT  # noqa: E402
from src.utils.config import load_experiment_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


@dataclass(frozen=True)
class Cell:
    scenario: str
    partition: Path
    data_yaml_dir: Path
    arm: str
    seed: int
    rho: float | None
    tau_elig: int
    run_class: str
    mlflow_experiment: str

    @property
    def mechanism(self) -> str:
        return mechanism_for_arm(self.arm)

    @property
    def exp_id(self) -> str:
        return exp_id_for_arm(self.arm, self.mechanism, self.rho if self.rho is not None else 1.0)


def _resolve(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base_dir / p)


def _partition_id(manifest: Path) -> str:
    data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    pid = data.get("partition_id")
    if not pid:
        raise ValueError(f"{manifest} has no partition_id")
    return str(pid)


def _normalise_arm(entry: Any, defaults: dict[str, Any]) -> tuple[str, float | None, int]:
    """Accept either ``A0`` or ``{id: A2b, rho: 0.25, tau_elig: 50}``."""
    if isinstance(entry, str):
        arm, opts = entry, {}
    else:
        opts = dict(entry)
        arm = str(opts.pop("id"))
    if arm not in _ABLATIONS and arm not in _STRATEGIES:
        raise SystemExit(
            f"Unknown arm {arm!r}. Use an ablation preset {sorted(_ABLATIONS)} "
            f"or a bare strategy name {list(_STRATEGIES)}."
        )
    rho = opts.get("rho", defaults.get("rho"))
    tau = int(opts.get("tau_elig", defaults.get("tau_elig", 1)))
    if mechanism_for_arm(arm) in _RHO_MECHANISMS and rho is None:
        raise SystemExit(
            f"Arm {arm} applies a client mechanism, so the sweep must state rho "
            "explicitly (plan.md §7.3 — rho is selected by the pre-registered sweep, "
            "it has no default)."
        )
    if mechanism_for_arm(arm) not in _RHO_MECHANISMS:
        rho = None                      # A5/A6/A6c and baselines take no --rho
    return arm, (float(rho) if rho is not None else None), tau


def build_cells(sweep: dict[str, Any], base_dir: Path) -> list[Cell]:
    defaults = dict(sweep.get("defaults", {}) or {})
    run_class = str(sweep["run_class"])
    cells: list[Cell] = []
    for group in sweep["cells"]:
        partition = _resolve(str(group["partition"]), base_dir)
        data_dir = _resolve(str(group["data_yaml_dir"]), base_dir)
        scenario = str(group.get("scenario", partition.parent.name))
        seeds = [int(s) for s in group.get("seeds", defaults.get("seeds", [42]))]
        mlflow_exp = str(
            group.get("mlflow_experiment",
                      defaults.get("mlflow_experiment", sweep["sweep_id"]))
        )
        for entry in group["arms"]:
            arm, rho, tau = _normalise_arm(entry, defaults)
            for seed in seeds:
                cells.append(Cell(
                    scenario=scenario, partition=partition, data_yaml_dir=data_dir,
                    arm=arm, seed=seed, rho=rho, tau_elig=tau,
                    run_class=run_class, mlflow_experiment=mlflow_exp,
                ))
    return cells


def cell_run_dir(cell: Cell, runs_root: Path) -> Path:
    return runs_root / default_run_id(
        cell.exp_id, cell.run_class, cell.seed, _partition_id(cell.partition)
    )


def cell_state(cell: Cell, runs_root: Path, total_rounds: int) -> tuple[str, int]:
    """('todo' | 'partial' | 'done', rounds_completed)."""
    run_dir = cell_run_dir(cell, runs_root)
    latest = find_latest_checkpoint(run_dir / "checkpoint")
    if latest is None:
        return ("todo", 0)
    done, _ = latest
    return ("done" if done >= total_rounds else "partial", done)


def build_command(
    cell: Cell,
    global_data_yaml: Path,
    mlflow_uri: str,
    resume: bool,
) -> list[str]:
    # An ablation preset goes through --ablation; a bare strategy name (a
    # reference baseline such as SCAFFOLD) goes through --algorithm.
    arm_flag = ["--ablation", cell.arm] if cell.arm in _ABLATIONS else ["--algorithm", cell.arm]
    cmd = [
        sys.executable, str(_REPO_ROOT / "scripts" / "run_fl_experiment.py"),
        "--partition", str(cell.partition),
        "--data-yaml-dir", str(cell.data_yaml_dir),
        *arm_flag,
        "--run-class", cell.run_class,
        "--seed", str(cell.seed),
        "--tau-elig", str(cell.tau_elig),
        "--mlflow-uri", mlflow_uri,
        "--mlflow-experiment", cell.mlflow_experiment,
        "--global-data-yaml", str(global_data_yaml),
    ]
    if cell.rho is not None and cell.mechanism != "none":
        cmd += ["--rho", str(cell.rho)]
    if resume:
        cmd += ["--resume"]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=Path, required=True, help="configs/sweeps/*.yaml")
    ap.add_argument("--base-dir", type=Path, default=Path.cwd(),
                    help="Base for relative paths in the sweep file")
    ap.add_argument("--global-data-yaml", type=Path, default=None)
    ap.add_argument("--mlflow-uri", default=None)
    ap.add_argument("--runs-root", type=Path, default=None,
                    help="Default: <repo>/artifacts/runs")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan and each cell's state; run nothing")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="Keep going after a failing cell (it stays 'partial'/'todo')")
    args = ap.parse_args()

    sweep = yaml.safe_load(args.sweep.read_text(encoding="utf-8")) or {}
    defaults = dict(sweep.get("defaults", {}) or {})

    global_data_yaml = args.global_data_yaml or (
        _resolve(str(defaults["global_data_yaml"]), args.base_dir)
        if "global_data_yaml" in defaults else None
    )
    if global_data_yaml is None:
        raise SystemExit(
            "--global-data-yaml is required (or set defaults.global_data_yaml). "
            "Without it the server skips centralized eval and no per-class "
            "metrics are produced (CLAUDE.md §13, §21)."
        )
    mlflow_uri = args.mlflow_uri or str(defaults.get("mlflow_uri", "http://mlflow:5000"))
    runs_root = args.runs_root or (ARTIFACTS_ROOT / "runs")

    exp_config = _REPO_ROOT / "configs" / "experiments" / f"{sweep['run_class']}.yaml"
    total_rounds = int(load_experiment_config(exp_config)["federated"]["num_rounds"])

    cells = build_cells(sweep, args.base_dir)
    print(f"Sweep {sweep['sweep_id']}: {len(cells)} cell(s), run_class="
          f"{sweep['run_class']}, num_rounds={total_rounds}")
    if sweep.get("description"):
        print(f"  {sweep['description'].strip()}")
    if sweep.get("baseline_arm"):
        print(f"  baseline arm: {sweep['baseline_arm']}")

    plan: list[tuple[Cell, str, int]] = []
    for cell in cells:
        state, done = cell_state(cell, runs_root, total_rounds)
        plan.append((cell, state, done))
        print(f"  [{state:7s}] {cell.scenario:12s} {cell.exp_id:16s} seed={cell.seed}"
              f"  rounds_done={done}/{total_rounds}")

    if args.dry_run:
        print("\n--dry-run: nothing executed.")
        return

    failures: list[tuple[Cell, int]] = []
    for cell, state, _done in plan:
        if state == "done":
            logger.info("Skipping completed cell %s seed=%d", cell.exp_id, cell.seed)
            continue
        cmd = build_command(cell, global_data_yaml, mlflow_uri, resume=(state == "partial"))
        print("\n" + "=" * 78)
        print(f"RUN {cell.exp_id} seed={cell.seed} scenario={cell.scenario} ({state})")
        print("=" * 78)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            failures.append((cell, result.returncode))
            logger.error(
                "Cell failed: %s seed=%d exit=%d", cell.exp_id, cell.seed, result.returncode
            )
            if not args.continue_on_error:
                raise SystemExit(
                    f"Sweep aborted at {cell.exp_id} seed={cell.seed}. Fix the cause, "
                    "then re-run the sweep — completed cells are skipped and partial "
                    "ones resume."
                )

    if failures:
        print("\nFailed cells (registered as incomplete, NOT as negative results):")
        for cell, code in failures:
            print(f"  {cell.exp_id} seed={cell.seed} exit={code}")
        raise SystemExit(1)

    print(f"\nSweep {sweep['sweep_id']} complete. Next: scripts/compare_runs.py")


if __name__ == "__main__":
    main()
