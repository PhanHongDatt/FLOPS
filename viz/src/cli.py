"""CLI entry point for FL visualization toolkit.

Usage:
    python -m viz.cli --runs runs/ --out figures/ [--only convergence,partition_heatmap] [--demo]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

from viz.src.demo import generate_demo_data
from viz.src.io import load_runs
from viz.src.style import apply_style, save_figure

logger = logging.getLogger("viz")

# Registry: chart_name -> (module_path, function_name, required_tables)
_CHART_REGISTRY: list[tuple[str, str, str, list[str]]] = [
    # (name, module, function, required_tables)
    ("convergence",          "viz.src.plots.convergence",     "plot_convergence",          ["rounds_global"]),
    ("convergence_vs_comm",  "viz.src.plots.convergence",     "plot_convergence_vs_comm",  ["rounds_global", "rounds_client"]),
    ("rounds_to_target",     "viz.src.plots.convergence",     "plot_rounds_to_target",     ["rounds_global"]),
    ("partition_heatmap",    "viz.src.plots.partition",       "plot_partition_heatmap",     ["partition"]),
    ("per_class_metrics",    "viz.src.plots.partition",       "plot_per_class_metrics",     ["per_class"]),
    ("per_client_performance","viz.src.plots.client",         "plot_per_client_performance",["rounds_client"]),
    ("client_participation", "viz.src.plots.client",          "plot_client_participation",  ["rounds_client"]),
    ("local_vs_global",      "viz.src.plots.client",          "plot_local_vs_global",       ["rounds_global", "rounds_client"]),
    ("comm_cost",            "viz.src.plots.communication",   "plot_comm_cost",             ["rounds_client"]),
    ("round_time_breakdown", "viz.src.plots.timing",          "plot_round_time_breakdown",  ["system"]),
    ("ablation",             "viz.src.plots.convergence",     "plot_ablation",              ["rounds_global"]),
    ("summary_table",        "viz.src.plots.summary",         "plot_summary_table",         ["rounds_global"]),
]


def _has_required_tables(
    runs: dict[str, dict[str, Any]], tables: list[str]
) -> bool:
    """Check if at least one run has all required tables."""
    for method, seeds in runs.items():
        for seed, data in seeds.items():
            if all(t in data for t in tables):
                return True
    return False


def _import_func(module_path: str, func_name: str):
    """Dynamically import a plot function."""
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)


def _load_experiment_config(runs_dir: Path) -> dict[str, Any]:
    """Try to load experiment.yaml from viz/config/."""
    import yaml
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "experiment.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def run_all_charts(
    runs_dir: Path,
    out_dir: Path,
    only: list[str] | None = None,
    is_demo: bool = False,
) -> None:
    """Generate all (or selected) charts."""
    apply_style()

    logger.info("Loading runs from %s", runs_dir)
    runs = load_runs(runs_dir)

    exp_cfg = _load_experiment_config(runs_dir)
    exp = exp_cfg.get("experiment", {})
    metrics_cfg = exp_cfg.get("metrics", {})
    targets_cfg = exp_cfg.get("targets", [0.5, 0.6, 0.7])

    metric = metrics_cfg.get("primary", "global_acc")
    metric_label = metrics_cfg.get("primary_label", "Accuracy")
    higher_is_better = metrics_cfg.get("higher_is_better", True)
    confidence = exp_cfg.get("confidence", 0.95)
    setting = exp.get("scenario", "")

    # Collect sidecar base info
    all_seeds = set()
    all_methods = set()
    git_commits = set()
    for method, seeds in runs.items():
        all_methods.add(method)
        for seed, data in seeds.items():
            all_seeds.add(seed)
            cfg = data.get("config", {})
            if "git_commit" in cfg:
                git_commits.add(cfg["git_commit"])

    sidecar_base = {
        "methods": sorted(all_methods),
        "seeds": sorted(all_seeds),
        "n_seeds": len(all_seeds),
        "git_commits": sorted(git_commits),
        "is_demo": is_demo,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0

    for chart_name, mod_path, func_name, required in _CHART_REGISTRY:
        if only and chart_name not in only:
            continue

        if not _has_required_tables(runs, required):
            logger.warning(
                "Skipping '%s': missing required tables %s", chart_name, required
            )
            skipped += 1
            continue

        logger.info("Generating chart: %s", chart_name)
        try:
            func = _import_func(mod_path, func_name)

            # Build kwargs based on function
            kwargs: dict[str, Any] = {"runs": runs, "setting": setting}

            if func_name in ("plot_convergence", "plot_convergence_vs_comm", "plot_ablation"):
                kwargs.update(metric=metric, metric_label=metric_label, confidence=confidence)
            if func_name == "plot_convergence":
                kwargs["higher_is_better"] = higher_is_better
            if func_name == "plot_rounds_to_target":
                kwargs.update(
                    targets=targets_cfg, metric=metric,
                    metric_label=metric_label,
                    higher_is_better=higher_is_better,
                )
            if func_name == "plot_summary_table":
                kwargs.update(
                    metric=metric, metric_label=metric_label,
                    higher_is_better=higher_is_better,
                )

            fig = func(**kwargs)

            filename = f"{chart_name}_{setting}".rstrip("_") if setting else chart_name
            filename = filename.lower().replace(" ", "_").replace("-", "_")

            sidecar = {**sidecar_base, "chart": chart_name}
            save_figure(fig, out_dir / filename, sidecar_data=sidecar, is_demo=is_demo)
            plt.close(fig)
            generated += 1

        except Exception:
            logger.exception("Error generating chart '%s'", chart_name)
            skipped += 1

    # Summary CSV
    try:
        from viz.src.plots.summary import save_summary_csv
        csv_path = out_dir / "summary_table.csv"
        save_summary_csv(runs, csv_path, metric=metric, metric_label=metric_label,
                         higher_is_better=higher_is_better)
    except Exception:
        logger.exception("Error saving summary CSV")

    logger.info("Done: %d charts generated, %d skipped.", generated, skipped)


import matplotlib.pyplot as plt


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="FL Visualization Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs", type=Path, default=Path("runs"),
        help="Path to runs directory (default: runs/)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("figures"),
        help="Output directory for figures (default: figures/)",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Comma-separated list of chart names to generate",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Generate demo data and plot with watermark",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    only = [s.strip() for s in args.only.split(",")] if args.only else None

    if args.demo:
        logger.info("Generating demo data...")
        demo_runs = generate_demo_data(args.out.parent)
        run_all_charts(demo_runs, args.out, only=only, is_demo=True)
    else:
        run_all_charts(args.runs, args.out, only=only, is_demo=False)


if __name__ == "__main__":
    main()
