"""Aggregate finished runs into the comparison tables and plots.

Reads run directories, enforces the §14/§20 comparability conditions, then writes
``comparison_per_run.csv``, ``comparison_summary.csv``, a terminal table, and the
plots that CLAUDE.md §21 requires in ``plots/``.

By default an invalid comparison is REFUSED. ``--allow-invalid`` downgrades the
refusal to a warning and stamps the reasons into the output directory, so an
exploratory look is possible but can never be mistaken for a reportable table.

Usage:
  python scripts/compare_runs.py --arms A0 A1 A3 --baseline A0 \\
      --partition-id s1_mc_seed42 --out artifacts/comparisons/c3_server_only

  # S1 vs S1-Control for one arm (the H1 comparison)
  python scripts/compare_runs.py --arms A0 --scenario-delta s1_mc_seed42 s1_control_seed42 \\
      --out artifacts/comparisons/c1_h1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation.compare import (  # noqa: E402
    PRIMARY_METRICS,
    aggregate_by_arm,
    check_comparable,
    delta_vs_baseline,
    discover_runs,
    format_table,
    metric_keys,
    min_seeds_warning,
    scenario_delta,
    write_per_run_csv,
    write_summary_csv,
)
from src.utils.artifacts import ARTIFACTS_ROOT  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=None,
                    help="Default: <repo>/artifacts/runs")
    ap.add_argument("--arms", nargs="+", default=None,
                    help="Arm (exp_id) names to include, e.g. A0 A1 A3 A4b_rho0.25")
    ap.add_argument("--baseline", default="A0",
                    help="Arm used for delta columns (CLAUDE.md §13 ΔAP_c)")
    ap.add_argument("--partition-id", default=None,
                    help="Restrict to one partition — required for a valid comparison")
    ap.add_argument("--run-class", default=None, choices=["smoke", "feasibility", "main"])
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--scenario-delta", nargs=2, metavar=("PARTITION_A", "PARTITION_B"),
                    default=None,
                    help="Paired per-seed delta of each arm between two partitions "
                         "(the S1 vs S1-Control comparison)")
    ap.add_argument("--min-seeds", type=int, default=3)
    ap.add_argument("--allow-invalid", action="store_true",
                    help="Warn instead of refusing when the comparison violates §14/§20")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--out", type=Path, default=None,
                    help="Default: <repo>/artifacts/comparisons/latest")
    args = ap.parse_args()

    runs_root = args.runs_root or (ARTIFACTS_ROOT / "runs")
    out_dir = args.out or (ARTIFACTS_ROOT / "comparisons" / "latest")

    records = discover_runs(runs_root)
    if not records:
        raise SystemExit(f"No run directories found under {runs_root}")

    if args.arms:
        records = [r for r in records if r.arm in set(args.arms)]
    if args.run_class:
        records = [r for r in records if r.run_class == args.run_class]
    if args.seeds:
        records = [r for r in records if r.seed in set(args.seeds)]

    # Scenario-delta compares ACROSS partitions on purpose, so the single-partition
    # filter and the partition check only apply to the within-partition mode.
    if args.scenario_delta:
        wanted = set(args.scenario_delta)
        records = [r for r in records if r.partition_id in wanted]
    elif args.partition_id:
        records = [r for r in records if r.partition_id == args.partition_id]

    if not records:
        raise SystemExit("No runs left after filtering — check --arms/--partition-id.")

    print(f"{len(records)} run(s): "
          + ", ".join(sorted({f'{r.arm}@{r.partition_id}' for r in records})))

    # ── validity ──────────────────────────────────────────────────────────
    if args.scenario_delta:
        problems: list[str] = []
        for pid in args.scenario_delta:
            problems += [f"[{pid}] {p}"
                         for p in check_comparable([r for r in records if r.partition_id == pid])]
    else:
        problems = check_comparable(records)

    warnings = min_seeds_warning(records, args.min_seeds)

    out_dir.mkdir(parents=True, exist_ok=True)
    if problems:
        text = "\n".join(f"- {p}" for p in problems)
        (out_dir / "COMPARISON_INVALID.md").write_text(
            "# This comparison does NOT satisfy CLAUDE.md §14/§20\n\n"
            f"{text}\n\nDo not report these numbers as a method comparison.\n",
            encoding="utf-8",
        )
        if not args.allow_invalid:
            raise SystemExit(
                "Refusing to produce a comparison table:\n" + text
                + "\n\nFix the runs, or pass --allow-invalid for an exploratory "
                  "look (the output is then stamped COMPARISON_INVALID.md)."
            )
        logger.warning("Proceeding with an INVALID comparison:\n%s", text)
    else:
        (out_dir / "COMPARISON_INVALID.md").unlink(missing_ok=True)

    for w in warnings:
        logger.warning("below --min-seeds: %s", w)

    # ── tables ────────────────────────────────────────────────────────────
    write_per_run_csv(records, out_dir / "comparison_per_run.csv")
    aggregated = aggregate_by_arm(records)

    deltas = None
    baseline = args.baseline if args.baseline in aggregated else None
    if baseline:
        deltas = delta_vs_baseline(aggregated, baseline)
    elif args.baseline:
        logger.warning(
            "baseline arm %r absent from these runs — delta columns omitted. "
            "A method comparison without its declared baseline is not a comparison.",
            args.baseline,
        )

    write_summary_csv(aggregated, out_dir / "comparison_summary.csv", deltas, baseline)

    keys = metric_keys(records)
    headline = [m for m in PRIMARY_METRICS if m in keys] + [k for k in keys if k.startswith("AP50_")]
    print("\n" + format_table(aggregated, headline, baseline))

    if args.scenario_delta:
        pa, pb = args.scenario_delta
        print(f"\nPaired scenario delta ({pa} − {pb}), per seed then averaged:")
        for arm in sorted({r.arm for r in records}):
            d = scenario_delta(records, arm, pa, pb)
            if not d:
                continue
            print(f"  arm {arm}:")
            for key in headline:
                if key in d:
                    mean, std, n = d[key]
                    print(f"    {key:22s} {mean:+.4f} ± {std:.4f}  (n={n} paired seeds)")

    # ── plots ─────────────────────────────────────────────────────────────
    if not args.no_plots:
        from src.evaluation import plots
        plots_dir = out_dir / "plots"
        written = [
            plots.plot_per_class_ap(aggregated, plots_dir / "per_class_ap50.png"),
            plots.plot_metric_over_rounds(records, "mAP50", plots_dir / "map50_over_rounds.png"),
        ]
        for cls_metric in ("AP50_bus", "AP50_truck"):
            if cls_metric in keys:
                written.append(plots.plot_metric_over_rounds(
                    records, cls_metric, plots_dir / f"{cls_metric.lower()}_over_rounds.png"
                ))
        if deltas:
            written.append(plots.plot_delta_bars(
                deltas, plots_dir / "delta_ap50_vs_baseline.png", baseline_arm=baseline or "A0"
            ))
        for p in [w for w in written if w]:
            print("plot:", p)

    print(f"\nWrote comparison to {out_dir}")
    if problems:
        print("WARNING: comparison is marked INVALID — see COMPARISON_INVALID.md")


if __name__ == "__main__":
    main()
