"""Baseline comparison report: Centralized (G2) vs FedAvg vs FedProx.

Reads one results folder — the extracted ``flops_results.zip`` of Kaggle session 2
(``artifacts/runs/<run>/{environment.json,config.yaml,metrics.csv,round_metrics.csv,run.log}``)
— and writes ``report.md``, ``baseline_table.csv`` and plots.

The FL runs must pass ``check_comparable`` (same partition / protocol / rounds);
the centralized run is a reference upper bound, not a federated arm. Single-seed
feasibility output is labelled as such (CLAUDE.md §14).

Usage:
  python scripts/analyze_baselines.py --results kaggle/output/s2 --out kaggle/output/s2/report
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.bdd100k import TARGET_CLASSES  # noqa: E402
from src.evaluation.compare import RunRecord, check_comparable, discover_runs  # noqa: E402

CENTRALIZED = "G2-centralized"
METRICS = (
    ["mAP50", "mAP50-95"]
    + [f"AP50_{c}" for c in TARGET_CLASSES]
    + [f"AP_{c}" for c in TARGET_CLASSES]
    + [f"precision_{c}" for c in TARGET_CLASSES]
    + [f"recall_{c}" for c in TARGET_CLASSES]
    + [f"FN_{c}" for c in TARGET_CLASSES]
    + [f"FP_{c}" for c in TARGET_CLASSES]
)
_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def _duration_min(run_dir: Path) -> float | None:
    log = run_dir / "run.log"
    if not log.exists():
        return None
    stamps = [m.group(1) for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
              if (m := _TS.match(line))]
    if len(stamps) < 2:
        return None
    fmt = "%Y-%m-%d %H:%M:%S"
    return (datetime.strptime(stamps[-1], fmt) - datetime.strptime(stamps[0], fmt)).total_seconds() / 60


def _convergence(rec: RunRecord, metric: str = "mAP50") -> dict[str, Any]:
    series = [(int(r["round"]), r[metric]) for r in rec.rounds if metric in r]
    if not series:
        return {}
    best_round, best = max(series, key=lambda t: t[1])
    return {"best_round": best_round, "best": best, "final_round": series[-1][0],
            "final": series[-1][1], "series": series}


def _delta(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {k: a[k] - b[k] for k in METRICS if k in a and k in b}


def _fmt(v: Any) -> str:
    return f"{v:.4f}" if isinstance(v, float) else ("—" if v is None else str(v))


def _write_markdown(report: dict[str, Any], out: Path) -> None:
    arms = list(report["table"])
    lines = ["# Baseline comparison — Centralized vs FedAvg vs FedProx", "",
             "**Feasibility scale, single seed (42)**: a sanity comparison, not a reportable result "
             "(CLAUDE.md §14 requires 3 seeds). AP at conf 0.001, FP/FN at 0.25 (ADR-009).", ""]
    if report["fl_comparable_issues"]:
        lines += ["## ⚠️ FL runs are NOT comparable", ""] + [f"- {p}" for p in report["fl_comparable_issues"]] + [""]
    lines += ["## Final metrics", "", "| metric | " + " | ".join(arms) + " |",
              "|---|" + "---|" * len(arms)]
    for k in METRICS:
        if any(k in report["table"][a] for a in arms):
            lines.append(f"| {k} | " + " | ".join(_fmt(report["table"][a].get(k)) for a in arms) + " |")
    for title, block in (("Gap vs centralized (arm − G2)", report["gap_vs_centralized"]),):
        if block:
            cols = list(block)
            lines += ["", f"## {title}", "", "| metric | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
            for k in METRICS:
                if any(k in block[c] for c in cols):
                    lines.append(f"| {k} | " + " | ".join(_fmt(block[c].get(k)) for c in cols) + " |")
    if report["fedprox_vs_fedavg"]:
        lines += ["", "## FedProx − FedAvg", "", "| metric | Δ |", "|---|---|"]
        lines += [f"| {k} | {_fmt(v)} |" for k, v in report["fedprox_vs_fedavg"].items()]
    if report["convergence"]:
        lines += ["", "## Convergence (mAP50 per round, server eval on global val)", "",
                  "| arm | best round | best | final round | final |", "|---|---|---|---|---|"]
        for arm, c in report["convergence"].items():
            lines.append(f"| {arm} | {c['best_round']} | {_fmt(c['best'])} | {c['final_round']} | {_fmt(c['final'])} |")
    lines += ["", "## Wall-clock (run.log first → last timestamp)", "", "| arm | minutes |", "|---|---|"]
    lines += [f"| {a} | {_fmt(m)} |" for a, m in report["duration_min"].items()]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(results: Path, out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    records = discover_runs(Path(results) / "artifacts" / "runs")
    by_arm = {r.arm: r for r in records}
    table = {arm: {k: r.final_metrics[k] for k in METRICS if k in r.final_metrics}
             for arm, r in by_arm.items()}
    fl = [r for r in records if r.arm != CENTRALIZED]
    report: dict[str, Any] = {
        "table": table,
        "gap_vs_centralized": ({a: _delta(table[a], table[CENTRALIZED]) for a in table if a != CENTRALIZED}
                               if CENTRALIZED in table else {}),
        "fedprox_vs_fedavg": (_delta(table["FedProx"], table["FedAvg"])
                              if {"FedProx", "FedAvg"} <= set(table) else {}),
        "convergence": {r.arm: _convergence(r) for r in fl if _convergence(r)},
        "duration_min": {arm: _duration_min(r.run_dir) for arm, r in by_arm.items()},
        "fl_comparable_issues": check_comparable(fl) if len(fl) > 1 else [],
    }
    with (out / "baseline_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", *table])
        for k in METRICS:
            if any(k in table[a] for a in table):
                w.writerow([k, *(table[a].get(k, "") for a in table)])
    _write_markdown(report, out)
    _plots(records, table, out)
    return report


def _plots(records: list[RunRecord], table: dict[str, dict[str, float]], out: Path) -> None:
    try:
        from src.evaluation.plots import plot_metric_over_rounds, plot_per_class_ap
    except ImportError:
        return
    try:
        plot_per_class_ap({a: {k: (v, 0.0, 1) for k, v in m.items()} for a, m in table.items()},
                          out / "per_class_ap50.png", title="AP50 per class (seed 42)")
        fl = [r for r in records if r.rounds]
        if fl:
            plot_metric_over_rounds(fl, "mAP50", out / "map50_over_rounds.png", title="mAP50 per round")
    except Exception as exc:   # plots are a convenience; the tables are the record
        print(f"plotting skipped: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True, help="extracted flops_results.zip folder")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.results / "report"
    build_report(args.results, out)
    print((out / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
