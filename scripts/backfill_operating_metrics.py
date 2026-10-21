"""Add operating-threshold metrics (conf 0.25) to completed YOLO11s registry entries.

Every run records, per round, the Ultralytics confusion-matrix counts TP/FP/FN per class. [YOLO-DOC: ultralytics
8.3.253 utils/metrics.py ConfusionMatrix.process_batch] they use conf > 0.25 (the default val conf 0.001 is
replaced by 0.25) and IoU > 0.45; FP includes background detections and detections of another class, FN includes
missed objects and objects detected as another class. From them:

  recall_at025_c    = TP_c / (TP_c + FN_c)
  precision_at025_c = TP_c / (TP_c + FP_c)

The ``precision_c`` / ``recall_c`` columns of the run are NOT at 0.25: they are ``results.box.p/r``, taken at the
confidence that maximises the class-mean F1 (``ap_per_class``), which differs per run; they are kept here as
``*_maxF1_c`` for reference only. This script writes the rule mean (rounds 10 and 20, ratios computed per round)
into ``results.operating_r10_20`` of the matching registry entry, so the figures keep reading the registry only. A run is matched by (method, scenario, seed); the match is accepted only if the
run's AP50 bus equals the registered one to 1e-3, otherwise the script stops without writing.

Usage:
  python scripts/backfill_operating_metrics.py [--outputs kaggle/output] [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation.registry_results import load_runs  # noqa: E402

CLASSES = ("bus", "car", "truck", "motorcycle")
COUNTS = tuple(f"{k}_{c}" for c in CLASSES for k in ("TP", "FP", "FN"))
KEYS = (*COUNTS, *(f"recall_at025_{c}" for c in CLASSES), *(f"precision_at025_{c}" for c in CLASSES),
        *(f"recall_maxF1_{c}" for c in CLASSES), *(f"precision_maxF1_{c}" for c in CLASSES))
RUN_DIR = re.compile(r"^(?P<method>[A-Za-z0-9]+)_feasibility_seed(?P<seed>\d+)_s1b5_(?P<sc>bus|pooled|bus1)_3k_moto_seed42$")
SCENARIO = {"bus": "s5", "pooled": "pooled", "bus1": "bus1"}


def _round_values(row: dict[str, str]) -> dict[str, float]:
    v = {k: float(row[k]) for k in ("AP50_bus", *COUNTS)}
    for c in CLASSES:
        tp, fp, fn = v[f"TP_{c}"], v[f"FP_{c}"], v[f"FN_{c}"]
        v[f"recall_at025_{c}"] = tp / (tp + fn) if tp + fn else 0.0
        v[f"precision_at025_{c}"] = tp / (tp + fp) if tp + fp else 0.0
        v[f"recall_maxF1_{c}"] = float(row[f"recall_{c}"])
        v[f"precision_maxF1_{c}"] = float(row[f"precision_{c}"])
    return v


def rule_mean(round_csv: Path) -> dict[str, float] | None:
    rows = {int(r["round"]): r for r in csv.DictReader(round_csv.open(encoding="utf-8"))}
    need = ("AP50_bus", *COUNTS, *(f"recall_{c}" for c in CLASSES), *(f"precision_{c}" for c in CLASSES))
    if 10 not in rows or 20 not in rows or any(k not in rows[r] for r in (10, 20) for k in need):
        return None
    a, b = _round_values(rows[10]), _round_values(rows[20])
    return {k: (a[k] + b[k]) / 2 for k in a}


def _fmt(k: str, v: float) -> str:
    return f"{k}: {v:.1f}" if k[:3] in ("TP_", "FP_", "FN_") else f"{k}: {v:.4f}"


def insert_block(text: str, exp_id: str, values: dict[str, float]) -> str:
    """Insert (or replace) ``    operating_r10_20: {...}`` as the last line of the entry's results."""
    start = text.index(f"- id: {exp_id}\n")
    nxt = text.find("\n- id: ", start + 1)
    end = len(text) if nxt < 0 else nxt + 1
    entry = re.sub(r"    operating_r10_20: \{.*\}\n", "", text[start:end])
    lines = entry.splitlines(keepends=True)
    res = next(i for i, ln in enumerate(lines) if ln.startswith("  results:"))
    last = res
    for i in range(res + 1, len(lines)):
        if lines[i].startswith("    ") or not lines[i].strip():
            last = i if lines[i].strip() else last
        else:
            break
    line = "    operating_r10_20: {" + ", ".join(_fmt(k, values[k]) for k in KEYS) + "}\n"
    lines.insert(last + 1, line)
    return text[:start] + "".join(lines) + text[end:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", type=Path, default=_REPO_ROOT / "research" / "experiment_registry" / "registry.yaml")
    ap.add_argument("--outputs", type=Path, default=_REPO_ROOT / "kaggle" / "output")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    index = {(r.method, r.scenario, r.seed): r for r in load_runs(args.registry)}
    text = args.registry.read_text(encoding="utf-8")
    done = 0
    for csv_path in sorted(args.outputs.glob("*/artifacts/runs/*/round_metrics.csv")):
        m = RUN_DIR.match(csv_path.parent.name)
        if not m:
            continue
        run = index.get((m["method"], SCENARIO[m["sc"]], int(m["seed"])))
        values = rule_mean(csv_path)
        if run is None or values is None:
            continue
        if abs(values["AP50_bus"] - run.metrics["AP50_bus"]) > 1e-3:
            sys.exit(f"AP50 bus mismatch for {run.exp_id}: output {values['AP50_bus']:.4f} "
                     f"vs registry {run.metrics['AP50_bus']:.4f} ({csv_path})")
        text = insert_block(text, run.exp_id, values)
        done += 1
        print(f"{run.exp_id}: recall@0.25 bus {values['recall_at025_bus']:.3f} "
              f"(max-F1 {values['recall_maxF1_bus']:.3f}), TP_bus {values['TP_bus']:.0f}")
    if not args.dry_run:
        args.registry.write_text(text, encoding="utf-8", newline="\n")
    print(f"{done} entries {'checked' if args.dry_run else 'updated'}")


if __name__ == "__main__":
    main()
