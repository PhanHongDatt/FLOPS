"""Extract small, committable report data from the downloaded Kaggle outputs (kaggle/output is git-ignored).

Writes under research/figures/data/:
  partition_class_counts.yaml  boxes per client and class for the three scenarios on the same 15,000 images
                               (pooled 0/5, S1b5 3/5, bus1 4/5), copied from the runs' partition manifests;
  run_times.csv                wall time per round of every YOLO11s run, from the Flower log timestamps
                               ("[ROUND k]" to the next round; rounds 10 and 20 also contain the global evaluation),
                               with the session GPU read from the Kaggle session log.

Usage:
  python scripts/extract_report_data.py [--outputs kaggle/output]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, median

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = _REPO_ROOT / "research" / "figures" / "data"
RUN_DIR = re.compile(r"^(?P<method>[A-Za-z0-9]+)_feasibility_seed(?P<seed>\d+)_s1b5_(?P<sc>bus|pooled|bus1)_3k_moto_seed42$")
SCENARIO = {"bus": "s5", "pooled": "pooled", "bus1": "bus1"}
CLASSES = ("car", "bus", "truck", "motorcycle")
STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) .*\[ROUND (\d+)\]")
END = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) .*FL run complete")
GPU = re.compile(r"Tesla [A-Z0-9]+|NVIDIA [A-Z][A-Za-z0-9 ]*?(?=[,\"\\])")


def class_counts(outputs: Path) -> dict:
    found = {}
    for man in sorted(outputs.glob("*/artifacts/runs/*/partition_manifest.yaml")):
        m = RUN_DIR.match(man.parent.name)
        if not m or SCENARIO[m["sc"]] in found:
            continue
        counts = yaml.safe_load(man.read_text(encoding="utf-8"))["class_counts"]
        found[SCENARIO[m["sc"]]] = {c: {k: int(counts[c].get(k, 0)) for k in CLASSES} for c in sorted(counts)}
    missing = {"pooled", "s5", "bus1"} - set(found)
    if missing:
        sys.exit(f"no manifest for {sorted(missing)}")
    totals = {s: {k: sum(v[k] for v in found[s].values()) for k in CLASSES} for s in found}
    if len({tuple(t.values()) for t in totals.values()}) != 1:
        sys.exit(f"box totals differ between scenarios: {totals}")
    return {"note": "boxes per client (C0-C4) and class; same 15,000 images and box totals in every scenario",
            "totals": totals["pooled"], "scenarios": found}


def round_times(run_log: Path) -> tuple[list[float], float] | None:
    stamps, end = {}, None
    for line in run_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if (m := STAMP.match(line)):
            stamps[int(m[2])] = datetime.fromisoformat(m[1])
        elif (m := END.match(line)):
            end = datetime.fromisoformat(m[1])
    if not stamps or end is None:
        return None
    rounds = sorted(stamps)
    bounds = [stamps[r] for r in rounds] + [end]
    per_round = [(bounds[i + 1] - bounds[i]).total_seconds() / 60 for i in range(len(rounds))]
    return per_round, (end - stamps[rounds[0]]).total_seconds() / 60


def session_gpu(session: Path) -> str:
    for log in session.glob("*.log"):
        m = GPU.search(log.read_text(encoding="utf-8", errors="replace"))
        if m:
            return m[0].strip()
    return "Tesla T4 (machine_shape in kaggle/sessions.yaml)"


def run_times(outputs: Path) -> list[list]:
    rows, seen = [], set()
    for log in sorted(outputs.glob("*/artifacts/runs/*/run.log")):
        m = RUN_DIR.match(log.parent.name)
        res = round_times(log) if m else None
        if res is None or len(res[0]) != 20 or log.parent.name in seen:   # the 20-round protocol only
            continue
        seen.add(log.parent.name)
        per_round, total = res
        train_only = [t for i, t in enumerate(per_round, start=1) if i not in (10, 20)]
        rows.append([log.parent.name, m["method"], SCENARIO[m["sc"]], int(m["seed"]), session_gpu(log.parents[3]),
                     len(per_round), f"{total:.1f}", f"{mean(train_only):.2f}", f"{median(train_only):.2f}",
                     f"{mean(per_round[i - 1] - mean(train_only) for i in (10, 20) if i <= len(per_round)):.2f}"])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", type=Path, default=_REPO_ROOT / "kaggle" / "output")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "partition_class_counts.yaml").write_text(yaml.safe_dump(class_counts(args.outputs), sort_keys=False),
                                                    encoding="utf-8")
    rows = run_times(args.outputs)
    with (OUT / "run_times.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "method", "scenario", "seed", "gpu", "rounds", "total_min", "train_round_mean_min",
                    "train_round_median_min", "eval_extra_min"])
        w.writerows(rows)
    print(f"wrote partition_class_counts.yaml and run_times.csv ({len(rows)} runs)")


if __name__ == "__main__":
    main()
