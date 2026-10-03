"""Baseline report: Centralized (G2) vs FedAvg vs FedProx from one results folder."""
from __future__ import annotations

import csv
import json

import yaml

from scripts.analyze_baselines import build_report

CLASSES = ("car", "bus", "truck", "motorcycle")


def _run(root, name, exp_id, final, rounds=None, partition="s0_iid_seed42"):
    d = root / "artifacts" / "runs" / name
    d.mkdir(parents=True)
    (d / "environment.json").write_text(json.dumps(
        {"exp_id": exp_id, "seed": 42, "partition_id": partition, "run_class": "feasibility",
         "git_commit": "abc"}))
    (d / "config.yaml").write_text(yaml.safe_dump(
        {"model": {"image_size": 640}, "train": {"batch_size": 16, "lr0": 0.01},
         "federated": {"num_rounds": 5, "local_epochs": 1, "num_clients": 4, "fraction_fit": 1.0},
         "evaluation": {"conf": 0.001, "iou": 0.7}}))
    with (d / "metrics.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in final.items():
            w.writerow([k, v])
    if rounds:
        keys = sorted({k for r in rounds for k in r})
        with (d / "round_metrics.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rounds)
    (d / "run.log").write_text("2026-10-03 10:00:00 [INFO] start\n2026-10-03 11:30:00 [INFO] end\n")
    return d


def _final(m50, bus):
    out = {"mAP50": m50, "mAP50-95": m50 / 2}
    for c in CLASSES:
        out[f"AP50_{c}"] = bus if c == "bus" else m50
        out[f"FN_{c}"] = 100.0
    return out


def test_report_tables_and_deltas(tmp_path):
    _run(tmp_path, "G2-centralized_feasibility_seed42_centralized", "G2-centralized",
         _final(0.60, 0.40), partition="centralized")
    _run(tmp_path, "FedAvg_feasibility_seed42_s0_iid_seed42", "FedAvg", _final(0.50, 0.30),
         rounds=[{"round": r, "mAP50": 0.1 * r} for r in range(0, 6)])
    _run(tmp_path, "FedProx_feasibility_seed42_s0_iid_seed42", "FedProx", _final(0.52, 0.33),
         rounds=[{"round": r, "mAP50": 0.1 * r + 0.02} for r in range(0, 6)])

    out = tmp_path / "report"
    report = build_report(tmp_path, out)

    table = report["table"]
    assert set(table) == {"G2-centralized", "FedAvg", "FedProx"}
    assert table["FedAvg"]["AP50_bus"] == 0.30
    gap = report["gap_vs_centralized"]
    assert round(gap["FedAvg"]["mAP50"], 6) == -0.10
    assert round(report["fedprox_vs_fedavg"]["AP50_bus"], 6) == 0.03
    assert report["convergence"]["FedAvg"]["best_round"] == 5
    assert report["duration_min"]["FedAvg"] == 90.0
    assert report["fl_comparable_issues"] == []          # same partition + protocol
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "FedProx" in text and "single seed" in text.lower()
    assert (out / "baseline_table.csv").exists()


def test_report_flags_incomparable_fl_runs(tmp_path):
    _run(tmp_path, "FedAvg_a", "FedAvg", _final(0.5, 0.3), partition="s0_iid_seed42")
    _run(tmp_path, "FedProx_b", "FedProx", _final(0.5, 0.3), partition="other_partition")
    report = build_report(tmp_path, tmp_path / "r")
    assert report["fl_comparable_issues"]
