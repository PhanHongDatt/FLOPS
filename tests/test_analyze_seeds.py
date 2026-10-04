"""Multi-seed summary with the pre-declared rule (mean of evals at rounds 20/25/30)."""
from __future__ import annotations

import csv
import json

import pytest

from scripts.analyze_seeds import paired_effect, rule_value, summarise


def _run(root, name, seed, partition, bus_by_round):
    d = root / "artifacts" / "runs" / name
    d.mkdir(parents=True)
    (d / "environment.json").write_text(json.dumps({"exp_id": name.split("_feasibility")[0], "seed": seed,
                                                    "partition_id": partition, "run_class": "feasibility"}))
    (d / "config.yaml").write_text("{}")
    with (d / "round_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["round", "AP50_bus", "mAP50"])
        w.writeheader()
        for r, v in bus_by_round.items():
            w.writerow({"round": r, "AP50_bus": v, "mAP50": v + 0.1})


def test_rule_value_uses_rounds_20_25_30():
    rows = [{"round": r, "AP50_bus": v} for r, v in ((5, 9.0), (20, 0.3), (25, 0.2), (30, 0.1))]
    assert rule_value(rows, "AP50_bus") == pytest.approx(0.2)
    assert rule_value(rows[:2], "AP50_bus") is None          # incomplete run


def test_summary_and_paired_effects(tmp_path):
    for seed, (a0, ctrl, a4b) in {42: (0.23, 0.28, 0.22), 123: (0.24, 0.27, 0.21), 2024: (0.22, 0.29, 0.23)}.items():
        _run(tmp_path, f"A0_feasibility_seed{seed}_s1b_bus_2k_seed42", seed, "s1b_bus_2k_seed42",
             {20: a0, 25: a0, 30: a0})
        _run(tmp_path, f"A0_feasibility_seed{seed}_s1_control_matched_2k_seed42", seed,
             "s1_control_matched_2k_seed42", {20: ctrl, 25: ctrl, 30: ctrl})
        _run(tmp_path, f"A4b_rho0.25_feasibility_seed{seed}_s1b_bus_2k_seed42", seed, "s1b_bus_2k_seed42",
             {20: a4b, 25: a4b, 30: a4b})
    table = summarise([tmp_path], "AP50_bus")
    assert table["A0"]["n"] == 3 and table["A0"]["mean"] == pytest.approx(0.23)
    assert table["A0@control"]["mean"] == pytest.approx(0.28)
    h1 = paired_effect(table, "A0", "A0@control")
    assert h1["n"] == 3 and h1["mean"] == pytest.approx(-0.05)
    assert h1["per_seed"] == {42: pytest.approx(-0.05), 123: pytest.approx(-0.03), 2024: pytest.approx(-0.07)}
    m = paired_effect(table, "A4b_rho0.25", "A0")
    assert m["mean"] == pytest.approx(-0.01) and m["std"] > 0
