"""Tests for cross-run comparison (src/evaluation/compare.py).

The guard rails matter as much as the arithmetic: CLAUDE.md §20 forbids comparing
methods across partitions, dropping a seed for one arm only, and reporting the
best round per method. Those are enforced in `check_comparable`, so they are
tested as behaviour, not left as documentation.

Pure stdlib/numpy — no GPU, torch or ultralytics.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from src.evaluation.compare import (
    aggregate_by_arm,
    check_comparable,
    delta_vs_baseline,
    discover_runs,
    format_table,
    load_run,
    metric_keys,
    min_seeds_warning,
    scenario_delta,
    write_per_run_csv,
    write_summary_csv,
)

BASE_PROTOCOL = {
    "model": {"image_size": 640, "weights": "yolov8n.pt"},
    "train": {"batch_size": 16, "lr0": 0.01},
    "federated": {"num_rounds": 10, "local_epochs": 1, "num_clients": 4,
                  "fraction_fit": 1.0, "fraction_evaluate": 0.0},
    "evaluation": {"conf": 0.25, "iou": 0.70},
}


def _make_run(
    root: Path,
    arm: str,
    seed: int,
    partition_id: str = "s1b_bus_seed42",
    run_class: str = "feasibility",
    metrics: dict[str, float] | None = None,
    rounds: int = 10,
    config_override: dict | None = None,
) -> Path:
    run_dir = root / f"{arm}_{run_class}_seed{seed}_{partition_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "environment.json").write_text(json.dumps({
        "exp_id": arm, "seed": seed, "partition_id": partition_id,
        "run_class": run_class, "git_commit": "abc123",
    }), encoding="utf-8")

    config = {k: dict(v) for k, v in BASE_PROTOCOL.items()}
    for section, values in (config_override or {}).items():
        config.setdefault(section, {}).update(values)
    (run_dir / "config.yaml").write_text(yaml.dump(config), encoding="utf-8")

    metrics = metrics or {"mAP50": 0.5, "AP50_bus": 0.4}
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value"])
        w.writeheader()
        for k, v in metrics.items():
            w.writerow({"metric": k, "value": v})

    with (run_dir / "round_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["round", "mAP50", "AP50_bus"])
        w.writeheader()
        for r in range(rounds + 1):
            frac = r / max(rounds, 1)
            w.writerow({
                "round": r,
                "mAP50": metrics.get("mAP50", 0.5) * frac,
                "AP50_bus": metrics.get("AP50_bus", 0.4) * frac,
            })
    return run_dir


# ── loading ───────────────────────────────────────────────────────────────
def test_load_run_reads_arm_seed_partition_and_metrics(tmp_path):
    rd = _make_run(tmp_path, "A0", 42, metrics={"mAP50": 0.6, "AP50_bus": 0.3})
    rec = load_run(rd)
    assert rec.arm == "A0"
    assert rec.seed == 42
    assert rec.partition_id == "s1b_bus_seed42"
    assert rec.final_metrics["mAP50"] == pytest.approx(0.6)
    assert rec.final_round == 10
    assert rec.protocol["num_rounds"] == 10
    assert rec.protocol["image_size"] == 640


def test_load_run_rejects_non_run_directory(tmp_path):
    (tmp_path / "junk").mkdir()
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path / "junk")


def test_discover_runs_skips_non_run_directories(tmp_path):
    _make_run(tmp_path, "A0", 42)
    _make_run(tmp_path, "A1", 42)
    (tmp_path / "not_a_run").mkdir()
    assert len(discover_runs(tmp_path)) == 2


# ── guard rails (§14 / §20) ───────────────────────────────────────────────
def test_comparable_when_everything_matches(tmp_path):
    recs = [load_run(_make_run(tmp_path, arm, seed))
            for arm in ("A0", "A3") for seed in (42, 123, 2024)]
    assert check_comparable(recs) == []


def test_refuses_runs_from_different_partitions(tmp_path):
    recs = [
        load_run(_make_run(tmp_path, "A0", 42, partition_id="s1b_bus_seed42")),
        load_run(_make_run(tmp_path, "A3", 42, partition_id="s0_iid_seed42")),
    ]
    problems = check_comparable(recs)
    assert any("multiple partitions" in p for p in problems)


def test_refuses_mismatched_seed_sets(tmp_path):
    recs = [load_run(_make_run(tmp_path, "A0", s)) for s in (42, 123, 2024)]
    recs += [load_run(_make_run(tmp_path, "A3", s)) for s in (42, 123)]
    problems = check_comparable(recs)
    assert any("same seed set" in p for p in problems)


def test_refuses_protocol_mismatch(tmp_path):
    recs = [
        load_run(_make_run(tmp_path, "A0", 42)),
        load_run(_make_run(
            tmp_path, "A3", 42,
            config_override={"federated": {"num_rounds": 20}}, rounds=20,
        )),
    ]
    problems = check_comparable(recs)
    assert any("num_rounds" in p for p in problems)


def test_refuses_image_size_mismatch(tmp_path):
    recs = [
        load_run(_make_run(tmp_path, "A0", 42)),
        load_run(_make_run(tmp_path, "A3", 42,
                           config_override={"model": {"image_size": 512}})),
    ]
    assert any("image_size" in p for p in check_comparable(recs))


def test_smoke_runs_are_not_reportable(tmp_path):
    recs = [load_run(_make_run(tmp_path, arm, 42, run_class="smoke"))
            for arm in ("A0", "A3")]
    assert any("smoke" in p for p in check_comparable(recs))


def test_refuses_mixed_run_classes(tmp_path):
    recs = [
        load_run(_make_run(tmp_path, "A0", 42, run_class="feasibility")),
        load_run(_make_run(tmp_path, "A3", 42, run_class="main")),
    ]
    assert any("mix run classes" in p for p in check_comparable(recs))


def test_empty_input_is_not_comparable():
    assert check_comparable([]) == ["no runs given"]


def test_min_seeds_warning_lists_underpowered_arms(tmp_path):
    recs = [load_run(_make_run(tmp_path, "A0", s)) for s in (42, 123, 2024)]
    recs += [load_run(_make_run(tmp_path, "A3", 42))]
    warnings = min_seeds_warning(recs, min_seeds=3)
    assert len(warnings) == 1 and "A3" in warnings[0]


# ── aggregation ───────────────────────────────────────────────────────────
def test_aggregate_mean_and_sample_std(tmp_path):
    for seed, ap in ((42, 0.30), (123, 0.40), (2024, 0.50)):
        _make_run(tmp_path, "A0", seed, metrics={"mAP50": 0.5, "AP50_bus": ap})
    agg = aggregate_by_arm(discover_runs(tmp_path))
    mean, std, n = agg["A0"]["AP50_bus"]
    assert mean == pytest.approx(0.40)
    assert std == pytest.approx(0.1)        # sample std of 0.3/0.4/0.5
    assert n == 3


def test_single_seed_std_is_zero(tmp_path):
    _make_run(tmp_path, "A0", 42, metrics={"AP50_bus": 0.4})
    agg = aggregate_by_arm(discover_runs(tmp_path))
    assert agg["A0"]["AP50_bus"] == (pytest.approx(0.4), 0.0, 1)


def test_metric_keys_order_puts_map_first_then_per_class(tmp_path):
    _make_run(tmp_path, "A0", 42, metrics={
        "AP50_bus": 0.1, "mAP50": 0.2, "FP_bus": 3.0, "mAP50-95": 0.15,
    })
    keys = metric_keys(discover_runs(tmp_path))
    assert keys[:2] == ["mAP50", "mAP50-95"]
    assert keys.index("AP50_bus") < keys.index("FP_bus")


def test_delta_vs_baseline(tmp_path):
    _make_run(tmp_path, "A0", 42, metrics={"AP50_bus": 0.30})
    _make_run(tmp_path, "A3", 42, metrics={"AP50_bus": 0.42})
    agg = aggregate_by_arm(discover_runs(tmp_path))
    deltas = delta_vs_baseline(agg, "A0")
    assert "A0" not in deltas
    assert deltas["A3"]["AP50_bus"] == pytest.approx(0.12)


def test_delta_requires_the_declared_baseline(tmp_path):
    _make_run(tmp_path, "A3", 42)
    agg = aggregate_by_arm(discover_runs(tmp_path))
    with pytest.raises(KeyError, match="A0"):
        delta_vs_baseline(agg, "A0")


def test_scenario_delta_pairs_by_seed(tmp_path):
    """The C1 comparison: one arm, two partitions, paired per seed."""
    for seed, s1, ctrl in ((42, 0.20, 0.30), (123, 0.25, 0.37)):
        _make_run(tmp_path, "A0", seed, partition_id="s1b_bus_seed42",
                  metrics={"AP50_bus": s1})
        _make_run(tmp_path, "A0", seed, partition_id="s1_control_seed42",
                  metrics={"AP50_bus": ctrl})
    d = scenario_delta(discover_runs(tmp_path), "A0",
                       "s1b_bus_seed42", "s1_control_seed42")
    mean, std, n = d["AP50_bus"]
    assert n == 2
    assert mean == pytest.approx(((0.20 - 0.30) + (0.25 - 0.37)) / 2)
    assert std > 0


def test_scenario_delta_skips_unpaired_seeds(tmp_path):
    _make_run(tmp_path, "A0", 42, partition_id="s1b_bus_seed42", metrics={"AP50_bus": 0.2})
    _make_run(tmp_path, "A0", 42, partition_id="s1_control_seed42", metrics={"AP50_bus": 0.3})
    _make_run(tmp_path, "A0", 123, partition_id="s1b_bus_seed42", metrics={"AP50_bus": 0.9})
    d = scenario_delta(discover_runs(tmp_path), "A0",
                       "s1b_bus_seed42", "s1_control_seed42")
    assert d["AP50_bus"][2] == 1          # only seed 42 is paired


# ── writers ───────────────────────────────────────────────────────────────
def test_write_per_run_csv_has_one_row_per_run(tmp_path):
    for arm in ("A0", "A3"):
        for seed in (42, 123):
            _make_run(tmp_path, arm, seed)
    out = tmp_path / "out" / "per_run.csv"
    write_per_run_csv(discover_runs(tmp_path), out)
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == 4
    assert {r["arm"] for r in rows} == {"A0", "A3"}
    assert all(r["final_round"] == "10" for r in rows)


def test_write_summary_csv_includes_mean_pm_std_and_delta(tmp_path):
    _make_run(tmp_path, "A0", 42, metrics={"AP50_bus": 0.3})
    _make_run(tmp_path, "A3", 42, metrics={"AP50_bus": 0.5})
    recs = discover_runs(tmp_path)
    agg = aggregate_by_arm(recs)
    out = tmp_path / "out" / "summary.csv"
    write_summary_csv(agg, out, delta_vs_baseline(agg, "A0"), "A0")
    rows = {(r["arm"], r["metric"]): r for r in csv.DictReader(out.open(encoding="utf-8"))}
    assert rows[("A3", "AP50_bus")]["delta_vs_baseline"] == "+0.2000"
    assert rows[("A0", "AP50_bus")]["mean_pm_std"].startswith("0.3000")


def test_format_table_marks_delta_against_baseline(tmp_path):
    _make_run(tmp_path, "A0", 42, metrics={"AP50_bus": 0.30})
    _make_run(tmp_path, "A3", 42, metrics={"AP50_bus": 0.42})
    agg = aggregate_by_arm(discover_runs(tmp_path))
    text = format_table(agg, ["AP50_bus"], baseline_arm="A0")
    assert "AP50_bus" in text and "+0.1200" in text
