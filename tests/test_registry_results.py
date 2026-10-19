"""Reading per-run results out of the experiment registry for the paper figures."""
from __future__ import annotations

from pathlib import Path

import yaml

from src.evaluation.registry_results import gap_by_seed, load_runs, select

REG = {"experiments": [
    {"id": "EXP-1-S12-A0c-y11s", "method": "A0c: YOLO11s, COCO head kept, FedAvg (control)",
     "scenario": "S1b5-3k, YOLO11s, 20 rounds", "status": "completed", "seed": 42,
     "results": {"round_10": {"AP50_bus": 0.30, "mAP50": 0.40}, "round_20": {"AP50_bus": 0.40, "mAP50": 0.50}}},
    {"id": "EXP-1-S13-A0c-y11s-pooled", "method": "A0c", "scenario": "pooled re-split of S1b5-3k, YOLO11s",
     "status": "completed", "seed": 42, "results": {"rule_mean_r10_20": {"AP50_bus": 0.41, "mAP50": 0.46}}},
    {"id": "EXP-2-S18-A0c-s5bus1-y11s-seed42", "method": "A0c", "scenario": "S1b5-bus1-3k (bus on 1/5), YOLO11s",
     "status": "completed", "seed": 42, "results": {"rule_mean_r10_20": {"AP50_bus": 0.33}}},
    {"id": "EXP-3-S15-P5f-y11s-s5", "method": "P5f (P4noKD + frozen)", "scenario": "S1b5-3k, YOLO11s",
     "status": "completed", "seed": 42, "results": {"rule_mean_r10_20": {"AP50_bus": 0.49}}},
    {"id": "EXP-4-S13-A0c-y11s-seed123", "method": "A0c", "scenario": "S1b5-3k, YOLO11s",
     "status": "failed-technical", "seed": 123},
    {"id": "EXP-5-S5-A0", "method": "A0", "scenario": "S1b-2k", "status": "completed", "seed": 42,
     "results": {"rule_mean_r20_30": {"AP50_bus": 0.23}}},
    {"id": "EXP-6-S18-FedProxc-s5pooled-y11s-seed42", "method": "FedProxc", "scenario": "S1b5-pooled-3k (IID), YOLO11s",
     "status": "running", "seed": 42},
]}


def _reg(tmp_path: Path) -> Path:
    p = tmp_path / "registry.yaml"
    p.write_text(yaml.safe_dump(REG), encoding="utf-8")
    return p


def test_only_completed_yolo11s_runs_with_metrics_are_loaded(tmp_path):
    runs = load_runs(_reg(tmp_path))
    assert sorted((r.method, r.scenario, r.seed) for r in runs) == [
        ("A0c", "bus1", 42), ("A0c", "pooled", 42), ("A0c", "s5", 42), ("P5f", "s5", 42)]


def test_round_10_20_mean_is_used_when_no_rule_mean(tmp_path):
    r = select(load_runs(_reg(tmp_path)), method="A0c", scenario="s5")[0]
    assert abs(r.metrics["AP50_bus"] - 0.35) < 1e-9 and abs(r.metrics["mAP50"] - 0.45) < 1e-9


def test_gap_is_pooled_minus_scenario_paired_by_seed(tmp_path):
    runs = load_runs(_reg(tmp_path))
    assert {s: round(v, 4) for s, v in gap_by_seed(runs, "A0c", "s5").items()} == {42: 0.06}
    assert {s: round(v, 4) for s, v in gap_by_seed(runs, "A0c", "bus1").items()} == {42: 0.08}
    assert gap_by_seed(runs, "P5f", "s5") == {}                      # no pooled partner -> no gap
