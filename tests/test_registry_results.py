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


def test_operating_metrics_are_merged_and_rule_metrics_win(tmp_path):
    reg = {"experiments": [
        {"id": "EXP-7-S18-A0c-s5bus1-y11s-seed123", "method": "A0c", "scenario": "S1b5-bus1-3k, YOLO11s",
         "status": "completed", "seed": 123,
         "results": {"rule_mean_r10_20": {"AP50_bus": 0.36, "FN_bus": 1522.0},
                     "operating_r10_20": {"recall_bus": 0.064, "FN_bus": 9.0}}}]}
    p = tmp_path / "registry.yaml"
    p.write_text(yaml.safe_dump(reg), encoding="utf-8")
    m = load_runs(p)[0].metrics
    assert m["recall_bus"] == 0.064 and m["FN_bus"] == 1522.0 and m["AP50_bus"] == 0.36


def test_backfill_inserts_one_line_at_the_end_of_results():
    import importlib.util
    spec = importlib.util.spec_from_file_location("bf", Path(__file__).parents[1] / "scripts" / "backfill_operating_metrics.py")
    bf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bf)
    text = ("experiments:\n- id: EXP-A\n  status: completed\n  results:\n    rule_mean_r10_20:\n      AP50_bus: 0.38\n"
            "  interpretation: x\n- id: EXP-B\n  results:\n    rule_mean_r10_20: {AP50_bus: 0.4}\n")
    vals = {k: 1.0 for k in bf.KEYS}
    out = bf.insert_block(bf.insert_block(text, "EXP-A", vals), "EXP-A", vals)     # idempotent
    a = yaml.safe_load(out)["experiments"][0]
    assert out.count("operating_r10_20") == 1 and a["interpretation"] == "x"
    assert a["results"]["operating_r10_20"]["recall_at025_bus"] == 1.0 and a["results"]["rule_mean_r10_20"]["AP50_bus"] == 0.38


def test_export_csv_has_runs_means_pending_and_notes(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("ex", Path(__file__).parents[1] / "scripts" / "export_results_csv.py")
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    reg = _reg(tmp_path)
    rows = ex.build_rows(load_runs(reg), reg)
    kinds = [r[0] for r in rows]
    assert kinds.count("run") == 4 and "mean" in kinds and kinds.count("pending") == 1 and kinds.count("excluded") == 1
    assert all(len(r) == len(ex.HEADER) for r in rows)
    bus1 = next(r for r in rows if r[0] == "run" and r[4] == "bus1")
    assert bus1[ex.HEADER.index("gap_AP50_bus_vs_pooled")] == "0.0800" and "ADR-022" in bus1[-1]


def test_backfill_recall_at_025_is_tp_over_tp_plus_fn(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("bf2", Path(__file__).parents[1] / "scripts" / "backfill_operating_metrics.py")
    bf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bf)
    cols = ["round", "AP50_bus", *bf.COUNTS, *(f"{k}_{c}" for k in ("recall", "precision") for c in bf.CLASSES)]
    rows = []
    for rnd, tp, fp, fn in ((10, 100, 300, 900), (20, 300, 100, 700)):
        r = {c: "0.5" for c in cols} | {"round": str(rnd), "AP50_bus": "0.4",
                                        "TP_bus": str(tp), "FP_bus": str(fp), "FN_bus": str(fn)}
        rows.append(r)
    p = tmp_path / "round_metrics.csv"
    p.write_text(",".join(cols) + "\n" + "\n".join(",".join(r[c] for c in cols) for r in rows) + "\n", encoding="utf-8")
    v = bf.rule_mean(p)
    assert abs(v["recall_at025_bus"] - (0.1 + 0.3) / 2) < 1e-12           # per-round ratio, then the rule mean
    assert abs(v["precision_at025_bus"] - (0.25 + 0.75) / 2) < 1e-12
    assert v["recall_maxF1_bus"] == 0.5                                   # Ultralytics max-F1 value kept apart
