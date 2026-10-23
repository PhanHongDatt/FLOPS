"""ADR-023: motorcycle arms — P4t1 preset, the all-motorcycle partition config and the 960 px config."""
from pathlib import Path

import yaml

from scripts.run_fl_experiment import _ABLATIONS
from src.utils.config import load_experiment_config

ROOT = Path(__file__).resolve().parents[1]


def test_p4t1_is_p4_with_threshold_one():
    p4, p4t1 = _ABLATIONS["P4"], _ABLATIONS["P4t1"]
    assert p4t1["rfs_t"] == 1.0
    assert {k: v for k, v in p4t1.items() if k != "rfs_t"} == {k: v for k, v in p4.items() if k != "rfs_t"}


def test_allmoto_partition_differs_from_s1b5_only_in_the_quota():
    ref = yaml.safe_load((ROOT / "configs/partition/s1b5_bus_3k_moto_seed42.yaml").read_text(encoding="utf-8"))
    new = yaml.safe_load((ROOT / "configs/partition/s1b5_bus_3k_allmoto_seed42.yaml").read_text(encoding="utf-8"))
    assert new["class_quota"] == {"motorcycle": 460}
    assert new["partition_id"] == "s1b5_bus_3k_allmoto_seed42"
    for key in ("scenario", "seed", "num_clients", "per_client", "missing_map", "rare_rule_id", "rare_threshold"):
        assert new[key] == ref[key], key


def test_960_config_changes_only_the_image_size():
    ref = load_experiment_config(ROOT / "configs/experiments/convergence_5c_y11s_20r.yaml")
    new = load_experiment_config(ROOT / "configs/experiments/convergence_5c_y11s_20r_960.yaml")
    assert ref["model"]["image_size"] == 640 and new["model"]["image_size"] == 960
    ref["model"]["image_size"] = new["model"]["image_size"]
    ref.pop("notes", None), new.pop("notes", None)
    assert ref == new


def test_nine_client_partition_contract():
    cfg = yaml.safe_load((ROOT / "configs/partition/s9_bus8_3k_moto_seed42.yaml").read_text(encoding="utf-8"))
    assert cfg["num_clients"] == 9 and cfg["per_client"] == 3000 and set(cfg["missing_map"]) == {f"C{i}" for i in range(9)}
    assert [c for c, m in cfg["missing_map"].items() if "bus" not in m] == ["C8"]
    assert sorted(c for c, m in cfg["missing_map"].items() if "truck" in m) == ["C5", "C6", "C7"]
    exp = load_experiment_config(ROOT / "configs/experiments/convergence_9c_y11s_20r_960.yaml")
    assert exp["federated"]["num_clients"] == 9 and exp["model"]["image_size"] == 960


def test_release_training_memory_keeps_the_model():
    from types import SimpleNamespace
    from src.model.yolo_wrapper import release_training_memory
    trainer, weights = object(), object()
    m = SimpleNamespace(trainer=trainer, model=weights)
    release_training_memory(m)
    assert m.trainer is trainer and m.model is weights


def test_partial_aggregation_is_refused():
    import pytest
    from types import SimpleNamespace
    from src.federated.server import _wrap_fail_on_client_failures
    strategy = SimpleNamespace(aggregate_fit=lambda r, res, fail: ("params", {}))
    _wrap_fail_on_client_failures(strategy)
    assert strategy.aggregate_fit(1, ["a"] * 5, []) == ("params", {})
    with pytest.raises(RuntimeError, match="2 client fit"):
        strategy.aggregate_fit(3, ["a"] * 3, [RuntimeError("CUDA out of memory"), RuntimeError("x")])
