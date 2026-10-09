"""Tests for the ablation-arm wiring in scripts/run_fl_experiment.

These guard the experiment matrix itself: before this wiring existed, only A0
(FedAvg) had an execution path, so A1/A2a/A2b/A3/A4b could not be run at all.
Also covers the per-(experiment, round, client) training seed, without which the
three experiment seeds never reached local training (CLAUDE.md §14).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_fl_experiment import (  # noqa: E402
    _ABLATIONS,
    _MECHANISMS,
    _STRATEGIES,
    _pick_client_class,
    _resolve_arms,
    exp_id_for_arm,
)
from src.federated.client import YOLOFlowerClient, derive_local_seed  # noqa: E402
from src.federated.client_variants import (  # noqa: E402
    EFLClient,
    FocalClient,
    NTDClient,
    TeacherKDClient,
    TeacherKDRhoClient,
    FedNovaClient,
    LossPreservationClient,
    PreservationClient,
    ScaffoldClient,
)


def _args(**kw) -> argparse.Namespace:
    base = dict(algorithm=None, ablation=None, client_mechanism=None, rho=None)
    base.update(kw)
    return argparse.Namespace(**base)


# ── ablation presets ──────────────────────────────────────────────────────
def test_every_plan_arm_has_a_preset():
    assert set(_ABLATIONS) == {"A0", "A1", "A2a", "A2b", "A3", "A4a", "A4b", "A5", "A6", "A6c",
                               "B1", "B2", "P1", "P2", "A0c", "P3", "P3LR", "P4"}


def test_coco_head_arms():
    assert _ABLATIONS["A0c"] == {"algorithm": "FedAvg", "mechanism": "none", "coco_head": True}
    assert _ABLATIONS["P3"]["mechanism"] == "kd_teacher" and _ABLATIONS["P3"]["teacher_from_init"]
    algo, m, rho = _resolve_arms(_args(ablation="P3"))
    assert (algo, m, rho) == ("FedAvg", "kd_teacher", 1.0)
    assert exp_id_for_arm("P3", m, rho) == "P3"


@pytest.mark.parametrize("arm,mech,client,rho,exp_id", [
    ("B1", "none", YOLOFlowerClient, None, "B1"),
    ("B2", "none", YOLOFlowerClient, None, "B2"),
    ("P1", "kd_teacher", TeacherKDClient, None, "P1"),
    ("P2", "kd_teacher_rho", TeacherKDRhoClient, 0.25, "P2_rho0.25"),
])
def test_server_sample_arms(arm, mech, client, rho, exp_id):
    algo, m, r = _resolve_arms(_args(ablation=arm, rho=rho))
    assert (algo, m) == ("FedAvg", mech)
    assert _pick_client_class(algo, m) is client
    assert exp_id_for_arm(arm, m, r) == exp_id
    assert _ABLATIONS[arm]["teacher_init"] is True
    assert bool(_ABLATIONS[arm].get("server_ft")) == (arm == "B2")


def test_p2_requires_rho():
    with pytest.raises(SystemExit):
        _resolve_arms(_args(ablation="P2"))


def test_teacher_kd_clients_pass_teacher_and_weights(tmp_path, monkeypatch):
    import src.federated.client as base
    monkeypatch.setattr(base, "build_model", lambda w: None)
    cfg = {"local_epochs": 1, "batch_size": 16, "image_size": 640, "lr0": 0.01,
           "teacher_params": str(tmp_path / "teacher.npz")}
    p2 = TeacherKDRhoClient(client_id="C0", partition_id=0, data_yaml=tmp_path / "d.yaml", weights="w",
                            train_config=cfg, run_dir=tmp_path, missing_classes=["bus"], rho=0.25)
    spec = p2._train_kwargs()["cls_loss"]
    assert spec["teacher_path"] == cfg["teacher_params"]
    assert spec["class_weights"] == [1.0, 0.25, 1.0, 1.0]
    p1 = TeacherKDClient(client_id="C2", partition_id=2, data_yaml=tmp_path / "d.yaml", weights="w",
                         train_config={**cfg, "teacher_params": None}, run_dir=tmp_path)
    with pytest.raises(ValueError, match="teacher_params"):
        p1._train_kwargs()


@pytest.mark.parametrize("arm,mech,client", [("A5", "ntd", NTDClient), ("A6", "efl", EFLClient),
                                             ("A6c", "focal", FocalClient)])
def test_loss_variant_arms_need_no_rho(arm, mech, client):
    algo, m, rho = _resolve_arms(_args(ablation=arm))
    assert (algo, m, rho) == ("FedAvg", mech, 1.0)
    assert _pick_client_class(algo, m) is client
    assert exp_id_for_arm(arm, m, rho) == arm
    with pytest.raises(SystemExit):
        _resolve_arms(_args(ablation=arm, rho=0.5))


def test_rho_arm_ids_unchanged():
    assert exp_id_for_arm("A4b", "preservation_loss", 0.25) == "A4b_rho0.25"


def test_loss_variant_client_passes_spec(tmp_path, monkeypatch):
    import src.federated.client as base
    monkeypatch.setattr(base, "build_model", lambda w: None)
    c = EFLClient(client_id="C0", partition_id=0, data_yaml=tmp_path / "d.yaml", weights="w",
                  train_config={"local_epochs": 1, "batch_size": 16, "image_size": 640, "lr0": 0.01},
                  run_dir=tmp_path)
    spec = c._train_kwargs()["cls_loss"]
    assert spec["kind"] == "efl"
    assert Path(spec["state_path"]) == tmp_path / "clients" / "C0" / "efl_state.json"


@pytest.mark.parametrize("arm", sorted(_ABLATIONS))
def test_presets_reference_known_strategies_and_mechanisms(arm):
    preset = _ABLATIONS[arm]
    assert preset["algorithm"] in _STRATEGIES
    assert preset["mechanism"] in _MECHANISMS


def test_a0_resolves_to_plain_fedavg():
    algo, mech, rho = _resolve_arms(_args(ablation="A0"))
    assert (algo, mech, rho) == ("FedAvg", "none", 1.0)


def test_a1_resolves_to_class_count_control():
    algo, mech, _ = _resolve_arms(_args(ablation="A1"))
    assert algo == "ClassCountFedAvg"
    assert mech == "none"


def test_a3_is_server_only():
    algo, mech, rho = _resolve_arms(_args(ablation="A3"))
    assert (algo, mech, rho) == ("ClassAwareAgg", "none", 1.0)


def test_a4b_is_full_method():
    algo, mech, rho = _resolve_arms(_args(ablation="A4b", rho=0.25))
    assert (algo, mech, rho) == ("ClassAwareAgg", "preservation_loss", 0.25)


def test_mechanism_requires_explicit_rho():
    """No literature-supported default exists for rho (plan.md §7.3), so a run
    must state it rather than inherit a silent value."""
    with pytest.raises(SystemExit, match="--rho is required"):
        _resolve_arms(_args(ablation="A2b"))


def test_mechanism_none_forces_rho_one():
    _, mech, rho = _resolve_arms(_args(ablation="A3", rho=0.0))
    assert mech == "none" and rho == 1.0


def test_algorithm_without_ablation_defaults_to_no_mechanism():
    algo, mech, rho = _resolve_arms(_args(algorithm="FedNova"))
    assert (algo, mech, rho) == ("FedNova", "none", 1.0)


def test_missing_both_algorithm_and_ablation_errors():
    with pytest.raises(SystemExit, match="Specify --algorithm or --ablation"):
        _resolve_arms(_args())


def test_scaffold_plus_preservation_is_rejected():
    """SCAFFOLD/FedNova clients override fit(), so preservation would silently
    not be applied — refuse instead of producing a mislabelled run (§22)."""
    with pytest.raises(SystemExit, match="not supported"):
        _resolve_arms(_args(algorithm="SCAFFOLD",
                            client_mechanism="preservation_param", rho=0.0))


def test_explicit_flag_overrides_preset():
    algo, mech, _ = _resolve_arms(
        _args(ablation="A0", algorithm="ClassAwareAgg")
    )
    assert algo == "ClassAwareAgg"
    assert mech == "none"


# ── client class selection ────────────────────────────────────────────────
@pytest.mark.parametrize("algorithm,mechanism,expected", [
    ("FedAvg", "none", YOLOFlowerClient),
    ("ClassAwareAgg", "none", YOLOFlowerClient),
    ("ClassCountFedAvg", "none", YOLOFlowerClient),
    ("SCAFFOLD", "none", ScaffoldClient),
    ("FedNova", "none", FedNovaClient),
    ("FedAvg", "preservation_param", PreservationClient),
    ("ClassAwareAgg", "preservation_param", PreservationClient),
    ("FedAvg", "preservation_loss", LossPreservationClient),
    ("ClassAwareAgg", "preservation_loss", LossPreservationClient),
])
def test_client_class_selection(algorithm, mechanism, expected):
    assert _pick_client_class(algorithm, mechanism) is expected


def test_loss_preservation_client_weights_missing_classes(monkeypatch):
    """A2b (ADR-006): the BCE of each locally-missing class is scaled by rho.
    Prerequisites now met: F1 runtime on the pinned stack + v8DetectionLoss.bce
    verified as BCEWithLogitsLoss(reduction='none') over [B, anchors, nc]."""
    from src.federated.client import YOLOFlowerClient

    monkeypatch.setattr(YOLOFlowerClient, "_train_kwargs", lambda self: {"epochs": 1})
    client = LossPreservationClient.__new__(LossPreservationClient)
    client.missing_classes, client.rho = ["bus"], 0.25
    kw = client._train_kwargs()
    assert kw == {"epochs": 1, "cls_loss_weights": [1.0, 0.25, 1.0, 1.0]}


# ── per-client/round training seed (B3) ───────────────────────────────────
def test_derive_local_seed_differs_across_rounds_and_clients():
    seeds = {
        derive_local_seed(42, rnd, cid)
        for rnd in range(1, 11)
        for cid in range(4)
    }
    assert len(seeds) == 40


def test_derive_local_seed_differs_across_experiment_seeds():
    a = derive_local_seed(42, 1, 0)
    b = derive_local_seed(123, 1, 0)
    c = derive_local_seed(2024, 1, 0)
    assert len({a, b, c}) == 3


def test_derive_local_seed_is_deterministic_and_in_int32_range():
    for base in (42, 123, 2024):
        for rnd in (1, 10):
            for cid in range(4):
                s = derive_local_seed(base, rnd, cid)
                assert s == derive_local_seed(base, rnd, cid)
                assert 0 <= s < 2 ** 31


# ── absolute round numbering (resume correctness) ─────────────────────────
class _Stub(YOLOFlowerClient):
    """Bypass __init__ so no YOLO model is built (no ultralytics needed)."""

    def __init__(self) -> None:  # noqa: D107
        self._round = 0


def test_client_takes_absolute_round_from_server_config():
    c = _Stub()
    assert c._advance_round({"server_round": 7}) == 7
    assert c._advance_round({"server_round": 8}) == 8


def test_client_falls_back_to_local_counter_without_server_round():
    c = _Stub()
    assert c._advance_round({}) == 1
    assert c._advance_round({}) == 2


def test_absolute_round_prevents_seed_replay_after_resume():
    """A resumed run must not reuse the training seeds of rounds 1..k: the client
    counter restarts at 1 inside a new start_simulation call, so the round number
    has to come from the server."""
    fresh = [derive_local_seed(42, r, 0) for r in range(1, 6)]
    resumed_from_3 = [
        derive_local_seed(42, _Stub()._advance_round({"server_round": r}), 0)
        for r in (4, 5)
    ]
    assert resumed_from_3 == fresh[3:]
