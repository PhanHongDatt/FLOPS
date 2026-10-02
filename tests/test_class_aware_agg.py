"""Tests for H3 class-aware aggregation (A3) and the A1 class-count control.

Mandated by CLAUDE.md §18 (Aggregation): 0 / 1 / many eligible clients, weight
normalisation, no-contributor behaviour, all-classes-present compatibility.

These run on numpy + flwr only — no torch, no ultralytics, no GPU.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import yaml
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays

from src.data.bdd100k import TARGET_CLASSES
from src.federated.strategies.class_aware_agg import (
    ClassAwareAggregation,
    build_class_aware_agg,
    build_class_count_fedavg,
    find_cls_head_indices,
)

NC = len(TARGET_CLASSES)
CAR, BUS, TRUCK, MOTO = TARGET_CLASSES

# A tiny stand-in for a YOLO state_dict: one shared param + one class head
# (single detection scale is enough for the aggregation logic).
PARAM_NAMES = [
    "model.0.conv.weight",          # shared
    "model.22.cv3.0.2.weight",      # class head (dim0 == nc)
    "model.22.cv3.0.2.bias",        # class head (dim0 == nc)
]


def _params(shared_val: float, head_rows: list[float]) -> list[np.ndarray]:
    assert len(head_rows) == NC
    return [
        np.full((2, 2), shared_val, dtype=np.float32),
        np.array([[r] for r in head_rows], dtype=np.float32),   # (nc, 1)
        np.array(head_rows, dtype=np.float32),                  # (nc,)
    ]


class _Proxy:
    def __init__(self, cid: str) -> None:
        self.cid = cid


class _ClientManager:
    """Minimal ClientManager so FedAvg.configure_fit can run in a unit test."""

    def __init__(self, n: int = 0) -> None:
        self._n = n

    def num_available(self) -> int:
        return self._n

    def sample(self, num_clients, min_num_clients=None, criterion=None):
        return []


def _fitres(params: list[np.ndarray], n: int, counts: dict[str, int] | None) -> FitRes:
    metrics: dict = {}
    if counts is not None:
        metrics["class_counts_json"] = json.dumps(counts)
    return FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=ndarrays_to_parameters(params),
        num_examples=n,
        metrics=metrics,
    )


def _results(specs):
    """specs: list of (shared_val, head_rows, n_examples, counts)."""
    return [
        (_Proxy(f"C{i}"), _fitres(_params(sv, rows), n, counts))
        for i, (sv, rows, n, counts) in enumerate(specs)
    ]


def _run(strategy, global_params, specs, server_round=1):
    strategy.configure_fit(
        server_round, ndarrays_to_parameters(global_params), _ClientManager()
    )
    out, _ = strategy.aggregate_fit(server_round, _results(specs), [])
    return parameters_to_ndarrays(out)


def _make(no_contributor_rule=True, **kw):
    factory = build_class_aware_agg if no_contributor_rule else build_class_count_fedavg
    return factory(num_rounds=1, param_names=PARAM_NAMES, **kw)


# ── parameter-map guard ───────────────────────────────────────────────────
def test_find_cls_head_indices_matches_expected_keys():
    shapes = [(2, 2), (NC, 1), (NC,)]
    assert find_cls_head_indices(PARAM_NAMES, shapes) == [1, 2]


def test_find_cls_head_indices_raises_when_nothing_matches():
    # Simulates Ultralytics renaming / moving the Detect layer: a silent no-op
    # here would make class-aware aggregation secretly identical to FedAvg.
    names = ["model.0.conv.weight", "model.99.foo.0.2.weight"]
    with pytest.raises(ValueError, match="No class-specific parameters matched"):
        find_cls_head_indices(names, [(2, 2), (NC, 1)])


def test_find_cls_head_indices_rejects_wrong_dim0():
    names = ["model.22.cv3.0.2.weight"]
    with pytest.raises(ValueError):
        find_cls_head_indices(names, [(NC + 7, 1)])


# ── builder signature (B2 regression) ─────────────────────────────────────
def test_builders_consume_num_rounds_without_forwarding_to_fedavg():
    # flwr's FedAvg has neither a num_rounds parameter nor **kwargs; forwarding
    # it raised TypeError and made this strategy unconstructible.
    for factory in (build_class_aware_agg, build_class_count_fedavg):
        s = factory(num_rounds=10, param_names=PARAM_NAMES)
        assert isinstance(s, ClassAwareAggregation)


# ── eligibility: 0 / 1 / many ─────────────────────────────────────────────
def test_no_contributor_keeps_exact_global_row():
    """S_c empty → theta_c^{t+1} = theta_c^t (CLAUDE.md §12)."""
    g = _params(0.0, [10.0, 20.0, 30.0, 40.0])
    # Nobody holds bus; everyone holds the rest.
    counts = {CAR: 5, BUS: 0, TRUCK: 5, MOTO: 5}
    specs = [(1.0, [1.0, 1.0, 1.0, 1.0], 10, counts),
             (1.0, [2.0, 2.0, 2.0, 2.0], 10, counts)]
    out = _run(_make(), g, specs)
    # bus row preserved bit-for-bit from the global model...
    assert out[1][TARGET_CLASSES.index(BUS)] == pytest.approx(g[1][TARGET_CLASSES.index(BUS)])
    assert out[2][TARGET_CLASSES.index(BUS)] == pytest.approx(g[2][TARGET_CLASSES.index(BUS)])
    # ...while a class with contributors is updated away from global.
    assert out[2][TARGET_CLASSES.index(CAR)] != pytest.approx(g[2][TARGET_CLASSES.index(CAR)])


def test_single_eligible_client_row_equals_that_client():
    g = _params(0.0, [10.0, 20.0, 30.0, 40.0])
    only_bus = {CAR: 5, BUS: 7, TRUCK: 5, MOTO: 5}
    no_bus = {CAR: 5, BUS: 0, TRUCK: 5, MOTO: 5}
    # head_rows is ordered by TARGET_CLASSES, so index 1 is the bus row.
    specs = [(1.0, [1.0, 9.0, 1.0, 1.0], 10, only_bus),
             (1.0, [2.0, 2.0, 2.0, 2.0], 90, no_bus)]
    out = _run(_make(), g, specs)
    bus = TARGET_CLASSES.index(BUS)
    # Client 0 is the sole contributor, so its row wins outright despite
    # client 1 holding 9x the data.
    assert out[2][bus] == pytest.approx(9.0)


def test_many_eligible_weights_are_class_count_normalised():
    g = _params(0.0, [0.0, 0.0, 0.0, 0.0])
    specs = [
        (1.0, [0.0, 10.0, 0.0, 0.0], 50, {CAR: 1, BUS: 30, TRUCK: 1, MOTO: 1}),
        (1.0, [0.0, 20.0, 0.0, 0.0], 50, {CAR: 1, BUS: 10, TRUCK: 1, MOTO: 1}),
    ]
    out = _run(_make(), g, specs)
    bus = TARGET_CLASSES.index(BUS)
    # (30/40)*10 + (10/40)*20 = 12.5 — weighted by BOX count, not image count.
    assert out[2][bus] == pytest.approx(12.5)


def test_eligibility_threshold_excludes_rare_contributor():
    g = _params(0.0, [0.0, 0.0, 0.0, 0.0])
    specs = [
        (1.0, [0.0, 10.0, 0.0, 0.0], 50, {CAR: 1, BUS: 100, TRUCK: 1, MOTO: 1}),
        (1.0, [0.0, 99.0, 0.0, 0.0], 50, {CAR: 1, BUS: 5, TRUCK: 1, MOTO: 1}),
    ]
    out = _run(_make(eligibility_threshold=50), g, specs)
    bus = TARGET_CLASSES.index(BUS)
    assert out[2][bus] == pytest.approx(10.0)   # the 5-box client is not eligible


# ── shared params stay FedAvg ─────────────────────────────────────────────
def test_shared_params_use_num_examples_weighted_fedavg():
    g = _params(0.0, [0.0, 0.0, 0.0, 0.0])
    counts = {c: 5 for c in TARGET_CLASSES}
    specs = [(1.0, [1.0] * NC, 25, counts), (5.0, [1.0] * NC, 75, counts)]
    out = _run(_make(), g, specs)
    # 0.25*1 + 0.75*5 = 4.0 — image-weighted, untouched by class logic.
    assert out[0] == pytest.approx(np.full((2, 2), 4.0))


# ── A1 vs A3 (plan.md §6.4) ───────────────────────────────────────────────
def test_a1_equals_a3_when_every_class_has_a_contributor():
    """Encodes the §6.4 finding: with tau_elig=1 and full participation the
    class-count baseline is numerically IDENTICAL to class-aware aggregation.
    If this test ever fails, the two arms have diverged and the novelty argument
    in plan.md §6.4 must be revisited."""
    g = _params(0.0, [10.0, 20.0, 30.0, 40.0])
    specs = [
        (1.0, [1.0, 2.0, 3.0, 4.0], 40, {CAR: 9, BUS: 3, TRUCK: 4, MOTO: 1}),
        (2.0, [5.0, 6.0, 7.0, 8.0], 60, {CAR: 8, BUS: 1, TRUCK: 2, MOTO: 6}),
    ]
    a3 = _run(_make(no_contributor_rule=True), g, specs)
    a1 = _run(_make(no_contributor_rule=False), g, specs)
    for x, y in zip(a3, a1):
        assert x == pytest.approx(y)


def test_a1_differs_from_a3_exactly_in_the_no_contributor_case():
    g = _params(0.0, [10.0, 20.0, 30.0, 40.0])
    counts = {CAR: 5, BUS: 0, TRUCK: 5, MOTO: 5}      # nobody holds bus
    specs = [(1.0, [1.0] * NC, 50, counts), (1.0, [3.0] * NC, 50, counts)]
    a3 = _run(_make(no_contributor_rule=True), g, specs)
    a1 = _run(_make(no_contributor_rule=False), g, specs)
    bus = TARGET_CLASSES.index(BUS)
    assert a3[2][bus] == pytest.approx(g[2][bus])                      # kept global
    assert a1[2][bus] == pytest.approx(2.0)                            # FedAvg fallback
    # Classes with contributors agree between the two arms.
    assert a3[2][TARGET_CLASSES.index(CAR)] == pytest.approx(a1[2][TARGET_CLASSES.index(CAR)])


# ── missing metadata must fail loudly ─────────────────────────────────────
def test_missing_class_counts_raises_instead_of_silently_eligible():
    g = _params(0.0, [0.0] * NC)
    specs = [(1.0, [1.0] * NC, 10, None), (1.0, [2.0] * NC, 10, None)]
    with pytest.raises(ValueError, match="class_counts_json"):
        _run(_make(), g, specs)


def test_missing_class_counts_tolerated_when_explicitly_allowed():
    g = _params(0.0, [0.0] * NC)
    specs = [(1.0, [1.0] * NC, 10, None)]
    out = _run(_make(require_class_counts=False), g, specs)
    # No counts → nobody eligible → every class row keeps global.
    assert out[2] == pytest.approx(g[2])


# ── trace ─────────────────────────────────────────────────────────────────
def test_aggregation_trace_is_written_and_records_keep_global(tmp_path):
    g = _params(0.0, [10.0, 20.0, 30.0, 40.0])
    counts = {CAR: 5, BUS: 0, TRUCK: 5, MOTO: 5}
    specs = [(1.0, [1.0] * NC, 10, counts)]
    _run(_make(trace_dir=tmp_path), g, specs)

    trace_path = tmp_path / "aggregation_trace.yaml"
    assert trace_path.exists()
    trace = yaml.safe_load(trace_path.read_text(encoding="utf-8"))
    bus_entries = [e for e in trace if e["class"] == BUS]
    assert bus_entries and all(e["action"] == "keep_global" for e in bus_entries)
    assert all(e["reason"] == "no_valid_contributor" for e in bus_entries)
    car_entries = [e for e in trace if e["class"] == CAR]
    assert all(e["action"] == "class_count_weighted" for e in car_entries)


def test_aggregate_without_configure_fit_refuses_to_guess_global():
    s = _make()
    specs = [(1.0, [1.0] * NC, 10, {c: 5 for c in TARGET_CLASSES})]
    with pytest.raises(RuntimeError, match="theta\\^t unavailable"):
        s.aggregate_fit(1, _results(specs), [])
