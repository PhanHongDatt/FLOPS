"""ADR-031 fit tracking: per-client rows, bytes, timing, weighted round aggregates; never raises."""
from __future__ import annotations

import csv
from types import SimpleNamespace

from src.mlops.fit_tracking import FIT_METRICS_FILE, FitTracker, payload_bytes


def _params(*sizes):
    return SimpleNamespace(tensors=[b"x" * n for n in sizes])


def _res(n, sizes, **metrics):
    return SimpleNamespace(num_examples=n, parameters=_params(*sizes), metrics=metrics)


class _Strategy:
    def configure_fit(self, server_round, parameters, client_manager):
        return ["fit-ins"]

    def aggregate_fit(self, server_round, results, failures):
        return _params(7, 3), {"n": len(results)}


class _Clock:
    def __init__(self, *ticks):
        self.ticks = list(ticks)

    def __call__(self):
        return self.ticks.pop(0)


class _Recorder:
    def __init__(self):
        self.calls = []

    def log_metrics(self, metrics, step=None):
        self.calls.append((step, dict(metrics)))


def _rows(path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_payload_bytes():
    assert payload_bytes(_params(3, 4)) == 7
    assert payload_bytes(None) == 0


def test_rows_bytes_timing_and_weighted_aggregates(tmp_path):
    rec = _Recorder()
    strategy = FitTracker(tmp_path, tracker=rec, round_offset=5, clock=_Clock(10.0, 70.0, 72.5)).wrap(_Strategy())

    assert strategy.configure_fit(1, _params(6, 4), None) == ["fit-ins"]
    results = [
        (SimpleNamespace(cid="0"), _res(100, (5, 5), client_id="C0", **{"train/box_loss": 1.0},
                                       class_counts_json='{"bus": 0}', **{"metrics/mAP50(B)": 0.9})),
        (SimpleNamespace(cid="1"), _res(300, (5, 5), client_id="C1", **{"train/box_loss": 2.0},
                                       class_counts_json='{"bus": 4}')),
    ]
    out = strategy.aggregate_fit(1, results, [])
    assert out[1] == {"n": 2}                                   # strategy result passed through

    rows = _rows(tmp_path / FIT_METRICS_FILE)
    assert [r["client_id"] for r in rows] == ["C0", "C1"]
    assert {r["round"] for r in rows} == {"6"}                  # absolute round (offset 5)
    assert rows[0]["uplink_bytes"] == "10" and rows[0]["downlink_bytes"] == "10"
    assert float(rows[0]["fit_phase_s"]) == 60.0 and float(rows[0]["aggregation_s"]) == 2.5
    assert "class_counts_json" not in rows[0]                   # payload, not a measurement
    assert not any(k.startswith("metrics/") for k in rows[0])   # 8-image val stub, not tracked

    step, agg = rec.calls[0]
    assert step == 6
    assert agg["comm/uplink_bytes"] == 20.0
    assert agg["comm/downlink_bytes"] == 20.0
    assert agg["comm/global_model_bytes"] == 10.0
    assert agg["fit/train/box_loss"] == (100 * 1.0 + 300 * 2.0) / 400
    assert agg["fit/num_clients"] == 2.0


def test_recording_errors_never_break_aggregation(tmp_path, caplog):
    class _Boom:
        def log_metrics(self, metrics, step=None):
            raise RuntimeError("tracker down")

    strategy = FitTracker(tmp_path, tracker=_Boom()).wrap(_Strategy())
    strategy.configure_fit(1, _params(1), None)
    out = strategy.aggregate_fit(1, [(SimpleNamespace(cid="0"), _res(1, (1,)))], [])
    assert out[1] == {"n": 1}
    assert "fit tracking failed" in caplog.text


def test_late_keys_extend_the_csv_header(tmp_path):
    strategy = FitTracker(tmp_path).wrap(_Strategy())
    strategy.configure_fit(1, _params(1), None)
    strategy.aggregate_fit(1, [(SimpleNamespace(cid="0"), _res(1, (1,)))], [])
    strategy.configure_fit(2, _params(1), None)
    strategy.aggregate_fit(2, [(SimpleNamespace(cid="0"), _res(1, (1,), **{"train/cls_loss": 0.5}))], [])
    rows = _rows(tmp_path / FIT_METRICS_FILE)
    assert len(rows) == 2 and rows[0]["train/cls_loss"] == "" and rows[1]["train/cls_loss"] == "0.5"
