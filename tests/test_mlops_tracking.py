"""ADR-031 tracking: tracking failures never end a run; a resumed run reopens its MLflow run."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from src.mlops import tracking
from src.mlops.tracking import Tracker, finite_metrics, flatten_params, start_tracking


def test_flatten_params_nests_sanitises_and_truncates():
    flat = flatten_params({"model": {"weights": "coco:yolo11s.pt", "image_size": 960},
                           "eval_rounds": [10, 20], "bad(key)": 1, "long": "x" * 900})
    assert flat["model.weights"] == "coco:yolo11s.pt"
    assert flat["model.image_size"] == "960"
    assert flat["eval_rounds"] == "[10, 20]"
    assert "bad_key_" in flat
    assert len(flat["long"]) == tracking.MAX_PARAM_LEN and flat["long"].endswith("...")


def test_finite_metrics_drops_non_numeric_and_non_finite():
    out = finite_metrics({"mAP50": 0.4, "n": 3, "flag": True, "nan": float("nan"),
                          "inf": float("inf"), "text": "a", "metrics/precision(B)": 0.5})
    assert out == {"mAP50": 0.4, "n": 3.0, "flag": 1.0, "metrics/precision_B_": 0.5}


@pytest.mark.parametrize("uri", ["none", "", "OFF", None])
def test_disabled_uri_gives_a_noop_tracker(tmp_path, uri):
    t = start_tracking(tmp_path, "exp", uri)
    assert not t.enabled
    t.log_params({"a": 1})
    t.log_metrics({"m": 1.0}, step=1)
    assert t.log_run_artifacts(tmp_path) == []
    t.end()
    assert not (tmp_path / tracking.RUN_ID_FILE).exists()


def test_unreachable_server_disables_tracking_unless_required(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", SimpleNamespace())
    monkeypatch.setattr(tracking, "_reachable", lambda uri, timeout: False)
    assert not start_tracking(tmp_path, "exp", "http://nowhere:5000").enabled
    with pytest.raises(RuntimeError, match="unreachable"):
        start_tracking(tmp_path, "exp", "http://nowhere:5000", required=True)


def test_local_stores_skip_the_health_probe():
    assert tracking._reachable("file:///tmp/mlruns", timeout=0.01)
    assert tracking._reachable("sqlite:///x.db", timeout=0.01)


def test_tracker_swallows_backend_errors(caplog):
    def boom(*a, **k):
        raise RuntimeError("store down")
    fake = SimpleNamespace(log_metrics=boom, log_params=boom, set_tags=boom, log_artifact=boom, end_run=boom)
    t = Tracker(fake, run_id="r1", uri="http://x")
    t.log_metrics({"m": 1.0})
    t.log_params({"a": 1})
    t.set_tags({"k": "v"})
    t.end()
    assert "store down" in caplog.text


def test_params_are_logged_in_batches_of_100():
    calls = []
    t = Tracker(SimpleNamespace(log_params=lambda d: calls.append(len(d))), run_id="r", uri="u")
    t.log_params({f"p{i}": i for i in range(250)})
    assert calls == [100, 100, 50]


def test_real_file_store_roundtrip_and_resume(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    uri = (tmp_path / "mlruns").as_uri()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.csv").write_text("metric,value\nmAP50,0.4\n", encoding="utf-8")

    t1 = start_tracking(run_dir, "exp", uri, run_name="r", tags={"arm": "A0"}, required=True)
    assert t1.enabled and not t1.resumed
    t1.log_params({"model": {"weights": "yolo11s.pt"}})
    t1.log_metrics({"eval/mAP50": 0.3}, step=1)
    assert t1.log_run_artifacts(run_dir) == ["metrics.csv"]
    t1.end()
    saved = json.loads((run_dir / tracking.RUN_ID_FILE).read_text(encoding="utf-8"))
    assert saved["run_id"] == t1.run_id

    t2 = start_tracking(run_dir, "exp", uri, required=True)      # --resume of the same run dir
    assert t2.resumed and t2.run_id == t1.run_id
    t2.log_metrics({"eval/mAP50": 0.4}, step=2)
    t2.end()

    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    run = client.get_run(t1.run_id)
    assert run.data.params["model.weights"] == "yolo11s.pt"
    assert run.data.tags["arm"] == "A0"
    assert [m.step for m in client.get_metric_history(t1.run_id, "eval/mAP50")] == [1, 2]
    assert [a.path for a in client.list_artifacts(t1.run_id, "run")] == ["run/metrics.csv"]
