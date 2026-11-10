"""Per-round client fit, communication and timing records (ADR-031). [ENGINEERING]

``FitTracker.wrap(strategy)`` replaces the instance's ``configure_fit`` / ``aggregate_fit`` with
wrappers that observe the exchanged parameters and the client fit metrics, so it works for every
strategy (Flower built-ins and the custom ones) without touching their code. Writes
``<run_dir>/fit_metrics.csv`` — one row per client per round — and logs round aggregates to the
tracker. Recording errors are logged, never raised: they must not cost a training round.
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Callable

from src.utils.artifacts import append_csv_rows

logger = logging.getLogger(__name__)

FIT_METRICS_FILE = "fit_metrics.csv"

# Fit-metric keys that are not training measurements: identity / payloads sent for aggregation.
_NON_METRIC_KEYS = {"class_counts_json", "client_id", "delta_c", "n_train"}
# Ultralytics' results_dict with run_val=False comes from the 8-image validation stub
# (src/model/yolo_wrapper.py val_stub_data_yaml): meaningless, so not tracked.
_STUB_PREFIXES = ("metrics/", "fitness")


def payload_bytes(parameters: Any) -> int:
    """Serialized size of a Flower ``Parameters`` object (sum of its byte tensors)."""
    tensors = getattr(parameters, "tensors", None) or []
    return int(sum(len(t) for t in tensors))


def _numeric_fit_metrics(metrics: dict[str, Any] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in (metrics or {}).items():
        if k in _NON_METRIC_KEYS or str(k).startswith(_STUB_PREFIXES):
            continue
        if isinstance(v, bool):
            v = float(v)
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            out[str(k)] = float(v)
    return out


class FitTracker:
    def __init__(self, run_dir: Path, tracker: Any = None, round_offset: int = 0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.csv_path = Path(run_dir) / FIT_METRICS_FILE
        self.tracker = tracker
        self.round_offset = int(round_offset)
        self._clock = clock
        self._fit_start: dict[int, float] = {}
        self._downlink: dict[int, int] = {}

    def wrap(self, strategy: Any) -> Any:
        configure = strategy.configure_fit
        aggregate = strategy.aggregate_fit

        def configure_fit(server_round: int, parameters: Any, client_manager: Any) -> Any:
            self._fit_start[server_round] = self._clock()
            self._downlink[server_round] = payload_bytes(parameters)
            return configure(server_round, parameters, client_manager)

        def aggregate_fit(server_round: int, results: Any, failures: Any) -> Any:
            t_agg = self._clock()
            out = aggregate(server_round, results, failures)
            t_end = self._clock()
            try:
                self._record(server_round, results, failures, out, t_agg, t_end)
            except Exception as exc:
                logger.warning("fit tracking failed in round %d (%r); continuing", server_round, exc)
            return out

        strategy.configure_fit = configure_fit
        strategy.aggregate_fit = aggregate_fit
        return strategy

    def _record(self, server_round: int, results: Any, failures: Any, out: Any,
                t_agg: float, t_end: float) -> None:
        abs_round = self.round_offset + int(server_round)
        start = self._fit_start.pop(server_round, None)
        fit_phase_s = (t_agg - start) if start is not None else float("nan")
        aggregation_s = t_end - t_agg
        downlink = self._downlink.pop(server_round, 0)

        rows: list[dict[str, Any]] = []
        for proxy, res in results or []:
            metrics = getattr(res, "metrics", None) or {}
            rows.append({
                "round": abs_round,
                "client_id": metrics.get("client_id", getattr(proxy, "cid", "?")),
                "num_examples": int(getattr(res, "num_examples", 0)),
                "uplink_bytes": payload_bytes(getattr(res, "parameters", None)),
                "downlink_bytes": downlink,
                "fit_phase_s": round(fit_phase_s, 3),
                "aggregation_s": round(aggregation_s, 3),
                **_numeric_fit_metrics(metrics),
            })
        if rows:
            append_csv_rows(self.csv_path, rows)

        params = out[0] if isinstance(out, tuple) and out else None
        agg: dict[str, float] = {
            "fit/num_clients": float(len(rows)),
            "fit/num_failures": float(len(failures or [])),
            "comm/uplink_bytes": float(sum(r["uplink_bytes"] for r in rows)),
            "comm/downlink_bytes": float(downlink * len(rows)),
            "comm/global_model_bytes": float(payload_bytes(params)),
            "time/fit_phase_s": fit_phase_s,
            "time/aggregation_s": aggregation_s,
        }
        # num_examples-weighted mean of every training measurement every client reported
        total = sum(r["num_examples"] for r in rows)
        fixed = {"round", "client_id", "num_examples", "uplink_bytes", "downlink_bytes",
                 "fit_phase_s", "aggregation_s"}
        shared = set.intersection(*(set(r) - fixed for r in rows)) if rows else set()
        for k in sorted(shared):
            if total > 0:
                agg[f"fit/{k}"] = sum(r[k] * r["num_examples"] for r in rows) / total
        if self.tracker is not None:
            self.tracker.log_metrics(agg, step=abs_round)
