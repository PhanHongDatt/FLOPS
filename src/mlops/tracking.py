"""MLflow experiment tracking for FL and centralized runs (ADR-031). [ENGINEERING]

The CSV artifacts in the run directory stay the results of record (CLAUDE.md §21); MLflow is
the browsable index over them. A tracking failure therefore never ends a run unless the caller
asks for ``required=True``.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

logger = logging.getLogger(__name__)

DISABLED_URIS = {"", "none", "off", "disabled"}
RUN_ID_FILE = "mlflow_run.json"
MAX_PARAM_LEN = 500          # MLflow 2.x accepts longer values on some stores, 500 is safe on all
MAX_TAG_LEN = 5000
_NAME_BAD = re.compile(r"[^A-Za-z0-9_\-. :/]")

# Files of a run directory logged under artifact path "run/" (CLAUDE.md §21 + ADR-031 additions).
RUN_ARTIFACTS: tuple[str, ...] = (
    "config.yaml",
    "environment.json",
    "partition_manifest.yaml",
    "eval_protocol.json",
    "metrics.csv",
    "per_class_metrics.csv",
    "round_metrics.csv",
    "fit_metrics.csv",
    "aggregation_trace.yaml",
    "final_metrics.yaml",
    "run.log",
    "README.md",
)


def default_tracking_uri() -> str:
    """``$MLFLOW_TRACKING_URI``, else the compose MLflow service as published on the host."""
    return os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def sanitize_key(key: Any) -> str:
    """MLflow names allow alphanumerics, ``_ - . : /`` and spaces (Ultralytics emits ``(B)``)."""
    return _NAME_BAD.sub("_", str(key))


def flatten_params(data: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """Nested mapping -> ``{"a.b.c": "value"}`` with MLflow-safe keys and bounded values."""
    out: dict[str, str] = {}
    for k, v in data.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(flatten_params(v, f"{key}."))
            continue
        text = json.dumps(list(v), default=str) if isinstance(v, (list, tuple)) else str(v)
        if len(text) > MAX_PARAM_LEN:
            text = text[: MAX_PARAM_LEN - 3] + "..."
        out[sanitize_key(key)] = text
    return out


def finite_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Numeric, finite values only, with MLflow-safe names."""
    out: dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, bool):
            v = float(v)
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            out[sanitize_key(k)] = float(v)
    return out


def _reachable(uri: str, timeout: float) -> bool:
    """HTTP(S) stores are probed at ``/health``; local stores (file:, sqlite:) always pass."""
    if not uri.startswith(("http://", "https://")):
        return True
    try:
        with urllib.request.urlopen(uri.rstrip("/") + "/health", timeout=timeout) as resp:
            return 200 <= int(resp.status) < 300
    except Exception as exc:  # URLError, timeout, connection refused, ...
        logger.warning("MLflow health check of %s failed: %r", uri, exc)
        return False


def _read_run_id(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / RUN_ID_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_run_id(run_dir: Path, run_id: str, uri: str, experiment: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_ID_FILE).write_text(
        json.dumps({"run_id": run_id, "tracking_uri": uri, "experiment": experiment}, indent=2),
        encoding="utf-8",
    )


class Tracker:
    """Thin MLflow facade; every method is a no-op when tracking is disabled and never raises."""

    def __init__(self, mlflow_module: Any = None, run_id: str | None = None, uri: str | None = None,
                 resumed: bool = False) -> None:
        self._mlflow = mlflow_module
        self.run_id = run_id
        self.uri = uri
        self.resumed = resumed      # True when an existing MLflow run was reopened

    @property
    def enabled(self) -> bool:
        return self._mlflow is not None and self.run_id is not None

    def _call(self, what: str, fn: Callable[[Any], Any]) -> Any:
        if not self.enabled:
            return None
        try:
            return fn(self._mlflow)
        except Exception as exc:  # tracking is auxiliary (CSV artifacts are canonical)
            logger.warning("MLflow %s failed (%r); continuing", what, exc)
            return None

    def log_params(self, params: Mapping[str, Any]) -> None:
        items = list(flatten_params(params).items())

        def _do(m: Any) -> None:
            for i in range(0, len(items), 100):        # MLflow batch limit is 100 params
                m.log_params(dict(items[i:i + 100]))

        self._call("log_params", _do)

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        clean = {sanitize_key(k): str(v)[:MAX_TAG_LEN] for k, v in tags.items()}
        self._call("set_tags", lambda m: m.set_tags(clean))

    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        clean = finite_metrics(metrics)
        if clean:
            self._call("log_metrics", lambda m: m.log_metrics(clean, step=step))

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        self._call("log_artifact", lambda m: m.log_artifact(str(path), artifact_path=artifact_path))

    def log_run_artifacts(self, run_dir: Path, names: Iterable[str] = RUN_ARTIFACTS,
                          artifact_path: str = "run") -> list[str]:
        """Log the run's record files that exist; returns the names that were logged."""
        present = [n for n in names if (Path(run_dir) / n).is_file()]

        def _do(m: Any) -> None:
            for n in present:
                m.log_artifact(str(Path(run_dir) / n), artifact_path=artifact_path)

        self._call("log_run_artifacts", _do)
        return present if self.enabled else []

    def end(self, status: str = "FINISHED") -> None:
        self._call("end_run", lambda m: m.end_run(status=status))


def start_tracking(
    run_dir: Path,
    experiment: str,
    uri: str | None,
    run_name: str | None = None,
    tags: Mapping[str, Any] | None = None,
    required: bool = False,
    probe_timeout: float = 5.0,
) -> Tracker:
    """Open (or, for a resumed run directory, reopen) an MLflow run.

    Returns a disabled ``Tracker`` when tracking is turned off (``uri`` in ``DISABLED_URIS``),
    mlflow is not installed, or the store is unreachable — unless ``required`` is True, in which
    case those failures raise ``RuntimeError``.
    """
    run_dir = Path(run_dir)
    uri = (uri or "").strip()
    if uri.lower() in DISABLED_URIS:
        logger.info("MLflow tracking disabled (uri=%r)", uri)
        return Tracker()

    def fail(msg: str, exc: BaseException | None = None) -> Tracker:
        if required:
            raise RuntimeError(msg) from exc
        logger.warning("%s — MLflow tracking disabled; CSV artifacts in %s remain the results of record",
                       msg, run_dir)
        return Tracker()

    try:
        import mlflow
    except ImportError as exc:
        return fail("mlflow is not installed", exc)
    if not _reachable(uri, probe_timeout):
        return fail(f"MLflow tracking server {uri} is unreachable")

    previous = _read_run_id(run_dir)
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        if mlflow.active_run() is not None:
            mlflow.end_run()
        run = None
        if previous and previous.get("tracking_uri") == uri and previous.get("run_id"):
            try:   # resumed run directory: continue the same MLflow run (steps stay absolute)
                run = mlflow.start_run(run_id=previous["run_id"])
                logger.info("Reopened MLflow run %s", previous["run_id"])
            except Exception as exc:
                logger.warning("Could not reopen MLflow run %s (%r); starting a new one",
                               previous["run_id"], exc)
        if run is None:
            run = mlflow.start_run(run_name=run_name)
    except Exception as exc:
        return fail(f"MLflow run could not be started on {uri}: {exc!r}", exc)

    _write_run_id(run_dir, run.info.run_id, uri, experiment)
    resumed = bool(previous) and run.info.run_id == previous.get("run_id")
    tracker = Tracker(mlflow, run.info.run_id, uri, resumed=resumed)
    if tags:
        tracker.set_tags(tags)
    logger.info("MLflow tracking: experiment=%s run_id=%s uri=%s", experiment, run.info.run_id, uri)
    return tracker
