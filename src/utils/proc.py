"""Run long notebook subprocesses with their output in a log file.

Kaggle s1 v5 hung for an hour inside Ultralytics' progress bar
(``ultralytics/utils/tqdm.py`` ``self.file.flush()``): the FL subprocess had
inherited the notebook kernel's stdout pipe, which stopped draining once Ray had
forwarded the clients' training output. Writing to a file cannot block like
that, keeps the full log as an artifact, and the notebook shows only the tail.

A stall watchdog kills a process whose run directory stops changing, so an
unknown hang ends the Kaggle session instead of burning GPU quota until the
12 h cap. The log file itself is not a progress signal: faulthandler writes
stack dumps to it periodically even while the process is stuck.
"""
from __future__ import annotations

import os
import subprocess
import time
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path


def _tail(path: Path, n: int) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return "".join(deque(f, maxlen=n))


def _newest_mtime(root: Path) -> float:
    newest = root.stat().st_mtime if root.exists() else 0.0
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            try:
                newest = max(newest, os.stat(os.path.join(dirpath, name)).st_mtime)
            except OSError:
                continue   # file removed between listing and stat
    return newest


def run_logged(
    cmd: Sequence[str],
    log_path: Path,
    timeout: float | None = None,
    tail_lines: int = 40,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    stall_timeout: float | None = None,
    watch_dir: Path | None = None,
    poll_interval: float = 30.0,
) -> str:
    """Run ``cmd`` with stdout+stderr appended to ``log_path``; return the log tail.

    ``timeout``: total wall-clock limit. ``stall_timeout`` + ``watch_dir``: kill when
    no file under ``watch_dir`` has changed for ``stall_timeout`` seconds. Both raise
    TimeoutExpired after killing the process; a non-zero exit raises
    CalledProcessError. The log tail is printed in every failure case.
    """
    if stall_timeout is not None and watch_dir is None:
        raise ValueError("stall_timeout needs watch_dir (the log file is not a progress signal)")
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    args = list(map(str, cmd))
    # unbuffered child output keeps stdout/stderr lines in time order in the log
    full_env = {**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})}
    start = time.monotonic()
    last_progress_mtime = _newest_mtime(Path(watch_dir)) if watch_dir else 0.0
    last_progress_at = start

    def _kill(reason: str) -> None:
        proc.kill()
        proc.wait()
        print(f"[{reason}] last lines of {log_path}:\n{_tail(log_path, tail_lines)}")

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"$ {' '.join(args)}\n")
        log.flush()
        proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=full_env, cwd=cwd)
        while True:
            try:
                code = proc.wait(timeout=poll_interval)
                break
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            if timeout is not None and now - start > timeout:
                _kill(f"TIMEOUT after {timeout}s")
                raise subprocess.TimeoutExpired(args, timeout)
            if stall_timeout is not None:
                mtime = _newest_mtime(Path(watch_dir))
                if mtime > last_progress_mtime:
                    last_progress_mtime, last_progress_at = mtime, now
                elif now - last_progress_at > stall_timeout:
                    _kill(f"STALLED: no progress under {watch_dir} for {stall_timeout}s")
                    raise subprocess.TimeoutExpired(
                        args, stall_timeout, output=f"no progress under {watch_dir}")
    tail = _tail(log_path, tail_lines)
    if code != 0:
        print(f"[FAILED exit {code}] last lines of {log_path}:\n{tail}")
        raise subprocess.CalledProcessError(code, args)
    return tail
