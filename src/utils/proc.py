"""Run long notebook subprocesses with their output in a log file.

Kaggle s1 v5 hung for an hour inside Ultralytics' progress bar
(``ultralytics/utils/tqdm.py`` ``self.file.flush()``): the FL subprocess had
inherited the notebook kernel's stdout pipe, which stopped draining once Ray had
forwarded the clients' training output. Writing to a file cannot block like
that, keeps the full log as an artifact, and the notebook shows only the tail.
"""
from __future__ import annotations

import os
import subprocess
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path


def _tail(path: Path, n: int) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return "".join(deque(f, maxlen=n))


def run_logged(
    cmd: Sequence[str],
    log_path: Path,
    timeout: float | None = None,
    tail_lines: int = 40,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> str:
    """Run ``cmd`` with stdout+stderr appended to ``log_path``; return the log tail.

    Prints the tail and re-raises on a non-zero exit (CalledProcessError) or on
    ``timeout`` (TimeoutExpired; the process is killed first).
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # unbuffered child output keeps stdout/stderr lines in time order in the log
    full_env = {**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})}
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"$ {' '.join(map(str, cmd))}\n")
        log.flush()
        proc = subprocess.Popen(list(map(str, cmd)), stdout=log, stderr=subprocess.STDOUT,
                                env=full_env, cwd=cwd)
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            print(f"[TIMEOUT after {timeout}s] last lines of {log_path}:\n{_tail(log_path, tail_lines)}")
            raise
    tail = _tail(log_path, tail_lines)
    if code != 0:
        print(f"[FAILED exit {code}] last lines of {log_path}:\n{tail}")
        raise subprocess.CalledProcessError(code, list(map(str, cmd)))
    return tail
