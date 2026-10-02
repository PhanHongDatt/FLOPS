"""run_logged: long notebook subprocesses write to a file, not the kernel's pipe.

Kaggle s1 v5 hung for 1 h inside Ultralytics' progress bar
(ultralytics/utils/tqdm.py:286 ``self.file.flush()``): the FL subprocess
inherited the notebook's stdout pipe, which stopped draining after Ray had
forwarded the clients' training output.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from src.utils.proc import run_logged


def test_output_goes_to_file_and_tail_is_returned(tmp_path):
    log = tmp_path / "logs" / "job.log"
    code = "import sys\nfor i in range(200): print('line', i)\nprint('err', file=sys.stderr)"
    tail = run_logged([sys.executable, "-c", code], log, tail_lines=3)
    text = log.read_text(encoding="utf-8")
    assert "line 0" in text and "line 199" in text and "err" in text
    assert tail.splitlines()[-1] == "err"
    assert len(tail.splitlines()) == 3


def test_large_output_does_not_block(tmp_path):
    code = "import sys\nsys.stdout.write('x' * 5_000_000)\nsys.stdout.flush()"
    run_logged([sys.executable, "-c", code], tmp_path / "big.log", timeout=60)
    assert (tmp_path / "big.log").stat().st_size >= 5_000_000


def test_failure_raises_with_tail(tmp_path, capsys):
    code = "print('before failure'); raise SystemExit(3)"
    with pytest.raises(subprocess.CalledProcessError) as exc:
        run_logged([sys.executable, "-c", code], tmp_path / "fail.log")
    assert exc.value.returncode == 3
    assert "before failure" in capsys.readouterr().out   # tail shown in the notebook


def test_timeout_kills_and_raises(tmp_path):
    code = "import time; print('started', flush=True); time.sleep(30)"
    with pytest.raises(subprocess.TimeoutExpired):
        run_logged([sys.executable, "-c", code], tmp_path / "slow.log", timeout=2)
    assert "started" in (tmp_path / "slow.log").read_text(encoding="utf-8")


def test_extra_env_is_passed(tmp_path):
    code = "import os; print(os.environ['YOLO_VERBOSE'])"
    tail = run_logged([sys.executable, "-c", code], tmp_path / "env.log", env={"YOLO_VERBOSE": "False"},
                      tail_lines=1)
    assert tail.strip() == "False"
