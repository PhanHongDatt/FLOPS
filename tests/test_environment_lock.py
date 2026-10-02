"""environment.lock must be readable in both formats it exists in.

The repo ships a YAML template; notebook 01 Cell 2 overwrites it with
``pip freeze`` output on Kaggle. yaml.safe_load turns freeze text into one
string, and ``{**lock, ...}`` in prepare_run / run_f2 / run_f3 then raised
TypeError — every run after G1 would have crashed (found 2026-10-02).
"""
from __future__ import annotations

from src.experiments import runner


def _read(tmp_path, monkeypatch, text: str) -> dict:
    lock = tmp_path / "environment.lock"
    lock.write_text(text, encoding="utf-8")
    monkeypatch.setattr(runner, "_environment_lock_path", lambda: lock)
    return runner._read_environment_lock()


def test_pip_freeze_lock_is_parsed(tmp_path, monkeypatch):
    env = _read(tmp_path, monkeypatch,
                "torch==2.7.1+cu128\nultralytics==8.3.253\nflwr==1.21.0\n"
                "some-pkg @ file:///tmp/x\n\n")
    assert env["packages"]["torch"] == "2.7.1+cu128"
    assert env["packages"]["ultralytics"] == "8.3.253"
    assert env["lock_format"] == "pip-freeze"
    assert {**env, "seed": 1}["seed"] == 1  # usable as a mapping


def test_yaml_template_lock_is_parsed(tmp_path, monkeypatch):
    env = _read(tmp_path, monkeypatch, "# comment\npython_version: '3.11'\nflwr_version: ''\n")
    assert env["python_version"] == "3.11"


def test_missing_lock_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_environment_lock_path", lambda: tmp_path / "nope.lock")
    assert runner._read_environment_lock() == {}


def test_git_commit_is_resolved_outside_the_repo(tmp_path, monkeypatch):
    """Kaggle runs scripts from /kaggle/working (not the repo): environment.json
    recorded git_commit 'unknown' in s1 v6, breaking CLAUDE.md §5 traceability."""
    import re
    monkeypatch.chdir(tmp_path)
    assert re.fullmatch(r"[0-9a-f]{40}", runner._git_commit())
