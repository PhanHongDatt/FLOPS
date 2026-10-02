"""Building Kaggle push folders (no network, no Kaggle account needed)."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("jupytext")

from scripts.kaggle_run import build_session, load_sessions  # noqa: E402

SHA = "0123456789abcdef0123456789abcdef01234567"


def _cfg(tmp_path):
    nb = tmp_path / "nb.py"
    nb.write_text(
        "# %% [markdown]\n# # Title\n\n# %%\nRUN_C1 = True\nSEEDS = [42, 123, 2024]\nprint(SEEDS)\n",
        encoding="utf-8",
    )
    return {
        "owner": "phdatt",
        "repo_url": "https://github.com/PhanHongDatt/FLOPS.git",
        "dataset_sources": ["phdatt/bdd100k-flops"],
        "machine_shape": "NvidiaTeslaT4",
        "sessions": {
            "s2": {"slug": "flops-s2", "notebook": str(nb)},
            "s3": {"slug": "flops-s3", "notebook": str(nb), "kernel_sources": ["s2"],
                   "replace": {"RUN_C1 = True": "RUN_C1 = False",
                               "SEEDS = [42, 123, 2024]": "SEEDS = [42]"}},
        },
    }


def test_build_writes_metadata(tmp_path):
    out = build_session(_cfg(tmp_path), "s3", SHA, tmp_path / "build")
    meta = json.loads((out / "kernel-metadata.json").read_text())
    assert meta["id"] == "phdatt/flops-s3"
    assert meta["title"] == "flops-s3"            # Kaggle derives the slug from the title
    assert meta["kernel_type"] == "notebook" and meta["language"] == "python"
    assert meta["enable_gpu"] == "true" and meta["enable_internet"] == "true"
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["is_private"] == "true"
    assert meta["dataset_sources"] == ["phdatt/bdd100k-flops"]
    assert meta["kernel_sources"] == ["phdatt/flops-s2"]   # session id → owner/slug
    assert (out / meta["code_file"]).exists()


def test_notebook_pins_commit_and_applies_overrides(tmp_path):
    out = build_session(_cfg(tmp_path), "s3", SHA, tmp_path / "build")
    meta = json.loads((out / "kernel-metadata.json").read_text())
    nb = json.loads((out / meta["code_file"]).read_text())
    sources = ["".join(c["source"]) for c in nb["cells"]]
    assert SHA in sources[0] and "git" in sources[0] and "checkout" in sources[0]
    body = "\n".join(sources)
    assert "RUN_C1 = False" in body and "RUN_C1 = True" not in body
    assert "SEEDS = [42]\n" in body
    assert "bdd100k_yolo" in sources[-1]          # cleanup cell keeps the output small


def test_override_must_match_exactly_once(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["sessions"]["s3"]["replace"] = {"NOT_IN_NOTEBOOK = 1": "x"}
    with pytest.raises(ValueError, match="NOT_IN_NOTEBOOK"):
        build_session(cfg, "s3", SHA, tmp_path / "build")


def test_unknown_session(tmp_path):
    with pytest.raises(KeyError, match="s9"):
        build_session(_cfg(tmp_path), "s9", SHA, tmp_path / "build")


def test_repo_sessions_file_is_valid():
    cfg = load_sessions()
    assert cfg["owner"] == "phdatt"
    for sid, s in cfg["sessions"].items():
        for src in s.get("kernel_sources", []):
            assert src in cfg["sessions"], f"{sid} depends on unknown session {src}"
