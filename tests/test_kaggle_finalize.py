"""Kaggle keeps at most 500 output items; a run that leaves more saves nothing.

Kaggle s2 v2 crashed with ~80k label files in /kaggle/working, so even the finished
G2 checkpoint was lost. finalize_outputs packs the results into one zip and removes
the bulky trees, and runs after a failed cell as well as at the end.
"""
from __future__ import annotations

import zipfile

from src.utils.kaggle_finalize import finalize_outputs


def _tree(root, n):
    for i in range(n):
        p = root / f"d{i % 7}" / f"f{i}.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(i))


def test_packs_results_and_leaves_few_items(tmp_path):
    work = tmp_path / "working"
    _tree(work / "FLOPS" / "artifacts" / "runs", 300)
    _tree(work / "flops_export", 300)
    _tree(work / "data" / "bdd100k_yolo", 900)
    _tree(work / "runs", 50)                       # Ultralytics val plots
    _tree(work / "mlruns", 450)                    # MLflow file store (s2 v3: 492 files)
    (work / "FLOPS" / "src").mkdir(parents=True)
    (work / "FLOPS" / "src" / "x.py").write_text("code")

    archive = finalize_outputs(work)

    assert archive == work / "flops_results.zip"
    names = zipfile.ZipFile(archive).namelist()
    assert sum(n.startswith("artifacts/runs/") for n in names) == 300
    assert sum(n.startswith("flops_export/") for n in names) == 300
    assert sum(n.startswith("mlruns/") for n in names) == 450
    assert not any("bdd100k_yolo" in n for n in names)
    remaining = [p for p in work.rglob("*") if p.is_file()]
    assert len(remaining) < 500
    assert not (work / "data").exists() and not (work / "FLOPS").exists()


def test_is_idempotent_and_merges_new_results(tmp_path):
    work = tmp_path / "working"
    _tree(work / "flops_export", 3)
    finalize_outputs(work)
    _tree(work / "flops_export" / "later", 2)      # a later cell produced more
    archive = finalize_outputs(work)
    names = zipfile.ZipFile(archive).namelist()
    assert sum(n.startswith("flops_export/") for n in names) == 5


def test_nothing_to_pack(tmp_path):
    work = tmp_path / "working"
    work.mkdir()
    assert finalize_outputs(work) is None
