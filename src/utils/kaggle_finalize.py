"""Keep a Kaggle session's output under the 500-item limit, even after a failure.

Kaggle saves a notebook's /kaggle/working only if it holds at most 500 items
(product-feedback/420538). Kaggle s2 v2 failed with the converted dataset (~80k
label files) still there, and nothing but the log was saved — including a
finished 2 h G2 checkpoint. ``finalize_outputs`` merges the results into one zip
(``flops_results.zip``) and deletes the bulky trees. The generated notebook calls
it from an IPython ``post_run_cell`` hook on any failed cell and again at the end.

Standard library only: it must run before the project's dependencies are installed.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ARCHIVE = "flops_results.zip"
# (path relative to /kaggle/working, name inside the zip)
_PACK = (("FLOPS/artifacts/runs", "artifacts/runs"), ("flops_export", "flops_export"))
_DROP = ("data", "runs", "FLOPS", "flops_export", ".ultralytics")


def finalize_outputs(working: Path = Path("/kaggle/working")) -> Path | None:
    """Add results to ``flops_results.zip`` (merging with an earlier archive) and
    remove the bulky directories. Returns the archive path, or None if nothing to pack."""
    working = Path(working)
    archive = working / ARCHIVE
    files: dict[str, Path] = {}
    for rel, arc_root in _PACK:
        src = working / rel
        if src.is_dir():
            for f in src.rglob("*"):
                if f.is_file():
                    files[f"{arc_root}/{f.relative_to(src).as_posix()}"] = f
    if not files and not archive.exists():
        return None

    tmp = archive.with_suffix(".tmp.zip")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as out:
        if archive.exists():                       # keep what an earlier call packed
            with zipfile.ZipFile(archive) as old:
                for name in old.namelist():
                    if name not in files:
                        out.writestr(name, old.read(name))
        for name, path in sorted(files.items()):
            out.write(path, name)
    tmp.replace(archive)

    for rel in _DROP:
        shutil.rmtree(working / rel, ignore_errors=True)
    return archive
