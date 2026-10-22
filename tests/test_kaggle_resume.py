import zipfile
from pathlib import Path

from src.utils.kaggle_resume import restore_run

RUN = "P4t1i960_feasibility_seed42_s9_bus8_3k_moto_seed42"


def _zip(path: Path, rounds: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for r in rounds:
            zf.writestr(f"artifacts/runs/{RUN}/checkpoint/global_round_{r:03d}.npz", b"x")
        zf.writestr(f"artifacts/runs/{RUN}/round_metrics.csv", "round\n10\n")
        zf.writestr("artifacts/runs/other_run/checkpoint/global_round_020.npz", b"y")


def test_restores_from_zip(tmp_path):
    _zip(tmp_path / "in" / "s20a" / "flops_results.zip", [10, 14, 15])
    dest = restore_run(RUN, tmp_path / "runs", tmp_path / "in")
    assert dest == tmp_path / "runs" / RUN
    assert sorted(p.name for p in (dest / "checkpoint").iterdir()) == [
        "global_round_010.npz", "global_round_014.npz", "global_round_015.npz"]
    assert (dest / "round_metrics.csv").read_text() == "round\n10\n"
    assert not (tmp_path / "runs" / "other_run").exists()


def test_restores_from_unpacked_dir_and_prefers_most_rounds(tmp_path):
    _zip(tmp_path / "in" / "a" / "flops_results.zip", [10, 12])
    d = tmp_path / "in" / "b" / "artifacts" / "runs" / RUN / "checkpoint"
    d.mkdir(parents=True)
    (d / "global_round_016.npz").write_bytes(b"z")
    dest = restore_run(RUN, tmp_path / "runs", tmp_path / "in")
    assert [p.name for p in (dest / "checkpoint").iterdir()] == ["global_round_016.npz"]


def test_noop_when_present_or_absent(tmp_path):
    assert restore_run(RUN, tmp_path / "runs", tmp_path / "in") is None      # no source
    (tmp_path / "runs" / RUN).mkdir(parents=True)
    _zip(tmp_path / "in" / "s20a" / "flops_results.zip", [10])
    assert restore_run(RUN, tmp_path / "runs", tmp_path / "in") is None      # already local
