# %% [markdown]
# # FLOPS — Notebook 04: Ablation plumbing smoke + F1 runtime + resume check
#
# **Scope:** CORRECTNESS ONLY. Nothing in this notebook produces a reportable
# research result. It verifies that the experiment matrix of
# `research/plan/plan.md` §7.1 can actually execute on Kaggle, which was not the
# case before ADR-007 (A1/A2a/A2b/A3/A4b had no execution path and
# SCAFFOLD/FedNova/ClassAwareAgg raised TypeError at construction).
#
# **Prerequisites:** notebook 01 (repo + YOLO dataset + a partition) has run in
# this session or its output dataset is attached.
#
# **What this notebook checks** — the §13.3 definition of done:
# 1. `verify_artifacts` passes per arm, including `aggregation_trace.yaml`
# 2. a smoke run completes for every *runnable* arm
# 3. kill-and-resume continues at the right round
# 4. **F1 runtime verification** of the YOLOv8 parameter map (gate G5 input)
# 5. the A4a ≡ A3 numerical identity predicted in plan.md §6.3
#
# **Gate discipline (CLAUDE.md §7):** A2b raises `NotImplementedError` by design
# (blocked on F1 + ADR-006). A2a/A4a execute here only as a plumbing check —
# they must NOT be used for reporting until G5 passes.

# %% [markdown]
# ## Cell 1 — Environment restore (mirror notebook 01 exactly)

# %%
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path("/kaggle/working/FLOPS")
if not REPO_ROOT.exists():
    raise RuntimeError(f"REPO_ROOT {REPO_ROOT} missing — run notebook 01 first.")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Kaggle sessions are ephemeral — mirror notebook 01's install stack EXACTLY
# per ADR-002 addendum 1 (mlflow<3.0 keeps protobuf<5 which flwr==1.21.0
# requires; numpy<2 breaks Ultralytics).
subprocess.check_call([
    sys.executable, "-m", "pip", "uninstall", "-q", "-y",
    "tensorflow", "tensorflow-cpu", "keras", "tf-keras",
])
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "torch==2.7.1", "torchvision==0.22.0",
    "--index-url", "https://download.pytorch.org/whl/cu128",
])
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "ultralytics==8.3.253",
    "flwr==1.21.0",
    "mlflow>=2.0,<3.0",
    "protobuf>=3.20,<5.0",
    "pandas>=2.2,<3.0",
    "PyYAML>=6.0",
    "scipy>=1.13,<2.0",
    "opencv-python-headless>=4.9,<5.0",
])
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "--no-deps", "-e", str(REPO_ROOT),
])

# Keep Ultralytics' settings inside /kaggle/working (plan.md §13 K7).
import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/kaggle/working/.ultralytics")

import torch
assert torch.cuda.is_available(), "CUDA required."
print("GPU:", torch.cuda.get_device_name(0))

# %% [markdown]
# ## Cell 2 — Paths

# %%
WORK = Path("/kaggle/working")
YOLO_ROOT = WORK / "data" / "bdd100k_yolo"
DATA_YAML = YOLO_ROOT / "data.yaml"
PARTITION_DIR = WORK / "partitions" / "s1_mc"
MANIFEST = PARTITION_DIR / "manifest.yaml"
MLFLOW_URI = f"file://{WORK}/mlruns"

for p in (DATA_YAML, MANIFEST):
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run notebook 01/02 first.")
print("manifest:", MANIFEST)

# %% [markdown]
# ## Cell 3 — F1 runtime verification (gate G5 input)
#
# `research/feasibility/F1/parameter_map.yaml` is still
# `runtime_confirmed: false`. Everything class-aware depends on the Detect-head
# key names being what source inspection claimed, so confirm against a real
# model **before** trusting any class-aware run.

# %%
import yaml

from src.data.bdd100k import TARGET_CLASSES
from src.federated.strategies.class_aware_agg import (
    EXPECTED_CLS_HEAD_KEYS,
    find_cls_head_indices,
)
from src.model.parameter_map import build_parameter_map, summarize
from src.model.yolo_wrapper import build_model

model = build_model("yolov8n.pt")
info = build_parameter_map(model)
param_names = [p.name for p in info]
shapes = [p.shape for p in info]

print("state_dict entries:", len(info))
print("module group summary:")
for group, stats in sorted(summarize(info).items()):
    print(f"  {group:14s} params={stats['params']:4d} numel={stats['numel']:,}")

cls_idx = find_cls_head_indices(param_names, shapes)
print("\nclass-head params matched:", len(cls_idx))
for i in cls_idx:
    print(f"  {param_names[i]:34s} {shapes[i]}")

assert len(cls_idx) == EXPECTED_CLS_HEAD_KEYS, (
    f"Expected {EXPECTED_CLS_HEAD_KEYS} class-head keys (3 scales x weight/bias), "
    f"got {len(cls_idx)}. Update research/feasibility/F1/parameter_map.yaml and "
    "open an ADR before running any class-aware arm."
)
# NOTE: nc here is whatever yolov8n.pt ships with (COCO = 80) until the model is
# built against the 4-class data.yaml. The ASSERTION above is about key names
# and count; the nc check belongs with the task-built model.
print("\nTARGET_CLASSES:", TARGET_CLASSES)
print("dim0 of class heads:", sorted({shapes[i][0] for i in cls_idx}))

# %% [markdown]
# ## Cell 4 — Smoke every runnable arm
#
# `--run-class smoke` = 2 rounds, batch 4, `fraction_evaluate: 0.0`. Correctness
# only; these numbers are not reportable (CLAUDE.md §14).

# %%
import shutil

def run_arm(ablation: str, rho: float | None = None, resume: bool = False,
            extra: list[str] | None = None) -> None:
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "run_fl_experiment.py"),
        "--partition", str(MANIFEST),
        "--data-yaml-dir", str(PARTITION_DIR),
        "--ablation", ablation,
        "--run-class", "smoke",
        "--seed", "42",
        "--mlflow-uri", MLFLOW_URI,
        "--mlflow-experiment", "G6G7-ablation-smoke",
        "--global-data-yaml", str(DATA_YAML),
    ]
    if rho is not None:
        cmd += ["--rho", str(rho)]
    if resume:
        cmd += ["--resume"]
    if extra:
        cmd += extra
    print("\n" + "=" * 78)
    print(" ".join(cmd[1:]))
    print("=" * 78)
    subprocess.check_call(cmd)


RUNS = Path(REPO_ROOT) / "artifacts" / "runs"

# A0 baseline, A1 class-count control, A3 server-only.
for arm in ("A0", "A1", "A3"):
    run_arm(arm)

# A2a / A4a: parameter-level preservation. PLUMBING CHECK ONLY — gated for
# reporting until G5 (parameter_map runtime_confirmed + Gate A/B/C in ADR-003).
for arm in ("A2a", "A4a"):
    run_arm(arm, rho=0.0)

# A2b / A4b must refuse to run: blocked on F1 + rho_loss.py + ADR-006.
for arm in ("A2b", "A4b"):
    try:
        run_arm(arm, rho=0.0)
    except subprocess.CalledProcessError:
        print(f"{arm} correctly refused to run (NotImplementedError) — expected.")
    else:
        raise AssertionError(
            f"{arm} ran, but A2b/A4b must be blocked until F1 + ADR-006. "
            "Check LossPreservationClient."
        )

# %% [markdown]
# ## Cell 5 — Artifact contract per arm (§21)

# %%
from src.utils.artifacts import verify_artifacts

ARM_ALGORITHM = {
    "A0": "FedAvg",
    "A1": "ClassCountFedAvg",
    "A2a": "FedAvg",
    "A3": "ClassAwareAgg",
    "A4a": "ClassAwareAgg",
}

def run_dir_for(arm: str, rho: float | None) -> Path:
    exp_id = f"{arm}_rho{rho:g}" if rho is not None else arm
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    return RUNS / f"{exp_id}_smoke_seed42_{manifest['partition_id']}"

problems = []
for arm, rho in (("A0", None), ("A1", None), ("A3", None),
                 ("A2a", 0.0), ("A4a", 0.0)):
    rd = run_dir_for(arm, rho)
    missing = verify_artifacts(rd, algorithm=ARM_ALGORITHM[arm])
    status = "OK" if not missing else f"MISSING {missing}"
    print(f"{arm:4s} {rd.name:55s} {status}")
    if missing:
        problems.append((arm, missing))
if problems:
    raise AssertionError(f"Artifact contract violated: {problems}")

# %% [markdown]
# ## Cell 6 — A4a ≡ A3 identity (plan.md §6.3)
#
# A2a protects exactly the rows ClassAwareAggregation already excludes for that
# client, so A4a cannot differ from A3. This is a numerical identity, not an
# experimental arm — if it fails, eligibility or the mask is wrong.

# %%
import numpy as np

from src.experiments.checkpoint import find_latest_checkpoint, load_global_checkpoint

a3_dir = run_dir_for("A3", None)
a4a_dir = run_dir_for("A4a", 0.0)

a3_ckpt = find_latest_checkpoint(a3_dir / "checkpoint")
a4a_ckpt = find_latest_checkpoint(a4a_dir / "checkpoint")
assert a3_ckpt and a4a_ckpt, "missing checkpoints — did Cell 4 complete?"
assert a3_ckpt[0] == a4a_ckpt[0], f"round mismatch: {a3_ckpt[0]} vs {a4a_ckpt[0]}"

p3 = load_global_checkpoint(a3_ckpt[1])
p4 = load_global_checkpoint(a4a_ckpt[1])
assert len(p3) == len(p4)
max_abs = max(float(np.abs(a - b).max()) for a, b in zip(p3, p4))
print(f"max |A3 - A4a| over {len(p3)} tensors at round {a3_ckpt[0]}: {max_abs:.3e}")
if max_abs > 1e-5:
    print(
        "WARNING: A4a differs from A3. Either eligibility does not match the "
        "missing-class definition, or the preservation mask touches rows outside "
        "the eligible set. Investigate before any ablation reporting."
    )
else:
    print("A4a == A3 as predicted by plan.md §6.3 — A4a stays an identity test.")

# %% [markdown]
# ## Cell 7 — Kill-and-resume (§13.3 criterion 4, plan.md §13 K4)
#
# Simulate a session that died mid-run: drop the last round checkpoint from the
# completed A0 run and resume. The run must continue at the dropped round and
# keep ABSOLUTE round numbers in `round_metrics.csv`.

# %%
import csv

a0_dir = run_dir_for("A0", None)
ckpt_dir = a0_dir / "checkpoint"
before = find_latest_checkpoint(ckpt_dir)
assert before, "A0 produced no checkpoints"
last_round, last_path = before
print("completed up to round", last_round)

# Refusing to overwrite without --resume is itself part of the contract.
try:
    run_arm("A0")
except subprocess.CalledProcessError:
    print("Re-run without --resume refused as expected (FileExistsError).")
else:
    raise AssertionError("Re-run without --resume should have been refused.")

last_path.unlink()
print("removed", last_path.name, "→ simulating interruption")

run_arm("A0", resume=True)

after = find_latest_checkpoint(ckpt_dir)
assert after and after[0] == last_round, f"resume did not reach round {last_round}"
rows = list(csv.DictReader((a0_dir / "round_metrics.csv").open(encoding="utf-8")))
print("round_metrics.csv rounds:", [r["round"] for r in rows])
assert str(last_round) in [r["round"] for r in rows], "absolute round numbering lost"
print("Resume OK — continued at round", last_round)

# %% [markdown]
# ## Cell 8 — Export

# %%
EXPORT_DIR = WORK / "flops_export"
EXPORT_DIR.mkdir(exist_ok=True)
shutil.copytree(RUNS, EXPORT_DIR / "runs", dirs_exist_ok=True)
print("Exported to", EXPORT_DIR, "— save as a Kaggle Output Dataset (ADR-002 §5).")

# %% [markdown]
# ## Cell 9 — Record outcome
#
# Update **locally**, then commit:
#
# 1. `research/feasibility/F1/parameter_map.yaml` → `runtime_confirmed: true`
#    plus `confirmed_by_script: notebooks/04_ablation_smoke.py` and the date, if
#    Cell 3 passed with no discrepancies.
# 2. `research/gates.yaml` → G5 `status: F1_runtime_confirmed` (F2/F3 still open,
#    so G5 stays unpassed and `sub_path` stays null).
# 3. `research/experiment_registry/registry.yaml` → one entry per smoke run with
#    `run_class: smoke`, `status: completed`, and a note that smoke results are
#    **not** reportable.
# 4. `research/plan/plan.md` §13.3 → tick criteria 2, 3, 4 and 6.
#
# Still open after this notebook: F2 (perturbation), F3 (matched missing-class
# training), ADR-003 Gate A/B/C, and A2b (`rho_loss.py` + ADR-006).
