# %% [markdown]
# # FLOPS — Notebook 01: Smoke Run on Kaggle
#
# **Scope:** G1 (environment) + end-to-end smoke FedAvg on S0 IID.
# **NOT a gate pass for G2/G3/G4.** Smoke is for correctness verification only
# (CLAUDE.md §14 run classes). Results are NOT recorded in the experiment registry.
#
# **Prerequisites:**
# - FLOPS repo cloned/uploaded to `/kaggle/working/FLOPS`
# - BDD100K uploaded as private Kaggle Dataset at `/kaggle/input/bdd100k/`
#   with official structure: `images/100k/{train,val}/`, `labels/det_20/det_{train,val}.json`
# - GPU accelerator enabled (T4/P100)
#
# **Exit criteria:**
# 1. `environment.lock` populated → G1 gate progress
# 2. Smoke FL run completes 2 rounds without error
# 3. Artifacts §21 present in `artifacts/runs/<run_id>/`
#
# **Not in scope:** G2 centralized, G3 baseline comparison, G4 Missing-Class,
# G5 F1 runtime verification, ablation. Those live in notebooks 02+ and require
# separate ADR (G5 needs ADR-003).

# %% [markdown]
# ## Cell 1 — Install pinned versions (ADR-001)
#
# Key constraints (Kaggle-specific, per ADR-002):
#   - mlflow<3.0  : flwr==1.21 needs protobuf<5; mlflow>=3 needs protobuf>=5 → conflict
#   - protobuf<5  : enforced via mlflow<3 pin
#   - numpy>=2.0  : Kaggle base image ships numpy 2.x; do NOT downgrade
#   - TensorFlow  : Kaggle image includes TF which imports protobuf>=5 → must uninstall

# %%
import subprocess
import sys

# Step 0: Remove TensorFlow (conflicts with protobuf<5 required by flwr+mlflow<3)
subprocess.check_call([
    sys.executable, "-m", "pip", "uninstall", "-q", "-y",
    "tensorflow", "tensorflow-cpu", "keras", "tf-keras",
    "torchaudio",  # Kaggle ships 2.10 built for torch 2.10; unused here, would mismatch 2.7.1
])

TORCH_INDEX = "https://download.pytorch.org/whl/cu128"

# Step 1: PyTorch + TorchVision (keep Kaggle CUDA 12.8 wheel)
# torchvision 0.22.1 is the release built for torch 2.7.1 (ADR-002-A2).
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "torch==2.7.1", "torchvision==0.22.1",  # 0.22.0 requires torch==2.7.0 (ADR-002-A2)
    "--index-url", TORCH_INDEX,
])

# Step 2: FL stack — mlflow<3 enforces protobuf<5 (compatible with flwr 1.21)
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "ultralytics==8.3.253",
    "flwr==1.21.0",
    "mlflow>=2.0,<3.0",       # <3.0 keeps protobuf<5
    "protobuf>=3.20,<5.0",    # explicit ceiling to prevent auto-upgrade
    "pandas>=2.2,<3.0",
    "PyYAML>=6.0",
    "scipy>=1.13,<2.0",
    "opencv-python-headless>=4.9,<5.0",
    # numpy: intentionally omitted — Kaggle ships numpy>=2, keep it
])

print("Install done.")

# %% [markdown]
# ## Cell 1b — Install FLOPS repo as package (no-deps)
#
# `pip install -e --no-deps`: registers `src` as importable package without
# re-pulling any dependency (preserves mlflow<3, protobuf<5, numpy>=2 set above).

# %%
REPO_ROOT_SETUP = "/kaggle/working/FLOPS"

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "--no-deps", "-e", REPO_ROOT_SETUP,
])
# The editable install's .pth hook only applies to NEW interpreters (the
# subprocess-run scripts). This kernel is already running, so add the repo
# to sys.path for the imports below.
if REPO_ROOT_SETUP not in sys.path:
    sys.path.insert(0, REPO_ROOT_SETUP)

# Verify: imports work + environment not broken
import importlib
import numpy as _np
from src.data.bdd100k import TARGET_CLASSES, BDD_NAME_MAP
import mlflow as _mlflow
import google.protobuf as _pb

print("numpy:       ", _np.__version__,    " (expect >=2.0)")
print("mlflow:      ", _mlflow.__version__, " (expect <3.0)")
print("protobuf:    ", _pb.__version__,    " (expect <5.0)")
print("TARGET_CLASSES:", TARGET_CLASSES)
print("motor ->", BDD_NAME_MAP.get("motor"), " (expect 'motorcycle')")
assert BDD_NAME_MAP["motor"] == "motorcycle", "BDD_NAME_MAP mapping wrong!"
assert _mlflow.__version__ < "3.0", f"mlflow too new: {_mlflow.__version__}"
print("\n✅ Repo installed, environment intact.")

# %% [markdown]
# ## Cell 2 — Environment audit + freeze (G1)
#
# Per CLAUDE.md §5: every experiment must record environment. We WARN on version
# mismatch (not assert), then dump `pip freeze` to `environment.lock`. If any
# version differs from ADR-001, ADR-002 supersedes the deviation for G1–G4.

# %%
import sys
from pathlib import Path

import torch
import ultralytics
import flwr
import mlflow

EXPECTED = {
    "ultralytics": "8.3.253",
    "flwr": "1.21.0",
}
# mlflow is a range, not a pin: ADR-002-A1 superseded ADR-001's 3.4.0 with <3.0
# (protobuf<5 for flwr 1.21.0); the old exact check always warned.
actual = {
    "ultralytics": ultralytics.__version__,
    "flwr": flwr.__version__,
    "mlflow": mlflow.__version__,
    "torch": torch.__version__,
}

print("=" * 60)
print("G1 — Environment audit")
print("=" * 60)
print(f"Python:         {sys.version.split()[0]}")
print(f"PyTorch:        {actual['torch']}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version:   {torch.version.cuda}")
    print(f"GPU:            {torch.cuda.get_device_name(0)}")
print(f"Ultralytics:    {actual['ultralytics']}")
print(f"Flower:         {actual['flwr']}")
print(f"MLflow:         {actual['mlflow']}")

warnings = []
for pkg, expected in EXPECTED.items():
    if actual[pkg] != expected:
        warnings.append(f"  {pkg}: expected {expected}, got {actual[pkg]}")

if not mlflow.__version__ < "3.0":
    warnings.append(f"  mlflow: expected <3.0 (ADR-002-A1), got {mlflow.__version__}")

if warnings:
    print("\n⚠️  Version mismatch vs ADR-001:")
    print("\n".join(warnings))
    print("  → Log this deviation in ADR-002 addendum before proceeding to G2+.")
else:
    print("\n✅ All pinned versions match ADR-001")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available — enable GPU accelerator in Kaggle settings.")

# Freeze environment
REPO_ROOT = Path("/kaggle/working/FLOPS")
if not REPO_ROOT.exists():
    raise RuntimeError(
        f"REPO_ROOT {REPO_ROOT} missing. Clone or upload FLOPS repo first."
    )

lock_path = REPO_ROOT / "environment.lock"
freeze_raw = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
# Strip editable/local-install pointers (e.g. "-e file:///kaggle/working/FLOPS")
# so environment.lock remains portable across machines per CLAUDE.md §5.
freeze_lines = [
    ln for ln in freeze_raw.splitlines()
    if ln and not ln.startswith("-e ") and not ln.startswith("# Editable")
]
freeze_clean = "\n".join(freeze_lines) + "\n"
lock_path.write_text(freeze_clean, encoding="utf-8")
print(f"\n✅ Wrote {lock_path} ({len(freeze_lines)} entries, editable installs stripped)")
print("   Commit this file to advance G1 status in research/gates.yaml.")

# %% [markdown]
# ## Cell 3 — Repo import path + workspace paths

# %%
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Kaggle dataset mount path (ADR-002)
# Format: /kaggle/input/datasets/<owner>/<dataset-slug>/<folder>
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.utils.kaggle_paths import find_bdd100k_root

# web editor mounts datasets/<owner>/<slug>/, `kaggle kernels push` may mount <slug>/
BDD100K_RAW = find_bdd100k_root()
print("BDD100K_RAW:", BDD100K_RAW)
if not BDD100K_RAW.exists():
    raise RuntimeError(
        f"BDD100K dataset not mounted at {BDD100K_RAW}. "
        "Attach the private Kaggle Dataset first."
    )

WORK = Path("/kaggle/working")
YOLO_ROOT = WORK / "data" / "bdd100k_yolo"
PARTITIONS_DIR = WORK / "data" / "partitions"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MLFLOW_URI = f"file://{WORK / 'mlruns'}"

for d in (YOLO_ROOT, PARTITIONS_DIR, ARTIFACTS_DIR, WORK / "mlruns"):
    d.mkdir(parents=True, exist_ok=True)

print(f"REPO_ROOT:       {REPO_ROOT}")
print(f"BDD100K_RAW:     {BDD100K_RAW}")
print(f"YOLO_ROOT:       {YOLO_ROOT}")
print(f"PARTITIONS_DIR:  {PARTITIONS_DIR}")
print(f"MLFLOW_URI:      {MLFLOW_URI}")

# %% [markdown]
# ## Cell 3b — F1 + ADR-008 checks on the pinned GPU stack (no dataset needed)
#
# 1. `verify_map.py`: F1 runtime parameter map on the pinned torch/ultralytics;
#    fails unless exactly 6 class-head keys exist. Output `runtime_map.yaml` is
#    exported in Cell 8 and replaces the local-CPU record (gates.yaml G5).
# 2. `check_fp32_upload.py`: one tiny training run on the GPU; fails if the
#    weights a client would upload are fp16-rounded (AMP path, ADR-008).

# %%
import subprocess
F1_DIR = REPO_ROOT / "research" / "feasibility" / "F1"
subprocess.check_call([sys.executable, str(F1_DIR / "verify_map.py")], cwd=WORK)
subprocess.check_call([sys.executable, str(F1_DIR / "check_fp32_upload.py"), "--device", "0"], cwd=WORK)

# %% [markdown]
# ## Cell 4 — Convert BDD100K → YOLO format
#
# Uses `scripts/prepare_bdd100k.py`.
#
# IMPORTANT: This cell MUST run (or re-run) after any change to src/data/bdd100k.py.
# The fix for "motor"->"motorcycle" (BDD_NAME_MAP) only takes effect on disk
# after conversion. If you have old YOLO labels from before the fix, delete
# YOLO_ROOT and re-run this cell to get correct motorcycle labels.
#
#   !rm -rf /kaggle/working/data/bdd100k_yolo  # only if re-converting

# %%
# Verify structure first
train_json = BDD100K_RAW / "labels" / "det_20" / "det_train.json"
val_json = BDD100K_RAW / "labels" / "det_20" / "det_val.json"
train_imgs = BDD100K_RAW / "images" / "100k" / "train"

if not train_json.exists():
    # Try common alternative structures
    candidates = list(BDD100K_RAW.rglob("det_train.json"))
    raise FileNotFoundError(
        f"det_train.json not at expected path {train_json}.\n"
        f"Found candidates: {candidates}\n"
        "Adjust prepare_bdd100k.py --train-ann arg accordingly."
    )

print(f"train JSON: {train_json}")
print(f"val JSON:   {val_json}")
print(f"train imgs: {train_imgs} ({sum(1 for _ in train_imgs.glob('*.jpg'))} files)")

# Run conversion
cmd = [
    sys.executable, str(REPO_ROOT / "scripts" / "prepare_bdd100k.py"),
    "--data-root", str(BDD100K_RAW),
    "--output-root", str(YOLO_ROOT),
    "--img-w", "1280", "--img-h", "720",
]
print("\nRunning:", " ".join(cmd))
subprocess.check_call(cmd)
print(f"\n✅ YOLO dataset at {YOLO_ROOT}")

# %% [markdown]
# ## Cell 5 — Generate S0 IID partition
#
# Uses `scripts/generate_partition.py` (tests: `tests/test_generate_partition.py`).
# Produces `manifest.yaml` + per-client `data_C{i}.yaml`.

# %%
cmd = [
    sys.executable, str(REPO_ROOT / "scripts" / "generate_partition.py"),
    "--partition-config", str(REPO_ROOT / "configs" / "partition" / "s0_iid_smoke_seed42.yaml"),
    "--yolo-root", str(YOLO_ROOT),
    "--output-dir", str(PARTITIONS_DIR),
]
print("Running:", " ".join(cmd))
subprocess.check_call(cmd)

partition_dir = PARTITIONS_DIR / "s0_iid_smoke_seed42"   # 250 images/client
manifest_path = partition_dir / "manifest.yaml"
data_yaml_dir = partition_dir  # data_C{i}.yaml files live alongside manifest

print(f"\n✅ Manifest: {manifest_path}")
print(f"   data.yamls in: {data_yaml_dir}")

# Inspect class counts to catch obvious partition bugs BEFORE running FL
import yaml
with manifest_path.open() as f:
    manifest_data = yaml.safe_load(f)
print("\nClass counts per client (sanity check):")
for cid, counts in manifest_data["class_counts"].items():
    total = sum(counts.values())
    print(f"  {cid}: total={total:>6d}  {counts}")

# %% [markdown]
# ## Cell 6 — Smoke FL run (FedAvg on S0 IID)
#
# Uses `scripts/run_fl_experiment.py` which goes through `runner.prepare_run` /
# `finalize_run` to produce §21 artifacts (config.yaml, environment.json,
# git_commit, run.log, checkpoint/).
#
# **Smoke config** (`configs/experiments/smoke.yaml`): 2 rounds, 2 clients, 1 epoch,
# batch 4. Correctness-only per CLAUDE.md §14. Results are NOT for reporting.

# %%
# Note: smoke.yaml has num_clients=2 but partition has 4 clients.
# For smoke we override to 2 clients by only passing C0, C1 data yamls — but
# run_fl_experiment.py uses manifest.client_assignments, which will fail if
# num_clients differs. Simplest: use s0_iid.yaml (4 clients) throughout and let
# smoke.yaml override rounds/epochs/batch only.

cmd = [
    sys.executable, str(REPO_ROOT / "scripts" / "run_fl_experiment.py"),
    "--partition", str(manifest_path),
    "--algorithm", "FedAvg",
    "--data-yaml-dir", str(data_yaml_dir),
    "--run-class", "smoke",
    "--seed", "42",
    "--mlflow-uri", MLFLOW_URI,
    "--mlflow-experiment", "smoke-FedAvg-S0",
    "--global-data-yaml", str(YOLO_ROOT / "data.yaml"),
]
print("Running:", " ".join(cmd))
# A smoke run on 4 x 250 images takes minutes; fail loudly instead of holding
# the GPU for hours if something hangs (s1 v4 stalled >90 min after round 1).
subprocess.check_call(cmd, timeout=3600)
print("\n✅ Smoke FL run complete.")

# %% [markdown]
# ## Cell 7 — Verify §21 artifact contract
#
# Every completed run must produce the artifacts listed in CLAUDE.md §21.
# `runner.finalize_run` already calls `verify_artifacts`, but we double-check
# here and surface missing files explicitly.

# %%
from src.utils.artifacts import verify_artifacts

# Find latest run dir
# Deterministic run id since ADR-007: the arm name ("FedAvg") is the prefix.
from src.experiments.runner import default_run_id

latest_run = ARTIFACTS_DIR / "runs" / default_run_id("FedAvg", "smoke", 42, manifest_data["partition_id"])
if not latest_run.exists():
    raise RuntimeError(f"Run dir {latest_run} not found under {ARTIFACTS_DIR / 'runs'}")
print(f"Latest run: {latest_run}")
print("\nRun contents:")
for f in sorted(latest_run.rglob("*")):
    if f.is_file():
        size = f.stat().st_size
        print(f"  {f.relative_to(latest_run)!s:<40s}  {size:>10d} bytes")

missing = verify_artifacts(latest_run)
if missing:
    print(f"\n⚠️  Missing artifacts (§21): {missing}")
    print("   Smoke run is technically valid but downstream gates require full artifacts.")
else:
    print("\n✅ All §21 artifacts present.")

# %% [markdown]
# ## Cell 8 — Export to Kaggle Output Dataset (persistence)
#
# ADR-002 §5: Kaggle sessions are ephemeral. Copy artifacts + environment.lock to
# `/kaggle/working/flops_export/` for saving as a Kaggle Output Dataset.
# Register the dataset (via Kaggle UI) as "FLOPS-artifacts" for future sessions.

# %%
import shutil
from datetime import datetime

EXPORT_DIR = WORK / "flops_export" / f"smoke_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Copy artifacts run + env lock
shutil.copytree(latest_run, EXPORT_DIR / "run", dirs_exist_ok=True)
shutil.copy2(REPO_ROOT / "environment.lock", EXPORT_DIR / "environment.lock")
shutil.copy2(F1_DIR / "runtime_map.yaml", EXPORT_DIR / "F1_runtime_map.yaml")  # Cell 3b

# Also save the partition manifest for reproducibility
shutil.copytree(partition_dir, EXPORT_DIR / "partition", dirs_exist_ok=True)

print(f"✅ Exported to {EXPORT_DIR}")
print("   Save /kaggle/working/flops_export/ as a Kaggle Output Dataset.")

# %% [markdown]
# ## Cell 9 — Exit summary
#
# **What passed:**
# - G1 environment reproducible: `environment.lock` populated
# - End-to-end smoke run: FedAvg on S0 IID completes 2 rounds
# - §21 artifact contract satisfied (or missing items surfaced)
#
# **What is NOT claimed:**
# - G2 (centralized baseline) — see notebook 02
# - G3 (FedAvg baseline formally) — smoke is not sufficient, needs feasibility scale
# - G4 (Missing-Class effect) — see notebook 03
# - G5 (F1/F2/F3 feasibility) — requires ADR-003 gate decision
# - Any H2/H3 method claim
#
# **Next steps:**
# 1. Commit `environment.lock` locally
# 2. Update `research/gates.yaml` G1 status → passed (with date)
# 3. Run notebook 02 for G2 + G3 (feasibility scale)

# %%
print("Smoke notebook complete. Do NOT log this run in experiment_registry as 'completed' — smoke runs are not experiments per CLAUDE.md §14.")
