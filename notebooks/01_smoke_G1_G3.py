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

# %%
import subprocess, sys

PINNED = [
    "torch==2.7.1", "torchvision==0.22.0",  # ADR-001
    "ultralytics==8.3.253",
    "flwr==1.21.0",
    "mlflow==3.4.0",
    "numpy>=1.26,<2.0", "pandas>=2.2,<3.0", "PyYAML>=6.0",
    "scipy>=1.13,<2.0", "opencv-python-headless>=4.9,<5.0",
]
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"

# Install torch first with correct CUDA wheel, then rest
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q",
     "torch==2.7.1", "torchvision==0.22.0", "--index-url", TORCH_INDEX]
)
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q"] + PINNED[2:]
)
print("Install done.")

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
    "mlflow": "3.4.0",
}
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
freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
lock_path.write_text(freeze, encoding="utf-8")
print(f"\n✅ Wrote {lock_path} ({len(freeze.splitlines())} entries)")
print("   Commit this file to advance G1 status in research/gates.yaml.")

# %% [markdown]
# ## Cell 3 — Repo import path + workspace paths

# %%
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Kaggle dataset mount (official BDD100K structure per ADR-002)
BDD100K_RAW = Path("/kaggle/input/bdd100k")
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
# ## Cell 4 — Convert BDD100K → YOLO format
#
# Uses `scripts/prepare_bdd100k.py`. If your Kaggle dataset structure differs
# from official, override `--train-ann` / `--val-ann` paths accordingly.

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
    "--partition-config", str(REPO_ROOT / "configs" / "partition" / "s0_iid.yaml"),
    "--yolo-root", str(YOLO_ROOT),
    "--output-dir", str(PARTITIONS_DIR),
]
print("Running:", " ".join(cmd))
subprocess.check_call(cmd)

partition_dir = PARTITIONS_DIR / "s0_iid_seed42"
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
]
print("Running:", " ".join(cmd))
subprocess.check_call(cmd)
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
run_dirs = sorted(
    (ARTIFACTS_DIR / "runs").glob("G3-FedAvg_smoke_seed42_*"),
    key=lambda p: p.stat().st_mtime,
)
if not run_dirs:
    raise RuntimeError(f"No run dir found under {ARTIFACTS_DIR / 'runs'}")

latest_run = run_dirs[-1]
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
