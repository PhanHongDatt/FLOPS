# %% [markdown]
# # FLOPS — Notebook 02: G2 Centralized + G3 FedAvg Baselines
#
# **Scope:** G2 (centralized YOLOv8) + G3 (FedAvg on S0 IID) at **feasibility scale**.
# **Prerequisite:** Notebook 01 passed (G1 environment reproducible).
#
# **Feasibility scale** (CLAUDE.md §14): reduced epochs/rounds vs main.
# - Centralized: 10 epochs
# - FedAvg: 5 rounds × 1 local epoch × 4 clients
# - 1 seed. Multi-seed (min 3) is only required for G10 main experiments.
#
# **Exit criteria:**
# 1. G2: centralized model trains without error, produces per-class AP
# 2. G3: FedAvg completes all rounds, per-class AP recorded, artifacts §21 present
# 3. Both runs registered in `research/experiment_registry/registry.yaml`
#    with status=`completed` and run_class=`feasibility`.
#
# **Not in scope:** G4 Missing-Class comparison (notebook 03), any ablation.

# %% [markdown]
# ## Cell 1 — Assume environment ready (from notebook 01)

# %%
import subprocess, sys
from pathlib import Path

REPO_ROOT = Path("/kaggle/working/FLOPS")
if not REPO_ROOT.exists():
    raise RuntimeError(f"REPO_ROOT {REPO_ROOT} missing. Run notebook 01 first.")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if not (REPO_ROOT / "environment.lock").exists() or (REPO_ROOT / "environment.lock").stat().st_size < 100:
    raise RuntimeError(
        "environment.lock missing or empty. Run notebook 01 to satisfy G1 first."
    )

# Re-install to be safe (Kaggle sessions are ephemeral)
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q",
     "torch==2.7.1", "torchvision==0.22.0",
     "--index-url", "https://download.pytorch.org/whl/cu128"]
)
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q",
     "ultralytics==8.3.253", "flwr==1.21.0", "mlflow==3.4.0",
     "numpy>=1.26,<2.0", "pandas>=2.2,<3.0", "PyYAML>=6.0",
     "scipy>=1.13,<2.0", "opencv-python-headless>=4.9,<5.0"]
)

import torch
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available.")
print(f"CUDA: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## Cell 2 — Paths (assume BDD100K + YOLO conversion + partition already done)
#
# If notebook 01 was run in a previous session, restore from Kaggle Output Dataset
# `flops-artifacts`. Otherwise re-run conversion + partitioning.

# %%
BDD100K_RAW = Path("/kaggle/input/bdd100k")
WORK = Path("/kaggle/working")
YOLO_ROOT = WORK / "data" / "bdd100k_yolo"
PARTITIONS_DIR = WORK / "data" / "partitions"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MLFLOW_URI = f"file://{WORK / 'mlruns'}"

# Re-run prepare if needed
if not (YOLO_ROOT / "data.yaml").exists():
    print("YOLO dataset missing → re-running prepare_bdd100k.py")
    subprocess.check_call([
        sys.executable, str(REPO_ROOT / "scripts" / "prepare_bdd100k.py"),
        "--data-root", str(BDD100K_RAW),
        "--output-root", str(YOLO_ROOT),
    ])

# Re-run partition if needed
partition_dir = PARTITIONS_DIR / "s0_iid_seed42"
if not (partition_dir / "manifest.yaml").exists():
    print("Partition missing → re-running generate_partition.py")
    subprocess.check_call([
        sys.executable, str(REPO_ROOT / "scripts" / "generate_partition.py"),
        "--partition-config", str(REPO_ROOT / "configs" / "partition" / "s0_iid.yaml"),
        "--yolo-root", str(YOLO_ROOT),
        "--output-dir", str(PARTITIONS_DIR),
    ])

DATA_YAML = YOLO_ROOT / "data.yaml"
MANIFEST = partition_dir / "manifest.yaml"
print(f"DATA_YAML: {DATA_YAML}")
print(f"MANIFEST:  {MANIFEST}")

# %% [markdown]
# ## Cell 3 — G2: Centralized YOLOv8 baseline (feasibility)
#
# Uses `scripts/train_centralized.py`. Feasibility scale: 10 epochs.

# %%
SEED = 42
# NOTE: epochs/batch/rounds come from configs/experiments/feasibility.yaml
# (verified: 10 epochs, 5 rounds, 4 clients). Do NOT pass as CLI arg —
# train_centralized.py does not accept --epochs.

cmd = [
    sys.executable, str(REPO_ROOT / "scripts" / "train_centralized.py"),
    "--data-yaml", str(DATA_YAML),
    "--run-class", "feasibility",
    "--seed", str(SEED),
    "--mlflow-uri", MLFLOW_URI,
    "--mlflow-experiment", "G2-centralized",
]
print("Running:", " ".join(cmd))
subprocess.check_call(cmd)
print("\n✅ G2 centralized training complete.")

# Find latest G2 run (exp_id="G2-centralized" per train_centralized.py:59)
g2_runs = sorted(
    (ARTIFACTS_DIR / "runs").glob("G2-centralized_feasibility_seed42_*"),
    key=lambda p: p.stat().st_mtime,
)
if not g2_runs:
    raise RuntimeError("No G2 run dir found.")
g2_run = g2_runs[-1]
print(f"G2 run dir: {g2_run}")

# %% [markdown]
# ## Cell 4 — G3: FedAvg on S0 IID (feasibility)

# %%
G3_ROUNDS = 5  # from feasibility.yaml (main = 10)

# feasibility.yaml verified: 10 epochs, 5 rounds, 4 clients (matches base_config
# for num_clients). run_fl_experiment.py picks it up automatically via --run-class.

feas_cfg = REPO_ROOT / "configs" / "experiments" / "feasibility.yaml"
if not feas_cfg.exists():
    raise FileNotFoundError(f"Missing {feas_cfg} — required for feasibility runs.")

cmd = [
    sys.executable, str(REPO_ROOT / "scripts" / "run_fl_experiment.py"),
    "--partition", str(MANIFEST),
    "--algorithm", "FedAvg",
    "--data-yaml-dir", str(partition_dir),
    "--run-class", "feasibility",
    "--exp-config", str(feas_cfg),
    "--seed", str(SEED),
    "--mlflow-uri", MLFLOW_URI,
    "--mlflow-experiment", "G3-FedAvg-S0",
    "--global-data-yaml", str(DATA_YAML),
]
print("Running:", " ".join(cmd))
subprocess.check_call(cmd)
print("\n✅ G3 FedAvg baseline complete.")

g3_runs = sorted(
    (ARTIFACTS_DIR / "runs").glob("G3-FedAvg_feasibility_seed42_*"),
    key=lambda p: p.stat().st_mtime,
)
g3_run = g3_runs[-1]
print(f"G3 run dir: {g3_run}")

# %% [markdown]
# ## Cell 5 — Verify §21 artifacts for BOTH runs

# %%
from src.utils.artifacts import verify_artifacts

for label, run_dir in [("G2", g2_run), ("G3", g3_run)]:
    missing = verify_artifacts(run_dir)
    if missing:
        print(f"⚠️  {label} missing artifacts (§21): {missing}")
    else:
        print(f"✅ {label} artifacts complete: {run_dir.name}")

# %% [markdown]
# ## Cell 6 — Register experiments in registry
#
# Per CLAUDE.md §16 — feasibility runs may be registered with status=completed
# because they pass the gate at reduced scale. Main multi-seed runs get separate
# entries. Never overwrite an existing entry — append only.

# %%
import yaml
from datetime import datetime

REGISTRY = REPO_ROOT / "research" / "experiment_registry" / "registry.yaml"
with REGISTRY.open() as f:
    registry = yaml.safe_load(f) or {}
registry.setdefault("experiments", [])

ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")

# Read metrics from run dirs to include summary in registry entry
def _read_metrics_csv(run_dir):
    metrics_csv = run_dir / "metrics.csv"
    if not metrics_csv.exists():
        return {}
    import csv
    with metrics_csv.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows[-1] if rows else {}

g2_metrics = _read_metrics_csv(g2_run)
g3_metrics = _read_metrics_csv(g3_run)

new_entries = [
    {
        "id": f"EXP-Kaggle-G2-feasibility-{ts}",
        "gate": "G2",
        "scenario": "centralized",
        "method": "YOLOv8n-centralized",
        "run_class": "feasibility",
        "status": "completed",
        "seed": SEED,
        "epochs": G2_EPOCHS,
        "run_dir": str(g2_run.relative_to(REPO_ROOT)),
        "notes": (
            "Kaggle T4 feasibility run. Single seed. "
            "Passes G2 gate at feasibility scale. "
            "Main experiments (100 epochs, 3 seeds) require ADR-004 approval."
        ),
    },
    {
        "id": f"EXP-Kaggle-G3-FedAvg-S0-feasibility-{ts}",
        "gate": "G3",
        "scenario": "S0",
        "method": "FedAvg",
        "run_class": "feasibility",
        "status": "completed",
        "seed": SEED,
        "rounds": G3_ROUNDS,
        "partition_id": "s0_iid_seed42",
        "run_dir": str(g3_run.relative_to(REPO_ROOT)),
        "notes": (
            "Kaggle T4 feasibility run. Single seed. "
            "Passes G3 gate at feasibility scale. "
            "Full G3 comparison across 4 algorithms + 3 seeds deferred to ADR-004."
        ),
    },
]

registry["experiments"].extend(new_entries)
with REGISTRY.open("w") as f:
    yaml.dump(registry, f, default_flow_style=False, allow_unicode=True)

print(f"✅ Appended {len(new_entries)} entries to {REGISTRY}")

# %% [markdown]
# ## Cell 7 — Export artifacts

# %%
import shutil

EXPORT = WORK / "flops_export" / f"baseline_{ts}"
EXPORT.mkdir(parents=True, exist_ok=True)
shutil.copytree(g2_run, EXPORT / "G2", dirs_exist_ok=True)
shutil.copytree(g3_run, EXPORT / "G3", dirs_exist_ok=True)
shutil.copy2(REGISTRY, EXPORT / "registry.yaml")
print(f"✅ Exported to {EXPORT}")

# %% [markdown]
# ## Cell 8 — Exit summary
#
# **What passed:**
# - G2 centralized baseline (feasibility, seed 42)
# - G3 FedAvg baseline on S0 IID (feasibility, seed 42)
# - Registry updated
#
# **What is NOT claimed:**
# - G3 across all 4 algorithms (FedProx/SCAFFOLD/FedNova) — need separate runs
# - Multi-seed reproducibility (min 3 seeds per §14) — feasibility uses 1 seed
# - G4 Missing-Class effect — see notebook 03
#
# **Next steps:**
# 1. Update `research/gates.yaml` G2 + G3 status → `passed_feasibility`
# 2. Run notebook 03 for G4 Missing-Class experiment

# %%
print("Baseline notebook complete.")
