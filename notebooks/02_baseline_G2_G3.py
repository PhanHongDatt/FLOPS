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
import subprocess
import sys
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

# Kaggle sessions are ephemeral — mirror notebook 01's install stack EXACTLY
# per ADR-002 addendum 1 (mlflow<3.0 to keep protobuf<5, required by
# flwr==1.21.0). Previous version pinned mlflow==3.4.0 and numpy<2 which
# broke flwr and Ultralytics respectively.
subprocess.check_call([
    sys.executable, "-m", "pip", "uninstall", "-q", "-y",
    "tensorflow", "tensorflow-cpu", "keras", "tf-keras",
    "torchaudio",  # Kaggle ships 2.10 built for torch 2.10; unused here, would mismatch 2.7.1
])
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "torch==2.7.1", "torchvision==0.22.1",  # 0.22.0 requires torch==2.7.0 (ADR-002-A2)
    "--index-url", TORCH_INDEX,
])
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "ultralytics==8.3.253",
    "flwr==1.21.0",
    "mlflow>=2.0,<3.0",       # ADR-002-A1: keep protobuf<5
    "protobuf>=3.20,<5.0",
    "pandas>=2.2,<3.0",
    "PyYAML>=6.0",
    "scipy>=1.13,<2.0",
    "opencv-python-headless>=4.9,<5.0",
])
# Register FLOPS as an editable package so `python scripts/*.py` invoked
# via subprocess can resolve `from src...` imports.
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "--no-deps", "-e", str(REPO_ROOT),
])

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
# Kaggle private-dataset mount path — kept in sync with notebook 01 + docs/KAGGLE_SETUP.md
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.utils.kaggle_paths import find_bdd100k_root

# web editor mounts datasets/<owner>/<slug>/, `kaggle kernels push` may mount <slug>/
BDD100K_RAW = find_bdd100k_root()
print("BDD100K_RAW:", BDD100K_RAW)
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
# (single source of truth per CLAUDE.md §5). Do NOT pass as CLI arg —
# train_centralized.py does not accept --epochs.
import yaml
_feas_cfg_path = REPO_ROOT / "configs" / "experiments" / "feasibility.yaml"
with _feas_cfg_path.open() as _f:
    _feas_cfg = yaml.safe_load(_f) or {}
# Read for later registry entry — previously referenced but never defined.
G2_EPOCHS = int(_feas_cfg.get("train", {}).get("epochs", 10))
G3_ROUNDS_FROM_CFG = int(_feas_cfg.get("federated", {}).get("num_rounds", 5))

cmd = [
    sys.executable, str(REPO_ROOT / "scripts" / "train_centralized.py"),
    "--data-yaml", str(DATA_YAML),
    "--run-class", "feasibility",
    "--seed", str(SEED),
    "--mlflow-uri", MLFLOW_URI,
    "--mlflow-experiment", "G2-centralized",
]
print("Running:", " ".join(cmd))
from src.utils.proc import run_logged  # output → file: the notebook stdout pipe can block (s1 v5)
run_logged(cmd, WORK / "flops_export" / "logs" / "g2_centralized.log", env={"YOLO_VERBOSE": "False"})
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
G3_ROUNDS = G3_ROUNDS_FROM_CFG  # single source: feasibility.yaml (currently 5)

# feasibility.yaml: 10 epochs, 5 rounds, 4 clients (num_clients inherited from
# base_config). run_fl_experiment.py picks it up automatically via --run-class.

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
run_logged(cmd, WORK / "flops_export" / "logs" / "g3_fedavg.log", env={"YOLO_VERBOSE": "False"})
print("\n✅ G3 FedAvg baseline complete.")

# Run ids are deterministic since ADR-007 (no timestamp), so the directory is
# computed rather than guessed from mtime, and the arm name is the run prefix
# ("FedAvg" here, not "G3-FedAvg").
import yaml as _yaml

from src.experiments.runner import default_run_id

_pid = (_yaml.safe_load((partition_dir / "manifest.yaml").read_text(encoding="utf-8")) or {})["partition_id"]
g3_run = ARTIFACTS_DIR / "runs" / default_run_id("FedAvg", "feasibility", SEED, _pid)
if not g3_run.exists():
    raise RuntimeError(f"Expected G3 run dir not found: {g3_run}")
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
