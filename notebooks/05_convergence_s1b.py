# %% [markdown]
# # FLOPS — Notebook 05: Convergence-length comparison on S1b-2k (baselines + proposed method)
#
# **Scope (ADR-011):** every arm on the SAME partition (S1b, 4 × 2000 images; C0/C1 have
# zero bus, C2 zero truck), seed 42, 30 rounds × 1 local epoch, full-val server eval every
# 5 rounds + final, same evaluator (AP at conf 0.001, FP/FN at 0.25 — ADR-009).
#
# Arms (`ARMS` below; sessions s5a / s5b split them to fit Kaggle's 12 h cap):
# - Baselines: A0 FedAvg, FedProx (mu 0.01, ADR-010)
# - Controls: A1 class-count FedAvg (H3 control), A0 on S1-Control-Matched (H1 control)
# - Proposed: A3 class-aware aggregation (server), A2b loss-level rho (client, ADR-006),
#   A4b = A2b + A3 (full method). rho = 0.25 fixed in advance, not tuned (§20).
#
# **Status:** exploratory — G4/G5 not yet passed (CLAUDE.md §7), single seed (§14).

# %% [markdown]
# ## Cell 1 — Environment restore

# %%
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path("/kaggle/working/FLOPS")
if not REPO_ROOT.exists():
    raise RuntimeError(f"REPO_ROOT {REPO_ROOT} missing.")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

subprocess.check_call([
    sys.executable, "-m", "pip", "uninstall", "-q", "-y",
    "tensorflow", "tensorflow-cpu", "keras", "tf-keras",
    "torchaudio",  # Kaggle ships 2.10 built for torch 2.10; unused here, would mismatch 2.7.1
])
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "torch==2.7.1", "torchvision==0.22.1",  # ADR-002-A2
    "--index-url", "https://download.pytorch.org/whl/cu128",
])
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "ultralytics==8.3.253", "flwr==1.21.0",
    "mlflow>=2.0,<3.0", "protobuf>=3.20,<5.0",   # ADR-002-A1
    "pandas>=2.2,<3.0", "PyYAML>=6.0", "scipy>=1.13,<2.0",
    "opencv-python-headless>=4.9,<5.0",
])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(REPO_ROOT)])

import torch
assert torch.cuda.is_available(), "CUDA required."
print(f"GPU: {torch.cuda.get_device_name(0)} x{torch.cuda.device_count()}")

# %% [markdown]
# ## Cell 2 — Paths, dataset, partitions

# %%
from src.utils.kaggle_paths import find_bdd100k_root

BDD100K_RAW = find_bdd100k_root()
WORK = Path("/kaggle/working")
DATA_TMP = Path("/kaggle/tmp/data")          # outside /kaggle/working (500-item output limit)
YOLO_ROOT = DATA_TMP / "bdd100k_yolo"
PARTITIONS_DIR = DATA_TMP / "partitions"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MLFLOW_URI = f"file://{WORK / 'mlruns'}"
LOGS = WORK / "flops_export" / "logs"
EXP_CFG = REPO_ROOT / "configs" / "experiments" / "convergence.yaml"

if not (YOLO_ROOT / "data.yaml").exists():
    subprocess.check_call([
        sys.executable, str(REPO_ROOT / "scripts" / "prepare_bdd100k.py"),
        "--data-root", str(BDD100K_RAW), "--output-root", str(YOLO_ROOT),
    ])
DATA_YAML = YOLO_ROOT / "data.yaml"

for cfg in ("s1b_bus_2k_seed42", "s1_control_matched_2k_seed42"):   # matched is built from S1b
    subprocess.check_call([
        sys.executable, str(REPO_ROOT / "scripts" / "generate_partition.py"),
        "--partition-config", str(REPO_ROOT / "configs" / "partition" / f"{cfg}.yaml"),
        "--yolo-root", str(YOLO_ROOT), "--output-dir", str(PARTITIONS_DIR),
    ])

import yaml
S1B = PARTITIONS_DIR / "s1b_bus_2k_seed42"
CTRL = PARTITIONS_DIR / "s1_control_matched_2k_seed42"
s1b = yaml.safe_load((S1B / "manifest.yaml").read_text())
for cid, cls in (("C0", "bus"), ("C1", "bus"), ("C2", "truck")):
    assert s1b["class_counts"][cid][cls] == 0, f"S1b-2k {cid} {cls} must be 0"
print("S1b-2k images/client:", {c: len(v) for c, v in s1b["client_assignments"].items()})
print("matched control:", yaml.safe_load((CTRL / "match_report.yaml").read_text()).get("matched"))

# %% [markdown]
# ## Cell 3 — Run the arms
#
# Each arm: watchdog kills it after 1 h without progress (a checkpoint is written every
# round) or after 5 h in total. A failed arm is logged and the next arm still runs, so
# one failure cannot cost the rest of the session; every result is zipped at the end.

# %%
from src.utils.proc import run_logged

ARMS = ["A0", "FedProx", "A1", "A3", "A2b", "A4b", "A0@control"]   # sessions override this
RHO = "0.25"

def arm_command(arm: str) -> tuple[list[str], Path]:
    partition = CTRL if arm.endswith("@control") else S1B
    name = arm.split("@")[0]
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "run_fl_experiment.py"),
        "--partition", str(partition / "manifest.yaml"),
        "--data-yaml-dir", str(partition),
        "--run-class", "feasibility", "--exp-config", str(EXP_CFG),
        "--seed", "42", "--mlflow-uri", MLFLOW_URI,
        "--mlflow-experiment", "S1b-2k-convergence",
        "--global-data-yaml", str(DATA_YAML), "--resume",
    ]
    cmd += ["--algorithm", "FedProx"] if name == "FedProx" else ["--ablation", name]
    if name in ("A2b", "A4b"):
        cmd += ["--rho", RHO]
    return cmd, partition

outcomes = {}
for arm in ARMS:
    cmd, _ = arm_command(arm)
    print(f"\n=== {arm} ===")
    try:
        run_logged(cmd, LOGS / f"{arm.replace('@', '_')}.log", env={"YOLO_VERBOSE": "False"},
                   timeout=5 * 3600, stall_timeout=3600, watch_dir=ARTIFACTS_DIR / "runs")
        outcomes[arm] = "completed"
    except Exception as exc:   # keep going: one arm must not cost the others
        outcomes[arm] = f"failed: {type(exc).__name__}"
        print(f"⚠️  {arm} failed: {exc}")
print(outcomes)

# %% [markdown]
# ## Cell 4 — Comparison table (same partition, same evaluator)

# %%
import pandas as pd
from src.evaluation.compare import check_comparable, discover_runs

records = [r for r in discover_runs(ARTIFACTS_DIR / "runs") if "_2k_" in r.partition_id]
keys = ["mAP50", "mAP50-95"] + [f"AP50_{c}" for c in ("car", "bus", "truck", "motorcycle")] \
       + [f"FP_{c}" for c in ("bus", "truck")] + [f"FN_{c}" for c in ("bus", "truck")]
table = pd.DataFrame({f"{r.arm} [{r.partition_id}]": {k: r.final_metrics.get(k) for k in keys}
                      for r in records})
print(table.round(4).to_string())
s1b_runs = [r for r in records if r.partition_id == "s1b_bus_2k_seed42"]
print("comparability issues (S1b arms):", check_comparable(s1b_runs) or "none")
(WORK / "flops_export").mkdir(exist_ok=True)
table.to_csv(WORK / "flops_export" / "s1b_2k_comparison.csv")
yaml.safe_dump(outcomes, open(WORK / "flops_export" / "arm_outcomes.yaml", "w"))

failed = {a: o for a, o in outcomes.items() if o != "completed"}
if failed:
    raise RuntimeError(f"arms failed (results of the others are kept): {failed}")
