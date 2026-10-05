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
# - Directions after the s5/s6 negative result (sessions s7a/s7b): A5 not-true distillation
#   (ADR-012), A6 Equalized Focal Loss + A6c plain focal control (ADR-013), and the
#   diagnostics D1 error decomposition + D3 = A0 on the pooled re-split of S1b (ADR-014).
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

for cfg in ("s1b_bus_2k_seed42", "s1_control_matched_2k_seed42",   # matched + pooled are built from S1b
            "s1b_pooled_iid_2k_seed42", "server_sample_1k_seed42"):  # server sample excludes all three
    subprocess.check_call([
        sys.executable, str(REPO_ROOT / "scripts" / "generate_partition.py"),
        "--partition-config", str(REPO_ROOT / "configs" / "partition" / f"{cfg}.yaml"),
        "--yolo-root", str(YOLO_ROOT), "--output-dir", str(PARTITIONS_DIR),
    ])

import yaml
S1B = PARTITIONS_DIR / "s1b_bus_2k_seed42"
CTRL = PARTITIONS_DIR / "s1_control_matched_2k_seed42"
POOLED = PARTITIONS_DIR / "s1b_pooled_iid_2k_seed42"
SERVER = PARTITIONS_DIR / "server_sample_1k_seed42"
s1b = yaml.safe_load((S1B / "manifest.yaml").read_text())
for cid, cls in (("C0", "bus"), ("C1", "bus"), ("C2", "truck")):
    assert s1b["class_counts"][cid][cls] == 0, f"S1b-2k {cid} {cls} must be 0"
print("S1b-2k images/client:", {c: len(v) for c, v in s1b["client_assignments"].items()})
print("matched control:", yaml.safe_load((CTRL / "match_report.yaml").read_text()).get("matched"))
pooled = yaml.safe_load((POOLED / "manifest.yaml").read_text())
pooled_imgs = sorted(sum(pooled["client_assignments"].values(), []))
assert pooled_imgs == sorted(sum(s1b["client_assignments"].values(), [])), "pooled must hold exactly the S1b images"
print("pooled re-split class counts:", pooled["class_counts"])
server = yaml.safe_load((SERVER / "manifest.yaml").read_text())
assert not set(server["client_assignments"]["S"]) & set(pooled_imgs), "server sample overlaps the clients"
print("server sample (ADR-015):", len(server["client_assignments"]["S"]), "images,", server["class_counts"]["S"])

# %% [markdown]
# ## Cell 2b — D1: error decomposition of finished runs (ADR-014)
#
# Needs the outputs of s5a/s5b/s6a/s6b attached as kernel sources (session s7a); other
# sessions skip it. Round-30 global checkpoints, conf 0.001, FP/FN split at 0.25.

# %%
RUN_D1 = False   # session s7a sets True
if RUN_D1:
    from src.utils.proc import run_logged
    try:   # a diagnostic must not cost the training arms below
        run_logged([sys.executable, str(REPO_ROOT / "scripts" / "analyze_errors.py"),
                    "--data-yaml", str(YOLO_ROOT / "data.yaml"), "--roots", "/kaggle/input",
                    "--out", str(WORK / "flops_export" / "d1_errors.csv"), "--targets", "bus", "truck",
                    "--batch", "8"],   # small batches: Ultralytics' NMS time limit drops boxes
                   LOGS / "d1_errors.log", timeout=3 * 3600, stall_timeout=3600,
                   watch_dir=WORK / "flops_export")
    except Exception as exc:
        print(f"⚠️  D1 failed: {exc}")

# %% [markdown]
# ## Cell 2c — Teacher T on the server's 1,000-image sample (ADR-015)
#
# Session s8t trains it (50 epochs, G2 schedule); s8a/s8b load that same teacher from the
# s8t output (kernel source) so every arm uses one teacher.

# %%
RUN_TEACHER = False   # session s8t sets True
TEACHER_GLOB = "artifacts/runs/T-teacher_*/checkpoint/teacher.npz"
if RUN_TEACHER:
    from src.utils.proc import run_logged
    run_logged([sys.executable, str(REPO_ROOT / "scripts" / "train_server_teacher.py"),
                "--data-yaml", str(SERVER / "data_S.yaml"), "--global-data-yaml", str(DATA_YAML),
                "--exp-config", str(EXP_CFG), "--epochs", "50", "--seed", "42"],
               LOGS / "teacher.log", env={"YOLO_VERBOSE": "False"}, timeout=3 * 3600,
               stall_timeout=3600, watch_dir=ARTIFACTS_DIR / "runs")
from src.utils.kaggle_paths import find_output_file
_live = sorted((REPO_ROOT).glob(TEACHER_GLOB))
TEACHER = _live[-1] if _live else find_output_file(TEACHER_GLOB)
if TEACHER is None:   # other accounts: the s8t teacher uploaded as a dataset (flops-teacher-s8t)
    _ds = [q for pat in ("*/teacher.npz", "datasets/*/*/teacher.npz") for q in sorted(Path("/kaggle/input").glob(pat))]
    TEACHER = _ds[0] if _ds else None
if TEACHER is not None:
    import hashlib
    print("teacher:", TEACHER, "sha256", hashlib.sha256(Path(TEACHER).read_bytes()).hexdigest())
else:
    print("teacher: none")

# %% [markdown]
# ## Cell 3 — Run the arms
#
# Each arm: watchdog kills it after 1 h without progress (a checkpoint is written every
# round) or after 5 h in total. A failed arm is logged and the next arm still runs, so
# one failure cannot cost the rest of the session; every result is zipped at the end.

# %%
from src.utils.proc import run_logged

ARMS = ["A0", "FedProx", "A1", "A3", "A2b", "A4b", "A0@control"]   # sessions override this
SEED = 42   # training seed; the partition is the same for every seed (CLAUDE.md §14)
RHO = "0.25"

PARTITIONS = {"": S1B, "control": CTRL, "pooled": POOLED}

def arm_command(arm: str) -> tuple[list[str], Path]:
    name, _, where = arm.partition("@")
    partition = PARTITIONS[where]
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "run_fl_experiment.py"),
        "--partition", str(partition / "manifest.yaml"),
        "--data-yaml-dir", str(partition),
        "--run-class", "feasibility", "--exp-config", str(EXP_CFG),
        "--seed", str(SEED), "--mlflow-uri", MLFLOW_URI,
        "--mlflow-experiment", "S1b-2k-convergence",
        "--global-data-yaml", str(DATA_YAML), "--resume",
    ]
    cmd += ["--algorithm", "FedProx"] if name == "FedProx" else ["--ablation", name]
    if name in ("A2b", "A4b", "P2"):
        cmd += ["--rho", RHO]
    if name in ("B1", "B2", "P1", "P2"):                       # ADR-015
        if TEACHER is None:
            raise FileNotFoundError("no teacher.npz: run s8t first and attach it as a kernel source")
        cmd += ["--teacher-params", str(TEACHER)]
    if name == "B2":
        cmd += ["--server-data-yaml", str(SERVER / "data_S.yaml")]
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
    # pack what exists so far: a session cut short (quota, crash) keeps finished arms
    from src.utils.kaggle_finalize import finalize_outputs
    print("packed so far:", finalize_outputs(WORK, drop=False))
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
