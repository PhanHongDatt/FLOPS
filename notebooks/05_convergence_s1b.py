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
import time
from pathlib import Path

SESSION_T0 = time.time()   # session start, for the budget guard in Cell 3

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
EXP_CFG_5C = REPO_ROOT / "configs" / "experiments" / "convergence_5c.yaml"   # 5 clients (ADR-017)

if not (YOLO_ROOT / "data.yaml").exists():
    subprocess.check_call([
        sys.executable, str(REPO_ROOT / "scripts" / "prepare_bdd100k.py"),
        "--data-root", str(BDD100K_RAW), "--output-root", str(YOLO_ROOT),
    ])
DATA_YAML = YOLO_ROOT / "data.yaml"

for cfg in ("s1b_bus_2k_seed42", "s1_control_matched_2k_seed42",   # matched + pooled are built from S1b
            "s1b_pooled_iid_2k_seed42", "server_sample_1k_seed42",   # server sample excludes all three
            "s1b5_bus_3k_moto_seed42", "s1b5_pooled_3k_moto_seed42",   # 5-client scenario (ADR-017)
            "s1b5_bus_3k_allmoto_seed42"):                               # every motorcycle image (ADR-023)
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
S1B5 = PARTITIONS_DIR / "s1b5_bus_3k_moto_seed42"
POOLED5 = PARTITIONS_DIR / "s1b5_pooled_3k_moto_seed42"
S1B5BUS1 = PARTITIONS_DIR / "s1b5_bus1_3k_moto_seed42"   # severity sweep (ADR-022): bus on 1 of 5 clients
S1B5MOTO = PARTITIONS_DIR / "s1b5_bus_3k_allmoto_seed42"  # ADR-023 idea 2: all 3,002 motorcycle boxes
try:   # never block the other scenarios: an infeasible re-deal only fails the arms that use it
    subprocess.check_call([
        sys.executable, str(REPO_ROOT / "scripts" / "generate_partition.py"),
        "--partition-config", str(REPO_ROOT / "configs" / "partition" / "s1b5_bus1_3k_moto_seed42.yaml"),
        "--yolo-root", str(YOLO_ROOT), "--output-dir", str(PARTITIONS_DIR),
    ])
    _ref5 = yaml.safe_load((S1B5 / "manifest.yaml").read_text())
    _bus1 = yaml.safe_load((S1B5BUS1 / "manifest.yaml").read_text())
    assert sorted(sum(_bus1["client_assignments"].values(), [])) == sorted(sum(_ref5["client_assignments"].values(), []))
    assert [c for c in sorted(_bus1["class_counts"]) if _bus1["class_counts"][c]["bus"] > 0] == ["C4"]
    print("bus1 re-deal class counts:", _bus1["class_counts"])
except Exception as exc:
    print(f"⚠️  s1b5_bus1 partition not available: {exc}")
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
s1b5 = yaml.safe_load((S1B5 / "manifest.yaml").read_text())
for cid in ("C0", "C1", "C2"):
    assert s1b5["class_counts"][cid]["bus"] == 0, f"S1b5 {cid} must have zero bus"
assert s1b5["class_counts"]["C3"]["truck"] == 0
print("S1b5-3k (ADR-017):", {c: len(v) for c, v in s1b5["client_assignments"].items()}, s1b5["class_counts"])
s5moto = yaml.safe_load((S1B5MOTO / "manifest.yaml").read_text())
for cid in ("C0", "C1", "C2"):
    assert s5moto["class_counts"][cid]["bus"] == 0, f"S1b5-allmoto {cid} must have zero bus"
assert s5moto["class_counts"]["C3"]["truck"] == 0
assert sum(v["motorcycle"] for v in s5moto["class_counts"].values()) == 3002, "allmoto must hold every motorcycle box"
print("S1b5-3k-allmoto (ADR-023):", s5moto["class_counts"])

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
# ## Cell 2d — ADR-020: step 0 (COCO-aware D1) and N1 (CCVR calibration) on P4noKD round 20
#
# Needs the outputs of s12c_g (P4noKD @ S1b5) and s13c_g (A0c / P4noKD @ pooled) as kernel sources
# (session s14_g). No federated training. Each step is isolated and packed as soon as it ends, so an
# interruption keeps every finished step.

# %%
RUN_N1 = False   # session s14_g sets True
if RUN_N1:
    import zipfile
    from src.utils.kaggle_finalize import finalize_outputs
    from src.utils.proc import run_logged
    N1_EXP_CFG = REPO_ROOT / "configs" / "experiments" / "convergence_5c_y11s_20r.yaml"
    N1_OUT = WORK / "flops_export" / "n1"
    N1_OUT.mkdir(parents=True, exist_ok=True)

    def n1_checkpoint(run_name: str, rnd: int = 20) -> Path:
        rel = f"artifacts/runs/{run_name}/checkpoint/global_round_{rnd:03d}.npz"
        hits = sorted(Path("/kaggle/input").glob(f"**/{rel}"))
        if hits:
            return hits[0]
        for z in sorted(Path("/kaggle/input").glob("**/flops_results.zip")):
            with zipfile.ZipFile(z) as zf:
                if rel in zf.namelist():
                    zf.extract(rel, DATA_TMP / "n1_ckpts")
                    return DATA_TMP / "n1_ckpts" / rel
        raise FileNotFoundError(rel)

    steps = {"d1": None}
    try:   # step 0: missed-bus decomposition with the COCO head (labels stay 0..3, predictions remapped)
        run_logged([sys.executable, str(REPO_ROOT / "scripts" / "analyze_errors.py"),
                    "--data-yaml", str(DATA_YAML), "--roots", "/kaggle/input", "--round", "20",
                    "--weights", "coco:yolo11s.pt", "--targets", "bus", "truck",
                    "--out", str(N1_OUT / "d1_errors_p4nokd.csv"), "--batch", "8",
                    "--extract-to", str(DATA_TMP / "d1_ckpts")],
                   LOGS / "d1_p4nokd.log", timeout=2 * 3600, stall_timeout=3600, watch_dir=N1_OUT)
        steps["d1"] = "completed"
    except Exception as exc:
        steps["d1"] = f"failed: {exc}"
        print(f"⚠️  D1 failed: {exc}")
    print("packed:", finalize_outputs(WORK, drop=False))

    for tag, part in (("s1b5", S1B5), ("s1b5pooled", POOLED5)):
        try:
            ckpt = n1_checkpoint(f"P4noKD_feasibility_seed42_{part.name}")
            run_logged([sys.executable, str(REPO_ROOT / "scripts" / "calibrate_classifier.py"),
                        "--checkpoint", str(ckpt), "--partition", str(part / "manifest.yaml"),
                        "--data-yaml-dir", str(part), "--global-data-yaml", str(DATA_YAML),
                        "--exp-config", str(N1_EXP_CFG), "--weights", "coco:yolo11s.pt",
                        "--out", str(N1_OUT / tag), "--seed", "42"],
                       LOGS / f"n1_{tag}.log", timeout=2 * 3600, stall_timeout=3600, watch_dir=N1_OUT)
            steps[f"n1_{tag}"] = "completed"
        except Exception as exc:
            steps[f"n1_{tag}"] = f"failed: {exc}"
            print(f"⚠️  N1 {tag} failed: {exc}")
        print("packed:", finalize_outputs(WORK, drop=False))
    yaml.safe_dump(steps, open(N1_OUT / "n1_steps.yaml", "w"))
    print(steps)

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
ARM_TIMEOUT_H = 5   # sessions with s-tier models raise this (a Kaggle session allows 12 h)
# Kaggle discards ALL output of a session it kills at its 12 h limit (s13a/s13b, 2026-10-12: 3 YOLO11s
# arms, nothing saved). Keep the whole session under SESSION_BUDGET_H: an arm only starts if its
# estimate fits, and a running arm is cut before the budget so the notebook still ends normally.
import time as _time
SESSION_BUDGET_H = 11.0
ARM_ESTIMATE_H = 3.5   # conservative per-arm estimate; sessions may override

PARTITIONS = {"": S1B, "control": CTRL, "pooled": POOLED, "s5": S1B5, "s5pooled": POOLED5, "s5bus1": S1B5BUS1,
              "s5moto": S1B5MOTO}
RUN_TAG = ""   # appended to the arm in the run directory name; sessions at 960 px set "i960" (ADR-023)

def arm_command(arm: str) -> tuple[list[str], Path]:
    name, _, where = arm.partition("@")
    partition = PARTITIONS[where]
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "run_fl_experiment.py"),
        "--partition", str(partition / "manifest.yaml"),
        "--data-yaml-dir", str(partition),
        "--run-class", "feasibility", "--exp-config", str(EXP_CFG_5C if where.startswith("s5") else EXP_CFG),
        "--seed", str(SEED), "--mlflow-uri", MLFLOW_URI,
        "--mlflow-experiment", "S1b-2k-convergence",
        "--global-data-yaml", str(DATA_YAML), "--resume",
    ]
    if RUN_TAG:   # keep 960 px runs apart from the 640 px runs of the same arm, seed and partition
        cmd += ["--run-id", f"{name}{RUN_TAG}_feasibility_seed{SEED}_{partition.name}"]
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
    left_h = SESSION_BUDGET_H - (_time.time() - SESSION_T0) / 3600
    print(f"\n=== {arm} === ({left_h:.1f} h left in the session budget)")
    if left_h < ARM_ESTIMATE_H:
        outcomes[arm] = f"skipped: {left_h:.1f} h left < estimate {ARM_ESTIMATE_H} h"
        print(f"⚠️  {arm} skipped to stay under the Kaggle limit")
        continue
    try:
        run_logged(cmd, LOGS / f"{arm.replace('@', '_')}.log", env={"YOLO_VERBOSE": "False"},
                   timeout=min(ARM_TIMEOUT_H, left_h) * 3600, stall_timeout=3600, watch_dir=ARTIFACTS_DIR / "runs")
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

records = [r for r in discover_runs(ARTIFACTS_DIR / "runs") if "_2k_" in r.partition_id or "_3k_" in r.partition_id]
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

failed = {a: o for a, o in outcomes.items() if o != "completed" and not o.startswith("skipped")}
if failed:
    raise RuntimeError(f"arms failed (results of the others are kept): {failed}")
