# %% [markdown]
# # FLOPS — Notebook 03: G4 Missing-Class Effect (S1 vs S1-Control)
#
# **Scope:** G4 (Missing-Class effect established) at feasibility scale.
# **Prerequisites:** notebooks 01 (G1) + 02 (G2/G3) passed.
#
# **CLAUDE.md §7 G4 requirement:** *"Requires S1 vs S1-Control with parameter +
# prediction evidence"*.
#
# **What this notebook covers:**
# 1. **Prediction evidence** (full): per-class AP on S1 vs S1-Control across ≥3 seeds
# 2. **Parameter evidence** (STUB): flagged `[NEEDS-VERIFICATION]` — proper F3
#    instrumentation not yet in codebase. G4 gate CANNOT be claimed passed with
#    prediction evidence alone (§8 CLAUDE.md).
#
# **Exit criteria (partial G4):**
# - S1 vs S1-Control per-class AP delta reported with mean ± std across 3 seeds
# - Runs registered with status=`completed`, run_class=`feasibility`
# - Parameter analysis flagged as incomplete → G4 remains `in_progress`
#   pending F3 instrumentation ADR (to be written separately)
#
# **Multi-seed requirement:** CLAUDE.md §14 — main = ≥3 seeds. Feasibility with
# 3 seeds is stronger than needed for gate check, but here we do 3 seeds so the
# ΔAP mean±std is meaningful.

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

# Kaggle sessions are ephemeral — mirror notebook 01's install stack EXACTLY
# per ADR-002 addendum 1 (mlflow<3.0 keeps protobuf<5 which flwr==1.21.0
# requires). Previous version pinned mlflow==3.4.0 + numpy<2 and broke
# flwr + Ultralytics respectively.
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
    "mlflow>=2.0,<3.0",       # ADR-002-A1
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

import torch
assert torch.cuda.is_available(), "CUDA required."
print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## Cell 2 — Paths + restore YOLO dataset

# %%
# Kaggle private-dataset mount path — kept in sync with notebook 01/02 + docs/KAGGLE_SETUP.md
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

if not (YOLO_ROOT / "data.yaml").exists():
    subprocess.check_call([
        sys.executable, str(REPO_ROOT / "scripts" / "prepare_bdd100k.py"),
        "--data-root", str(BDD100K_RAW),
        "--output-root", str(YOLO_ROOT),
    ])

DATA_YAML = YOLO_ROOT / "data.yaml"
print(f"DATA_YAML: {DATA_YAML}")

# %% [markdown]
# ## Cell 3 — Generate S1b + matched S1-Control partitions (comparison C1)
#
# S1b (`configs/partition/s1b_bus_seed42.yaml`): C0, C1 have zero bus; C2 zero
# truck; C3 all classes. S1-Control-Matched is built FROM S1b (same image count
# per client, non-target box vector matched, bus present) — generate S1b first.
# Replaces the old s1_missing_class / s1_control pair, whose "control" was a
# plain IID split (CLAUDE.md §10). Same partitions as configs/sweeps/c1_h1_missing_class.yaml.

# %%
for cfg_name in ("s1b_bus_seed42.yaml", "s1_control_matched_seed42.yaml"):
    subprocess.check_call([
        sys.executable, str(REPO_ROOT / "scripts" / "generate_partition.py"),
        "--partition-config", str(REPO_ROOT / "configs" / "partition" / cfg_name),
        "--yolo-root", str(YOLO_ROOT),
        "--output-dir", str(PARTITIONS_DIR),
    ])

S1_DIR = PARTITIONS_DIR / "s1b_bus_seed42"
S1C_DIR = PARTITIONS_DIR / "s1_control_matched_seed42"
S1_MANIFEST = S1_DIR / "manifest.yaml"
S1C_MANIFEST = S1C_DIR / "manifest.yaml"
MISSING_IN_S1 = ("bus", "truck")   # bus: C0, C1 (primary target); truck: C2

# Sanity checks (§18): the vacant classes really have zero boxes
import yaml
s1 = yaml.safe_load(S1_MANIFEST.read_text())
for cid, cls in (("C0", "bus"), ("C1", "bus"), ("C2", "truck")):
    assert s1["class_counts"][cid][cls] == 0, f"S1b {cid} {cls} should be 0"
print("✅ S1b passes missing-class sanity checks (§18)")

report = yaml.safe_load((S1C_DIR / "match_report.yaml").read_text())
print(f"S1-Control-Matched matched = {report.get('matched')}")
if not report.get("matched"):
    print("  ⚠️  Control NOT fully matched — publish match_report.yaml with any result; "
          "ΔAP is an observation, not a causal Missing-Class effect (§10).")

# %% [markdown]
# ## Cell 4 — Multi-seed FedAvg loop (S1 + S1-Control × 3 seeds)
#
# Uses `scripts/run_fl_experiment.py` end-to-end (through prepare_run/finalize_run).
# Same evaluator, same rounds, same partition-config across scenarios — only the
# partition (S1 vs S1-Control) varies. §22 anti-cherry-picking compliant.

# %%
SEEDS = [42, 123, 2024]  # from base_config.yaml
FEAS_CFG = REPO_ROOT / "configs" / "experiments" / "feasibility.yaml"

def run_one(scenario_name: str, manifest_path: Path, data_yaml_dir: Path, seed: int) -> Path:
    """Run one FedAvg experiment; return run dir."""
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "run_fl_experiment.py"),
        "--partition", str(manifest_path),
        "--algorithm", "FedAvg",
        "--data-yaml-dir", str(data_yaml_dir),
        "--run-class", "feasibility",
        "--exp-config", str(FEAS_CFG),
        "--seed", str(seed),
        "--mlflow-uri", MLFLOW_URI,
        "--mlflow-experiment", f"G4-FedAvg-{scenario_name}",
        "--global-data-yaml", str(DATA_YAML),
        # safe on a fresh run (starts at round 1) and on a finished one (no-op);
        # re-running this cell after an interruption continues from the last round
        "--resume",
    ]
    print(f"\n[{scenario_name} seed={seed}] {' '.join(cmd[-8:])}")
    subprocess.check_call(cmd)
    # Deterministic run id since ADR-007: compute the directory instead of
    # picking the newest match by mtime. Arm name "FedAvg" is the run prefix.
    import yaml as _yaml

    from src.experiments.runner import default_run_id

    pid = (_yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {})["partition_id"]
    run_dir = ARTIFACTS_DIR / "runs" / default_run_id("FedAvg", "feasibility", seed, pid)
    return run_dir if run_dir.exists() else None

# C1 = 6 FL runs (several GPU hours). Set False in the F2/F3 session and run C1
# in its own session(s); SEEDS can be split across sessions (e.g. [42] then [123, 2024]).
RUN_C1 = True

results: dict[str, dict[int, Path]] = {"S1": {}, "S1-Control": {}}
if RUN_C1:
    for seed in SEEDS:
        results["S1"][seed] = run_one("S1", S1_MANIFEST, S1_DIR, seed)
        results["S1-Control"][seed] = run_one("S1-Control", S1C_MANIFEST, S1C_DIR, seed)
    print(f"\n✅ C1 runs complete: 2 scenarios × {len(SEEDS)} seeds.")
else:
    print("[SKIP] RUN_C1 = False — C1 FedAvg runs not executed in this session.")

# %% [markdown]
# ## Cell 5 — Per-class AP aggregation across seeds
#
# Read `metrics.csv` / `per_class_metrics.csv` from each run and compute
# ΔAP = AP_S1 - AP_S1Control per class, mean ± std across seeds.

# %%
import csv
import pandas as pd
import numpy as np
from src.data.bdd100k import TARGET_CLASSES

def read_metrics(run_dir: Path) -> dict:
    """Read final metrics from a run dir. Prefer per_class_metrics.csv."""
    per_class_csv = run_dir / "per_class_metrics.csv"
    metrics_csv = run_dir / "metrics.csv"
    out: dict[str, float] = {}
    # per_class_metrics.csv: rows have keys [class, AP50, AP, precision, recall]
    if per_class_csv.exists():
        with per_class_csv.open() as f:
            for row in csv.DictReader(f):
                cls = row.get("class")
                if not cls:
                    continue
                ap50 = row.get("AP50")
                if ap50 not in (None, "", "None"):
                    try:
                        out[f"AP50_{cls}"] = float(ap50)
                    except ValueError:
                        pass
    # metrics.csv: two columns [metric, value], one row per metric
    if metrics_csv.exists():
        with metrics_csv.open() as f:
            for row in csv.DictReader(f):
                name, val = row.get("metric"), row.get("value")
                if not name or val in (None, "", "None"):
                    continue
                try:
                    out.setdefault(name, float(val))
                except ValueError:
                    pass
    return out

rows = []
for scenario, seed_runs in results.items():
    for seed, run_dir in seed_runs.items():
        if run_dir is None:
            print(f"⚠️  Missing run dir for {scenario} seed {seed}")
            continue
        m = read_metrics(run_dir)
        rows.append({"scenario": scenario, "seed": seed, **m})

df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["scenario", "seed"])
print("\nRaw per-run metrics:")
print(df.to_string())

# Aggregate: mean ± std per scenario per class
print("\n" + "=" * 70)
print("G4 — Per-class AP (mean ± std across 3 seeds)")
print("=" * 70)

summary_rows = []
for cls in TARGET_CLASSES:
    key = f"AP50_{cls}"
    s1_vals = df[df["scenario"] == "S1"].get(key, pd.Series(dtype=float)).dropna()
    s1c_vals = df[df["scenario"] == "S1-Control"].get(key, pd.Series(dtype=float)).dropna()
    if len(s1_vals) == 0 or len(s1c_vals) == 0:
        summary_rows.append({"class": cls, "note": "metrics missing — check per_class_metrics.csv format"})
        continue
    delta = s1_vals.mean() - s1c_vals.mean()
    delta_std = np.sqrt(s1_vals.var() + s1c_vals.var())  # independent samples
    summary_rows.append({
        "class": cls,
        "S1_mean": s1_vals.mean(), "S1_std": s1_vals.std(),
        "S1C_mean": s1c_vals.mean(), "S1C_std": s1c_vals.std(),
        "ΔAP": delta, "ΔAP_std_est": delta_std,
        "is_missing_in_S1": cls in MISSING_IN_S1,
    })

summary = pd.DataFrame(summary_rows)
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Save
G4_RESULTS_DIR = WORK / "flops_export" / "G4_prediction_evidence"
G4_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(G4_RESULTS_DIR / "per_run_metrics.csv", index=False)
summary.to_csv(G4_RESULTS_DIR / "delta_ap_summary.csv", index=False)
print(f"\n✅ Saved to {G4_RESULTS_DIR}")

# %% [markdown]
# ## Cell 5b — F2 controlled perturbation (`scripts/run_f2.py`)
#
# **CLAUDE.md §8 F2**: perturb one class-head weight row at a time (all 4 rows →
# effect matrix), a bias-shift positive control, and a norm-matched shared-conv
# control, all from the G2 checkpoint on one seeded 2000-image val subset.
# Pass/fail: pre-registered rule in `research/feasibility/F2/README.md`.

# %%
def find_g2_weights() -> Path:
    """G2 checkpoint from this session (notebook 02) or an attached output dataset.

    train_centralized.py writes it to
    artifacts/runs/G2-centralized_<run_class>_seed42_centralized/checkpoint/train/weights/best.pt;
    notebook 02 Cell 7 exports that run as flops_export/baseline_*/G2/.
    """
    patterns = [
        (ARTIFACTS_DIR / "runs", "G2-centralized_*/checkpoint/train/weights/best.pt"),
        (WORK / "flops_export", "baseline_*/G2/checkpoint/train/weights/best.pt"),
        (Path("/kaggle/input"), "**/G2/checkpoint/train/weights/best.pt"),
    ]
    for root, pattern in patterns:
        hits = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime) if root.exists() else []
        if hits:
            return hits[-1]
    return WORK / "g2" / "best.pt"   # placeholder → cells below print [SKIP]

G2_WEIGHTS = find_g2_weights()   # override manually if needed (also used by Cell 6)
print("G2_WEIGHTS:", G2_WEIGHTS, "(exists)" if G2_WEIGHTS.exists() else "(NOT FOUND)")
F2_OUT = WORK / "flops_export" / "F2"

if not G2_WEIGHTS.exists():
    print(f"[SKIP] {G2_WEIGHTS} not found — run notebook 02 (G2) first.")
else:
    subprocess.check_call([
        sys.executable, str(REPO_ROOT / "scripts" / "run_f2.py"),
        "--f2-config", str(REPO_ROOT / "configs" / "feasibility" / "f2_perturb_bus.yaml"),
        "--weights", str(G2_WEIGHTS),
        "--eval-data-yaml", str(DATA_YAML),
        "--output-dir", str(F2_OUT),
    ], cwd=REPO_ROOT)
    print("F2 artifacts:", sorted(str(p) for p in F2_OUT.rglob("summary.yaml")))

# %% [markdown]
# ## Cell 6 — Parameter + prediction evidence: F3 matched pair (`scripts/run_f3.py`)
#
# **CLAUDE.md §8 F3**: from ONE global checkpoint, train locally on
# S1b C0 (zero bus) vs its S1-Control-Matched twin (same images except the
# swaps that add bus) for 3 seeds; record Δθ by module, the 6-key class head
# and each class row (signed Δbias), plus per-class ΔAP / FP / FN.
#
# **Global checkpoint must already detect bus** — set `G2_WEIGHTS` to the G2
# centralized `best.pt` (or an FL `global_round_XXX.npz`). Raw `yolov8n.pt` is
# not valid: at nc=4 the whole cv3 branch is re-initialised (F1 runtime,
# 2026-10-02), so there is nothing to forget. The run records
# `target_known_before` and warns if the start model has zero bus AP.
#
# Pass/fail is NOT decided here: apply the pre-registered rule in
# `research/feasibility/F3/README.md` to `summary.yaml` across seeds.


# %%
F3_OUT = WORK / "flops_export" / "F3"

for cfg in ("s1b_bus_seed42", "s1_control_matched_seed42"):   # matched needs s1b first
    subprocess.check_call([
        sys.executable, str(REPO_ROOT / "scripts" / "generate_partition.py"),
        "--partition-config", str(REPO_ROOT / "configs" / "partition" / f"{cfg}.yaml"),
        "--yolo-root", str(YOLO_ROOT),
        "--output-dir", str(PARTITIONS_DIR),
    ], cwd=REPO_ROOT)

if not G2_WEIGHTS.exists():
    print(f"[SKIP] {G2_WEIGHTS} not found — run notebook 02 (G2) first. G4 stays in_progress.")
else:
    subprocess.check_call([
        sys.executable, str(REPO_ROOT / "scripts" / "run_f3.py"),
        "--f3-config", str(REPO_ROOT / "configs" / "feasibility" / "f3_matched_bus.yaml"),
        "--partition-dir", str(PARTITIONS_DIR),
        "--global-weights", str(G2_WEIGHTS),
        "--eval-data-yaml", str(DATA_YAML),
        "--output-dir", str(F3_OUT),
    ], cwd=REPO_ROOT)
    print("F3 artifacts:", sorted(str(p) for p in F3_OUT.rglob("summary.yaml")))

# %% [markdown]
# ## Cell 7 — Register runs in experiment registry
#
# 6 experiments (S1 × 3 seeds + S1-Control × 3 seeds) at feasibility scale.
# Each is a legitimate `completed` experiment per §16. The G4 GATE remains
# in_progress because parameter evidence is missing (Cell 6).

# %%
from datetime import datetime

REGISTRY = REPO_ROOT / "research" / "experiment_registry" / "registry.yaml"
with REGISTRY.open() as f:
    registry = yaml.safe_load(f) or {}
registry.setdefault("experiments", [])

ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
new_entries = []
for scenario, seed_runs in results.items():
    for seed, run_dir in seed_runs.items():
        if run_dir is None:
            continue
        pid = "s1b_bus_seed42" if scenario == "S1" else "s1_control_matched_seed42"
        new_entries.append({
            "id": f"EXP-Kaggle-G4-FedAvg-{scenario}-seed{seed}-{ts}",
            "gate": "G4-partial",  # partial: prediction evidence only
            "scenario": scenario,
            "method": "FedAvg",
            "run_class": "feasibility",
            "status": "completed",
            "seed": seed,
            "partition_id": pid,
            "run_dir": str(run_dir.relative_to(REPO_ROOT)),
            "notes": (
                "Prediction-evidence run for G4. "
                "Parameter evidence (F3 Δθ analysis) NOT included — see ADR-003 (planned). "
                "G4 gate remains in_progress until parameter evidence available."
            ),
        })

registry["experiments"].extend(new_entries)
with REGISTRY.open("w") as f:
    yaml.dump(registry, f, default_flow_style=False, allow_unicode=True)

print(f"✅ Appended {len(new_entries)} entries to {REGISTRY}")

# %% [markdown]
# ## Cell 8 — Export

# %%
import shutil
EXPORT = WORK / "flops_export" / f"G4_partial_{ts}"
EXPORT.mkdir(parents=True, exist_ok=True)
for scenario, seed_runs in results.items():
    for seed, run_dir in seed_runs.items():
        if run_dir is not None:
            dest = EXPORT / f"{scenario}_seed{seed}"
            shutil.copytree(run_dir, dest, dirs_exist_ok=True)
shutil.copytree(G4_RESULTS_DIR, EXPORT / "analysis", dirs_exist_ok=True)
shutil.copy2(REGISTRY, EXPORT / "registry.yaml")
print(f"✅ Exported to {EXPORT}")

# %% [markdown]
# ## Cell 9 — Exit summary
#
# **What is claimed:**
# - Prediction evidence for Missing-Class effect: per-class ΔAP across 3 seeds
# - 6 feasibility runs registered
#
# **What is NOT claimed:**
# - G4 gate PASSED — parameter evidence missing (§8)
# - H1 hypothesis SUPPORTED — needs parameter + prediction evidence
# - Any H2/H3 method effectiveness
#
# **Interpretation guidance (§20 anti-cherry-picking):**
# - Report *all* seeds, not only best
# - Report *all* 4 classes, including where S1 ≥ S1-Control (positive or negative)
# - If ΔAP for bus (and truck) is not consistently negative across seeds,
#   H1 is *weakened*, not strengthened by silence
#
# **Next steps:**
# 1. Write ADR-003 for F3 instrumentation contract
# 2. Implement server + client parameter capture hooks
# 3. Re-run to obtain parameter evidence
# 4. Only after both evidence types present → G4 gate → passed

# %%
print("G4 partial notebook complete. G4 gate NOT claimed passed.")
