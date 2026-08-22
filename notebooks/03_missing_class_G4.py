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
import subprocess, sys
from pathlib import Path

REPO_ROOT = Path("/kaggle/working/FLOPS")
if not REPO_ROOT.exists():
    raise RuntimeError(f"REPO_ROOT {REPO_ROOT} missing.")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reinstall pinned versions (Kaggle sessions are ephemeral per ADR-002)
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
assert torch.cuda.is_available(), "CUDA required."
print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## Cell 2 — Paths + restore YOLO dataset

# %%
BDD100K_RAW = Path("/kaggle/input/bdd100k")
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
# ## Cell 3 — Generate S1 + S1-Control partitions
#
# Use existing configs. S1 has C0 missing motorcycle, C1 missing truck.
# S1-Control uses matched IID split (no forced exclusions per §10).

# %%
for cfg_name in ("s1_missing_class.yaml", "s1_control.yaml"):
    cfg_path = REPO_ROOT / "configs" / "partition" / cfg_name
    subprocess.check_call([
        sys.executable, str(REPO_ROOT / "scripts" / "generate_partition.py"),
        "--partition-config", str(cfg_path),
        "--yolo-root", str(YOLO_ROOT),
        "--output-dir", str(PARTITIONS_DIR),
    ])

S1_DIR = PARTITIONS_DIR / "s1_mc_seed42"
S1C_DIR = PARTITIONS_DIR / "s1_control_seed42"
S1_MANIFEST = S1_DIR / "manifest.yaml"
S1C_MANIFEST = S1C_DIR / "manifest.yaml"

# Sanity-check S1 truly has zero motorcycle for C0, zero truck for C1 (§18)
import yaml
with S1_MANIFEST.open() as f:
    s1 = yaml.safe_load(f)
assert s1["class_counts"]["C0"]["motorcycle"] == 0, \
    f"S1 C0 motorcycle should be 0, got {s1['class_counts']['C0']['motorcycle']}"
assert s1["class_counts"]["C1"]["truck"] == 0, \
    f"S1 C1 truck should be 0, got {s1['class_counts']['C1']['truck']}"
print("✅ S1 partition passes missing-class sanity checks (§18)")

with S1C_MANIFEST.open() as f:
    s1c = yaml.safe_load(f)
# S1-Control should NOT have systemic missing classes (matched control)
for cid, missing in s1c["missing_classes"].items():
    if missing:
        print(f"  ⚠️  S1-Control {cid} accidentally missing {missing} — check partition size")

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
    ]
    print(f"\n[{scenario_name} seed={seed}] {' '.join(cmd[-8:])}")
    subprocess.check_call(cmd)
    # Latest run dir
    runs = sorted(
        (ARTIFACTS_DIR / "runs").glob(f"G3-FedAvg_feasibility_seed{seed}_*"),
        key=lambda p: p.stat().st_mtime,
    )
    return runs[-1] if runs else None

results: dict[str, dict[int, Path]] = {"S1": {}, "S1-Control": {}}
for seed in SEEDS:
    results["S1"][seed] = run_one("S1", S1_MANIFEST, S1_DIR, seed)
    results["S1-Control"][seed] = run_one("S1-Control", S1C_MANIFEST, S1C_DIR, seed)

print("\n✅ All 6 runs (2 scenarios × 3 seeds) complete.")

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
    out = {}
    if per_class_csv.exists():
        with per_class_csv.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                cls = row.get("class")
                ap = row.get("AP50") or row.get("AP") or row.get("mAP50")
                if cls and ap:
                    try:
                        out[f"AP50_{cls}"] = float(ap)
                    except ValueError:
                        pass
    if metrics_csv.exists():
        with metrics_csv.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                for k, v in row.items():
                    try:
                        out.setdefault(k, float(v))
                    except (ValueError, TypeError):
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

df = pd.DataFrame(rows)
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
        "is_missing_in_S1": cls in ("motorcycle", "truck"),
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
# ## Cell 6 — Parameter evidence STUB (`[NEEDS-VERIFICATION]`)
#
# **CLAUDE.md §8 F3** requires tracking Δθ = θ_local - θ_global by module
# (backbone / neck / classification branch / regression branch / validated
# class-associated group) with L2 norm + cosine similarity + confidence + FP/FN.
#
# **The current codebase does NOT instrument this.** `run_fl_experiment.py`
# does not save initial θ_global or per-client Δθ. Adding this requires:
#
# 1. Server-side hook in `src/federated/server.py` to save initial parameters
#    at round 0 as `initial_params.npz`
# 2. Client-side hook in `YOLOFlowerClient.fit()` to save `params_local_C{i}.npz`
#    after local training but before returning
# 3. Post-run analysis script computing per-module L2 / cosine using
#    `src.model.parameter_map.build_parameter_map()` to group params
#
# Without this, G4 CANNOT be claimed passed (§8: parameter + prediction evidence).
# See: ADR-003 (planned) for F3 instrumentation contract.

# %%
print("=" * 70)
print("[NEEDS-VERIFICATION] G4 parameter evidence — INCOMPLETE")
print("=" * 70)
print("""
Parameter/update analysis (Δθ by module) required by CLAUDE.md §8 is NOT
implemented in the current codebase. This notebook produces prediction
evidence (per-class ΔAP) only.

G4 gate status: still 'in_progress' — parameter evidence pending.

To complete G4:
  1. Write ADR-003 specifying F3 instrumentation contract
  2. Add hooks to server.py + YOLOFlowerClient.fit()
  3. Re-run this notebook or a follow-up notebook 03b
  4. Compare parameter drift patterns between S1 and S1-Control
""")

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
        pid = "s1_mc_seed42" if scenario == "S1" else "s1_control_seed42"
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
# - If ΔAP for motorcycle/truck is not consistently negative across seeds,
#   H1 is *weakened*, not strengthened by silence
#
# **Next steps:**
# 1. Write ADR-003 for F3 instrumentation contract
# 2. Implement server + client parameter capture hooks
# 3. Re-run to obtain parameter evidence
# 4. Only after both evidence types present → G4 gate → passed

# %%
print("G4 partial notebook complete. G4 gate NOT claimed passed.")
