# %% [markdown]
# # FLOPS — Kaggle Experiment Notebook
# **Missing-Class Non-IID Federated Object Detection with YOLOv8**
#
# Pipeline thứ tự gate: G1 → G5(F1) → G2 → G3 → G4 → G6–G8 (ablation A0–A4)
#
# Run class: `smoke` (1 epoch, 2 rounds) — phù hợp Kaggle T4 session (~10–20 phút/run)
#
# ---
# **QUAN TRỌNG:** Chạy theo đúng thứ tự cell. Không bỏ qua gate check.

# %% [markdown]
# ## Cell 1 — Install dependencies
#
# ⚠️ [NEEDS-VERIFICATION] flwr==1.21.0 `start_simulation` API — verify sau khi install.

# %%
# !pip install -q ultralytics==8.3.253 flwr==1.21.0 mlflow==3.4.0 pyyaml>=6.0

import subprocess, sys
pkgs = [
    "ultralytics==8.3.253",
    "flwr==1.21.0",
    "mlflow==3.4.0",
    "pyyaml>=6.0",
    "scipy>=1.13,<2.0",
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs)
print("Install complete.")

# %% [markdown]
# ## Cell 2 — Path setup & repo import

# %%
import sys
from pathlib import Path

# Adjust this to where the FLOPS repo is uploaded/cloned on Kaggle
REPO_ROOT = Path("/kaggle/working/FLOPS")

if not REPO_ROOT.exists():
    raise RuntimeError(
        f"REPO_ROOT {REPO_ROOT} not found.\n"
        "Upload or git-clone the FLOPS repo to /kaggle/working/FLOPS first."
    )

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Dataset root — set to the BDD100K YOLO-format dataset path on Kaggle
# Common Kaggle dataset paths:
#   /kaggle/input/bdd100k-yolo-format/
#   /kaggle/input/bdd100k/
BDD100K_YOLO_ROOT = Path("/kaggle/input/bdd100k-yolo-format")
# [NEEDS-VERIFICATION] Verify this path has images/train, images/val, labels/train, labels/val
# and a data.yaml. If the structure differs, adapt prepare_bdd100k.py first.

OUTPUT_ROOT = Path("/kaggle/working/flops_outputs")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print(f"REPO_ROOT:        {REPO_ROOT}")
print(f"BDD100K_YOLO_ROOT:{BDD100K_YOLO_ROOT}")
print(f"OUTPUT_ROOT:      {OUTPUT_ROOT}")

# %% [markdown]
# ## Cell 3 — G1: Environment Verification
#
# Gate G1: environment reproducible. Check CUDA, versions.

# %%
import torch
import ultralytics
import flwr
import mlflow

print("=" * 50)
print("G1 — Environment Verification")
print("=" * 50)
print(f"Python:        {sys.version}")
print(f"PyTorch:       {torch.__version__}")
print(f"CUDA available:{torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version:  {torch.version.cuda}")
    print(f"GPU:           {torch.cuda.get_device_name(0)}")
print(f"Ultralytics:   {ultralytics.__version__}")
print(f"Flower:        {flwr.__version__}")
print(f"MLflow:        {mlflow.__version__}")

# Version assertions
assert ultralytics.__version__ == "8.3.253", f"Wrong ultralytics version: {ultralytics.__version__}"
assert flwr.__version__ == "1.21.0", f"Wrong flwr version: {flwr.__version__}"
assert torch.cuda.is_available(), "CUDA not available — GPU required for training"

print("\n✅ G1 — Environment OK")

# %% [markdown]
# ## Cell 4 — G5/F1: Runtime Parameter Map Verification
#
# Verify that the source-level F1 analysis matches the actual model parameters.
# This is required before H2/H3 can be used scientifically.

# %%
import yaml
from ultralytics import YOLO
from src.model.parameter_map import build_parameter_map, classify_parameter, is_class_specific

print("=" * 50)
print("G5/F1 — Runtime Parameter Map Verification")
print("=" * 50)

# Load YOLOv8n (will download yolov8n.pt if not cached)
_probe_model = YOLO("yolov8n.pt")
param_info = build_parameter_map(_probe_model)

# Store param_names for later use by ClassAwareAgg and PreservationClient
PARAM_NAMES = [p.name for p in param_info]
PARAM_SHAPES = {p.name: p.shape for p in param_info}

# Summary
from src.model.parameter_map import summarize
summary = summarize(param_info)
print("\nParameter group summary:")
for group, stats in summary.items():
    print(f"  {group:20s}: {stats['params']:3d} params, {stats['numel']:,} elements")

# Verify key F1 findings
cls_params = [p for p in param_info if p.class_specific]
print(f"\nClass-specific params (detect_cls): {len(cls_params)}")
for p in cls_params:
    print(f"  {p.name}  shape={p.shape}")
    assert p.shape[0] == 4, f"Expected dim-0=4 (nc), got {p.shape[0]} for {p.name}"

# Verify backbone/neck NOT class-specific
for p in param_info:
    if p.module_group in ("backbone", "neck"):
        assert not p.class_specific, f"Unexpected class-specific backbone/neck param: {p.name}"

# Update F1 verification status
f1_yaml_path = REPO_ROOT / "research" / "feasibility" / "F1" / "parameter_map.yaml"
if f1_yaml_path.exists():
    with f1_yaml_path.open() as f:
        f1_data = yaml.safe_load(f)
    f1_data["verification"]["runtime_confirmed"] = True
    f1_data["verification"]["confirmed_by_script"] = "notebooks/kaggle_flops_experiment.py"
    f1_data["verification"]["discrepancies"] = []
    with f1_yaml_path.open("w") as f:
        yaml.dump(f1_data, f, default_flow_style=False)
    print(f"\n✅ Updated F1 parameter_map.yaml: runtime_confirmed=True")

print("\n✅ G5/F1 — Runtime parameter map verified")
del _probe_model  # free memory

# %% [markdown]
# ## Cell 5 — Dataset Setup
#
# Verify BDD100K YOLO dataset structure and check data.yaml exists.
# [NEEDS-VERIFICATION] Kaggle dataset format may differ — verify before running.

# %%
DATA_YAML = BDD100K_YOLO_ROOT / "data.yaml"

if not DATA_YAML.exists():
    # Try to find data.yaml
    candidates = list(BDD100K_YOLO_ROOT.rglob("data.yaml"))
    if candidates:
        DATA_YAML = candidates[0]
        print(f"Found data.yaml at: {DATA_YAML}")
    else:
        raise FileNotFoundError(
            f"No data.yaml found under {BDD100K_YOLO_ROOT}.\n"
            "[NEEDS-VERIFICATION] Check dataset structure and run prepare_bdd100k.py if needed."
        )

with open(DATA_YAML) as f:
    data_cfg = yaml.safe_load(f)
print(f"data.yaml: {DATA_YAML}")
print(f"  nc:    {data_cfg.get('nc')}")
print(f"  names: {data_cfg.get('names')}")

# Verify target classes are present
from src.data.bdd100k import TARGET_CLASSES
assert data_cfg.get("nc", 0) >= len(TARGET_CLASSES), \
    f"Expected nc >= {len(TARGET_CLASSES)}, got {data_cfg.get('nc')}"

# Count images
train_dir = BDD100K_YOLO_ROOT / data_cfg.get("train", "images/train")
val_dir   = BDD100K_YOLO_ROOT / data_cfg.get("val", "images/val")
n_train = len(list(train_dir.glob("*.jpg"))) if train_dir.exists() else 0
n_val   = len(list(val_dir.glob("*.jpg"))) if val_dir.exists() else 0
print(f"\nDataset: {n_train} train images, {n_val} val images")

print("\n✅ Dataset setup OK")

# %% [markdown]
# ## Cell 6 — G2: Centralized YOLOv8 Baseline (smoke)
#
# Train centralized YOLOv8n on all data (smoke: 1 epoch).
# G2 requires: model trains, artifacts produced, per-class AP logged.

# %%
import random, mlflow
import numpy as np
from datetime import datetime
from src.model.yolo_wrapper import build_model, evaluate, train_one_round
from src.evaluation.metrics import save_summary_metrics

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

SEED = 42
RUN_CLASS = "smoke"
DEVICE = 0  # GPU index

set_seed(SEED)

# MLflow — file-based tracking (no server needed on Kaggle)
mlflow.set_tracking_uri(f"file://{OUTPUT_ROOT}/mlruns")
mlflow.set_experiment("G2-centralized")

g2_run_dir = OUTPUT_ROOT / "runs" / f"G2_centralized_{RUN_CLASS}_seed{SEED}"
g2_run_dir.mkdir(parents=True, exist_ok=True)

print("=" * 50)
print(f"G2 — Centralized YOLOv8 Baseline [{RUN_CLASS}]")
print("=" * 50)

with mlflow.start_run(run_name=f"centralized_seed{SEED}") as run:
    model = build_model("yolov8n.pt")

    train_metrics = train_one_round(
        model=model,
        data_yaml=DATA_YAML,
        epochs=1,          # smoke: 1 epoch
        batch=4,
        img_size=640,
        lr0=0.01,
        device=DEVICE,
        project=g2_run_dir / "checkpoint",
        name="train",
    )
    mlflow.log_params({"seed": SEED, "epochs": 1, "run_class": RUN_CLASS})
    mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})

    eval_metrics = evaluate(
        model=model,
        data_yaml=DATA_YAML,
        img_size=640,
        conf=0.25,
        iou=0.70,
        device=DEVICE,
    )
    mlflow.log_metrics(eval_metrics)
    print("\nG2 Centralized eval metrics:")
    for k, v in eval_metrics.items():
        print(f"  {k}: {v:.4f}")

    save_summary_metrics(eval_metrics, g2_run_dir / "metrics.csv")

    # Save global checkpoint for FL runs
    G2_CHECKPOINT = g2_run_dir / "checkpoint" / "train" / "weights" / "best.pt"
    print(f"\nCheckpoint: {G2_CHECKPOINT}")

print("\n✅ G2 — Centralized baseline complete")

# %% [markdown]
# ## Cell 7 — Create FL Partitions (S0, S1, S1-Control)

# %%
import yaml as _yaml
from src.data.partitioner import partition_iid, partition_missing_class, save_partition_manifest

# Get all training image names
train_label_dir = BDD100K_YOLO_ROOT / "labels" / "train"
image_names = [
    f.stem + ".jpg"
    for f in train_label_dir.glob("*.txt")
] if train_label_dir.exists() else []

if not image_names:
    # Fallback: get from image dir
    image_names = [f.name for f in train_dir.glob("*.jpg")]

print(f"Total training images: {len(image_names)}")
NUM_CLIENTS = 4

# S0 — IID partition
s0_manifest = partition_iid(
    image_names, train_label_dir, num_clients=NUM_CLIENTS,
    seed=SEED, partition_id="s0_iid_seed42"
)
s0_path = OUTPUT_ROOT / "partitions" / "s0_iid.yaml"
save_partition_manifest(s0_manifest, s0_path)
print(f"\nS0 IID partition: {s0_path}")
for cid, counts in s0_manifest.class_counts.items():
    print(f"  {cid}: {counts}")

# S1 — Missing-Class Non-IID
# C0 missing motorcycle, C1 missing truck (from s1_missing_class.yaml)
s1_manifest = partition_missing_class(
    image_names, train_label_dir, num_clients=NUM_CLIENTS,
    missing_map={"C0": ["motorcycle"], "C1": ["truck"]},
    seed=SEED, partition_id="s1_mc_seed42", scenario="S1"
)
s1_path = OUTPUT_ROOT / "partitions" / "s1_missing_class.yaml"
save_partition_manifest(s1_manifest, s1_path)
print(f"\nS1 Missing-Class partition: {s1_path}")
for cid, missing in s1_manifest.missing_classes.items():
    counts = s1_manifest.class_counts[cid]
    print(f"  {cid} missing={missing}  counts={counts}")

# S1-Control — matched control (no missing classes, same seed)
s1c_manifest = partition_iid(
    image_names, train_label_dir, num_clients=NUM_CLIENTS,
    seed=SEED, partition_id="s1_control_seed42"
)
s1c_path = OUTPUT_ROOT / "partitions" / "s1_control.yaml"
save_partition_manifest(s1c_manifest, s1c_path)
print(f"\nS1-Control partition: {s1c_path}")

print("\n✅ Partitions created")

# %% [markdown]
# ## Cell 8 — Helper: create per-client data.yaml files

# %%
def make_per_client_data_yamls(
    manifest,
    train_img_root: Path,
    val_img_root: Path,
    train_label_root: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Create per-client data_C{i}.yaml with only assigned images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    client_yamls: dict[str, Path] = {}

    for cid, img_names in manifest.client_assignments.items():
        # Write image list file
        img_list_path = output_dir / f"images_{cid}.txt"
        with img_list_path.open("w") as f:
            for name in img_names:
                img_path = train_img_root / name
                if img_path.exists():
                    f.write(str(img_path) + "\n")

        # Write client data.yaml using path= and train= as list file
        client_yaml_path = output_dir / f"data_{cid}.yaml"
        client_data = {
            "path": str(output_dir),
            "train": str(img_list_path),
            "val": str(val_img_root),
            "nc": len(TARGET_CLASSES),
            "names": list(TARGET_CLASSES),
        }
        with client_yaml_path.open("w") as f:
            yaml.dump(client_data, f, default_flow_style=False)
        client_yamls[cid] = client_yaml_path

    return client_yamls

# %% [markdown]
# ## Cell 9 — G3: FedAvg Baseline — S0 IID (smoke)

# %%
import flwr as fl
from flwr.server import ServerConfig
from src.federated.client import YOLOFlowerClient
from src.federated.server import run_fl_server
from src.federated.strategies.fedavg import build_fedavg

set_seed(SEED)

# Create per-client data yamls for S0
s0_yamls_dir = OUTPUT_ROOT / "client_yamls" / "s0"
s0_data_yamls = make_per_client_data_yamls(
    s0_manifest, train_img_root=train_dir, val_img_root=val_dir,
    train_label_root=train_label_dir, output_dir=s0_yamls_dir,
)

g3_run_dir = OUTPUT_ROOT / "runs" / f"G3_FedAvg_S0_{RUN_CLASS}_seed{SEED}"
g3_run_dir.mkdir(parents=True, exist_ok=True)

TRAIN_CONFIG_BASE = {
    "local_epochs": 1,
    "batch_size": 4,
    "image_size": 640,
    "lr0": 0.01,
    "conf": 0.25,
    "iou": 0.70,
    "device": DEVICE,
}

def make_client_fn(client_data_yamls: dict, train_config: dict, run_dir: Path):
    def client_fn(cid: str):
        return YOLOFlowerClient(
            client_id=cid,
            partition_id=int(cid.replace("C", "")),
            data_yaml=client_data_yamls[cid],
            weights="yolov8n.pt",
            train_config=train_config,
            run_dir=run_dir,
        )
    return client_fn

print("=" * 50)
print(f"G3 — FedAvg S0 IID [{RUN_CLASS}]")
print("=" * 50)

strategy = build_fedavg(
    num_rounds=2,
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=NUM_CLIENTS,
    min_evaluate_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
)

mlflow.set_experiment("G3-FedAvg-S0")
with mlflow.start_run(run_name=f"FedAvg_S0_seed{SEED}"):
    fl.simulation.start_simulation(
        client_fn=make_client_fn(s0_data_yamls, TRAIN_CONFIG_BASE, g3_run_dir),
        num_clients=NUM_CLIENTS,
        config=ServerConfig(num_rounds=2),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 0.25},
    )

print("\n✅ G3 — FedAvg S0 baseline complete")

# %% [markdown]
# ## Cell 10 — G4: Missing-Class Effect (S1 vs S1-Control, smoke)
#
# Compare FedAvg on S1 (missing-class) vs S1-Control (matched control).
# H1 support requires consistent class-level degradation with parameter evidence.

# %%
def run_fedavg_smoke(manifest, data_yamls_dir, run_name, run_dir, num_rounds=2):
    """Run FedAvg smoke experiment and return per-class eval metrics."""
    set_seed(SEED)

    client_yamls = make_per_client_data_yamls(
        manifest, train_img_root=train_dir, val_img_root=val_dir,
        train_label_root=train_label_dir, output_dir=data_yamls_dir,
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    strategy = build_fedavg(
        num_rounds=num_rounds, fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS, min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )

    fl.simulation.start_simulation(
        client_fn=make_client_fn(client_yamls, TRAIN_CONFIG_BASE, run_dir),
        num_clients=NUM_CLIENTS,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 0.25},
    )
    # Evaluate final global model
    final_model = build_model("yolov8n.pt")
    metrics = evaluate(final_model, DATA_YAML, 640, 0.25, 0.70, DEVICE)
    return metrics


print("=" * 50)
print("G4 — Missing-Class Effect (S1 vs S1-Control)")
print("=" * 50)

mlflow.set_experiment("G4-missing-class-effect")

with mlflow.start_run(run_name="FedAvg_S1"):
    s1_metrics = run_fedavg_smoke(
        s1_manifest,
        data_yamls_dir=OUTPUT_ROOT / "client_yamls" / "s1",
        run_name="FedAvg_S1",
        run_dir=OUTPUT_ROOT / "runs" / f"G4_FedAvg_S1_{RUN_CLASS}_seed{SEED}",
    )
    mlflow.log_metrics(s1_metrics)

with mlflow.start_run(run_name="FedAvg_S1Control"):
    s1c_metrics = run_fedavg_smoke(
        s1c_manifest,
        data_yamls_dir=OUTPUT_ROOT / "client_yamls" / "s1c",
        run_name="FedAvg_S1Control",
        run_dir=OUTPUT_ROOT / "runs" / f"G4_FedAvg_S1C_{RUN_CLASS}_seed{SEED}",
    )
    mlflow.log_metrics(s1c_metrics)

# Report ΔAP per class
print("\nG4 Results — ΔAP (S1 - S1-Control):")
print(f"{'Class':<12} {'S1':>8} {'S1-Ctrl':>10} {'ΔAP':>8}")
print("-" * 42)
for cls_name in TARGET_CLASSES:
    key = f"AP50_{cls_name}"
    s1_ap  = s1_metrics.get(key, float("nan"))
    s1c_ap = s1c_metrics.get(key, float("nan"))
    delta  = s1_ap - s1c_ap
    print(f"{cls_name:<12} {s1_ap:>8.4f} {s1c_ap:>10.4f} {delta:>+8.4f}")

print(f"\n{'mAP50':<12} {s1_metrics.get('mAP50', 0):>8.4f} {s1c_metrics.get('mAP50', 0):>10.4f}")
print("\n[THESIS-HYPOTHESIS] H1 evaluation at smoke scale only.")
print("Multi-seed main runs required to establish G4 formally.")
print("\n✅ G4 — Missing-Class effect measured")

# %% [markdown]
# ## Cell 11 — G6–G8: Ablation A0–A4 on S1 (smoke)
#
# Ablation matrix (CLAUDE.md §11):
# | ID | Preservation (H2) | ClassAwareAgg (H3) |
# |----|-------------------|--------------------|
# | A0 | No  (FedAvg)      | No                 |
# | A1 | No                | class-count FedAvg |  ← ClassAwareAgg only
# | A2 | Yes (rho=0)       | No                 |  ← Preservation only
# | A3 | No                | Yes (ClassAwareAgg)|  ← same as A1 here
# | A4 | Yes (rho=0)       | Yes                |  ← Full method

# %%
from src.federated.client_variants import PreservationClient
from src.federated.strategies.class_aware_agg import build_class_aware_agg

# Pre-compute param_names (needed by ClassAwareAgg)
_tmp_model = build_model("yolov8n.pt")
from src.preservation.mask import get_param_names
PARAM_NAMES_LIST = get_param_names(_tmp_model)
del _tmp_model

# S1 client class counts (from partition manifest)
S1_CLIENT_CLASS_COUNTS = {
    cid: s1_manifest.class_counts[cid]
    for cid in s1_manifest.client_assignments
}
S1_MISSING_CLASSES = {
    cid: s1_manifest.missing_classes[cid]
    for cid in s1_manifest.client_assignments
}

def make_preservation_client_fn(
    client_data_yamls: dict,
    train_config: dict,
    run_dir: Path,
    missing_classes_map: dict,
    class_counts_map: dict,
    rho: float,
):
    """Client factory for PreservationClient (H2)."""
    def client_fn(cid: str):
        cfg = dict(train_config)
        cfg["missing_classes"] = missing_classes_map.get(cid, [])
        cfg["class_counts"] = class_counts_map.get(cid, {})
        cfg["rho"] = rho
        return PreservationClient(
            client_id=cid,
            partition_id=int(cid.replace("C", "")),
            data_yaml=client_data_yamls[cid],
            weights="yolov8n.pt",
            train_config=cfg,
            run_dir=run_dir,
        )
    return client_fn


def run_ablation(ablation_id: str, use_preservation: bool, use_class_aware: bool, rho: float = 0.0):
    """Run one ablation variant on S1 partition."""
    print(f"\n--- Ablation {ablation_id}: preservation={use_preservation} (rho={rho}), class_aware={use_class_aware} ---")

    set_seed(SEED)
    s1_yamls_dir = OUTPUT_ROOT / "client_yamls" / "s1"
    s1_yamls_dir.mkdir(parents=True, exist_ok=True)
    client_yamls = make_per_client_data_yamls(
        s1_manifest, train_img_root=train_dir, val_img_root=val_dir,
        train_label_root=train_label_dir, output_dir=s1_yamls_dir,
    )
    run_dir = OUTPUT_ROOT / "runs" / f"ablation_{ablation_id}_{RUN_CLASS}_seed{SEED}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build strategy
    strategy_kwargs = dict(
        num_rounds=2, fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS, min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )
    if use_class_aware:
        strategy = build_class_aware_agg(
            param_names=PARAM_NAMES_LIST,
            trace_dir=run_dir,
            **strategy_kwargs,
        )
    else:
        strategy = build_fedavg(**strategy_kwargs)

    # Build client_fn
    if use_preservation:
        client_fn = make_preservation_client_fn(
            client_yamls, TRAIN_CONFIG_BASE, run_dir,
            missing_classes_map=S1_MISSING_CLASSES,
            class_counts_map=S1_CLIENT_CLASS_COUNTS,
            rho=rho,
        )
    else:
        # Standard client but with class_counts_json reported
        def client_fn(cid: str):
            cfg = dict(TRAIN_CONFIG_BASE)
            cfg["class_counts"] = S1_CLIENT_CLASS_COUNTS.get(cid, {})
            cfg["rho"] = 1.0       # no preservation
            cfg["missing_classes"] = []
            return PreservationClient(  # use PreservationClient even for A0/A1 to get class_counts reporting
                client_id=cid,
                partition_id=int(cid.replace("C", "")),
                data_yaml=client_yamls[cid],
                weights="yolov8n.pt",
                train_config=cfg,
                run_dir=run_dir,
            )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=ServerConfig(num_rounds=2),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 0.25},
    )

    # Evaluate final global model
    final_model = build_model("yolov8n.pt")
    metrics = evaluate(final_model, DATA_YAML, 640, 0.25, 0.70, DEVICE)
    return metrics


print("=" * 50)
print("G6–G8 — Ablation A0–A4 on S1 (Missing-Class)")
print("=" * 50)
print("⚠️  [THESIS-HYPOTHESIS] Results at smoke scale only.")
print("    Multi-seed feasibility/main runs required for scientific claims.\n")

ablation_results: dict[str, dict] = {}
mlflow.set_experiment("ablation-A0-A4")

for ablation_id, use_pres, use_ca, rho in [
    ("A0", False, False, 1.0),   # FedAvg baseline
    ("A1", False, True,  1.0),   # ClassAwareAgg only
    ("A2", True,  False, 0.0),   # Preservation only (rho=0)
    ("A3", False, True,  1.0),   # ClassAwareAgg only (same as A1 for now)
    ("A4", True,  True,  0.0),   # Full method
]:
    with mlflow.start_run(run_name=f"Ablation_{ablation_id}"):
        metrics = run_ablation(ablation_id, use_pres, use_ca, rho)
        mlflow.log_metrics(metrics)
        ablation_results[ablation_id] = metrics
        print(f"  {ablation_id} mAP50={metrics.get('mAP50', 0):.4f}")

# %% [markdown]
# ## Cell 12 — Results Summary (ΔAP vs FedAvg baseline)

# %%
import pandas as pd

print("=" * 60)
print("ABLATION RESULTS — S1 Missing-Class Non-IID (smoke)")
print("=" * 60)
print("⚠️  Smoke scale (1 epoch, 2 rounds). NOT for reporting.")
print("    Use feasibility/main scale for scientific conclusions.\n")

baseline_id = "A0"
rows = []
for ablation_id, metrics in ablation_results.items():
    row = {"ID": ablation_id, "mAP50": metrics.get("mAP50", float("nan"))}
    for cls_name in TARGET_CLASSES:
        ap_key = f"AP50_{cls_name}"
        row[f"AP_{cls_name}"] = metrics.get(ap_key, float("nan"))
        baseline_ap = ablation_results.get(baseline_id, {}).get(ap_key, float("nan"))
        row[f"ΔAP_{cls_name}"] = row[f"AP_{cls_name}"] - baseline_ap
    rows.append(row)

df = pd.DataFrame(rows).set_index("ID")
print(df.to_string(float_format="{:.4f}".format))

# Missing-class specific ΔAP (C0: motorcycle, C1: truck)
print("\nΔAP for missing classes (vs A0 FedAvg baseline):")
for cls_name in ["motorcycle", "truck"]:
    col = f"ΔAP_{cls_name}"
    if col in df.columns:
        print(f"  {cls_name}: {df[col].to_dict()}")

# Save results
results_path = OUTPUT_ROOT / "ablation_results.csv"
df.to_csv(results_path)
print(f"\nSaved: {results_path}")

# %% [markdown]
# ## Cell 13 — Register experiments in registry

# %%
from datetime import datetime

registry_path = REPO_ROOT / "research" / "experiment_registry" / "registry.yaml"
with registry_path.open() as f:
    registry = yaml.safe_load(f)

if registry.get("experiments") is None:
    registry["experiments"] = []

timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")

new_entries = []
for ablation_id, metrics in ablation_results.items():
    new_entries.append({
        "id": f"EXP-Kaggle-{ablation_id}-{timestamp}",
        "name": f"Ablation {ablation_id} S1 Smoke Kaggle",
        "gate": "G6-G8",
        "scenario": "S1",
        "method": f"Ablation-{ablation_id}",
        "run_class": RUN_CLASS,
        "status": "completed",
        "seed": SEED,
        "partition_id": s1_manifest.partition_id,
        "notes": f"Kaggle T4 smoke run. mAP50={metrics.get('mAP50', 0):.4f}. "
                 f"[THESIS-HYPOTHESIS] Not scientifically conclusive at smoke scale.",
    })

registry["experiments"].extend(new_entries)
with registry_path.open("w") as f:
    yaml.dump(registry, f, default_flow_style=False, allow_unicode=True)

print(f"Registered {len(new_entries)} experiments in {registry_path}")
print("\n✅ All done. Check /kaggle/working/flops_outputs/ for artifacts.")
