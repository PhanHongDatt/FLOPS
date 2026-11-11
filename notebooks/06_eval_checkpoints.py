# %% [markdown]
# # FLOPS — Notebook 06: evaluate saved checkpoints on GPU (no training)
#
# ADR-032: (1) the round-20 global eval of s29 (YOLO11m + gated CBAM), cut by the arm timeout after training
# finished; (2) AP50 by COCO object size for S26 / S27 (YOLO11m) and S29, rounds 10 and 20. Checkpoints come
# from a private dataset (an ERROR kernel's output cannot be mounted). Same evaluator as the runs.

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
# ## Cell 2 — Data and evaluations

# %%
from src.utils.kaggle_paths import find_bdd100k_root
from src.utils.proc import run_logged

WORK = Path("/kaggle/working")
DATA_TMP = Path("/kaggle/tmp/data")
YOLO_ROOT = DATA_TMP / "bdd100k_yolo"
OUT = WORK / "flops_export"
LOGS = OUT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
if not (YOLO_ROOT / "data.yaml").exists():
    subprocess.check_call([sys.executable, str(REPO_ROOT / "scripts" / "prepare_bdd100k.py"),
                           "--data-root", str(find_bdd100k_root()), "--output-root", str(YOLO_ROOT)])
DATA_YAML = YOLO_ROOT / "data.yaml"
CKPTS = Path("/kaggle/input")

import zipfile


def _run_dir(prefix: str) -> Path:
    """The run folder from the dataset, whether Kaggle kept the upload as folders or as zip archives."""
    for d in CKPTS.glob(f"**/{prefix}*"):
        if (d / "config.yaml").exists():
            return d
    for z in CKPTS.glob("**/*.zip"):
        with zipfile.ZipFile(z) as f:
            members = [n for n in f.namelist() if f"/{prefix}" in f"/{n}"]
            if members:
                f.extractall(DATA_TMP / "ckpt_zip", members)
                return next(p for p in (DATA_TMP / "ckpt_zip").glob(f"**/{prefix}*") if (p / "config.yaml").exists())
    raise FileNotFoundError(f"no run {prefix}* under {CKPTS}")


S29 = _run_dir("P4t1mcbam960_")
_run_dir("P4t1m960_")   # unpack S26 / S27 too if they came as zips (eval_size_split reads folders and zips)
(DATA_TMP / "ckpt_zip").mkdir(parents=True, exist_ok=True)
CKPTS_ALL = [CKPTS, DATA_TMP / "ckpt_zip"]
steps = {
    "s29_round20": [sys.executable, str(REPO_ROOT / "scripts" / "eval_global_checkpoint.py"),
                    "--run", str(S29), "--round", "20", "--weights", "coco:yolo11m.pt+cbam",
                    "--data-yaml", str(DATA_YAML), "--out", str(OUT / "s29_round20.json")],
    "size_y11m": [sys.executable, str(REPO_ROOT / "scripts" / "eval_size_split.py"),
                  "--data-yaml", str(DATA_YAML), "--roots", *map(str, CKPTS_ALL), "--rounds", "10", "20",
                  "--weights", "coco:yolo11m.pt", "--include", "P4t1m960", "--imgsz", "960",
                  "--out", str(OUT / "size_split_y11m.csv"), "--extract-to", str(DATA_TMP / "x1")],
    "size_y11m_cbam": [sys.executable, str(REPO_ROOT / "scripts" / "eval_size_split.py"),
                       "--data-yaml", str(DATA_YAML), "--roots", *map(str, CKPTS_ALL), "--rounds", "10", "20",
                       "--weights", "coco:yolo11m.pt+cbam", "--include", "P4t1mcbam960", "--imgsz", "960",
                       "--out", str(OUT / "size_split_y11m_cbam.csv"), "--extract-to", str(DATA_TMP / "x2")],
}
from src.utils.kaggle_finalize import finalize_outputs
for name, cmd in steps.items():
    try:   # one failed step must not cost the others
        run_logged(cmd, LOGS / f"{name}.log", timeout=3 * 3600, stall_timeout=3600, watch_dir=OUT)
        print(f"{name}: completed")
    except Exception as exc:
        print(f"⚠️  {name} failed: {exc}")
    print("packed:", finalize_outputs(WORK, drop=False))
