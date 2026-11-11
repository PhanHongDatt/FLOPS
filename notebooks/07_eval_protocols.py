# %% [markdown]
# # FLOPS — Notebook 07: evaluation-protocol study on saved checkpoints (ADR-034, no training)
#
# Round-20 checkpoints of S26 (YOLO11m, missing), S27 (YOLO11m, ideal), S29 (YOLO11m + gated CBAM, missing):
# P0 the registered protocol at 960 (size split, per-detection dumps for bootstrap CIs, fixed multi-label NMS),
# P1 NMS over the four target classes only, P2 evaluation at 1280 px. Two lanes, one per T4.

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
# ## Cell 2 — Data, checkpoints, two evaluation lanes

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
import threading

from src.utils.kaggle_finalize import finalize_outputs

OUT_P = OUT / "eval_protocols"
DUMP = OUT_P / "dumps"
S26 = _run_dir("P4t1m960_feasibility_seed42_s1b5_bus")
_run_dir("P4t1m960_feasibility_seed42_s1b5_pooled")
SS, GE = str(REPO_ROOT / "scripts" / "eval_size_split.py"), str(REPO_ROOT / "scripts" / "eval_global_checkpoint.py")
Y11M, CBAM = "coco:yolo11m.pt", "coco:yolo11m.pt+cbam"


def size(dev: int, weights: str, include: str, imgsz: int, t4: bool, name: str) -> tuple[str, list[str]]:
    cmd = [sys.executable, SS, "--data-yaml", str(DATA_YAML), "--roots", *map(str, CKPTS_ALL), "--rounds", "20",
           "--weights", weights, "--include", include, "--imgsz", str(imgsz), "--device", str(dev),
           "--dump", str(DUMP), "--out", str(OUT_P / f"{name}.csv"), "--extract-to", str(DATA_TMP / f"x_{name}")]
    return name, cmd + (["--only-targets"] if t4 else [])


def glob_eval(dev: int, run: Path, weights: str, imgsz: int, t4: bool, name: str) -> tuple[str, list[str]]:
    cmd = [sys.executable, GE, "--run", str(run), "--round", "20", "--weights", weights, "--imgsz", str(imgsz),
           "--device", str(dev), "--data-yaml", str(DATA_YAML), "--view-dir", str(DATA_TMP / f"view{dev}"),
           "--out", str(OUT_P / f"{name}.json")]
    return name, cmd + (["--only-targets"] if t4 else [])


LANES = {
    0: [size(0, Y11M, "P4t1m960", 960, False, "ss_y11m_960"),            # S26 + S27, P0 (also A5: bus by size)
        size(0, CBAM, "P4t1mcbam960", 960, False, "ss_cbam_960"),
        size(0, Y11M, "P4t1m960_feasibility_seed42_s1b5_bus", 960, True, "ss_s26_960_t4"),
        glob_eval(0, S26, Y11M, 960, True, "ge_s26_960_t4"),
        glob_eval(0, S29, CBAM, 960, True, "ge_s29_960_t4")],
    1: [size(1, Y11M, "P4t1m960", 1280, False, "ss_y11m_1280"),
        size(1, CBAM, "P4t1mcbam960", 1280, False, "ss_cbam_1280"),
        glob_eval(1, S26, Y11M, 1280, False, "ge_s26_1280"),
        glob_eval(1, S29, CBAM, 1280, False, "ge_s29_1280")],
}
from src.data.coco_view import ensure_coco_root
ensure_coco_root(YOLO_ROOT)   # built once here: the two lanes would otherwise race to create it
status: dict[str, str] = {}
lock = threading.Lock()


def lane(steps: list[tuple[str, list[str]]]) -> None:
    for name, cmd in steps:
        try:   # one failed step must not cost the others
            run_logged(cmd, LOGS / f"{name}.log", timeout=3 * 3600, stall_timeout=3600, watch_dir=OUT)
            status[name] = "completed"
        except Exception as exc:
            status[name] = f"failed: {exc}"
        with lock:
            print(name, status[name], "| packed:", finalize_outputs(WORK, drop=False))


threads = [threading.Thread(target=lane, args=(steps,)) for steps in LANES.values()]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(status)
print("packed:", finalize_outputs(WORK, drop=False))
