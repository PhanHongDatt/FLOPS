# %% [markdown]
# # FLOPS — Notebook 11: ADR-040 diagnostic on saved round-20 checkpoints (no training)
#
# D1 runs twice: multi-label NMS (as the global evaluator) and top-1 NMS (as S14).
# YOLO11m 960 px: D1 bus table + truck FP split for A0c (3 seeds x 2 scenarios) and P4t1 (2 seeds x 2), and
# per-detection PR dumps for A0c. YOLO11s 640 px (labelled YOLO11s): D1 for A0c / P4noKD on s5, s5bus1 and pooled.
# Two lanes, one per T4.

# ## Cell 1 — Environment restore


# %%
import subprocess
import sys
import time
from pathlib import Path

SESSION_T0 = time.time()

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
# ## Cell 2 — Data, checkpoints, two lanes

# %%
import threading
import zipfile

from src.utils.kaggle_finalize import finalize_outputs
from src.utils.kaggle_paths import find_bdd100k_root
from src.utils.proc import run_logged

WORK = Path("/kaggle/working")
DATA_TMP = Path("/kaggle/tmp/data")
YOLO_ROOT = DATA_TMP / "bdd100k_yolo"
OUT = WORK / "flops_export" / "adr040"
LOGS = OUT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
if not (YOLO_ROOT / "data.yaml").exists():
    subprocess.check_call([sys.executable, str(REPO_ROOT / "scripts" / "prepare_bdd100k.py"),
                           "--data-root", str(find_bdd100k_root()), "--output-root", str(YOLO_ROOT)])
DATA_YAML = YOLO_ROOT / "data.yaml"
CKPTS = Path("/kaggle/input")


def size_root(tag: str) -> Path:
    """The y11m / y11s folder of the checkpoint dataset, whether Kaggle kept it as a folder or a zip."""
    for d in CKPTS.glob(f"**/{tag}"):
        if d.is_dir() and any(d.glob("**/environment.json")):
            return d
    for z in CKPTS.glob(f"**/{tag}.zip"):
        dest = DATA_TMP / "ckpt" / tag
        with zipfile.ZipFile(z) as f:
            f.extractall(dest)
        return dest
    raise FileNotFoundError(f"no {tag} folder or zip under {CKPTS}")


MODE = "s40d"   # sessions override: "r1" reads only the YOLO11m P4noKDt1 checkpoints (ADR-040 R1)
Y11M = size_root("y11m")
Y11S = size_root("y11s") if MODE == "s40d" else None
print("runs:", sorted(p.parent.name for p in Y11M.glob("**/environment.json")),
      sorted(p.parent.name for p in Y11S.glob("**/environment.json")) if Y11S else [])
AE, SS = str(REPO_ROOT / "scripts" / "analyze_errors.py"), str(REPO_ROOT / "scripts" / "eval_size_split.py")


def d1(dev: int, root: Path, weights: str, imgsz: int, name: str, ml: bool) -> tuple[str, list[str]]:
    """D1 at conf 0.25; ml = multi-label NMS as in the global evaluator, else top-1 class as in S14."""
    name = f"{name}_{'ml' if ml else 'top1'}"
    return name, [sys.executable, AE, "--data-yaml", str(DATA_YAML), "--roots", str(root), "--round", "20",
                  "--targets", "bus", "truck", "--imgsz", str(imgsz), "--weights", weights, "--batch", "16",
                  "--device", str(dev), "--out", str(OUT / f"{name}.csv"), "--extract-to", str(DATA_TMP / f"x_{name}")] + (["--multi-label"] if ml else [])


def dumps(dev: int, root: Path, include: str, name: str) -> tuple[str, list[str]]:
    return name, [sys.executable, SS, "--data-yaml", str(DATA_YAML), "--roots", str(root), "--rounds", "20",
                  "--weights", "coco:yolo11m.pt", "--include", include, "--imgsz", "960", "--device", str(dev),
                  "--dump", str(OUT / "dumps"), "--out", str(OUT / f"{name}.csv"),
                  "--extract-to", str(DATA_TMP / f"x_{name}")]


if MODE == "s40d":
    LANES = {
        0: [d1(0, Y11M, "coco:yolo11m.pt", 960, "d1_y11m_r20", True),
            d1(0, Y11S, "coco:yolo11s.pt", 640, "d1_y11s_r20", True),
            d1(0, Y11S, "coco:yolo11s.pt", 640, "d1_y11s_r20", False)],
        1: [dumps(1, Y11M, "A0cm960", "ss_a0c_y11m_r20"),
            d1(1, Y11M, "coco:yolo11m.pt", 960, "d1_y11m_r20", False)],
    }
else:   # r1: D1 in both NMS modes on lane 0, PR dumps for the per-class thresholds on lane 1
    LANES = {
        0: [d1(0, Y11M, "coco:yolo11m.pt", 960, "d1_r1_y11m_r20", True),
            d1(0, Y11M, "coco:yolo11m.pt", 960, "d1_r1_y11m_r20", False)],
        1: [dumps(1, Y11M, "P4noKDt1m960", "ss_r1_y11m_r20")],
    }
from src.data.coco_view import ensure_coco_root
ensure_coco_root(YOLO_ROOT)   # built once here: the two lanes would otherwise race to create it
status: dict[str, str] = {}
lock = threading.Lock()


def lane(steps: list[tuple[str, list[str]]]) -> None:
    for name, cmd in steps:
        try:   # one failed step must not cost the others
            run_logged(cmd, LOGS / f"{name}.log", timeout=4 * 3600, stall_timeout=3600, watch_dir=OUT)
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
print(status, f"{(time.time() - SESSION_T0) / 3600:.2f} h")
print("packed:", finalize_outputs(WORK, drop=False))
