# %% [markdown]
# # FLOPS — Notebook 08: degradation-localisation probe features (ADR-036, no training)
#
# Box-centre features (neck, cv3[l][1], readout) on BDD100K val for: COCO initial, S26 (missing) and S27 (ideal)
# on YOLO11m; COCO initial, S21b (missing) and S28 (ideal) on YOLO11s; round-20 checkpoints. Two lanes, one per
# T4. The probes and the decision rule then run on CPU in the same session (also re-runnable locally).

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
# ## Cell 2 — Data, checkpoints, feature extraction in two lanes, probe analysis

# %%
import threading
import zipfile

from src.utils.kaggle_finalize import finalize_outputs
from src.utils.kaggle_paths import find_bdd100k_root
from src.utils.proc import run_logged

WORK = Path("/kaggle/working")
DATA_TMP = Path("/kaggle/tmp/data")
YOLO_ROOT = DATA_TMP / "bdd100k_yolo"
OUT = WORK / "flops_export" / "probe"
FEAT = OUT / "features"
LOGS = WORK / "flops_export" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
FEAT.mkdir(parents=True, exist_ok=True)
if not (YOLO_ROOT / "data.yaml").exists():
    subprocess.check_call([sys.executable, str(REPO_ROOT / "scripts" / "prepare_bdd100k.py"),
                           "--data-root", str(find_bdd100k_root()), "--output-root", str(YOLO_ROOT)])
DATA_YAML = YOLO_ROOT / "data.yaml"
CKPTS = Path("/kaggle/input")
UNZIP = DATA_TMP / "ckpt_zip"


def ckpt(prefix: str) -> Path:
    """global_round_020.npz of the run whose folder starts with ``prefix`` (folders or zip uploads)."""
    for base in (CKPTS, UNZIP):
        hits = [p for p in base.glob("**/global_round_020.npz") if p.parent.parent.name.startswith(prefix)]
        if hits:
            return hits[0]
    for z in CKPTS.glob("**/*.zip"):
        with zipfile.ZipFile(z) as f:
            if z.stem.startswith(prefix) or any(f"/{prefix}" in f"/{n}" for n in f.namelist()):
                f.extractall(UNZIP / z.stem)
    hits = [p for p in UNZIP.glob("**/global_round_020.npz")
            if prefix in str(p.relative_to(UNZIP))]
    if not hits:
        raise FileNotFoundError(f"no round-20 checkpoint for {prefix} under {CKPTS}")
    return hits[0]


EX = str(REPO_ROOT / "scripts" / "extract_probe_features.py")
RUNS = {   # tag: (weights, checkpoint or 'init')
    "coco11m": ("coco:yolo11m.pt", "init"),
    "s26": ("coco:yolo11m.pt", ckpt("P4t1m960_feasibility_seed42_s1b5_bus")),
    "s27": ("coco:yolo11m.pt", ckpt("P4t1m960_feasibility_seed42_s1b5_pooled")),
    "coco11s": ("coco:yolo11s.pt", "init"),
    "s21b": ("coco:yolo11s.pt", ckpt("P4t1i960_feasibility_seed42_s1b5_bus")),
    "s28": ("coco:yolo11s.pt", ckpt("P4t1i960_feasibility_seed42_s1b5_pooled")),
}
for tag, (w, c) in RUNS.items():
    print(tag, w, c)


def step(dev: int, tag: str) -> tuple[str, list[str]]:
    w, c = RUNS[tag]
    return tag, [sys.executable, EX, "--data-yaml", str(DATA_YAML), "--weights", w, "--ckpt", str(c),
                 "--tag", tag, "--out", str(FEAT), "--device", str(dev)]


LANES = {0: [step(0, t) for t in ("s26", "s27", "coco11m")],
         1: [step(1, t) for t in ("s21b", "s28", "coco11s")]}
status: dict[str, str] = {}
lock = threading.Lock()


def lane(steps: list[tuple[str, list[str]]]) -> None:
    for name, cmd in steps:
        try:
            run_logged(cmd, LOGS / f"{name}.log", timeout=3 * 3600, stall_timeout=3600, watch_dir=WORK / "flops_export")
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

if all(v == "completed" for v in status.values()):
    run_logged([sys.executable, str(REPO_ROOT / "scripts" / "probe_degradation.py"), "--features", str(FEAT),
                "--pair", "11m", "s26", "s27", "--pair", "11s", "s21b", "s28",
                "--ref", "11m", "coco11m", "--ref", "11s", "coco11s", "--out", str(OUT)],
               LOGS / "probe_analysis.log", timeout=3 * 3600, stall_timeout=3600, watch_dir=WORK / "flops_export")
    print((OUT / "decision.json").read_text())
print("packed:", finalize_outputs(WORK, drop=False))
print(f"session {(time.time() - SESSION_T0) / 3600:.2f} h")
