# %% [markdown]
# # FLOPS — Notebook 09: bus-row transplant (ADR-037, no training)
#
# Round-20 checkpoints M (missing) and I (ideal) of YOLO11m (S26/S27) and YOLO11s (S21b/S28): M, I, M+rowI, I+rowM,
# where rowX is the bus row of Detect.cv3[l][2] from X. Global evaluator + size-split dumps for bootstrap CIs.
# Lane 0 = YOLO11m on GPU 0, lane 1 = YOLO11s on GPU 1.

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
# ## Cell 2 — Transplants, evaluation lanes, bootstrap

# %%
import json
import threading
import zipfile

from src.utils.kaggle_finalize import finalize_outputs
from src.utils.kaggle_paths import find_bdd100k_root
from src.utils.proc import run_logged

WORK = Path("/kaggle/working")
DATA_TMP = Path("/kaggle/tmp/data")
YOLO_ROOT = DATA_TMP / "bdd100k_yolo"
OUT = WORK / "flops_export" / "transplant"
DUMP = OUT / "dumps"
LOGS = WORK / "flops_export" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
if not (YOLO_ROOT / "data.yaml").exists():
    subprocess.check_call([sys.executable, str(REPO_ROOT / "scripts" / "prepare_bdd100k.py"),
                           "--data-root", str(find_bdd100k_root()), "--output-root", str(YOLO_ROOT)])
DATA_YAML = YOLO_ROOT / "data.yaml"
CKPTS = Path("/kaggle/input")
UNZIP = DATA_TMP / "ckpt_zip"
RUNS = DATA_TMP / "transplant_runs"


def run_dir(prefix: str) -> Path:
    """Run folder (config.yaml + checkpoint/global_round_020.npz) whose name starts with ``prefix``."""
    def hits(base: Path) -> list[Path]:
        return [p.parent.parent for p in base.glob("**/global_round_020.npz")
                if p.parent.parent.name.startswith(prefix) and (p.parent.parent / "config.yaml").exists()]
    for base in (CKPTS, UNZIP):
        if hits(base):
            return hits(base)[0]
    for z in CKPTS.glob("**/*.zip"):
        with zipfile.ZipFile(z) as f:
            if z.stem.startswith(prefix) or any(f"/{prefix}" in f"/{n}" for n in f.namelist()):
                f.extractall(UNZIP / z.stem)
    found = hits(UNZIP)
    if not found:
        raise FileNotFoundError(f"no round-20 run {prefix} under {CKPTS}")
    return found[0]


TP = str(REPO_ROOT / "scripts" / "transplant_rows.py")
PAIRS = {"11m": ("coco:yolo11m.pt", run_dir("P4t1m960_feasibility_seed42_s1b5_bus"),
                 run_dir("P4t1m960_feasibility_seed42_s1b5_pooled")),
         "11s": ("coco:yolo11s.pt", run_dir("P4t1i960_feasibility_seed42_s1b5_bus"),
                 run_dir("P4t1i960_feasibility_seed42_s1b5_pooled"))}
for model, (w, m, i) in PAIRS.items():
    for name, rec, don in ((f"T{model}_M", m, m), (f"T{model}_I", i, i),
                           (f"T{model}_MrowI", m, i), (f"T{model}_IrowM", i, m)):
        subprocess.check_call([sys.executable, TP, "--recipient", str(rec), "--donor", str(don), "--weights", w,
                               "--cls", "bus", "--out", str(RUNS / model / name)])

GE = str(REPO_ROOT / "scripts" / "eval_global_checkpoint.py")
SS = str(REPO_ROOT / "scripts" / "eval_size_split.py")
from src.data.coco_view import ensure_coco_root
ensure_coco_root(YOLO_ROOT)   # built once: the two lanes would otherwise race to create it


def steps(dev: int, model: str) -> list[tuple[str, list[str]]]:
    w = PAIRS[model][0]
    out = [(f"ge_{d.name}", [sys.executable, GE, "--run", str(d), "--round", "20", "--weights", w,
                             "--device", str(dev), "--data-yaml", str(DATA_YAML),
                             "--view-dir", str(DATA_TMP / f"view{dev}"), "--out", str(OUT / f"ge_{d.name}.json")])
           for d in sorted((RUNS / model).iterdir())]
    out.append((f"ss_{model}", [sys.executable, SS, "--data-yaml", str(DATA_YAML), "--roots", str(RUNS / model),
                                "--rounds", "20", "--weights", w, "--include", f"T{model}_", "--imgsz", "960",
                                "--device", str(dev), "--dump", str(DUMP), "--out", str(OUT / f"ss_{model}.csv"),
                                "--extract-to", str(DATA_TMP / f"x{dev}")]))
    return out


status: dict[str, str] = {}
lock = threading.Lock()


def lane(todo: list[tuple[str, list[str]]]) -> None:
    for name, cmd in todo:
        try:
            run_logged(cmd, LOGS / f"{name}.log", timeout=3 * 3600, stall_timeout=3600, watch_dir=WORK / "flops_export")
            status[name] = "completed"
        except Exception as exc:
            status[name] = f"failed: {exc}"
        with lock:
            print(name, status[name], "| packed:", finalize_outputs(WORK, drop=False))


threads = [threading.Thread(target=lane, args=(steps(dev, model),)) for dev, model in ((0, "11m"), (1, "11s"))]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(status)

BC = str(REPO_ROOT / "scripts" / "bootstrap_ci.py")
pairs = []
for model in PAIRS:
    tag = lambda s: f"T{model}_{s}_r20_960"   # noqa: E731
    pairs += [("--pair", tag("M"), tag("MrowI")), ("--pair", tag("IrowM"), tag("I")), ("--pair", tag("M"), tag("I"))]
if all((DUMP / f"{p[1]}.npz").exists() and (DUMP / f"{p[2]}.npz").exists() for p in pairs):
    run_logged([sys.executable, BC, "--dump", str(DUMP), *[x for p in pairs for x in p], "--classes", "bus",
                "--sizes", "all", "--n", "1000", "--out", str(OUT / "bootstrap_ci.csv")],
               LOGS / "bootstrap.log", timeout=3 * 3600, stall_timeout=3600, watch_dir=WORK / "flops_export")
for f in sorted(OUT.glob("ge_*.json")):
    r = json.loads(f.read_text())
    print(f.stem, {k: round(r[k], 4) for k in ("mAP50", "AP50_car", "AP50_bus", "AP50_truck", "AP50_motorcycle")})
print("packed:", finalize_outputs(WORK, drop=False))
print(f"session {(time.time() - SESSION_T0) / 3600:.2f} h")
