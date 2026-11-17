# %% [markdown]
# # FLOPS — Notebook 10: ceiling test 1a, retrain Detect.cv3 on pooled labelled data (ADR-038)
#
# Round-20 pairs M (missing) / I (ideal): YOLO11m S26 / S27 (lane 0, GPU 0), YOLO11s S21b / S28 (lane 1, GPU 1).
# Per model: identity copies of M and I, M_rt and I_rt (cv3 retrained, recipe fixed in ADR-038), global evals,
# size-split dumps, bootstrap. Lane 1 then measures the ADR-036 readout on A0c S32a / S32b (descriptive).

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
# ## Cell 2 — Data, central partition, lanes, bootstrap

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
PARTITIONS_DIR = DATA_TMP / "partitions"
OUT = WORK / "flops_export" / "retrain"
DUMP = OUT / "dumps"
FEAT = OUT / "features"
LOGS = WORK / "flops_export" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
if not (YOLO_ROOT / "data.yaml").exists():
    subprocess.check_call([sys.executable, str(REPO_ROOT / "scripts" / "prepare_bdd100k.py"),
                           "--data-root", str(find_bdd100k_root()), "--output-root", str(YOLO_ROOT)])
DATA_YAML = YOLO_ROOT / "data.yaml"
for cfg in ("s1b5_bus_3k_allmoto_seed42",       # the central partition re-deals this one's images (ADR-026)
            "s1b5_central_3k_allmoto_seed42"):
    subprocess.check_call([sys.executable, str(REPO_ROOT / "scripts" / "generate_partition.py"), "--partition-config",
                           str(REPO_ROOT / "configs" / "partition" / f"{cfg}.yaml"),
                           "--yolo-root", str(YOLO_ROOT), "--output-dir", str(PARTITIONS_DIR)])
CENTRAL_YAML = PARTITIONS_DIR / "s1b5_central_3k_allmoto_seed42" / "data_C0.yaml"
assert CENTRAL_YAML.exists(), CENTRAL_YAML
import yaml
_imgs = lambda p: sorted(sum(yaml.safe_load((PARTITIONS_DIR / p / "manifest.yaml").read_text())["client_assignments"].values(), []))   # noqa: E731
assert _imgs("s1b5_central_3k_allmoto_seed42") == _imgs("s1b5_bus_3k_allmoto_seed42"), "central != union of clients"
print("central partition:", len(_imgs("s1b5_central_3k_allmoto_seed42")), "images = union of the 5 clients")
CKPTS = Path("/kaggle/input")
UNZIP = DATA_TMP / "ckpt_zip"
RUNS = DATA_TMP / "retrain_runs"


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
RT = str(REPO_ROOT / "scripts" / "retrain_cls_head.py")
GE = str(REPO_ROOT / "scripts" / "eval_global_checkpoint.py")
SS = str(REPO_ROOT / "scripts" / "eval_size_split.py")
EX = str(REPO_ROOT / "scripts" / "extract_probe_features.py")
PAIRS = {"11m": ("coco:yolo11m.pt", run_dir("P4t1m960_feasibility_seed42_s1b5_bus"),
                 run_dir("P4t1m960_feasibility_seed42_s1b5_pooled")),
         "11s": ("coco:yolo11s.pt", run_dir("P4t1i960_feasibility_seed42_s1b5_bus"),
                 run_dir("P4t1i960_feasibility_seed42_s1b5_pooled"))}
A0C = {"a0c_m": run_dir("A0cm960_feasibility_seed42_s1b5_bus"),
       "a0c_i": run_dir("A0cm960_feasibility_seed42_s1b5_pooled")}
for model, (w, m, i) in PAIRS.items():   # identity copies: in-session M and I go through the same scripts
    for name, rec in ((f"T{model}_M", m), (f"T{model}_I", i)):
        subprocess.check_call([sys.executable, TP, "--recipient", str(rec), "--donor", str(rec), "--weights", w,
                               "--cls", "bus", "--out", str(RUNS / model / name)])
from src.data.coco_view import ensure_coco_root
ensure_coco_root(YOLO_ROOT)   # built once: the two lanes would otherwise race to create it


def steps(dev: int, model: str) -> list[tuple[str, list[str]]]:
    w, m, i = PAIRS[model]
    out = [(f"rt_{model}_{tag}", [sys.executable, RT, "--run", str(src), "--weights", w,
                                  "--data-yaml", str(CENTRAL_YAML), "--out", str(RUNS / model / f"T{model}_RT{tag}"),
                                  "--device", str(dev)])
           for tag, src in (("M", m), ("I", i))]
    names = [f"T{model}_{s}" for s in ("M", "I", "RTM", "RTI")]
    out += [(f"ge_{n}", [sys.executable, GE, "--run", str(RUNS / model / n), "--round", "20", "--weights", w,
                         "--device", str(dev), "--data-yaml", str(DATA_YAML),
                         "--view-dir", str(DATA_TMP / f"view{dev}"), "--out", str(OUT / f"ge_{n}.json")])
            for n in names]
    out.append((f"ss_{model}", [sys.executable, SS, "--data-yaml", str(DATA_YAML), "--roots", str(RUNS / model),
                                "--rounds", "20", "--weights", w, "--include", f"T{model}_", "--imgsz", "960",
                                "--device", str(dev), "--dump", str(DUMP), "--out", str(OUT / f"ss_{model}.csv"),
                                "--extract-to", str(DATA_TMP / f"x{dev}")]))
    return out


def a0c_steps(dev: int) -> list[tuple[str, list[str]]]:
    feats = [("coco11m", "init")] + [(t, str(d / "checkpoint" / "global_round_020.npz")) for t, d in A0C.items()]
    return [(f"feat_{t}", [sys.executable, EX, "--data-yaml", str(DATA_YAML), "--weights", "coco:yolo11m.pt",
                           "--ckpt", c, "--tag", t, "--out", str(FEAT), "--device", str(dev)]) for t, c in feats]


status: dict[str, str] = {}
lock = threading.Lock()


def lane(todo: list[tuple[str, list[str]]]) -> None:
    for name, cmd in todo:
        try:
            run_logged(cmd, LOGS / f"{name}.log", timeout=4 * 3600, stall_timeout=3600,
                       watch_dir=WORK / "flops_export")
            status[name] = "completed"
        except Exception as exc:
            status[name] = f"failed: {exc}"
        with lock:
            print(name, status[name], f"{(time.time() - SESSION_T0) / 3600:.2f} h", "| packed:",
                  finalize_outputs(WORK, drop=False))


LANES = {0: steps(0, "11m"), 1: steps(1, "11s") + a0c_steps(1)}
threads = [threading.Thread(target=lane, args=(todo,)) for todo in LANES.values()]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(status)

if all(status.get(f"feat_{t}") == "completed" for t in ("coco11m", "a0c_m", "a0c_i")):
    run_logged([sys.executable, str(REPO_ROOT / "scripts" / "probe_degradation.py"), "--features", str(FEAT),
                "--pair", "11mA0c", "a0c_m", "a0c_i", "--ref", "11m", "coco11m", "--out", str(OUT / "a0c_readout")],
               LOGS / "a0c_readout.log", timeout=3 * 3600, stall_timeout=3600, watch_dir=WORK / "flops_export")
BC = str(REPO_ROOT / "scripts" / "bootstrap_ci.py")
pairs = []
for model in PAIRS:
    def tag(s: str, model: str = model) -> str:
        return f"T{model}_{s}_r20_960"
    pairs += [("--pair", tag("M"), tag("I")), ("--pair", tag("RTM"), tag("RTI")), ("--pair", tag("M"), tag("RTM"))]
if all((DUMP / f"{p[1]}.npz").exists() and (DUMP / f"{p[2]}.npz").exists() for p in pairs):
    run_logged([sys.executable, BC, "--dump", str(DUMP), *[x for p in pairs for x in p], "--classes", "bus",
                "--sizes", "all", "--n", "1000", "--out", str(OUT / "bootstrap_ci.csv")],
               LOGS / "bootstrap.log", timeout=3 * 3600, stall_timeout=3600, watch_dir=WORK / "flops_export")
for f in sorted(OUT.glob("ge_*.json")):
    r = json.loads(f.read_text())
    print(f.stem, {k: round(r[k], 4) for k in ("mAP50", "AP50_car", "AP50_bus", "AP50_truck", "AP50_motorcycle")})
print("packed:", finalize_outputs(WORK, drop=False))
print(f"session {(time.time() - SESSION_T0) / 3600:.2f} h")
