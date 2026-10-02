"""ADR-008 runtime check: do trained client weights stay fp32 on this device?

On CUDA with AMP, Ultralytics fp16-rounds the EMA during the final-epoch
validation; train_one_round must return the fp32 snapshot taken before it.
CPU tests can only simulate that path — run this once on the Kaggle GPU.

Trains one epoch on a few synthetic 64-px images (no dataset needed) and
exits 1 if the returned weights are fp16-representable or did not change.

Usage:  python research/feasibility/F1/check_fp32_upload.py --device 0
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data.bdd100k import TARGET_CLASSES
from src.model.yolo_wrapper import build_model, train_one_round


def _synthetic_dataset(root: Path, n: int = 8) -> Path:
    from PIL import Image

    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(n):
            Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(
                root / "images" / split / f"{i}.jpg")
            (root / "labels" / split / f"{i}.txt").write_text(
                f"{i % len(TARGET_CLASSES)} 0.5 0.5 0.4 0.4\n")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(root.resolve()), "train": "images/train", "val": "images/val",
        "nc": len(TARGET_CLASSES), "names": list(TARGET_CLASSES),
    }))
    return data_yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="0")
    ap.add_argument("--weights", default="yolov8n.pt")
    args = ap.parse_args()
    device = int(args.device) if args.device.isdigit() else args.device

    with tempfile.TemporaryDirectory() as tmp:
        data_yaml = _synthetic_dataset(Path(tmp) / "ds")
        model = build_model(args.weights)
        before = model.model.state_dict()["model.0.conv.weight"].detach().cpu().clone()
        # nbs == batch → the optimizer steps on every batch
        train_one_round(model, data_yaml, epochs=1, batch=4, img_size=64, lr0=0.01,
                        device=device, project=Path(tmp) / "runs", name="fp32check",
                        workers=0, nbs=4)
        amp = bool(getattr(model.trainer, "amp", False))
        w = model.model.state_dict()["model.0.conv.weight"].detach().cpu()

    changed = not bool((w == before).all())
    rounded = bool((w == w.half().float()).all())
    print(f"device={device} amp={amp} changed={changed} fp16_rounded={rounded}")
    if not changed or rounded:
        print("FAIL: uploaded weights are not trained fp32 values (ADR-008)")
        return 1
    print("OK: trained client weights are fp32 (ADR-008 confirmed on this device)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
