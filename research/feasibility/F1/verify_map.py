"""F1 runtime verification — CLAUDE.md §8 F1, plan.md §6.2.

Builds the project's model exactly as FL clients/server do (src.model.yolo_wrapper.build_model)
and writes the runtime parameter table to research/feasibility/F1/runtime_map.yaml.
Fails (exit 1) unless exactly 6 class-head keys with dim0 == nc exist.

Usage:  python research/feasibility/F1/verify_map.py [--weights yolov8n.pt]
"""
from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data.bdd100k import TARGET_CLASSES
from src.model.parameter_map import (
    build_parameter_map,
    class_head_keys,
    summarize,
)
from src.model.yolo_wrapper import build_model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("runtime_map.yaml"))
    args = ap.parse_args()

    import torch
    import ultralytics

    model = build_model(args.weights)
    info = build_parameter_map(model)
    nc = len(TARGET_CLASSES)
    head = class_head_keys([p.name for p in info], [p.shape for p in info], nc)  # raises unless 6

    detect = model.model.model[-1]
    record = {
        "environment": {
            "ultralytics": str(ultralytics.__version__),
            "torch": str(torch.__version__),
            "python": platform.python_version(),
            "weights": args.weights,
        },
        "detect": {"type": type(detect).__name__, "nc": int(detect.nc), "reg_max": int(detect.reg_max),
                   "stride": [float(s) for s in detect.stride]},
        "summary": summarize(info),
        "class_head_keys": {p.name: list(p.shape) for p in info if p.name in head},
        "class_specific_numel": sum(p.numel for p in info if p.class_specific),
        "detect_head_table": [
            {"name": p.name, "shape": list(p.shape), "group": p.module_group,
             "class_specific": p.class_specific}
            for p in info if p.name.startswith(f"model.{len(model.model.model) - 1}.")
        ],
    }
    args.out.write_text(yaml.safe_dump(record, sort_keys=False), encoding="utf-8")
    print(f"OK: {len(head)} class-head keys, {record['class_specific_numel']} class-specific elements "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
