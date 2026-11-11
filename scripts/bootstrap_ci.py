"""Paired bootstrap 95 % CIs of AP50 differences from eval_size_split.py dumps (ADR-034 A1). CPU only.

  python scripts/bootstrap_ci.py --dump research/feasibility/eval_protocols/dumps \
      --pair <tag_a> <tag_b> [--pair ...] --classes bus motorcycle --sizes all small --n 1000 --out ci.csv

A tag is a dump file name without ``.npz``. The CI covers val-set sampling only (which images were drawn), not
training seeds: an interval excluding 0 says the difference is not an artefact of this val set, not that it
would replicate with another training seed.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402

from src.evaluation.size_split import Matched, paired_bootstrap  # noqa: E402


def load(dump: Path, tag: str, cls: str, size: str) -> tuple[np.ndarray, Matched]:
    with np.load(dump / f"{tag}.npz", allow_pickle=False) as d:
        return d["images"], Matched(*(d[f"{cls}/{size}/{f}"] for f in ("img", "score", "flag", "n_gt")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--pair", nargs=2, action="append", required=True, metavar=("A", "B"))
    ap.add_argument("--classes", nargs="+", default=["car", "bus", "truck", "motorcycle"])
    ap.add_argument("--sizes", nargs="+", default=["all", "small", "medium", "large"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for a, b in args.pair:
        for cls in args.classes:
            for size in args.sizes:
                ia, ma = load(args.dump, a, cls, size)
                ib, mb = load(args.dump, b, cls, size)
                if not np.array_equal(ia, ib):
                    raise SystemExit(f"{a} and {b} were evaluated on different images")
                r = paired_bootstrap(ma, mb, n=args.n, seed=args.seed)
                rows.append({"a": a, "b": b, "class": cls, "size": size, "n_gt": int(ma.n_gt.sum()), **r})
                print(rows[-1])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
