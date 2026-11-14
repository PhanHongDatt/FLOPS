"""ADR-036 analysis: linear probes on extracted box features, readout scores, paired bootstrap, decision. CPU only.

  python scripts/probe_degradation.py --features <dir> \
      --pair 11m s26 s27 --pair 11s s21b s28 --ref 11m coco11m --ref 11s coco11s --out <dir>

``--pair MODEL MISSING IDEAL`` names feature files (``<tag>.npz`` from extract_probe_features.py). Probes are
trained on image half 0 (classes balanced), evaluated on half 1; the 'primary' level trains one probe per level on
the boxes whose primary level it is. Δ = ideal − missing; CIs resample half-1 images (the probe itself is not
refitted, so the CI covers evaluation sampling only).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from src.evaluation.probe import (
    PROBE_CLASSES,
    STRIDES,
    balanced_index,
    confusion,
    decide,
    fit_probe,
    paired_image_bootstrap,
    recall_metrics,
)

DEPTHS = ("neck", "cv3_1")
LEVELS = ("primary", "P3", "P4", "P5")


def probe_correct(d: dict, depth: str, level: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-box correctness on half 1 (nan where the box is not evaluated) and predictions (-1 if not)."""
    n = len(d["cls"])
    correct, pred = np.full(n, np.nan), np.full(n, -1)
    for lvl in range(len(STRIDES)):
        sel = d["level"] == lvl if level == "primary" else np.ones(n, bool)
        if level != "primary" and level != f"P{lvl + 3}":
            continue
        tr = np.flatnonzero(sel & (d["half"] == 0))
        ev = np.flatnonzero(sel & (d["half"] == 1))
        y = d["cls"]
        if len(np.unique(y[tr])) < len(PROBE_CLASSES) or not len(ev):
            continue
        tr = tr[balanced_index(y[tr])]
        probe = fit_probe(d[f"{depth}_{lvl}"][tr], y[tr])
        p = probe.predict(d[f"{depth}_{lvl}"][ev])
        pred[ev], correct[ev] = p, (p == y[ev])
    return correct, pred


def readout(d: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per box at its primary level: sigmoid bus score, and whether the bus logit beats car and truck."""
    lg = np.stack([d[f"logit_{lvl}"] for lvl in range(len(STRIDES))], 1)[np.arange(len(d["cls"])), d["level"]]
    return 1 / (1 + np.exp(-lg[:, 1])), (lg[:, 1] > np.maximum(lg[:, 0], lg[:, 2])).astype(float)


def load(features: Path, tag: str) -> dict:
    with np.load(features / f"{tag}.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def describe(tag: str, d: dict, rows: list[dict]) -> dict:
    """Per-checkpoint metrics; returns the per-box vectors used for the paired deltas."""
    vec = {}
    for depth in DEPTHS:
        for level in LEVELS:
            correct, pred = probe_correct(d, depth, level)
            ev = ~np.isnan(correct)
            m = recall_metrics(d["cls"][ev], correct[ev].astype(bool))
            cm = confusion(d["cls"][ev], pred[ev]).tolist()
            rows.append({"tag": tag, "depth": depth, "level": level, **m, "confusion": json.dumps(cm)})
            vec[(depth, level)] = correct
    score, wins = readout(d)
    bus = d["cls"] == 1
    rows.append({"tag": tag, "depth": "readout", "level": "primary", "bus_score": float(score[bus].mean()),
                 "bus_wins": float(wins[bus].mean())})
    vec[("readout", "primary")], vec[("readout_wins", "primary")] = score, wins
    return vec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--pair", nargs=3, action="append", required=True, metavar=("MODEL", "MISSING", "IDEAL"))
    ap.add_argument("--ref", nargs=2, action="append", default=[], metavar=("MODEL", "TAG"))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows: list[dict] = []
    deltas: list[dict] = []
    per_model: dict[str, dict[str, dict[str, float]]] = {}
    for model, miss_tag, ideal_tag in args.pair:
        miss, ideal = load(args.features, miss_tag), load(args.features, ideal_tag)
        for k in ("image", "cls", "half", "level"):
            if not np.array_equal(miss[k], ideal[k]):
                raise SystemExit(f"{miss_tag} and {ideal_tag} differ in {k}: not the same boxes")
        vm, vi = describe(miss_tag, miss, rows), describe(ideal_tag, ideal, rows)
        bus = miss["cls"] == 1
        for key in vm:
            a, b = vm[key], vi[key]
            mask = bus & ~np.isnan(a) & ~np.isnan(b)
            r = paired_image_bootstrap(miss["image"], np.nan_to_num(a), np.nan_to_num(b), mask, n=args.n)
            deltas.append({"model": model, "depth": key[0], "level": key[1], "n_bus": int(mask.sum()), **r})
            print(deltas[-1], flush=True)
        prim = {d["depth"]: d for d in deltas if d["model"] == model and d["level"] == "primary"}
        per_model[model] = {"neck": prim["neck"], "cv3_1": prim["cv3_1"], "readout": prim["readout"]}
    for model, tag in args.ref:
        describe(f"{tag} ({model} reference)", load(args.features, tag), rows)

    decision = decide(per_model)
    args.out.mkdir(parents=True, exist_ok=True)
    import csv
    for name, data in (("probe_metrics.csv", rows), ("probe_deltas.csv", deltas)):
        fields = list(dict.fromkeys(k for r in data for k in r))
        with (args.out / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)
    (args.out / "decision.json").write_text(json.dumps({"decision": decision, "rule": "ADR-036",
                                                        "per_model": per_model}, indent=2), encoding="utf-8")
    print("DECISION:", decision)


if __name__ == "__main__":
    main()
