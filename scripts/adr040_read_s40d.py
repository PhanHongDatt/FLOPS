"""ADR-040 section 5 / addendum A5 reading of the s40d diagnostic (round 20).

Prints, per model / seed / NMS mode, the bus D1 excess (missing - ideal) and the category shares, the truck FP
split, and the threshold-free bus metrics from the per-detection dumps.

Usage: python scripts/adr040_read_s40d.py [--root kaggle/output/s40d/flops_export/adr040]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CATS = ["confused_car", "confused_truck", "low_score", "missed"]


def run_key(df: pd.DataFrame) -> pd.DataFrame:
    scen = df.partition_id.str.split("_").str[1]   # bus / bus1 / pooled
    return df.assign(arm=df.exp_id.str.split("_").str[0], scen=scen)


def d1_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Excess per category and the A5 shares, for every (arm, seed, missing scenario) with a pooled pair."""
    rows = []
    for (arm, seed), g in df.groupby(["arm", "seed"]):
        ideal = g[g.scen == "pooled"]
        if ideal.empty:
            continue
        ideal = ideal.iloc[0]
        for _, m in g[g.scen != "pooled"].iterrows():
            ex = {c: int(m[f"bus_{c}"] - ideal[f"bus_{c}"]) for c in CATS}
            pos = sum(max(v, 0) for v in ex.values())
            share = {c: max(ex[c], 0) / pos if pos else 0.0 for c in CATS}
            conf_miss = share["confused_car"] + share["confused_truck"] + share["missed"]
            rows.append({"arm": arm, "seed": seed, "scen": m.scen, "L": int(ideal.bus_tp - m.bus_tp), **ex,
                         "share_conf_miss": round(conf_miss, 2), "share_low": round(share["low_score"], 2)})
    return pd.DataFrame(rows)


def pr_metrics(path: Path, cls: str = "bus") -> dict[str, float]:
    p = np.load(path)
    s, f, n = p[f"{cls}/all/score"], p[f"{cls}/all/flag"], p[f"{cls}/all/n_gt"].sum()
    o = np.argsort(-s)
    s, f = s[o], f[o]
    tp, fp = np.cumsum(f), np.cumsum(1 - f)
    r, pr = tp / n, tp / (tp + fp)
    f1 = 2 * pr * r / (pr + r + 1e-9)
    k = int(f1.argmax())
    j = int(np.searchsorted(-s, -0.25, side="right") - 1)

    def rec_at(prec: float) -> float:
        ok = pr >= prec
        return float(r[ok].max()) if ok.any() else 0.0

    return {"R@P0.7": rec_at(0.7), "R@P0.8": rec_at(0.8), "R@conf0.25": float(r[j]),
            "bestF1": float(f1[k]), "bestF1_thr": float(s[k]), "R@bestF1": float(r[k])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("kaggle/output/s40d/flops_export/adr040"))
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    for f in sorted(args.root.glob("d1_*.csv")):
        df = run_key(pd.read_csv(f))
        print(f"\n## {f.name}: bus D1 excess (missing - ideal) and A5 shares")
        print(d1_shares(df).to_string(index=False))
        if "y11m" in f.name:
            cols = ["truck_fp", "truck_fp_background", "truck_fp_loc", "truck_fp_on_car", "truck_fp_on_bus",
                    "truck_tp", "truck_confused_car"]
            print(f"\n## {f.name}: truck FP split")
            print(df[["arm", "scen", "seed", *cols]].to_string(index=False))
    rows = []
    for d in sorted((args.root / "dumps").glob("*.npz")):
        parts = d.stem.split("_")
        rows.append({"arm": parts[0], "seed": parts[2].replace("seed", ""), "scen": parts[4], **pr_metrics(d)})
    if rows:
        print("\n## bus, threshold-free (dumps)")
        print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
