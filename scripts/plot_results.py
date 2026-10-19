"""Paper figures for the YOLO11s / COCO-head experiments, drawn only from the experiment registry.

Every figure is written as PNG + PDF together with a CSV of exactly the numbers it plots (the table view).
Runs that are not yet completed in the registry are simply absent; re-run after updating the registry.

Usage:
  python scripts/plot_results.py                       # Vietnamese labels -> research/figures/
  python scripts/plot_results.py --lang en --out DIR   # English labels
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

from src.evaluation.registry_results import (  # noqa: E402
    MISSING_BUS_CLIENTS, Run, by_seed, gap_by_seed, load_runs, summary,
)

# ── look (dataviz reference palette, light surface; validated: adjacent CVD ΔE 9.2, normal-vision 27.6) ──
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLOR = {"P4noKD": "#2a78d6", "A0c": "#52514e", "A1c": "#a3a29c", "FedProxc": "#eb6834"}   # entity -> colour, every figure
MARKER = {"P4noKD": "o", "A0c": "s", "A1c": "^", "FedProxc": "D"}
D1_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

TEXT = {
    "vi": {
        "classes": {"mAP50": "mAP50", "AP50_bus": "bus", "AP50_car": "car", "AP50_truck": "truck",
                    "AP50_motorcycle": "motorcycle"},
        "ap50": "AP50 (trung bình vòng 10/20)",
        "seed_note": "điểm nhỏ = từng seed, ký hiệu lớn = trung bình",
        "f1_title": "P4noKD hơn baseline tốt nhất {d:+.3f} mAP50 trên S1b5-3k ({n} seed)",
        "f2_title": "P4noKD − baseline, ghép cặp theo seed: dương ở {pos}/{n} cặp",
        "f2_y": "Chênh lệch (P4noKD − baseline)",
        "f3_title": "AP50 bus theo số client thiếu bus (cùng 15.000 ảnh; mỗi đường chỉ dùng seed có ở mọi điểm)",
        "f3_x": "Số client không có bus (trên 5)",
        "f4_title": "Gap thiếu lớp = AP50 bus(pooled) − AP50 bus(kịch bản)",
        "f4_y": "Gap AP50 bus",
        "f5_title": "Đóng góp từng thành phần (seed 42)",
        "f6_title": "Kết cục của {n} bus thật ở vòng 20 (ngưỡng điểm 0.25)",
        "d1": {"tp": "phát hiện đúng", "confused_car": "nhầm thành car", "confused_truck": "nhầm thành truck",
               "missed": "không phát hiện", "low_score": "điểm thấp"},
        "share": "Tỉ lệ bus thật",
    },
    "en": {
        "classes": {"mAP50": "mAP50", "AP50_bus": "bus", "AP50_car": "car", "AP50_truck": "truck",
                    "AP50_motorcycle": "motorcycle"},
        "ap50": "AP50 (mean of rounds 10/20)",
        "seed_note": "small dots = seeds, large marker = mean",
        "f1_title": "P4noKD beats the best baseline by {d:+.3f} mAP50 on S1b5-3k ({n} seeds)",
        "f2_title": "P4noKD − baseline, paired by seed: positive in {pos}/{n} pairs",
        "f2_y": "Difference (P4noKD − baseline)",
        "f3_title": "AP50 bus vs clients missing bus (same 15,000 images; each line uses only seeds present at every point)",
        "f3_x": "Clients without bus (of 5)",
        "f4_title": "Missing-class gap = AP50 bus(pooled) − AP50 bus(scenario)",
        "f4_y": "AP50 bus gap",
        "f5_title": "Contribution of each component (seed 42)",
        "f6_title": "Outcome of the {n} ground-truth buses at round 20 (score 0.25)",
        "d1": {"tp": "detected", "confused_car": "as car", "confused_truck": "as truck",
               "missed": "not detected", "low_score": "low score"},
        "share": "Share of ground-truth buses",
    },
}
BASELINES = ("A0c", "A1c", "FedProxc")
CLASS_KEYS = ("mAP50", "AP50_bus", "AP50_truck", "AP50_car", "AP50_motorcycle")
ABLATION = ("A0c", "P3", "P3LR", "P4", "P4noKD")


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2,
        "legend.frameon": False,
    })


def _save(fig, out: Path, name: str, header: list[str], rows: list[list]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    with (out / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("wrote", out / f"{name}.png")


def fig_baselines(runs: list[Run], t: dict, out: Path) -> None:
    """Dot plot per class: mean over seeds (tick) and every seed (dot), one row of methods per class."""
    methods = [m for m in (*BASELINES, "P4noKD") if by_seed(runs, m, "s5", "mAP50")]
    if "P4noKD" not in methods or len(methods) < 2:
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    rows, step = [], 0.18
    for ci, key in enumerate(CLASS_KEYS):
        for mi, m in enumerate(methods):
            vals = by_seed(runs, m, "s5", key)
            if not vals:
                continue
            y = ci + (mi - (len(methods) - 1) / 2) * step
            mu, lo, hi = summary(vals)
            ax.plot([lo, hi], [y, y], color=COLOR[m], lw=1.25, alpha=0.6)
            ax.scatter(list(vals.values()), [y] * len(vals), s=14, color=COLOR[m], alpha=0.55, zorder=3)
            ax.scatter([mu], [y], s=56, marker=MARKER[m], color=COLOR[m], edgecolor=SURFACE, lw=1.2, zorder=4,
                       label=m if ci == 0 else None)
            rows += [[key, m, s, f"{v:.4f}"] for s, v in vals.items()]
    best = max(summary(by_seed(runs, m, "s5", "mAP50"))[0] for m in methods if m != "P4noKD")
    d = summary(by_seed(runs, "P4noKD", "s5", "mAP50"))[0] - best
    n = len(by_seed(runs, "P4noKD", "s5", "mAP50"))
    ax.set_yticks(range(len(CLASS_KEYS)), [t["classes"][k] for k in CLASS_KEYS])
    ax.invert_yaxis()
    ax.set_xlabel(t["ap50"])
    ax.grid(axis="y", visible=False)
    ax.set_title(t["f1_title"].format(d=d, n=n))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncols=len(methods), fontsize=9,
              title=t["seed_note"], title_fontsize=8.5)
    _save(fig, out, "fig1_baselines_per_class", ["metric", "method", "seed", "value"], rows)


def fig_paired_deltas(runs: list[Run], t: dict, out: Path) -> None:
    """Small multiples (mAP50 | AP50 bus): P4noKD − baseline per seed about zero."""
    bases = [b for b in BASELINES if by_seed(runs, b, "s5", "mAP50")]
    if not bases or not by_seed(runs, "P4noKD", "s5", "mAP50"):
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True)
    rows, pos, tot = [], 0, 0
    for ax, key in zip(axes, ("mAP50", "AP50_bus")):
        p = by_seed(runs, "P4noKD", "s5", key)
        for i, b in enumerate(bases):
            q = by_seed(runs, b, "s5", key)
            diffs = {s: p[s] - q[s] for s in sorted(p.keys() & q.keys())}
            if not diffs:
                continue
            mu = summary(diffs)[0]
            ax.bar(i, mu, width=0.55, color=COLOR["P4noKD"], alpha=0.18, edgecolor=COLOR["P4noKD"], lw=1)
            ax.scatter([i] * len(diffs), list(diffs.values()), s=22, color=COLOR["P4noKD"], zorder=3)
            ax.annotate(f"{mu:+.3f}", (i, max(mu, *diffs.values())), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=9, color=INK)
            pos += sum(v > 0 for v in diffs.values())
            tot += len(diffs)
            rows += [[key, b, s, f"{v:.4f}"] for s, v in diffs.items()]
        ax.margins(y=0.12)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(range(len(bases)), [f"− {b}" for b in bases])
        ax.set_title(t["classes"][key], fontsize=10.5)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(t["f2_y"])
    fig.suptitle(t["f2_title"].format(pos=pos, n=tot), x=0.01, ha="left", fontsize=11.5, fontweight="semibold")
    fig.tight_layout()
    _save(fig, out, "fig2_paired_deltas", ["metric", "baseline", "seed", "p4nokd_minus_baseline"], rows)


def fig_severity(runs: list[Run], t: dict, out: Path) -> None:
    """AP50 bus against the number of clients without bus: one line per method, dots per seed."""
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    rows, drawn = [], False
    for m in ("A0c", "FedProxc", "P4noKD"):
        xs, mus = [], []
        present = [by_seed(runs, m, sc, "AP50_bus") for sc in MISSING_BUS_CLIENTS]
        common = set.intersection(*(set(v) for v in present if v)) if any(present) else set()
        for sc, k in sorted(MISSING_BUS_CLIENTS.items(), key=lambda kv: kv[1]):
            vals = {s: v for s, v in by_seed(runs, m, sc, "AP50_bus").items() if s in common}   # same seeds at every point
            if not vals:
                continue
            ax.scatter([k] * len(vals), list(vals.values()), s=16, color=COLOR[m], alpha=0.5, zorder=3)
            xs.append(k)
            mus.append(summary(vals)[0])
            rows += [[m, k, s, f"{v:.4f}"] for s, v in vals.items()]
        if xs:
            drawn = True
            ax.plot(xs, mus, marker=MARKER[m], ms=7, color=COLOR[m], mec=SURFACE, label=m)
            ax.annotate(f"{m} (n={len(common)})", (xs[-1], mus[-1]), xytext=(8, 0), textcoords="offset points", va="center",
                        fontsize=9.5, color=INK)
    if not drawn:
        plt.close(fig)
        return
    ax.set_xticks([0, 3, 4], ["0/5\n(pooled)", "3/5\n(S1b5)", "4/5\n(bus1)"])
    ax.set_xlim(-0.4, 4.9)
    ax.set_xlabel(t["f3_x"])
    ax.set_ylabel(t["ap50"].replace("AP50", "AP50 bus"))
    ax.set_title(t["f3_title"], fontsize=10.5)
    ax.legend(loc="lower left", fontsize=9, title=t["seed_note"], title_fontsize=8.5)
    _save(fig, out, "fig3_bus_vs_severity", ["method", "clients_without_bus", "seed", "AP50_bus"], rows)


def fig_gap(runs: list[Run], t: dict, out: Path) -> None:
    """Missing-class gap per method, at each severity, paired by seed."""
    groups = [(m, sc) for sc in ("s5", "bus1") for m in ("A0c", "FedProxc", "P4noKD") if gap_by_seed(runs, m, sc)]
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    rows, labels = [], []
    for i, (m, sc) in enumerate(groups):
        g = gap_by_seed(runs, m, sc)
        mu = summary(g)[0]
        ax.bar(i, mu, width=0.6, color=COLOR[m], alpha=0.2, edgecolor=COLOR[m], lw=1)
        ax.scatter([i] * len(g), list(g.values()), s=22, color=COLOR[m], zorder=3)
        ax.annotate(f"{mu:+.3f}\nn={len(g)}", (i, max(mu, max(g.values()))), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK)
        labels.append(f"{m}\n{MISSING_BUS_CLIENTS[sc]}/5")
        rows += [[m, MISSING_BUS_CLIENTS[sc], s, f"{v:.4f}"] for s, v in g.items()]
    ax.axhline(0, color=INK2, lw=1)
    ax.margins(y=0.25)
    ax.set_xticks(range(len(groups)), labels)
    ax.set_ylabel(t["f4_y"])
    ax.grid(axis="x", visible=False)
    ax.set_title(t["f4_title"])
    _save(fig, out, "fig4_missing_class_gap", ["method", "clients_without_bus", "seed", "gap_AP50_bus"], rows)


def fig_ablation(runs: list[Run], t: dict, out: Path) -> None:
    """A0c -> P3 -> P3LR -> P4 -> P4noKD at seed 42: mAP50 and AP50 bus, one panel each."""
    steps = [m for m in ABLATION if 42 in by_seed(runs, m, "s5", "mAP50")]
    if len(steps) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharex=True)
    rows = []
    for ax, key in zip(axes, ("mAP50", "AP50_bus")):
        ys = [by_seed(runs, m, "s5", key)[42] for m in steps]
        ax.plot(range(len(steps)), ys, color=INK2, lw=1.25, zorder=2)
        ax.scatter(range(len(steps)), ys, s=40, zorder=3,
                   color=[COLOR["P4noKD"] if m == "P4noKD" else INK2 for m in steps])
        for i, y in enumerate(ys):
            ax.annotate(f"{y:.3f}", (i, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8.5)
        ax.set_xticks(range(len(steps)), steps, fontsize=9)
        ax.set_title(t["classes"][key], fontsize=10.5)
        ax.margins(y=0.18)
        ax.grid(axis="x", visible=False)
        rows += [[key, m, f"{y:.4f}"] for m, y in zip(steps, ys)]
    fig.suptitle(t["f5_title"], x=0.01, ha="left", fontsize=11.5, fontweight="semibold")
    fig.tight_layout()
    _save(fig, out, "fig5_ablation_seed42", ["metric", "arm", "value"], rows)


def fig_d1(registry: Path, t: dict, out: Path) -> None:
    """100 % bars: what happened to every ground-truth bus (D1, ADR-020), S1b5 vs pooled."""
    data = yaml.safe_load(registry.read_text(encoding="utf-8"))
    entry = next((e for e in data["experiments"] if e.get("id") == "EXP-2026-10-14-S14-D1-P4noKD"), None)
    if entry is None:
        return
    parts = ("tp", "confused_car", "confused_truck", "missed", "low_score")
    bars = list(entry["results"].items())
    fig, ax = plt.subplots(figsize=(7.6, 0.9 + 0.62 * len(bars)))
    rows = []
    for yi, (name, res) in enumerate(bars):
        total, left = sum(res[p] for p in parts), 0.0
        for p, c in zip(parts, D1_COLORS):
            w = res[p] / total
            ax.barh(yi, w, left=left, height=0.6, color=c, edgecolor=SURFACE, lw=2,
                    label=t["d1"][p] if yi == 0 else None)
            if w > 0.06:
                ax.text(left + w / 2, yi, f"{res[p]}", ha="center", va="center", fontsize=8.5, color=INK)
            left += w
            rows.append([name, p, res[p], f"{res[p] / total:.4f}"])
    ax.set_yticks(range(len(bars)), [f"{n.split(' (')[0]} · {n.split('(')[-1].split()[0]}" for n, _ in bars])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(t["share"])
    ax.grid(False)
    n_gt = sum(bars[0][1][p] for p in parts)
    ax.set_title(t["f6_title"].format(n=n_gt))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncols=5, fontsize=8.5)
    _save(fig, out, "fig6_d1_bus_outcomes", ["run", "outcome", "count", "share"], rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", type=Path, default=_REPO_ROOT / "research" / "experiment_registry" / "registry.yaml")
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "research" / "figures")
    ap.add_argument("--lang", choices=sorted(TEXT), default="vi")
    args = ap.parse_args()
    _style()
    t = TEXT[args.lang]
    runs = load_runs(args.registry)
    print(f"{len(runs)} completed YOLO11s runs in the registry")
    for draw in (fig_baselines, fig_paired_deltas, fig_severity, fig_gap, fig_ablation):
        draw(runs, t, args.out)
    fig_d1(args.registry, t, args.out)


if __name__ == "__main__":
    main()
