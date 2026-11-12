"""Progress figures for the YOLO11m / architecture / protocol experiments (ADR-030 .. ADR-035), Vietnamese.

Same style and conventions as scripts/plot_report_figures.py (one question per figure; PNG 200 dpi + PDF + the CSV
of exactly what is plotted; no two y-axes). Sources, all committed: the experiment registry (``rule_mean_r10_20``,
seed 42) and research/feasibility/eval_protocols/ (round-20 size split and paired bootstrap CIs, ADR-034).

  python scripts/plot_progress_figures.py [--out research/figures/progress]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (_REPO_ROOT, _REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import yaml  # noqa: E402

from plot_report_figures import GRID, INK, INK2, SURFACE, _save, _style  # noqa: E402

REGISTRY = _REPO_ROOT / "research" / "experiment_registry" / "registry.yaml"
PROTO = _REPO_ROOT / "research" / "feasibility" / "eval_protocols"
CLASSES = ("car", "bus", "truck", "motorcycle")
# categorical order of the report palette (validated: scripts/validate_palette.js, light surface #fcfcfb)
CLASS_COLOR = {"car": "#2a78d6", "bus": "#eb6834", "truck": "#1baf7a", "motorcycle": "#eda100"}
CLASS_MARK = {"car": "o", "bus": "s", "truck": "^", "motorcycle": "D"}
IDEAL, MISSING = "#0b0b0b", "#eb6834"
RUNS = {
    "s_miss": "EXP-2026-10-23-S21-P4t1-s5moto-y11s960-seed42",
    "s_b1": "EXP-2026-11-06-S25-P4t1AKadd-s5moto-y11s960-seed42",
    "m_miss": "EXP-2026-11-07-S26-P4t1-s5moto-y11m960-seed42",
    "m_ideal": "EXP-2026-11-09-S27-P4t1-s5motopooled-y11m960-seed42",
    "s_ideal": "EXP-2026-11-09-S28-P4t1-s5motopooled-y11s960-seed42",
    "m_cbam": "EXP-2026-11-10-S29-P4t1-s5moto-y11mcbam960-seed42",
}
NOTE = "seed 42, trung bình vòng 10 và 20"


def _head(fig, text: str) -> None:
    """Figure headline above the legend row and the panel titles (no overlap)."""
    fig.suptitle(text, x=0.01, y=1.12, ha="left", fontsize=11, fontweight="semibold", color=INK)


def _rule_means() -> dict[str, dict[str, float]]:
    entries = {e["id"]: e for e in yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["experiments"]}
    return {k: entries[v]["results"]["rule_mean_r10_20"] for k, v in RUNS.items()}


def hinh18_thieu_lop_theo_mo_hinh(m: dict, out: Path) -> None:
    """Dumbbell: ideal vs missing AP50 per class, YOLO11s and YOLO11m side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True)
    rows = []
    for ax, (model, ideal, miss) in zip(axes, (("YOLO11s", "s_ideal", "s_miss"), ("YOLO11m", "m_ideal", "m_miss"))):
        for x, c in enumerate(CLASSES):
            a, b = m[ideal][f"AP50_{c}"], m[miss][f"AP50_{c}"]
            ax.plot([x, x], [b, a], color=GRID, lw=3, zorder=1, solid_capstyle="round")
            ax.scatter(x, a, s=46, color=IDEAL, zorder=3, edgecolor=SURFACE, linewidth=1.5)
            ax.scatter(x, b, s=46, color=MISSING, marker="s", zorder=3, edgecolor=SURFACE, linewidth=1.5)
            ax.annotate(f"−{a - b:.3f}", (x + 0.12, (a + b) / 2), va="center", fontsize=8,
                        color=INK if c in ("bus", "truck") else INK2,
                        fontweight="semibold" if c == "bus" else "normal")
            rows.append([model, c, round(a, 4), round(b, 4), round(a - b, 4)])
        ax.set_xticks(range(4), CLASSES)
        ax.set_xlim(-0.4, 3.6)
        ax.set_title(model)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("AP50")
    axes[0].scatter([], [], color=IDEAL, s=40, label="Lý tưởng (mọi client đủ lớp)")
    axes[0].scatter([], [], color=MISSING, marker="s", s=40, label="Thiếu lớp (3/5 client không có bus)")
    fig.legend(loc="upper left", bbox_to_anchor=(0.01, 1.07), ncol=2)
    _head(fig, "Mức tụt do thiếu lớp gần như không đổi khi tăng kích thước mô hình")
    fig.text(0.01, -0.03, f"Cùng 15.000 ảnh, cùng P4t1, 960 px; {NOTE}. Nhãn: AP50(lý tưởng) − AP50(thiếu lớp).",
             fontsize=8, color=INK2)
    _save(fig, out, "hinh18_thieu_lop_theo_mo_hinh", ["model", "class", "AP50_ideal", "AP50_missing", "gap"], rows)


def hinh19_lo_trinh_ap(m: dict, out: Path) -> None:
    """Connected dots: AP50 per class across the configurations tried on the missing-class scenario."""
    steps = (("s_miss", "YOLO11s\nP4t1"), ("s_b1", "YOLO11s\n+ B1 (ADR-028)"), ("m_miss", "YOLO11m\nP4t1"),
             ("m_cbam", "YOLO11m\n+ CBAM"))
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    rows = []
    for c in CLASSES:
        ys = [m[k][f"AP50_{c}"] for k, _ in steps]
        ax.plot(range(len(steps)), ys, color=CLASS_COLOR[c], marker=CLASS_MARK[c], ms=7, mec=SURFACE, mew=1.5)
        ax.annotate(f"{c}  {ys[-1]:.3f}", (len(steps) - 1 + 0.08, ys[-1]), va="center", fontsize=8, color=INK)
        rows += [[lab.replace("\n", " "), c, round(y, 4)] for (_, lab), y in zip(steps, ys)]
    ax.axhline(0.70, color=INK2, lw=1, ls="--", zorder=0)
    ax.annotate("mục tiêu lớp hiếm 0,70", (0, 0.705), fontsize=8, color=INK2)
    ax.set_xticks(range(len(steps)), [lab for _, lab in steps])
    ax.set_xlim(-0.3, len(steps) - 0.2)
    ax.set_ylabel("AP50 (kịch bản thiếu lớp)")
    ax.grid(axis="x", visible=False)
    ax.legend([plt.Line2D([], [], color=CLASS_COLOR[c], marker=CLASS_MARK[c], ms=6) for c in CLASSES], CLASSES,
              loc="lower right", ncol=4)
    ax.set_title("Đổi mô hình mang lại phần tăng lớn nhất; B1 và CBAM thay đổi rất ít")
    fig.text(0.01, -0.04, f"{NOTE}. CBAM: chênh lệch moto nằm trong nhiễu tập val (ADR-034).",
             fontsize=8, color=INK2)
    _save(fig, out, "hinh19_lo_trinh_ap", ["configuration", "class", "AP50"], rows)


def _ci() -> list[dict]:
    with (PROTO / "bootstrap_ci.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pair(row: dict) -> str:
    a = "cbam" if "mcbam" in row["a"] else "s26"
    b = "cbam" if "mcbam" in row["b"] else ("s27" if "pooled" in row["b"] else "s26")
    return f"{a}{'@1280' if row['b'].endswith('1280') else ''}->{b}"


SIZES = ("all", "small", "medium", "large")
SIZE_NAME = {"all": "tất cả", "small": "nhỏ\n< 32²", "medium": "vừa", "large": "lớn\n≥ 96²"}


def _forest(ax, rows: list[dict], color: str, offset: float = 0.0, label: str | None = None) -> None:
    for i, s in enumerate(SIZES):
        r = next(x for x in rows if x["size"] == s)
        d, lo, hi = float(r["delta"]), float(r["ci_low"]), float(r["ci_high"])
        sig = lo > 0 or hi < 0
        ax.plot([lo, hi], [i + offset] * 2, color=color, lw=2, solid_capstyle="round")
        ax.scatter(d, i + offset, s=40, color=color if sig else SURFACE, edgecolor=color, linewidth=1.8, zorder=3,
                   label=label if i == 0 else None)
    ax.axvline(0, color=INK2, lw=1)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.025))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "0" if abs(v) < 1e-9 else f"{v:+.2f}"))
    ax.grid(axis="x", which="minor", color=GRID, linewidth=0.5)
    ax.set_yticks(range(len(SIZES)), [SIZE_NAME[s] for s in SIZES])
    ax.set_ylim(len(SIZES) - 0.5, -0.5)                     # "all" on top; idempotent across calls
    ax.grid(axis="y", visible=False)


def hinh20_muc_tut_theo_kich_thuoc(out: Path) -> None:
    """Forest: ideal − missing (S27 − S26, YOLO11m, round 20) per size, bootstrap 95 % CI."""
    ci = [r for r in _ci() if _pair(r) == "s26->s27"]
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.0), sharey=True, sharex=True)
    rows = []
    for ax, c in zip(axes, ("bus", "truck", "motorcycle")):
        sub = [r for r in ci if r["class"] == c]
        _forest(ax, sub, CLASS_COLOR[c])
        ax.set_title(c)
        rows += [[c, r["size"], r["n_gt"], round(float(r["delta"]), 4), round(float(r["ci_low"]), 4),
                  round(float(r["ci_high"]), 4)] for r in sub]
    axes[1].set_xlabel("AP50(lý tưởng) − AP50(thiếu lớp), khoảng tin cậy 95 %")
    _head(fig, "Mức tụt bus có ở mọi kích thước vật thể, lớn nhất ở bus cỡ vừa")
    fig.text(0.01, -0.06, "YOLO11m, vòng 20, seed 42; bootstrap ghép cặp theo ảnh val (1.000 lần). Chấm đặc = "
             "khoảng tin cậy không chứa 0. CI chỉ phản ánh nhiễu tập val, không phản ánh nhiễu seed.",
             fontsize=8, color=INK2)
    _save(fig, out, "hinh20_muc_tut_theo_kich_thuoc", ["class", "size", "n_gt", "delta", "ci_low", "ci_high"], rows)


def hinh21_cbam_va_1280(out: Path) -> None:
    """Forest: what CBAM and 1280-px evaluation change, per size, for motorcycle and bus."""
    ci = _ci()
    comps = (("s26->cbam", "CBAM − YOLO11m (960)", "#2a78d6", -0.14),
             ("s26@1280->s26", "Đánh giá 1280 − 960 (YOLO11m)", "#7a7975", 0.14))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True, sharex=True)
    rows = []
    for ax, c in zip(axes, ("motorcycle", "bus")):
        for key, label, color, off in comps:
            sub = [r for r in ci if _pair(r) == key and r["class"] == c]
            _forest(ax, sub, color, off, label)
            rows += [[label, c, r["size"], round(float(r["delta"]), 4), round(float(r["ci_low"]), 4),
                      round(float(r["ci_high"]), 4)] for r in sub]
        ax.set_title(c)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.01, 1.07), ncol=2)
    axes[0].set_xlabel("Chênh lệch AP50, khoảng tin cậy 95 %")
    axes[1].set_xlabel("Chênh lệch AP50, khoảng tin cậy 95 %")
    _head(fig, "CBAM chưa vượt nhiễu; đánh giá ở 1280 px giúp vật thể nhỏ nhưng hại vật thể lớn")
    fig.text(0.01, -0.06, "Vòng 20, seed 42, kịch bản thiếu lớp; bootstrap ghép cặp theo ảnh val. Chấm đặc = "
             "khoảng tin cậy không chứa 0 (ADR-034).", fontsize=8, color=INK2)
    _save(fig, out, "hinh21_cbam_va_1280", ["comparison", "class", "size", "delta", "ci_low", "ci_high"], rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "research" / "figures" / "progress")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    _style()
    m = _rule_means()
    hinh18_thieu_lop_theo_mo_hinh(m, args.out)
    hinh19_lo_trinh_ap(m, args.out)
    hinh20_muc_tut_theo_kich_thuoc(args.out)
    hinh21_cbam_va_1280(args.out)
    print("written to", args.out)


if __name__ == "__main__":
    main()
