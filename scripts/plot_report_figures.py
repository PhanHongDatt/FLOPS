"""Report figures (Vietnamese), one question per figure, numbered as in the report outline.

Sources, all committed: the experiment registry (rule metrics, mean of rounds 10/20; ``operating_r10_20``),
research/figures/data/partition_class_counts.yaml and run_times.csv (scripts/extract_report_data.py), and
research/figures/data/local_eval/*.json (round-20 checkpoints re-evaluated by scripts/eval_checkpoints_local.py).
A figure whose data is missing is skipped. Each figure: PNG (200 dpi) + PDF + a CSV of exactly what is plotted.

Conventions: one fixed colour per method in every figure; the ideal scenario first; every seed drawn as a dot
next to the mean, with n stated; no two y-axes; no pooling across scenarios.

Usage:
  python scripts/plot_report_figures.py [--out research/figures/report]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src.evaluation.registry_results import Run, by_seed, load_runs  # noqa: E402

DATA = _REPO_ROOT / "research" / "figures" / "data"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
METHODS = ("A0c", "FedProxc", "A1c", "P4noKD")
NAME = {"A0c": "FedAvg", "FedProxc": "FedProx", "A1c": "Gộp theo số box", "P4noKD": "Công thức cải tiến"}
COLOR = {"A0c": "#2a78d6", "FedProxc": "#eb6834", "A1c": "#1baf7a", "P4noKD": "#eda100"}
MARKER = {"A0c": "o", "FedProxc": "D", "A1c": "^", "P4noKD": "s"}
SCEN = ("pooled", "s5", "bus1")                                   # ideal first
SCEN_NAME = {"pooled": "Lý tưởng (0/5)", "s5": "Thiếu bus 3/5", "bus1": "Thiếu bus 4/5"}
SCEN_X = {"pooled": "0/5\n(lý tưởng)", "s5": "3/5", "bus1": "4/5"}
SCEN_STYLE = {"pooled": ("#0b0b0b", "-"), "s5": ("#7a7975", "--"), "bus1": ("#b5b4ae", ":")}
CLASSES = ("car", "bus", "truck", "motorcycle")
RULE = "trung bình vòng 10 và 20"


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.titlesize": 10, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2, "legend.frameon": False,
        "legend.fontsize": 8,
    })


def _save(fig, out: Path, name: str, header: list[str], rows: list[list]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    with (out / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("wrote", out / f"{name}.png")


def _suptitle(fig, text: str) -> None:
    fig.suptitle(text, x=0.01, ha="left", fontsize=10.5, fontweight="semibold")


def _local_eval() -> dict[tuple[str, str, int], dict]:
    out = {}
    for p in sorted((DATA / "local_eval").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        method, rest = d["run"].split("_feasibility_seed")
        sc = "bus1" if "_bus1_" in rest else "pooled" if "_pooled_" in rest else "s5"
        out[(method, sc, int(d["seed"]))] = d
    return out


# ── Hình 2, 3: dữ liệu ──────────────────────────────────────────────────────────────────────────────────────
def hinh02_kich_ban(out: Path) -> None:
    counts = yaml.safe_load((DATA / "partition_class_counts.yaml").read_text(encoding="utf-8"))["scenarios"]
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.9), sharey=True)
    rows = []
    for ax, sc in zip(axes, SCEN):
        cl = sorted(counts[sc])
        for i, c in enumerate(cl):
            for j, k in enumerate(CLASSES):
                has = counts[sc][c][k] > 0
                ax.add_patch(plt.Rectangle((j - 0.42, i - 0.42), 0.84, 0.84, lw=1.2,
                                           fc="#2a78d6" if has else SURFACE, ec="#2a78d6" if has else INK2,
                                           alpha=0.85 if has else 1, ls="-" if has else (0, (2, 2))))
                if not has:
                    ax.text(j, i, "×", ha="center", va="center", fontsize=11, color=INK2)
                rows.append([sc, c, k, int(has), counts[sc][c][k]])
        ax.set_xlim(-0.6, 3.6)
        ax.set_ylim(len(cl) - 0.4, -0.6)
        ax.set_xticks(range(4), CLASSES, rotation=0, fontsize=8)
        ax.set_yticks(range(len(cl)), cl)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        n_missing = sum(counts[sc][c]["bus"] == 0 for c in cl)
        ax.set_title(f"{SCEN_NAME[sc]}\n{n_missing}/5 client không có bus", fontsize=9)
    _suptitle(fig, "Ba kịch bản dùng cùng 15.000 ảnh và cùng tổng số box; chỉ khác client nào giữ bus (ô xanh = có lớp, × = 0 box)")
    fig.tight_layout()
    _save(fig, out, "hinh02_ba_kich_ban", ["scenario", "client", "class", "present", "boxes"], rows)


def hinh03_phan_bo_box(out: Path) -> None:
    data = yaml.safe_load((DATA / "partition_class_counts.yaml").read_text(encoding="utf-8"))
    counts, totals = data["scenarios"], data["totals"]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.5), sharey=True, layout="constrained")
    rows = []
    for ax, sc in zip(axes, SCEN):
        cl = sorted(counts[sc])
        share = np.array([[counts[sc][c][k] / totals[k] for k in CLASSES] for c in cl])
        im = ax.imshow(share, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i, c in enumerate(cl):
            for j, k in enumerate(CLASSES):
                v = counts[sc][c][k]
                ax.text(j, i, f"{v:,}".replace(",", "."), ha="center", va="center", fontsize=7.5,
                        color="white" if share[i, j] > 0.55 else INK)
                rows.append([sc, c, k, v, f"{share[i, j]:.4f}"])
        ax.set_xticks(range(4), ["car", "bus", "truck", "motor-\ncycle"], fontsize=8)
        ax.set_yticks(range(len(cl)), cl)
        ax.grid(False)
        ax.set_title(SCEN_NAME[sc], fontsize=9)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("Tỉ lệ box của lớp nằm ở client", fontsize=8)
    tot = ", ".join(f"{k} {totals[k]:,}".replace(",", ".") for k in CLASSES)
    _suptitle(fig, f"Số box theo client và lớp (tổng mỗi kịch bản: {tot})")
    _save(fig, out, "hinh03_phan_bo_box", ["scenario", "client", "class", "boxes", "share_of_class_total"], rows)


# ── Hình 4, 5, 6, 8: registry ───────────────────────────────────────────────────────────────────────────────
def _dots_and_mean(ax, x, vals: dict[int, float], m: str, jitter: float = 0.0, size: float = 16):
    xs = [x + jitter * (i - (len(vals) - 1) / 2) for i in range(len(vals))]
    ax.scatter(xs, list(vals.values()), s=size, color=COLOR[m], alpha=0.45, zorder=3, lw=0)
    return mean(vals.values())


def hinh04_ap50_bus_theo_muc_thieu(runs: list[Run], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    rows = []
    for m in METHODS:
        xs, mus, ns = [], [], []
        for x, sc in enumerate(SCEN):
            vals = by_seed(runs, m, sc, "AP50_bus")
            if not vals:
                continue
            mus.append(_dots_and_mean(ax, x, vals, m, jitter=0.06))
            xs.append(x)
            ns.append(len(vals))
            rows += [[NAME[m], SCEN_NAME[sc], s, f"{v:.4f}"] for s, v in vals.items()]
        if xs:
            ax.plot(xs, mus, marker=MARKER[m], ms=6.5, color=COLOR[m], mec=SURFACE, mew=1,
                    ls="-" if len(xs) > 1 else "none", label=f"{NAME[m]} (n = {'/'.join(map(str, ns))})")
            dy = {"FedProxc": 7, "A1c": -8}.get(m, 0)
            ax.annotate(f"{mus[-1]:.3f}", (xs[-1], mus[-1]), xytext=(9, dy), textcoords="offset points",
                        va="center", fontsize=8)
    ax.set_xticks(range(3), [SCEN_X[s] for s in SCEN])
    ax.set_xlim(-0.3, 2.55)
    ax.set_xlabel("Số client không có ảnh bus (trên 5)")
    ax.set_ylabel(f"AP50 bus ({RULE})")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), title="chấm mờ = từng seed; n theo từng điểm",
              title_fontsize=7.5)
    ax.set_title("AP50 bus giảm khi nhiều client thiếu bus hơn (cùng 15.000 ảnh)")
    _save(fig, out, "hinh04_ap50_bus_theo_muc_thieu", ["method", "scenario", "seed", "AP50_bus"], rows)


def _gap(runs: list[Run], m: str, sc: str) -> dict[int, float]:
    a, b = by_seed(runs, m, "pooled", "AP50_bus"), by_seed(runs, m, sc, "AP50_bus")
    return {s: a[s] - b[s] for s in sorted(a.keys() & b.keys())}


def hinh05_muc_suy_giam(runs: list[Run], out: Path) -> None:
    panels = [sc for sc in ("s5", "bus1") if _gap(runs, "A0c", sc)]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.6, 3.4), sharey=True, squeeze=False)
    rows = []
    for ax, sc in zip(axes[0], panels):
        ms = [m for m in METHODS if _gap(runs, m, sc)]
        ref = mean(_gap(runs, "A0c", sc).values())
        ax.axhline(ref, color=COLOR["A0c"], lw=1, ls="--", zorder=1)
        ax.annotate(f"FedAvg {ref:+.3f}", (len(ms) - 0.5, ref), xytext=(0, 3), textcoords="offset points",
                    ha="right", fontsize=7.5, color=INK2)
        for i, m in enumerate(ms):
            g = _gap(runs, m, sc)
            mu = _dots_and_mean(ax, i, g, m, jitter=0.12, size=26)
            ax.plot([i - 0.22, i + 0.22], [mu, mu], color=COLOR[m], lw=2.5, zorder=4)
            ax.annotate(f"{mu:+.3f}\nn = {len(g)}", (i, max(g.values())), xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=7.5)
            rows += [[SCEN_NAME[sc], NAME[m], s, f"{v:.4f}"] for s, v in g.items()]
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(len(ms)), [NAME[m].replace(" ", "\n", 1) if len(NAME[m]) > 10 else NAME[m] for m in ms],
                      fontsize=8)
        ax.set_xlim(-0.6, len(ms) - 0.4)
        ax.margins(y=0.2)
        ax.grid(axis="x", visible=False)
        ax.set_title(SCEN_NAME[sc], fontsize=9.5)
    axes[0][0].set_ylabel("Suy giảm AP50 bus = lý tưởng − thiếu lớp")
    _suptitle(fig, "Không phương pháp nào có mức suy giảm nhỏ hơn FedAvg (ghép cặp theo seed; thanh = trung bình)")
    fig.tight_layout()
    _save(fig, out, "hinh05_muc_suy_giam_theo_phuong_phap", ["scenario", "method", "seed", "gap_AP50_bus"], rows)


def hinh06_ap50_theo_lop(runs: list[Run], out: Path) -> None:
    ms = [m for m in METHODS if by_seed(runs, m, "s5", "AP50_bus")]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    w, rows = 0.8 / len(ms), []
    for ci, c in enumerate(CLASSES):
        for mi, m in enumerate(ms):
            vals = by_seed(runs, m, "s5", f"AP50_{c}")
            x = ci + (mi - (len(ms) - 1) / 2) * w
            mu = mean(vals.values())
            ax.bar(x, mu, width=w - 0.03, color=COLOR[m], alpha=0.8, label=NAME[m] if ci == 0 else None, zorder=2)
            ax.scatter([x] * len(vals), list(vals.values()), s=9, color=INK, zorder=3, lw=0)
            rows += [[c, NAME[m], s, f"{v:.4f}"] for s, v in vals.items()]
    n = len(by_seed(runs, "P4noKD", "s5", "AP50_bus"))
    ax.set_xticks(range(4), CLASSES)
    ax.set_ylabel(f"AP50 ({RULE})")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncols=2)
    ax.set_ylim(0, 0.85)
    ax.set_title(f"AP50 theo lớp, kịch bản thiếu bus 3/5 (n = {n} seed; chấm = từng seed)")
    _save(fig, out, "hinh06_ap50_theo_lop_thieu_3_5", ["class", "method", "seed", "AP50"], rows)


def hinh08_chenh_lech_ghep_cap(runs: list[Run], out: Path) -> None:
    bases = [b for b in ("A0c", "FedProxc", "A1c") if by_seed(runs, b, "s5", "mAP50")]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.6), sharey=True)
    rows, pos, tot = [], 0, 0
    for ax, key, lab in zip(axes, ("mAP50", "AP50_bus"), ("mAP50", "AP50 bus")):
        p = by_seed(runs, "P4noKD", "s5", key)
        for i, b in enumerate(bases):
            q = by_seed(runs, b, "s5", key)
            d = {s: p[s] - q[s] for s in sorted(p.keys() & q.keys())}
            ax.scatter(list(d.values()), [i] * len(d), s=26, color=COLOR[b], zorder=3)
            mu = mean(d.values())
            ax.plot([mu, mu], [i - 0.25, i + 0.25], color=COLOR[b], lw=2.5)
            ax.annotate(f"{mu:+.3f}", (max(d.values()), i), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=7.5)
            pos += sum(v > 0 for v in d.values())
            tot += len(d)
            rows += [[key, NAME[b], s, f"{v:.4f}"] for s, v in d.items()]
        ax.axvline(0, color=INK2, lw=1)
        ax.set_yticks(range(len(bases)), [f"cải tiến − {NAME[b]}" for b in bases])
        ax.set_ylim(len(bases) - 0.5, -0.5)
        ax.set_xlim(min(-0.01, ax.get_xlim()[0]), ax.get_xlim()[1] * 1.15)
        ax.grid(axis="y", visible=False)
        ax.set_title(lab, fontsize=9.5)
        ax.set_xlabel("Chênh lệch (ghép cặp theo seed)")
    _suptitle(fig, f"Công thức cải tiến hơn mọi baseline ở {pos}/{tot} cặp so sánh (thiếu bus 3/5)")
    fig.tight_layout()
    _save(fig, out, "hinh08_chenh_lech_ghep_cap", ["metric", "baseline", "seed", "improved_minus_baseline"], rows)


def hinh11_ablation(runs: list[Run], out: Path) -> None:
    steps = [("A0c", "FedAvg"), ("P3", "+ chưng cất"), ("P3LR", "+ LR cosine"), ("P4", "+ lặp lại\nlớp hiếm"),
             ("P4noKD", "bỏ chưng cất")]
    steps = [(m, lab) for m, lab in steps if 42 in by_seed(runs, m, "s5", "mAP50")]
    if len(steps) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.2))
    rows = []
    for ax, key, lab in zip(axes, ("mAP50", "AP50_bus"), ("mAP50", "AP50 bus")):
        ys = [by_seed(runs, m, "s5", key)[42] for m, _ in steps]
        ax.step(range(len(ys)), ys, where="mid", color=INK2, lw=1.2, zorder=2)
        ax.scatter(range(len(ys)), ys, s=34, zorder=3,
                   color=[COLOR["P4noKD"] if m == "P4noKD" else COLOR["A0c"] if m == "A0c" else INK2 for m, _ in steps])
        for i, y in enumerate(ys):
            d = "" if i == 0 else f"\n({y - ys[i - 1]:+.3f})"
            ax.annotate(f"{y:.3f}{d}", (i, y), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7)
        ax.set_xticks(range(len(steps)), [s for _, s in steps], fontsize=7.5)
        ax.margins(y=0.3)
        ax.grid(axis="x", visible=False)
        ax.set_title(lab, fontsize=9.5)
        rows += [[key, m, s.replace("\n", " "), f"{y:.4f}"] for (m, s), y in zip(steps, ys)]
    _suptitle(fig, "Đóng góp từng thành phần (thiếu bus 3/5, seed 42, n = 1): mỗi bước đều tăng; bỏ chưng cất ở cuối chỉ mất 0,003 mAP50 / 0,008 AP50 bus")
    fig.tight_layout()
    _save(fig, out, "hinh11_ablation", ["metric", "arm", "step", "value"], rows)


def hinh13_giu_lai(runs: list[Run], out: Path) -> None:
    rows, items = [], []
    for m in ("A0c", "P4noKD"):
        for sc in ("s5", "bus1"):
            a = {k: by_seed(runs, m, s_, k) for s_ in ("pooled", sc) for k in ("AP50_bus", "TP_bus")}
            ap_p, ap_s = by_seed(runs, m, "pooled", "AP50_bus"), by_seed(runs, m, sc, "AP50_bus")
            tp_p, tp_s = by_seed(runs, m, "pooled", "TP_bus"), by_seed(runs, m, sc, "TP_bus")
            seeds = sorted(ap_p.keys() & ap_s.keys() & tp_p.keys() & tp_s.keys())
            if not seeds:
                continue
            ap = {s: 100 * ap_s[s] / ap_p[s] for s in seeds}
            tp = {s: 100 * tp_s[s] / tp_p[s] for s in seeds}
            items.append((m, sc, ap, tp))
            rows += [[NAME[m], SCEN_NAME[sc], s, f"{ap[s]:.1f}", f"{tp[s]:.1f}"] for s in seeds]
            del a
    if not items:
        return
    fig, ax = plt.subplots(figsize=(7.2, 0.9 + 0.55 * len(items)))
    for y, (m, sc, ap, tp) in enumerate(items):
        a, t = mean(ap.values()), mean(tp.values())
        ax.plot([t, a], [y, y], color=COLOR[m], lw=2, alpha=0.5, zorder=1)
        ax.scatter(list(ap.values()), [y] * len(ap), s=10, color=COLOR[m], alpha=0.4, zorder=2)
        ax.scatter(list(tp.values()), [y] * len(tp), s=10, color=COLOR[m], alpha=0.4, zorder=2)
        ax.scatter([a], [y], s=60, color=COLOR[m], zorder=3, label="AP50 bus" if y == 0 else None)
        ax.scatter([t], [y], s=60, facecolor=SURFACE, edgecolor=COLOR[m], lw=2, zorder=3,
                   label="Số bus phát hiện đúng (TP @0,25)" if y == 0 else None)
        ax.annotate(f"{a:.0f} %", (a, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
        ax.annotate(f"{t:.0f} %", (t, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.set_yticks(range(len(items)), [f"{NAME[m]} · {SCEN_NAME[sc]} (n = {len(ap)})" for m, sc, ap, _ in items],
                  fontsize=8)
    ax.set_ylim(len(items) - 0.5, -0.8)
    ax.set_xlim(0, 105)
    ax.axvline(100, color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("% so với kịch bản lý tưởng, cùng phương pháp và seed")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=2, fontsize=7.5)
    ax.set_title("AP50 bus giữ lại phần lớn, nhưng số bus phát hiện đúng ở ngưỡng 0,25 giảm mạnh")
    _save(fig, out, "hinh13_giu_lai_ap50_va_tp", ["method", "scenario", "seed", "AP50_bus_pct", "TP_bus_pct"], rows)


def hinh14_thoi_gian(out: Path) -> None:
    path = DATA / "run_times.csv"
    if not path.exists():
        return
    rec = list(csv.DictReader(path.open(encoding="utf-8")))
    order = ("A0c", "FedProxc", "A1c", "P3", "P3LR", "P4", "P4noKD")
    label = {**NAME, "P3": "+ chưng cất (P3)", "P3LR": "P3 + LR cosine", "P4": "P3LR + lặp lại lớp hiếm"}
    groups = [(m, [r for r in rec if r["method"] == m]) for m in order]
    groups = [(m, g) for m, g in groups if g]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), sharey=True)
    rows = []
    for ax, key, lab in zip(axes, ("train_round_mean_min", "total_min"),
                            ("Phút / vòng huấn luyện (5 client)", "Phút / run (20 vòng + 2 lần đánh giá)")):
        for y, (m, g) in enumerate(groups):
            vals = [float(r[key]) for r in g]
            c = COLOR.get(m, INK2)
            ax.scatter(vals, [y] * len(vals), s=18, color=c, alpha=0.5, lw=0, zorder=3)
            mu = mean(vals)
            ax.plot([mu, mu], [y - 0.28, y + 0.28], color=c, lw=2.5, zorder=4)
            ax.annotate(f"{mu:.1f} (n = {len(vals)})", (max(vals), y), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=7.5)
            if key == "total_min":
                rows += [[r["run"], m, r["scenario"], r["seed"], r["train_round_mean_min"], r["total_min"]] for r in g]
        ax.set_yticks(range(len(groups)), [label[m] for m, _ in groups], fontsize=8)
        ax.set_ylim(len(groups) - 0.5, -0.5)
        ax.set_xlim(0, ax.get_xlim()[1] * 1.25)
        ax.grid(axis="y", visible=False)
        ax.set_title(lab, fontsize=9)
    _suptitle(fig, "Thời gian huấn luyện trên Kaggle (GPU Tesla T4; mỗi chấm một run, mọi kịch bản và seed)")
    fig.tight_layout()
    _save(fig, out, "hinh14_thoi_gian_huan_luyen",
          ["run", "method", "scenario", "seed", "train_round_mean_min", "total_min"], rows)


# ── Hình 7, 9, 10, 12: checkpoint vòng 20 đánh giá lại ───────────────────────────────────────────────────────
OUTCOMES = (("tp", "phát hiện đúng", "#1baf7a"), ("confused_car", "nhầm thành car", "#eb6834"),
            ("confused_truck", "nhầm thành truck", "#eda100"), ("low_score", "điểm thấp (< 0,25)", "#a3a29c"),
            ("missed", "không phát hiện", "#52514e"))


def _configs(le: dict) -> list[tuple[str, str, int]]:
    return [(m, sc, s) for m in ("A0c", "P4noKD") for sc in SCEN for s in sorted({k[2] for k in le})
            if (m, sc, s) in le]


def hinh07_ket_cuc_bus(le: dict, out: Path) -> None:
    cfg = _configs(le)
    if not cfg:
        return
    fig, ax = plt.subplots(figsize=(7.8, 0.9 + 0.42 * len(cfg)))
    rows = []
    for y, k in enumerate(cfg):
        oc = le[k]["bus_outcomes"]
        other = sum(v for kk, v in oc.items() if kk.startswith("confused_") and kk not in ("confused_car", "confused_truck"))
        total = sum(oc.values())
        left = 0.0
        for key, lab, c in OUTCOMES:
            v = oc.get(key, 0) + (other if key == "missed" else 0)
            w = v / total
            ax.barh(y, w, left=left, height=0.62, color=c, edgecolor=SURFACE, lw=1.5, label=lab if y == 0 else None)
            if w > 0.055:
                ax.text(left + w / 2, y, f"{w * 100:.0f}%", ha="center", va="center", fontsize=7,
                        color="white" if key in ("missed", "confused_car") else INK)
            left += w
            rows.append([NAME[k[0]], SCEN_NAME[k[1]], k[2], key, v, f"{w:.4f}"])
    ax.set_yticks(range(len(cfg)), [f"{NAME[m]} · {SCEN_NAME[sc]} · seed {s}" for m, sc, s in cfg], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=5, fontsize=7.5)
    ax.set_title("Kết cục của 1.597 bus thật ở vòng 20 (ngưỡng điểm 0,25, IoU ≥ 0,5)")
    _save(fig, out, "hinh07_ket_cuc_1597_bus", ["method", "scenario", "seed", "outcome", "count", "share"], rows)


def hinh09_pr_bus(le: dict, out: Path) -> None:
    ms = [m for m in ("A0c", "P4noKD") if any(k[0] == m for k in le)]
    if not ms:
        return
    fig, axes = plt.subplots(1, len(ms), figsize=(8.0, 3.5), sharey=True, squeeze=False)
    rows = []
    for ax, m in zip(axes[0], ms):
        for sc in SCEN:
            ks = [k for k in le if k[0] == m and k[1] == sc]
            if not ks:
                continue
            d = le[ks[0]]
            c, ls = SCEN_STYLE[sc]
            ax.plot(d["pr_recall"], d["pr_precision"], color=c, ls=ls, lw=1.8,
                    label=f"{SCEN_NAME[sc]} · AP50 {d['ap50_bus']:.3f}")
            ax.scatter([d["r_at_025"]], [d["p_at_025"]], s=40, color=c, edgecolor=SURFACE, zorder=4)
            ax.annotate(f"0,25: R={d['r_at_025']:.2f}", (d["r_at_025"], d["p_at_025"]), xytext=(5, 4),
                        textcoords="offset points", fontsize=7, color=INK)
            rows += [[NAME[m], SCEN_NAME[sc], ks[0][2], f"{r:.3f}", f"{p:.4f}"]
                     for r, p in zip(d["pr_recall"], d["pr_precision"])]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Recall bus")
        ax.set_title(f"{NAME[m]} (seed {ks[0][2]}, vòng 20)", fontsize=9.5)
        ax.legend(loc="upper right", fontsize=7)
    axes[0][0].set_ylabel("Precision bus")
    _suptitle(fig, "Đường precision–recall của bus: điểm tròn = ngưỡng 0,25; thiếu bus đẩy điểm vận hành về recall thấp")
    fig.tight_layout()
    _save(fig, out, "hinh09_pr_bus", ["method", "scenario", "seed", "recall", "precision"], rows)


def hinh10_confusion(le: dict, out: Path, method: str = "A0c") -> None:
    ks = [k for sc in ("pooled", "s5") for k in le if k[0] == method and k[1] == sc][:2]
    if len(ks) < 2:
        return
    labels = [*CLASSES, "nền"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9))
    rows = []
    for ax, k in zip(axes, ks):
        m = np.array(le[k]["confusion_5x5"], dtype=float)
        norm = m / np.maximum(m.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                if i == 4 and j == 4:
                    ax.text(j, i, "–", ha="center", va="center", fontsize=8, color=INK2)
                    continue
                ax.text(j, i, f"{int(m[i, j]):,}\n{norm[i, j] * 100:.0f}%".replace(",", "."), ha="center",
                        va="center", fontsize=6.5, color="white" if norm[i, j] > 0.55 else INK)
                rows.append([SCEN_NAME[k[1]], k[2], labels[i], labels[j], int(m[i, j]), f"{norm[i, j]:.4f}"])
        ax.set_xticks(range(5), labels, fontsize=8)
        ax.set_yticks(range(5), labels, fontsize=8)
        ax.set_xlabel("Dự đoán (nền = không phát hiện hoặc lớp ngoài 4 lớp)")
        ax.set_ylabel("Thực tế (nền = dự đoán trên nền, FP)")
        ax.grid(False)
        ax.set_title(f"{SCEN_NAME[k[1]]} · {NAME[method]} · seed {k[2]}", fontsize=9)
    _suptitle(fig, "Confusion matrix ở ngưỡng 0,25, IoU 0,45 (Ultralytics), chuẩn hóa theo hàng thực tế")
    fig.tight_layout()
    _save(fig, out, f"hinh10_confusion_matrix_{method}", ["scenario", "seed", "actual", "predicted", "count", "row_share"], rows)


def hinh12_histogram_diem(le: dict, out: Path) -> None:
    ms = [m for m in ("A0c", "P4noKD") if any(k[0] == m for k in le)]
    if not ms:
        return
    fig, axes = plt.subplots(1, len(ms), figsize=(8.0, 3.3), sharey=True, squeeze=False)
    bins = np.linspace(0, 1, 41)
    rows = []
    for ax, m in zip(axes[0], ms):
        for sc in SCEN:
            ks = [k for k in le if k[0] == m and k[1] == sc]
            if not ks:
                continue
            s = np.array(le[ks[0]]["bus_best_scores"])
            c, ls = SCEN_STYLE[sc]
            zero = float((s < 0.001).mean())
            ge = float((s >= 0.25).mean())
            h, _ = np.histogram(s[s >= 0.001], bins=bins)
            ax.stairs(h / len(s), bins, color=c, ls=ls, lw=1.8,
                      label=f"{SCEN_NAME[sc]}: ≥0,25 {ge * 100:.0f}% · trung vị {np.median(s):.2f}")
            rows += [[NAME[m], SCEN_NAME[sc], ks[0][2], f"{bins[i]:.3f}", f"{bins[i + 1]:.3f}", int(h[i]),
                      f"{h[i] / len(s):.4f}"] for i in range(len(h))]
            rows.append([NAME[m], SCEN_NAME[sc], ks[0][2], "no_bus_prediction", "", int((s < 0.001).sum()), f"{zero:.4f}"])
        ax.axvline(0.25, color=INK2, lw=1, ls="--")
        ax.set_xlabel("Điểm bus cao nhất trên mỗi bus thật (IoU ≥ 0,5)")
        ax.set_title(f"{NAME[m]} (seed {ks[0][2]}, vòng 20)", fontsize=9.5)
        ax.legend(loc="upper right", fontsize=6.8)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0][0].set_ylabel("Tỉ lệ trên 1.597 bus thật")
    _suptitle(fig, "Phân bố điểm tin cậy của bus thật: thiếu bus đẩy phân bố sang trái, dưới ngưỡng 0,25")
    fig.tight_layout()
    _save(fig, out, "hinh12_histogram_diem_bus", ["method", "scenario", "seed", "bin_lo", "bin_hi", "count", "share"], rows)


# ADR-023 (motorcycle), seed 42. Colour = image size, not method: the other figures' method colours do not apply.
ADR023 = (("P4noKD", "s5", "Mốc: công thức cải tiến", 640),
          ("P4noKD", "s5moto", "Cách 2: đủ ảnh motorcycle", 640),
          ("P4t1", "s5", "Cách 4: KD + RFS t = 1,0", 640),
          ("P4noKD", "s5_960", "Cách 3: ảnh 960 px", 960),
          ("P4t1", "s5moto_960", "Kết hợp 2 + 3 + 4", 960))
SIZE_COLOR = {640: "#7a7975", 960: "#eda100"}


def hinh15_motorcycle(runs: list[Run], out: Path, seed: int = 42) -> None:
    items = []
    for m, sc, label, size in ADR023:
        r = next((r for r in runs if r.method == m and r.scenario == sc and r.seed == seed), None)
        if r is not None:
            items.append((label, size, r.metrics))
    if len(items) < 2:
        return
    panels = (("AP50_motorcycle", "motorcycle"), ("AP50_bus", "bus"), ("AP50_truck", "truck"),
              ("AP50_car", "car"), ("mAP50", "mAP50 (4 lớp)"))
    ref = items[0][2]
    fig, axes = plt.subplots(1, len(panels), figsize=(11.5, 0.9 + 0.45 * len(items)), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1]})
    y = np.arange(len(items))
    for ax, (key, title) in zip(axes, panels):
        vals = [it[2][key] for it in items]
        ax.barh(y, vals, height=0.62, color=[SIZE_COLOR[it[1]] for it in items], edgecolor=SURFACE, linewidth=2)
        for yi, v in zip(y, vals):
            d = round(v - ref[key], 3) + 0.0                 # no "-0.000"
            text = f"{v:.3f}" if yi == 0 else f"{v:.3f}  ({d:+.3f})"
            ax.annotate(text, (v, yi), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7.5,
                        color=INK if key == "AP50_motorcycle" else INK2)
        ax.axvline(ref[key], color=INK2, lw=0.8, ls="--")
        ax.set_xlim(0, 1.3)                                    # one scale in every panel
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [it[0] for it in items], fontsize=8.5)
    axes[0].set_ylim(len(items) - 0.5, -0.5)
    axes[2].set_xlabel(f"AP50, {RULE}; trong ngoặc: chênh lệch so với mốc (đường đứt)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SIZE_COLOR.values()]
    fig.legend(handles, ["Train và đánh giá ở 640 px", "Train và đánh giá ở 960 px"], loc="lower right",
               bbox_to_anchor=(0.99, -0.06), ncols=2, fontsize=8)
    _suptitle(fig, f"Cải thiện lớp motorcycle (ADR-023): kết hợp đưa AP50 motorcycle từ "
                   f"{ref['AP50_motorcycle']:.3f} lên {items[-1][2]['AP50_motorcycle']:.3f}"
                   f" · thiếu bus 3/5 · seed {seed} (n = 1, khả thi)")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    rows = [[it[0], it[1], *(f"{it[2][k]:.4f}" for k, _ in panels)] for it in items]
    _save(fig, out, "hinh15_cai_thien_motorcycle", ["config", "image_size", *(k for k, _ in panels)], rows)


R40_ID = "EXP-2026-10-25-S22-P4t1-s9bus8-y11s960-40r-seed42"     # ADR-025, 40 rounds
R20_ID = "EXP-2026-10-23-S21-P4t1-s9bus8-y11s960-seed42"         # ADR-024, 20 rounds (s21c)
CLASS_COLOR = dict(zip(CLASSES, COLOR.values()))                  # same 4 hues, fixed order


def _rounds(entry: dict) -> dict[int, dict]:
    res = entry.get("results", {})
    return {int(k.split("_")[1]): v for k, v in res.items() if k.startswith("round_")}


def hinh16_40_vong(registry: Path, out: Path) -> None:
    entries = {e.get("id"): e for e in yaml.safe_load(registry.read_text(encoding="utf-8"))["experiments"]}
    if R40_ID not in entries or entries[R40_ID].get("status") != "completed":
        return
    r40 = _rounds(entries[R40_ID])
    r20 = _rounds(entries[R20_ID]) if R20_ID in entries else {}
    series = [(f"AP50_{c}", c, CLASS_COLOR[c]) for c in CLASSES] + [("mAP50", "mAP50 (4 lớp)", INK)]
    rounds = sorted(r40)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})

    for key, label, color in series:
        ys = [r40[r][key] for r in rounds]
        ax1.plot(rounds, ys, color=color, marker="o", ms=5, mec=SURFACE, mew=1.5, label=label,
                 ls="-" if key != "mAP50" else (0, (4, 2)))
        ax1.annotate(f"{label.split(' ')[0]} {ys[-1]:.3f}", (rounds[-1], ys[-1]), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=7.5, color=INK2)
        if r20:
            xs = sorted(r20)
            ax1.scatter(xs, [r20[r][key] for r in xs], s=34, facecolors=SURFACE, edgecolors=color, lw=1.4, zorder=3)
    ax1.axvspan(29, 41, color=GRID, alpha=0.45, lw=0, zorder=0)
    ax1.text(35, 0.43, "vòng báo cáo\n30 và 40", ha="center", fontsize=7.5, color=INK2)
    ax1.set_xticks(rounds)
    ax1.set_xlim(7, 47)
    ax1.set_ylim(0.4, 0.85)
    ax1.set_xlabel("Vòng FL")
    ax1.set_ylabel("AP50")
    ax1.set_title("Đường cong 40 vòng (chấm rỗng: lần chạy 20 vòng, s21c)")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=5, fontsize=7.5)

    groups = [("Vòng 20 (s21c, 20 vòng)", r20.get(20), "#b5b4ae"), ("Vòng 30", r40.get(30), "#7a7975"),
              ("Vòng 40", r40.get(40), INK)]
    groups = [g for g in groups if g[1] is not None]
    x = np.arange(len(series))
    w = 0.8 / len(groups)
    for i, (label, vals, color) in enumerate(groups):
        ys = [vals[k] for k, _, _ in series]
        xs = x - 0.4 + w * (i + 0.5)
        ax2.bar(xs, ys, width=w, color=color, edgecolor=SURFACE, linewidth=2, label=label)
        for xi, v in zip(xs, ys):
            ax2.annotate(f"{v:.3f}", (xi, v), xytext=(0, 3), textcoords="offset points", ha="center",
                         rotation=90, fontsize=6.8, color=INK2)
    ax2.set_xticks(x, [lab.replace(" (4 lớp)", "\n(4 lớp)") for _, lab, _ in series])
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis="x", visible=False)
    ax2.set_ylabel("AP50")
    ax2.set_title("So sánh theo lớp")
    ax2.legend(loc="upper right", fontsize=7.5)

    _suptitle(fig, "9 client, thiếu bus 8/9, YOLO11s 960 px, P4t1: 40 vòng so với 20 vòng · seed 42 (n = 1, mô tả)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    rows = [["run_40r", r, *(f"{r40[r][k]:.4f}" for k, _, _ in series)] for r in rounds]
    rows += [["run_20r_s21c", r, *(f"{r20[r][k]:.4f}" for k, _, _ in series)] for r in sorted(r20)]
    _save(fig, out, "hinh16_40_vong", ["run", "round", *(k for k, _, _ in series)], rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", type=Path, default=_REPO_ROOT / "research" / "experiment_registry" / "registry.yaml")
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "research" / "figures" / "report")
    args = ap.parse_args()
    _style()
    runs = load_runs(args.registry)
    le = _local_eval()
    print(f"{len(runs)} registry runs; {len(le)} re-evaluated checkpoints")
    hinh02_kich_ban(args.out)
    hinh03_phan_bo_box(args.out)
    hinh04_ap50_bus_theo_muc_thieu(runs, args.out)
    hinh05_muc_suy_giam(runs, args.out)
    hinh06_ap50_theo_lop(runs, args.out)
    hinh07_ket_cuc_bus(le, args.out)
    hinh08_chenh_lech_ghep_cap(runs, args.out)
    hinh09_pr_bus(le, args.out)
    hinh10_confusion(le, args.out, "A0c")
    hinh10_confusion(le, args.out, "P4noKD")
    hinh11_ablation(runs, args.out)
    hinh12_histogram_diem(le, args.out)
    hinh13_giu_lai(runs, args.out)
    hinh14_thoi_gian(args.out)
    hinh15_motorcycle(runs, args.out)
    hinh16_40_vong(args.registry, args.out)


if __name__ == "__main__":
    main()
