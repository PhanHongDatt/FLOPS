"""Report figures for the supervisor meeting (2026-11-18). Output: analysis_charts/report_20261118/."""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path("D:/FLOPS")
KO = ROOT / "kaggle/output"
OUT = ROOT / "analysis_charts/report_20261118"
OUT.mkdir(parents=True, exist_ok=True)

C_OURS, C_FEDAVG, C_OTHER, C_BAD, C_REF = "#1f5fbf", "#8c8c8c", "#b9c7dc", "#c9653b", "#2e8b57"
CLASSES = ["car", "bus", "truck", "motorcycle"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
                     "axes.axisbelow": True, "savefig.dpi": 300, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


# ---------- data: per-run mean of rounds 10 and 20 ----------
def load_runs():
    rows = []
    for d in glob.glob(str(KO / "*/artifacts/runs/*m960_feasibility_seed*_s1b5_*_3k_allmoto_seed42")):
        d = Path(d)
        name = d.name
        if not (name.startswith("A0cm960") or name.startswith("P4t1m960")) or "_s1b5_central_" in name:
            continue
        rm = pd.read_csv(d / "round_metrics.csv").set_index("round").loc[[10, 20]]
        r = {"method": "Ours" if name.startswith("P4t1") else "FedAvg",
             "scenario": "missing" if "_s1b5_bus_" in name else "ideal",
             "seed": int(name.split("_seed")[1].split("_")[0]), "mAP50": rm["mAP50"].mean()}
        for c in CLASSES:
            r[c] = rm[f"AP50_{c}"].mean()
            r[f"rec_{c}"] = rm[f"recall_{c}"].mean()        # Ultralytics P/R: one confidence, max mean F1
            r[f"prec_{c}"] = rm[f"precision_{c}"].mean()
            for k in ("TP", "FP", "FN"):
                r[f"{k}_{c}"] = rm[f"{k}_{c}"].mean()   # confusion matrix at confidence 0.25
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["method", "scenario", "seed"])


runs = load_runs()
runs.to_csv(OUT / "runs_mean_r10_r20.csv", index=False)
print(runs.round(4).to_string())


def stat(method, scenario, col):
    v = runs[(runs.method == method) & (runs.scenario == scenario)][col]
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0), len(v)


def gap(method, col):
    """ideal - missing, paired by seed."""
    a = runs[runs.method == method].pivot(index="seed", columns="scenario", values=col).dropna()
    g = a["ideal"] - a["missing"]
    return g.mean(), (g.std(ddof=1) if len(g) > 1 else 0.0), len(g)


# ---------- Fig 1: data distribution ----------
def manifest(pattern):
    f = glob.glob(str(KO / f"*/artifacts/runs/{pattern}/partition_manifest.yaml"))[0]
    return yaml.safe_load(open(f, encoding="utf-8"))["class_counts"]


ideal = manifest("P4t1m960_feasibility_seed42_s1b5_pooled_3k_allmoto_seed42")
missing = manifest("P4t1m960_feasibility_seed42_s1b5_bus_3k_allmoto_seed42")
clients = [f"C{i}" for i in range(5)]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, cc, title in ((axes[0], ideal, "Kịch bản lý tưởng"), (axes[1], missing, "Kịch bản thiếu lớp")):
    m = np.array([[cc[k][c] for c in CLASSES] for k in clients], float)
    share = m / m.sum(axis=0, keepdims=True) * 100
    ax.imshow(share, cmap="Blues", vmin=0, vmax=60, aspect="auto")
    for i in range(5):
        for j in range(4):
            v = int(m[i, j])
            ax.text(j, i, "0\n(thiếu)" if v == 0 else f"{v:,}".replace(",", "."), ha="center", va="center",
                    fontsize=10, color=C_BAD if v == 0 else ("white" if share[i, j] > 35 else "black"),
                    fontweight="bold" if v == 0 else "normal")
    ax.set_xticks(range(4), ["car", "bus", "truck", "motorcycle"])
    ax.set_yticks(range(5), [f"Client {k[1]}" for k in clients])
    ax.set_title(title, fontweight="bold")
    ax.grid(False)
fig.suptitle("Số hộp mỗi lớp ở từng client (mỗi client 3.000 ảnh, tổng 15.000 ảnh như nhau ở hai kịch bản)",
             fontsize=11)
fig.text(0.5, -0.03, "Màu: tỷ lệ % hộp của lớp đó nằm ở client. Lý tưởng: mỗi client ~20 %. "
         "Thiếu lớp: bus chỉ ở client 3, 4; truck vắng ở client 3.", ha="center", fontsize=9.5, color="#444")
save(fig, "fig1_phan_phoi_du_lieu")

# ---------- Fig 2: main result per class, missing scenario ----------
cols = ["mAP50"] + CLASSES
labels = ["Trung bình\n(mAP50)", "car", "bus", "truck", "motorcycle"]
x = np.arange(len(cols))
w = 0.38
fig, ax = plt.subplots(figsize=(9.5, 4.6))
for k, (meth, col, off) in enumerate((("FedAvg", C_FEDAVG, -w / 2), ("Ours", C_OURS, w / 2))):
    mu = [stat(meth, "missing", c)[0] for c in cols]
    sd = [stat(meth, "missing", c)[1] for c in cols]
    n = stat(meth, "missing", "mAP50")[2]
    ax.bar(x + off, mu, w, yerr=sd, capsize=3, color=col, label=f"{meth} ({n} seed)", edgecolor="white")
    for xi, v, e in zip(x + off, mu, sd):
        ax.text(xi, v + e + 0.012, f"{v:.3f}", ha="center", fontsize=9)
ax.axhline(0.70, color=C_REF, ls="--", lw=1.2)
ax.text(len(cols) - 0.5, 0.705, "mục tiêu 0,70", color=C_REF, ha="right", fontsize=9)
ax.set_xticks(x, labels)
ax.set_ylim(0, 0.92)
ax.set_ylabel("AP50")
ax.set_title("Kết quả trên kịch bản thiếu lớp (YOLO11m, 960 px, 20 vòng)", fontweight="bold")
ax.legend(loc="upper right", frameon=False)
save(fig, "fig2_ket_qua_theo_lop")

# ---------- Fig 3: drop caused by missing classes ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
for meth, col, off in (("FedAvg", C_FEDAVG, -w / 2), ("Ours", C_OURS, w / 2)):
    g = [gap(meth, c) for c in cols]
    ax.bar(x + off, [a for a, _, _ in g], w, yerr=[b for _, b, _ in g], capsize=3, color=col,
           label=f"{meth} ({g[0][2]} seed)", edgecolor="white")
    for xi, (v, e, _) in zip(x + off, g):
        ax.text(xi, max(v, 0) + e + 0.0008, f"{v:.3f}", ha="center", fontsize=9)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x, labels)
ax.set_ylabel("AP50 lý tưởng − AP50 thiếu lớp")
ax.set_title("Mức tụt AP50 do thiếu lớp (càng thấp càng tốt)", fontweight="bold")
ax.legend(frameon=False)
ax = axes[1]
for i, (meth, col) in enumerate((("FedAvg", C_FEDAVG), ("Ours", C_OURS))):
    mi, ii = stat(meth, "missing", "rec_bus")[0], stat(meth, "ideal", "rec_bus")[0]
    ax.plot([mi, ii], [i, i], color=col, lw=3, zorder=1)
    ax.scatter([ii], [i], s=90, color="white", edgecolor=col, lw=2, zorder=2, label="Lý tưởng" if i == 0 else None)
    ax.scatter([mi], [i], s=90, color=col, zorder=2, label="Thiếu lớp" if i == 0 else None)
    ax.text(mi, i + 0.18, f"{mi:.2f}", ha="center", fontsize=9)
    ax.text(ii, i + 0.18, f"{ii:.2f}", ha="center", fontsize=9)
    ax.text((mi + ii) / 2, i - 0.28, f"tụt {ii - mi:.2f}", ha="center", fontsize=9, color=col)
ax.set_yticks([0, 1], ["FedAvg", "Ours"])
ax.set_ylim(-0.6, 1.6)
ax.set_xlim(0, 0.65)
ax.set_xlabel("Recall bus")
ax.set_title("Bus bị bỏ sót khi thiếu lớp", fontweight="bold")
leg = ax.legend(frameon=False, loc="lower right")
for h in leg.legend_handles:
    h.set_edgecolor("black")
save(fig, "fig3_muc_tut_do_thieu_lop")

# ---------- Fig 4 / 5: model comparison on the same data (960 px, same partitions) ----------
def run_rounds(pattern):
    for d in sorted(glob.glob(str(KO / f"*/artifacts/runs/{pattern}"))):
        rm = pd.read_csv(Path(d) / "round_metrics.csv").set_index("round")
        if 20 in rm.index:   # skip partial sessions
            return rm
    raise FileNotFoundError(pattern)


def mean_r(rm, col, rounds=(10, 20)):
    return rm.loc[list(rounds), col].mean()


MODELS = {"YOLO11s": "P4t1i960_feasibility_seed42_s1b5_{}_3k_allmoto_seed42",
          "YOLO11m": "P4t1m960_feasibility_seed{}_s1b5_{}_3k_allmoto_seed42"}
s_miss = run_rounds(MODELS["YOLO11s"].format("bus"))
s_ideal = run_rounds(MODELS["YOLO11s"].format("pooled"))
m_miss = run_rounds(MODELS["YOLO11m"].format(42, "bus"))
cc = {"mAP50": "mAP50", **{c: f"AP50_{c}" for c in CLASSES}}
series = [
    ("YOLO11s gốc COCO (chưa huấn luyện, không phương pháp)", [s_miss.loc[0, cc[c]] for c in cols], "#d5dbe5", None),
    ("YOLO11s + Ours", [mean_r(s_miss, cc[c]) for c in cols], C_OTHER, None),
    ("YOLO11m gốc COCO (chưa huấn luyện, không phương pháp)", [m_miss.loc[0, cc[c]] for c in cols], "#d5dbe5", "//"),
    ("YOLO11m + Ours", [stat("Ours", "missing", c)[0] for c in cols], C_OURS, None),
]
fig, ax = plt.subplots(figsize=(11.5, 4.8))
wb = 0.2
for k, (lab, vals, col, hatch) in enumerate(series):
    xo = x + (k - 1.5) * wb
    ax.bar(xo, vals, wb, color=col, hatch=hatch, edgecolor="white" if hatch is None else "#7a8799", lw=0.6,
           label=lab)
    for xi, v in zip(xo, vals):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
ax.axhline(0.70, color=C_REF, ls="--", lw=1.2)
ax.text(len(cols) - 0.5, 0.71, "mục tiêu 0,70", color=C_REF, ha="right", fontsize=9)
ax.set_xticks(x, labels)
ax.set_ylim(0, 1.08)
ax.set_ylabel("AP50")
ax.set_title("Đổi mô hình YOLO11s → YOLO11m (cùng dữ liệu, 960 px, kịch bản thiếu lớp)", fontweight="bold")
ax.legend(frameon=False, ncol=2, loc="upper center", fontsize=9.5)
fig.text(0.5, -0.03, "Cột nhạt: trọng số COCO tải về, đánh giá thẳng trên BDD100K (vòng 0), chưa áp dụng phương pháp nào. Cột đậm: sau 20 vòng FL bằng Ours. YOLO11s 1 seed, YOLO11m 2 seed.",
         ha="center", fontsize=9.5, color="#444")
save(fig, "fig4_so_sanh_mo_hinh")

# Fig 5: missing-class drop per model
drops = [
    ("YOLO11s + Ours (1 seed)", [mean_r(s_ideal, cc[c]) - mean_r(s_miss, cc[c]) for c in cols], [0] * len(cols),
     C_OTHER),
    ("YOLO11m + FedAvg (3 seed)", [gap("FedAvg", c)[0] for c in cols], [gap("FedAvg", c)[1] for c in cols],
     C_FEDAVG),
    ("YOLO11m + Ours (2 seed)", [gap("Ours", c)[0] for c in cols], [gap("Ours", c)[1] for c in cols], C_OURS),
]
fig, ax = plt.subplots(figsize=(10.5, 4.6))
wb = 0.26
for k, (lab, vals, err, col) in enumerate(drops):
    xo = x + (k - 1) * wb
    ax.bar(xo, vals, wb, yerr=err, capsize=3, color=col, label=lab, edgecolor="white")
    for xi, v, e in zip(xo, vals, err):
        ax.text(xi, max(v, 0) + e + 0.0008, f"{v:.3f}", ha="center", fontsize=8.5)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x, labels)
ax.set_ylabel("AP50 lý tưởng − AP50 thiếu lớp")
ax.set_title("Mức tụt do thiếu lớp theo mô hình (càng thấp càng tốt)", fontweight="bold")
ax.set_ylim(0, 0.052)
ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=9.5)
fig.text(0.5, -0.03, "Đổi YOLO11s sang YOLO11m nâng AP mọi lớp nhưng mức tụt bus của Ours giữ nguyên (0,038).",
         ha="center", fontsize=9.5, color="#444")
save(fig, "fig5_muc_tut_theo_mo_hinh")

# ---------- Fig 6: diagnosis ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
ax = axes[0]
grp = ["FedAvg", "Ours"]
sc = {"FedAvg": (0.076, 0.290), "Ours": (0.267, 0.365)}
xx = np.arange(2)
ax.bar(xx - w / 2, [sc[g][0] for g in grp], w, color=[C_FEDAVG, C_OURS], label="Thiếu lớp")
ax.bar(xx + w / 2, [sc[g][1] for g in grp], w, color="white", edgecolor=[C_FEDAVG, C_OURS], lw=2,
       label="Lý tưởng")
for xi, g in zip(xx, grp):
    ax.text(xi - w / 2, sc[g][0] + 0.008, f"{sc[g][0]:.3f}", ha="center", fontsize=9)
    ax.text(xi + w / 2, sc[g][1] + 0.008, f"{sc[g][1]:.3f}", ha="center", fontsize=9)
ax.axhline(0.264, color=C_REF, ls="--", lw=1.2)
ax.text(-0.45, 0.27, "mô hình COCO ban đầu", color=C_REF, ha="left", fontsize=9)
ax.set_xticks(xx, grp)
ax.set_ylim(0, 0.42)
ax.set_ylabel("Điểm tin cậy bus trung bình tại vật thể bus")
ax.set_title("Điểm tin cậy bus: thiếu lớp làm giảm,\nKD trong Ours giữ ở mức COCO", fontweight="bold")
leg = ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.92))
for h in leg.legend_handles:
    h.set_edgecolor("black")
ax = axes[1]
steps = ["Ban đầu", "Thay hàng bus\n(lớp cuối)", "Huấn luyện lại\nnhánh phân loại"]
v11m, v11s = [100, 86, 64], [100, 76, 66]
xs = np.arange(3)
ax.bar(xs - w / 2, v11m, w, color=C_OURS, label="YOLO11m")
ax.bar(xs + w / 2, v11s, w, color=C_OTHER, label="YOLO11s")
for xi, a, b in zip(xs, v11m, v11s):
    ax.text(xi - w / 2, a + 1.5, f"{a}%", ha="center", fontsize=9)
    ax.text(xi + w / 2, b + 1.5, f"{b}%", ha="center", fontsize=9)
ax.axhline(50, color=C_BAD, ls="--", lw=1.2)
ax.text(2.55, 52, "ngưỡng đạt (50 %)", color=C_BAD, ha="right", fontsize=9)
ax.set_xticks(xs, steps)
ax.set_ylim(0, 115)
ax.set_ylabel("Mức tụt bus còn lại (% so với ban đầu)")
ax.set_title("Chỉnh riêng phần phân loại\nkhông đóng được mức tụt bus", fontweight="bold")
ax.legend(frameon=False, loc="upper right")
save(fig, "fig6_chan_doan_nguyen_nhan")

# ---------- Fig 7: progress towards 0.70 (Ours, missing scenario) ----------
prog = [("YOLO11s\n640 px", 0.545, 0.501, 0.423),
        ("YOLO11s\n960 px", 0.6095, 0.565, 0.504),
        ("YOLO11m\n960 px", stat("Ours", "missing", "mAP50")[0], stat("Ours", "missing", "bus")[0],
         stat("Ours", "missing", "motorcycle")[0])]
fig, ax = plt.subplots(figsize=(8, 4.4))
xs = np.arange(len(prog))
for idx, lab, col, mk in ((1, "Trung bình (mAP50)", C_OURS, "o"), (2, "bus", C_FEDAVG, "s"),
                          (3, "motorcycle", C_BAD, "^")):
    vals = [p[idx] for p in prog]
    ax.plot(xs, vals, marker=mk, color=col, lw=2, ms=8, label=lab)
    for xi, v in zip(xs, vals):
        ax.text(xi + 0.06, v - 0.004, f"{v:.3f}", fontsize=9, va="top")
ax.axhline(0.70, color=C_REF, ls="--", lw=1.2)
ax.text(2.25, 0.705, "mục tiêu 0,70", color=C_REF, ha="right", fontsize=9)
ax.set_xticks(xs, [p[0] for p in prog])
ax.set_xlim(-0.3, 2.45)
ax.set_ylim(0.38, 0.74)
ax.set_ylabel("AP50")
ax.set_title("Ours qua các bước nâng cấu hình (kịch bản thiếu lớp)", fontweight="bold")
ax.legend(frameon=False, loc="lower right")
fig.text(0.5, -0.03, "Bước 640 px dùng phân chia dữ liệu khác bước 960 px; chỉ đọc xu hướng.", ha="center",
         fontsize=9.5, color="#444")
save(fig, "fig7_tien_do_den_muc_tieu")

# ---------- Fig 8: precision and recall per class ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
wb = 0.2
arms = [("FedAvg", "missing", C_FEDAVG, None, "FedAvg – thiếu lớp"), ("FedAvg", "ideal", "white", C_FEDAVG, "FedAvg – lý tưởng"),
        ("Ours", "missing", C_OURS, None, "Ours – thiếu lớp"), ("Ours", "ideal", "white", C_OURS, "Ours – lý tưởng")]
xc = np.arange(len(CLASSES))
for ax, key, title in ((axes[0], "prec", "Precision (tỷ lệ phát hiện đúng)"), (axes[1], "rec", "Recall (tỷ lệ vật thể được tìm thấy)")):
    for k, (meth, scen, fc, ec, lab) in enumerate(arms):
        mu = [stat(meth, scen, f"{key}_{c}")[0] for c in CLASSES]
        sd = [stat(meth, scen, f"{key}_{c}")[1] for c in CLASSES]
        xo = xc + (k - 1.5) * wb
        ax.bar(xo, mu, wb, yerr=sd, capsize=2, color=fc, edgecolor=ec or "white", lw=1.6 if ec else 0.6, label=lab)
        for xi, v, e in zip(xo, mu, sd):
            ax.text(xi, v + e + 0.012, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(xc, CLASSES)
    ax.set_ylim(0, 1.08)
    ax.set_title(title, fontweight="bold")
axes[0].set_ylabel("Giá trị")
leg = axes[0].legend(frameon=False, ncol=2, loc="upper right", fontsize=9)
fig.suptitle("Precision và recall theo lớp (YOLO11m, 960 px)", fontsize=12, fontweight="bold", y=1.03)
fig.text(0.5, -0.03, "Cùng một ngưỡng tin cậy cho mọi lớp (ngưỡng cho F1 trung bình cao nhất, cách Ultralytics báo cáo). "
         "FedAvg 3 seed, Ours 2 seed.", ha="center", fontsize=9.5, color="#444")
save(fig, "fig8_precision_recall")
print(runs.groupby(["method", "scenario"])[[f"{k}_{c}" for k in ("prec", "rec") for c in CLASSES]].mean().round(3).T.to_string())

# ---------- Fig 9: AP50 ideal vs missing, side by side ----------
fig, ax = plt.subplots(figsize=(12, 5))
wb = 0.2
order = [("FedAvg", "ideal", "white", C_FEDAVG, "FedAvg – lý tưởng (đủ lớp)"), ("FedAvg", "missing", C_FEDAVG, None, "FedAvg – thiếu lớp"),
         ("Ours", "ideal", "white", C_OURS, "Ours – lý tưởng (đủ lớp)"), ("Ours", "missing", C_OURS, None, "Ours – thiếu lớp")]
for k, (meth, scen, fc, ec, lab) in enumerate(order):
    mu = [stat(meth, scen, c)[0] for c in cols]
    sd = [stat(meth, scen, c)[1] for c in cols]
    xo = x + (k - 1.5) * wb
    ax.bar(xo, mu, wb, yerr=sd, capsize=2, color=fc, edgecolor=ec or "white", lw=1.6 if ec else 0.6, label=lab)
    for xi, v, e in zip(xo, mu, sd):
        ax.text(xi, v + e + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=90)
ax.axhline(0.70, color=C_REF, ls="--", lw=1.2)
ax.text(len(cols) - 0.5, 0.71, "mục tiêu 0,70", color=C_REF, ha="right", fontsize=9)
ax.set_xticks(x, labels)
ax.set_ylim(0, 1.05)
ax.set_ylabel("AP50")
ax.set_title("AP50: kịch bản lý tưởng (đủ lớp) so với thiếu lớp (YOLO11m, 960 px)", fontweight="bold")
leg = ax.legend(frameon=False, ncol=2, loc="upper center", fontsize=9.5)
for h in leg.legend_handles:
    h.set_linewidth(1.6)
fig.text(0.5, -0.03, "Cột rỗng: đủ lớp ở mọi client. Cột đặc: client 0–2 thiếu bus, client 3 thiếu truck. "
         "FedAvg 3 seed, Ours 2 seed.", ha="center", fontsize=9.5, color="#444")
save(fig, "fig9_ly_tuong_vs_thieu_lop")
print(runs.groupby(["method", "scenario"])[cols].mean().round(4).to_string())

# ---------- Fig 10: TP / FP / FN per class at confidence 0.25 ----------
C_TP, C_FP, C_FN = "#2e8b57", "#c9653b", "#8c8c8c"
arms10 = [("FedAvg", "ideal", "FedAvg\nđủ lớp"), ("FedAvg", "missing", "FedAvg\nthiếu lớp"),
          ("Ours", "ideal", "Ours\nđủ lớp"), ("Ours", "missing", "Ours\nthiếu lớp")]
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
xa = np.arange(len(arms10))
wb = 0.26
for ax, c in zip(axes.flat, CLASSES):
    top = 0
    for j, (k, col, lab) in enumerate((("TP", C_TP, "Phát hiện đúng (TP)"), ("FP", C_FP, "Báo nhầm (FP)"),
                                       ("FN", C_FN, "Bỏ sót (FN)"))):
        vals = [stat(m, sc, f"{k}_{c}")[0] for m, sc, _ in arms10]
        xo = xa + (j - 1) * wb
        ax.bar(xo, vals, wb, color=col, label=lab, edgecolor="white")
        for xi, v in zip(xo, vals):
            ax.text(xi, v * 1.01, f"{v:,.0f}".replace(",", "."), ha="center", va="bottom", fontsize=7.5)
        top = max(top, max(vals))
    n_gt = stat("Ours", "ideal", f"TP_{c}")[0] + stat("Ours", "ideal", f"FN_{c}")[0]
    ax.set_title(f"{c}  (số vật thể thật: {n_gt:,.0f})".replace(",", "."), fontweight="bold")
    ax.set_xticks(xa, [a[2] for a in arms10], fontsize=9.5)
    ax.set_ylim(0, top * 1.15)
    ax.axvline(1.5, color="#bbbbbb", lw=0.8)
axes[0, 0].set_ylabel("Số lượng")
axes[1, 0].set_ylabel("Số lượng")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.965))
fig.suptitle("Phát hiện đúng, báo nhầm và bỏ sót theo lớp (YOLO11m, 960 px, ngưỡng tin cậy 0,25)",
             fontsize=12, fontweight="bold", y=1.0)
fig.text(0.5, -0.01, "Trung bình vòng 10 và 20, FedAvg 3 seed, Ours 2 seed. Một bus bị đoán thành truck được tính là "
         "bỏ sót của bus và báo nhầm của truck.", ha="center", fontsize=9.5, color="#444")
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "fig10_tp_fp_fn")
print(runs.groupby(["method", "scenario"])[[f"{k}_{c}" for c in CLASSES for k in ("TP", "FP", "FN")]].mean().round(0).T.to_string())
print("saved to", OUT)
