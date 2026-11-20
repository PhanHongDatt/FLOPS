"""ADR-039 baseline comparison, YOLO11m 960 px, missing-class scenario, seed 42. Output: analysis_charts/report_20261118/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KO = Path("D:/FLOPS/kaggle/output")
OUT = Path("D:/FLOPS/analysis_charts/report_20261118")
CLASSES = ["car", "bus", "truck", "motorcycle"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True, "savefig.dpi": 300, "savefig.bbox": "tight"})
MISS = "s1b5_bus_3k_allmoto_seed42"
# label, run dir, colour; order = from plain to ours, centralized last as the reference
METHODS = [
    ("FedAvg", f"A0cm960_feasibility_seed42_{MISS}", "#8c8c8c"),
    ("FedAvg theo số box", f"A1cm960_feasibility_seed42_{MISS}", "#b0b0b0"),
    ("FedProx", f"FedProxcm960_feasibility_seed42_{MISS}", "#9fb3c8"),
    ("FedNTD", f"A5cm960_feasibility_seed42_{MISS}", "#6f8fb0"),
    ("Ours", f"P4t1m960_feasibility_seed42_{MISS}", "#1f5fbf"),
    ("Tập trung", "A0cm960_feasibility_seed42_s1b5_central_3k_allmoto_seed42", "#2e8b57"),
]


def load(run):
    hits = sorted(KO.glob(f"*/artifacts/runs/{run}/round_metrics.csv"))
    if not hits:
        return None
    rm = pd.read_csv(hits[0]).set_index("round")
    if 20 not in rm.index:
        return None
    return rm.loc[[10, 20]].mean(numeric_only=True)


rows = {lab: load(run) for lab, run, _ in METHODS}
avail = [(lab, col) for lab, _, col in METHODS if rows[lab] is not None]
missing_runs = [lab for lab, _, _ in METHODS if rows[lab] is None]
print("missing runs:", missing_runs)
tab = pd.DataFrame({lab: rows[lab] for lab, _ in avail}).T
keep = ["mAP50"] + [f"{k}_{c}" for k in ("AP50", "precision", "recall") for c in CLASSES] \
       + [f"{k}_{c}" for c in CLASSES for k in ("TP", "FP", "FN")]
tab[keep].to_csv(OUT / "baselines_seed42_mean_r10_r20.csv")
print(tab[keep].T.round(3).to_string())
note = "Kịch bản thiếu lớp, YOLO11m 960 px, 20 vòng, seed 42, trung bình vòng 10 và 20. " \
       "Tập trung = một máy giữ đủ 15.000 ảnh, huấn luyện như FedAvg (không KD, không LR cosine, không RFS)."


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


# ---------- Fig 11: AP50 per class ----------
cols = ["mAP50"] + [f"AP50_{c}" for c in CLASSES]
labels = ["Trung bình\n(mAP50)", "car", "bus", "truck", "motorcycle"]
n = len(avail)
wb = 0.8 / n
x = np.arange(len(cols))
fig, ax = plt.subplots(figsize=(13, 5.2))
for k, (lab, col) in enumerate(avail):
    v = [tab.loc[lab, c] for c in cols]
    xo = x + (k - (n - 1) / 2) * wb
    ax.bar(xo, v, wb, color=col, label=lab, edgecolor="white",
           hatch="//" if lab == "Tập trung" else None)
    for xi, vi in zip(xo, v):
        ax.text(xi, vi + 0.01, f"{vi:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=90)
ax.axhline(0.70, color="#2e8b57", ls="--", lw=1.1)
ax.text(len(cols) - 0.5, 0.705, "mục tiêu 0,70", color="#2e8b57", ha="right", fontsize=9)
ax.set_xticks(x, labels)
ax.set_ylim(0, 1.05)
ax.set_ylabel("AP50")
ax.set_title("So sánh với các baseline: AP50 theo lớp", fontweight="bold")
ax.legend(frameon=False, ncol=n, loc="upper center", fontsize=9.5)
fig.text(0.5, -0.03, note, ha="center", fontsize=9.5, color="#444")
save(fig, "fig11_so_sanh_baseline_ap50")

# ---------- Fig 12: precision and recall ----------
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
xc = np.arange(len(CLASSES))
for ax, key, title in ((axes[0], "precision", "Precision (tỷ lệ phát hiện đúng)"),
                       (axes[1], "recall", "Recall (tỷ lệ vật thể được tìm thấy)")):
    for k, (lab, col) in enumerate(avail):
        v = [tab.loc[lab, f"{key}_{c}"] for c in CLASSES]
        xo = xc + (k - (n - 1) / 2) * wb
        ax.bar(xo, v, wb, color=col, label=lab, edgecolor="white", hatch="//" if lab == "Tập trung" else None)
        for xi, vi in zip(xo, v):
            ax.text(xi, vi + 0.01, f"{vi:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(xc, CLASSES)
    ax.set_ylim(0, 1.12)
    ax.set_title(title, fontweight="bold")
axes[0].set_ylabel("Giá trị")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, ncol=n, loc="upper center", fontsize=9.5, bbox_to_anchor=(0.5, 1.0))
fig.suptitle("So sánh với các baseline: precision và recall theo lớp", fontsize=12, fontweight="bold", y=1.06)
fig.text(0.5, -0.03, note + " Ngưỡng tin cậy cho F1 trung bình cao nhất (cách Ultralytics báo cáo).",
         ha="center", fontsize=9, color="#444")
save(fig, "fig12_so_sanh_baseline_precision_recall")

# ---------- Fig 13: TP / FP / FN ----------
C_TP, C_FP, C_FN = "#2e8b57", "#c9653b", "#8c8c8c"
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
xa = np.arange(n)
w3 = 0.27
for ax, c in zip(axes.flat, CLASSES):
    top = 0
    for j, (k, col, lab) in enumerate((("TP", C_TP, "Phát hiện đúng (TP)"), ("FP", C_FP, "Báo nhầm (FP)"),
                                       ("FN", C_FN, "Bỏ sót (FN)"))):
        v = [tab.loc[m, f"{k}_{c}"] for m, _ in avail]
        xo = xa + (j - 1) * w3
        ax.bar(xo, v, w3, color=col, label=lab, edgecolor="white")
        for xi, vi in zip(xo, v):
            ax.text(xi, vi * 1.01, f"{vi:,.0f}".replace(",", "."), ha="center", va="bottom", fontsize=7, rotation=90)
        top = max(top, max(v))
    gt = tab.iloc[0][f"TP_{c}"] + tab.iloc[0][f"FN_{c}"]
    ax.set_title(f"{c}  (số vật thể thật: {gt:,.0f})".replace(",", "."), fontweight="bold")
    ax.set_xticks(xa, [m.replace(" theo ", "\ntheo ") for m, _ in avail], fontsize=9)
    ax.set_ylim(0, top * 1.25)
axes[0, 0].set_ylabel("Số lượng")
axes[1, 0].set_ylabel("Số lượng")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.965))
fig.suptitle("So sánh với các baseline: phát hiện đúng, báo nhầm và bỏ sót (ngưỡng tin cậy 0,25)",
             fontsize=12, fontweight="bold", y=1.0)
fig.text(0.5, -0.01, note, ha="center", fontsize=9.5, color="#444")
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "fig13_so_sanh_baseline_tp_fp_fn")
print("saved to", OUT)
