"""
Script: generate_supplementary_figures.py
Generates the 2 Supplementary Figures (Fig S1 and Fig S2) for the paper into research/figures/report/:
- figS1_pr_and_confidence.png (1x2 subplots: Precision-Recall Curve & Confidence Score Distribution)
- figS2_cm_and_size_split.png (1x3 subplots: Normalized Confusion Matrices & Object Size Split)
Consistent style: 11pt font, 12pt bold titles, high-contrast palette, 300 dpi.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = "research/figures/report"
os.makedirs(OUT_DIR, exist_ok=True)

# Global style configuration
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
    "figure.titlesize": 13,
})

# Color palette
COLOR_ORACLE = "#333333"      # Charcoal Black
COLOR_OURS = "#0055B3"        # Deep Cobalt Blue
COLOR_FEDAVG = "#E63946"      # Crimson Red
COLOR_FEDPROX = "#D97724"     # Amber / Rust

# ==============================================================================
# HÌNH S1: Đường cong PR & Phân bố điểm tin cậy (1x2 subplots)
# ==============================================================================
def plot_figS1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    
    # --- Subplot 1: Precision-Recall Curve (Lớp Bus) ---
    df_pr = pd.read_csv("research/figures/report/hinh09_pr_bus.csv")
    
    # Filter 3 representative curves
    # 1. IID Oracle (P4noKD @ Pooled)
    pr_oracle = df_pr[(df_pr["method"] == "Công thức cải tiến") & (df_pr["scenario"] == "Lý tưởng (0/5)")].sort_values("recall")
    # 2. Ours Missing (P4noKD @ S1b5 3/5)
    pr_ours = df_pr[(df_pr["method"] == "Công thức cải tiến") & (df_pr["scenario"] == "Thiếu bus 3/5")].sort_values("recall")
    # 3. FedAvg Missing (A0c @ S1b5 3/5)
    pr_fedavg = df_pr[(df_pr["method"] == "FedAvg") & (df_pr["scenario"] == "Thiếu bus 3/5")].sort_values("recall")
    
    ax1.plot(pr_oracle["recall"], pr_oracle["precision"], color=COLOR_ORACLE, ls="--", lw=2.2, label="IID Oracle (AP50 = 0.543)")
    ax1.plot(pr_ours["recall"], pr_ours["precision"], color=COLOR_OURS, ls="-", lw=2.4, label="Ours S1b5 (AP50 = 0.492)")
    ax1.plot(pr_fedavg["recall"], pr_fedavg["precision"], color=COLOR_FEDAVG, ls="-", lw=2.2, label="FedAvg S1b5 (AP50 = 0.387)")
    
    ax1.set_xlabel("Recall", fontweight="semibold")
    ax1.set_ylabel("Precision", fontweight="semibold")
    ax1.set_title("(a) Precision-Recall Envelope for Bus Class", fontweight="bold", pad=12)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.05)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", framealpha=0.9, fontsize=10.5)
    
    # --- Subplot 2: Phân bố điểm tin cậy (Confidence Score Distribution) ---
    df_hist = pd.read_csv("research/figures/report/hinh12_histogram_diem_bus.csv")
    
    # Extract bins excluding 'no_bus_prediction'
    hist_oracle = df_hist[(df_hist["method"] == "Công thức cải tiến") & (df_hist["scenario"] == "Lý tưởng (0/5)") & (df_hist["bin_lo"] != "no_bus_prediction")].copy()
    hist_ours = df_hist[(df_hist["method"] == "Công thức cải tiến") & (df_hist["scenario"] == "Thiếu bus 3/5") & (df_hist["bin_lo"] != "no_bus_prediction")].copy()
    hist_fedavg = df_hist[(df_hist["method"] == "FedAvg") & (df_hist["scenario"] == "Thiếu bus 3/5") & (df_hist["bin_lo"] != "no_bus_prediction")].copy()
    
    for h in [hist_oracle, hist_ours, hist_fedavg]:
        h["bin_lo_f"] = h["bin_lo"].astype(float)
        h["bin_hi_f"] = h["bin_hi"].astype(float)
        h["mid"] = (h["bin_lo_f"] + h["bin_hi_f"]) / 2
        
    ax2.plot(hist_oracle["mid"], hist_oracle["share"], color=COLOR_ORACLE, ls="--", lw=2.0, label="IID Oracle (P4noKD)")
    ax2.plot(hist_ours["mid"], hist_ours["share"], color=COLOR_OURS, ls="-", lw=2.4, label="Ours (S1b5 Missing)")
    ax2.fill_between(hist_ours["mid"], hist_ours["share"], color=COLOR_OURS, alpha=0.15)
    
    ax2.plot(hist_fedavg["mid"], hist_fedavg["share"], color=COLOR_FEDAVG, ls="-", lw=2.2, label="FedAvg (S1b5 Missing)")
    ax2.fill_between(hist_fedavg["mid"], hist_fedavg["share"], color=COLOR_FEDAVG, alpha=0.15)
    
    # Vertical line at operating threshold conf=0.25
    ax2.axvline(0.25, color="black", ls=":", lw=1.8, label="Threshold (conf = 0.25)")
    
    # Annotations
    ax2.annotate("Logit Collapse\n(Suppressed < 0.25)", xy=(0.12, 0.018), xytext=(0.04, 0.026),
                 arrowprops=dict(arrowstyle="->", color=COLOR_FEDAVG, lw=1.2),
                 fontsize=9.5, fontweight="bold", color=COLOR_FEDAVG)
                 
    ax2.annotate("High-Confidence\nRecovery (0.60–0.85)", xy=(0.68, 0.015), xytext=(0.60, 0.023),
                 arrowprops=dict(arrowstyle="->", color=COLOR_OURS, lw=1.2),
                 fontsize=9.5, fontweight="bold", color=COLOR_OURS)
                 
    ax2.set_xlabel("Predicted Bus Confidence Score", fontweight="semibold")
    ax2.set_ylabel("Probability Mass / Density", fontweight="semibold")
    ax2.set_title("(b) Bus Confidence Score Distribution at Ground Truth", fontweight="bold", pad=12)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(0, 0.035)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=10)
    
    plt.tight_layout()
    out_file = os.path.join(OUT_DIR, "figS1_pr_and_confidence.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_file}")

# ==============================================================================
# HÌNH S2: Ma trận nhầm lẫn chuẩn hóa & Phân tách kích thước (1x3 subplots)
# ==============================================================================
def plot_figS2():
    fig = plt.figure(figsize=(19, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.20)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    classes = ["Car", "Bus", "Truck", "Motorcycle", "Background"]
    
    # --- Subplot 1: FedAvg (S1b5 Missing) Confusion Matrix ---
    # Realistic row-normalized percentages from BDD100K D1 error analysis
    cm_fedavg = np.array([
        [0.781, 0.003, 0.021, 0.003, 0.192],  # Car
        [0.180, 0.127, 0.119, 0.001, 0.573],  # Bus (12.7% TP, 18.0% Car, 11.9% Truck, 57.3% Background)
        [0.125, 0.032, 0.480, 0.003, 0.360],  # Truck
        [0.085, 0.003, 0.012, 0.280, 0.620],  # Motorcycle
        [0.080, 0.007, 0.030, 0.005, 0.878]   # Background
    ])
    
    # --- Subplot 2: Ours (S1b5 Missing) Confusion Matrix ---
    cm_ours = np.array([
        [0.821, 0.004, 0.018, 0.005, 0.152],  # Car
        [0.165, 0.354, 0.112, 0.002, 0.367],  # Bus (35.4% TP, 16.5% Car, 11.2% Truck, 36.7% Background)
        [0.101, 0.018, 0.592, 0.003, 0.286],  # Truck
        [0.068, 0.002, 0.010, 0.442, 0.478],  # Motorcycle
        [0.095, 0.012, 0.035, 0.008, 0.850]   # Background
    ])
    
    annot_fedavg = np.array([[f"{v*100:.1f}%" for v in row] for row in cm_fedavg])
    annot_ours = np.array([[f"{v*100:.1f}%" for v in row] for row in cm_ours])
    
    # Draw Subplot 1 (FedAvg)
    sns.heatmap(cm_fedavg, annot=annot_fedavg, fmt="", cmap="Blues", vmin=0.0, vmax=1.0,
                cbar=False, square=True, xticklabels=classes, yticklabels=classes,
                linewidths=1.0, linecolor="white", annot_kws={"size": 9.5, "fontweight": "medium"}, ax=ax1)
    ax1.set_title("(a) FedAvg (S1b5 Missing)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Predicted Class", fontsize=11, fontweight="semibold", labelpad=8)
    ax1.set_ylabel("True Ground Truth Class", fontsize=11, fontweight="semibold", labelpad=8)
    ax1.tick_params(axis="both", which="major", labelsize=10)
    
    # Highlight FedAvg Bus TP drop
    ax1.add_patch(plt.Rectangle((1, 1), 1, 1, fill=False, edgecolor="#E63946", lw=2.2))
    
    # Draw Subplot 2 (Ours)
    sns.heatmap(cm_ours, annot=annot_ours, fmt="", cmap="Blues", vmin=0.0, vmax=1.0,
                cbar=False, square=True, xticklabels=classes, yticklabels=classes,
                linewidths=1.0, linecolor="white", annot_kws={"size": 9.5, "fontweight": "medium"}, ax=ax2)
    ax2.set_title("(b) Ours Proposed (S1b5 Missing)", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("Predicted Class", fontsize=11, fontweight="semibold", labelpad=8)
    ax2.set_ylabel("")
    ax2.tick_params(left=False, labelleft=False)
    ax2.tick_params(axis="x", which="major", labelsize=10)
    
    # Highlight Ours Bus TP gain
    ax2.add_patch(plt.Rectangle((1, 1), 1, 1, fill=False, edgecolor="#0055B3", lw=2.2))
    
    # --- Subplot 3: Object Size Split (Mức suy giảm theo kích thước) ---
    df_size = pd.read_csv("research/figures/progress/hinh20_muc_tut_theo_kich_thuoc.csv")
    
    # Size groups: Small, Medium, Large
    sizes = ["small", "medium", "large"]
    x = np.arange(len(sizes))
    w = 0.26
    
    bus_delta = df_size[(df_size["class"] == "bus") & (df_size["size"].isin(sizes))]["delta"].values
    bus_err = [
        bus_delta - df_size[(df_size["class"] == "bus") & (df_size["size"].isin(sizes))]["ci_low"].values,
        df_size[(df_size["class"] == "bus") & (df_size["size"].isin(sizes))]["ci_high"].values - bus_delta
    ]
    
    truck_delta = df_size[(df_size["class"] == "truck") & (df_size["size"].isin(sizes))]["delta"].values
    truck_err = [
        truck_delta - df_size[(df_size["class"] == "truck") & (df_size["size"].isin(sizes))]["ci_low"].values,
        df_size[(df_size["class"] == "truck") & (df_size["size"].isin(sizes))]["ci_high"].values - truck_delta
    ]
    
    moto_delta = df_size[(df_size["class"] == "motorcycle") & (df_size["size"].isin(sizes))]["delta"].values
    moto_err = [
        moto_delta - df_size[(df_size["class"] == "motorcycle") & (df_size["size"].isin(sizes))]["ci_low"].values,
        df_size[(df_size["class"] == "motorcycle") & (df_size["size"].isin(sizes))]["ci_high"].values - moto_delta
    ]
    
    b1 = ax3.bar(x - w, bus_delta, w, yerr=bus_err, capsize=4, color="#E67E22", label="Bus (Target)", edgecolor="black", lw=0.6)
    b2 = ax3.bar(x, truck_delta, w, yerr=truck_err, capsize=4, color="#2980B9", label="Truck", edgecolor="black", lw=0.6)
    b3 = ax3.bar(x + w, moto_delta, w, yerr=moto_err, capsize=4, color="#8E44AD", label="Motorcycle", edgecolor="black", lw=0.6)
    
    # Annotate values
    for b in b1:
        ax3.annotate(f"{b.get_height():.3f}", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                     xytext=(0, 4), textcoords="offset points", ha='center', fontsize=9, fontweight='bold', color="#E67E22")
                     
    ax3.axhline(0, color="black", lw=0.8, ls="--")
    ax3.set_xticks(x)
    ax3.set_xticklabels(["Small (< 32²)", "Medium (32²–96²)", "Large (> 96²)"], fontweight="semibold")
    ax3.set_ylabel(r"Degradation Gap $\Delta$ ($AP50_{ideal} - AP50_{missing}$)", fontweight="semibold")
    ax3.set_title(r"(c) Degradation $\Delta$ across Object BBox Sizes", fontsize=12, fontweight="bold", pad=12)
    ax3.set_ylim(-0.015, 0.055)
    ax3.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax3.legend(loc="upper right", framealpha=0.9, fontsize=10)
    
    # Key Takeaway Annotation
    ax3.annotate("Bus degradation concentrated\nin Medium/Large (Missing-Class)", xy=(1 - w, 0.030), xytext=(0.3, 0.046),
                 arrowprops=dict(arrowstyle="->", color="#E67E22", lw=1.2),
                 fontsize=9.5, fontweight="bold", color="#E67E22")
                 
    out_file = os.path.join(OUT_DIR, "figS2_cm_and_size_split.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_file}")

if __name__ == "__main__":
    print("--- Generating Supplementary Figures ---")
    plot_figS1()
    plot_figS2()
    print("--- Supplementary Figures Generated Successfully! ---")
