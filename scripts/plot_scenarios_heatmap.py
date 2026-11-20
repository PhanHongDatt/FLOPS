import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Dữ liệu đầu vào (Ma trận 5x4)
data1 = np.array([
    [31107, 499, 1334, 383],
    [30990, 522, 1274, 383],
    [31233, 538, 1279, 407],
    [31121, 519, 1240, 392],
    [30799, 550, 1294, 398]
])

data2 = np.array([
    [30974, 0,    1489, 399],
    [31216, 0,    1296, 379],
    [31293, 0,    1264, 410],
    [30482, 1380, 0,    395],
    [31285, 1248, 2372, 380]
])

data3 = np.array([
    [31322, 0,    1323, 345],
    [31515, 0,    1334, 346],
    [31176, 0,    1302, 338],
    [30634, 0,    0,    322],
    [30603, 2628, 2462, 612]
])

datasets = [data1, data2, data3]
titles = [
    "Pooled / IID (0/5 missing Bus)",
    "Missing Bus 3/5 (S1b5-3k)",
    "Missing Bus 4/5 (S1b5-bus1)"
]
classes = ['Car', 'Bus', 'Truck', 'Motorcycle']
clients = [f"Client {i}" for i in range(5)]

# 2. Khởi tạo Figure 1x3 subplots
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)

for i, (data, title, ax) in enumerate(zip(datasets, titles, axes)):
    # Chuẩn hóa theo từng cột
    norm_data = data / data.sum(axis=0)
    
    # Hiển thị số lượng box kèm tỷ lệ %
    annot_labels = np.array([
        [f"{v:,}\n({p*100:.1f}%)" for v, p in zip(row_d, row_p)]
        for row_d, row_p in zip(data, norm_data)
    ])
    
    sns.heatmap(
        norm_data,
        annot=annot_labels,
        fmt="",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=0.6,
        cbar=False,
        square=True,
        xticklabels=classes,
        yticklabels=clients,
        linewidths=1.2,
        linecolor="white",
        annot_kws={"size": 10.5, "fontweight": "medium"},
        ax=ax
    )
    
    # Tiêu đề và nhãn trục
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=12)
    ax.set_xlabel("Class Label", fontsize=11, fontweight="semibold", labelpad=8)
    
    if i == 0:
        ax.set_ylabel("Client ID", fontsize=11, fontweight="semibold", labelpad=8)
        ax.tick_params(axis="both", which="major", labelsize=11)
    else:
        ax.set_ylabel("")
        # Điểm tinh chỉnh 2: Xóa tick marks trên trục Y của biểu đồ 2 và 3
        ax.tick_params(left=False)
        ax.tick_params(axis="x", which="major", labelsize=11)

# Điểm tinh chỉnh 1: Thu hẹp khoảng cách giữa các biểu đồ
plt.subplots_adjust(wspace=0.08)

# Điểm tinh chỉnh 3: Xuất ảnh độ phân giải cao chuẩn bài báo
output_path = "analysis_charts/scenarios_data_distribution_heatmap_paper.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Biểu đồ đã được lưu tại: {output_path}")
