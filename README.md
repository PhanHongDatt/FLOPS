# FLOPS — Federated Learning for Object Detection under Missing-Class Non-IID

> **Đề tài:** Nghiên cứu các phương pháp bảo toàn tri thức và tổng hợp thích ứng trong Học liên kết (Federated Learning) cho bài toán Phát hiện đối tượng (Object Detection) khi xảy ra hiện tượng **Non-IID thiếu lớp (Missing-Class)** — tức một số client hoàn toàn không có mẫu dữ liệu nào của một hoặc nhiều lớp giao thông.

---

## 📌 Mục Lục Dành Cho Thành Viên Mới
1. [Tổng Quan Bài Toán & Kiến Trúc](#1-tổng-quan-bài-toán--kiến-trúc)
2. [Cấu Trúc Thư Mục Dự Án](#2-cấu-trúc-thư-mục-dự-án)
3. [Các Kịch Bản Dữ Liệu (Data Scenarios)](#3-các-kịch-bản-dữ-liệu-data-scenarios)
4. [Các Phương Pháp Huấn Luyện (Methods)](#4-các-phương-pháp-huấn-luyện-methods)
5. [Kết Quả Cốt Lõi & Phát Hiện Khoa Học](#5-kết-quả-cốt-lõi--phát-hiện-khoa-học)
6. [Bộ Hình Báo Cáo Luận Văn (Figures)](#6-bộ-hình-báo-cáo-luận-văn-figures)
7. [Hướng Dẫn Tái Hiện & Chạy Thử (Quickstart)](#7-hướng-dẫn-tái-hiện--chạy-thử-quickstart)

---

## 1. Tổng Quan Bài Toán & Kiến Trúc

* **Bài toán:** Khi huấn luyện Federated Learning với mô hình Object Detection, nếu một client không có ảnh chứa xe buýt (`bus`), cập nhật cục bộ của client đó không còn tín hiệu dương cho `bus`; giả thuyết H1 là điều này làm suy giảm khả năng phát hiện lớp bị thiếu trên mô hình toàn cục (kiểm chứng ở mục 5).
* **Tập dữ liệu:** **BDD100K** (Berkeley DeepDrive) — tập dữ liệu giao thông thực tế.
  * Tổng tập huấn luyện: **15.000 ảnh** (chia đều 3.000 ảnh/client cho 5 client).
  * Tập kiểm thử toàn cục (Global Test): **9.905 ảnh** (đánh giá cố định trên cùng tập này cho mọi phương pháp).
* **4 lớp mục tiêu:** `car` (xe hơi - đa số), `bus` (xe buýt - **lớp thiếu mục tiêu**), `truck` (xe tải), `motorcycle` (xe máy - lớp hiếm).
* **Mô hình chính thức:** **YOLO11s** (theo quyết định kiến trúc [ADR-016](research/decisions/ADR-016-model-selection.md)). Giữ nguyên trọng số pre-trained phân loại COCO (COCO head kept) để tránh phá hủy biểu diễn không gian đặc trưng.
* **Khung FL:** **Flower (flwr)** + PyTorch + Ultralytics YOLO.
* **Quy chuẩn thực nghiệm:** 20 rounds giao tiếp, 5 clients, 3 seed (`42`, `123`, `2024`); chỉ số chính = trung bình vòng 10 và 20. Kịch bản 4/5 (`bus1`) hiện mới có 2 seed (phiên seed 2024 chưa xong).

---

## 2. Cấu Trúc Thư Mục Dự Án

Dự án được tổ chức rất khoa học và minh bạch, tuân thủ nguyên tắc không cherry-picking kết quả:

```text
D:\FLOPS\
├── research/                         # 🧠 TRUNG TÂM KẾT QUẢ KHOA HỌC
│   ├── results_summary.csv           # File CSV tổng hợp toàn bộ metrics của mọi run (3 seed, mean, AP50, TP, FP, FN...)
│   ├── figures/
│   │   └── report/                   # 📊 Toàn bộ hình dùng cho báo cáo / luận văn (PDF vector + PNG + CSV số liệu)
│   ├── experiment_registry/
│   │   └── registry.yaml             # Nhật ký chi tiết toàn bộ các thử nghiệm từ đầu đến cuối
│   └── decisions/                    # Hồ sơ quyết định kỹ thuật (ADR-001 -> ADR-022)
│
├── kaggle/                           # 🚀 MÃ NGUỒN CHẠY TRÊN CLOUD GPU (Kaggle)
│   ├── output/                       # Nhật ký log, checkpoint, kết quả tải về từ các phiên chạy s1 -> s18
│   ├── build/                        # Mã đóng gói phiên chạy Kaggle
│   └── sessions.yaml                 # Danh sách phiên chạy (GPU Tesla T4)
│
├── src/                              # 💻 SOURCE CODE FRAMEWORK FL
│   ├── data/                         # Code phân chia dữ liệu (partitioner), tạo manifest Non-IID
│   ├── federated/                    # Cài đặt Client, Server, chiến lược tổng hợp (Strategy) của Flower
│   ├── model/                        # YOLO wrapper (load weights, train, validate, custom loss)
│   └── evaluation/                   # Tính toán mAP50, AP50 từng lớp, Confusion Matrix, D1 error analysis
│
├── configs/                          # ⚙️ CẤU HÌNH THỰC NGHIỆM
│   ├── base_config.yaml              # Hyperparameter chuẩn
│   └── partition/                    # Cấu hình kịch bản phân bổ dữ liệu
│
├── scripts/                          # Script tổng hợp registry, xuất CSV, vẽ hình báo cáo
├── CLAUDE.md                         # Hiến pháp dự án & quy tắc ứng xử code nghiêm ngặt
└── README.md                         # Tài liệu hướng dẫn này
```

---

## 3. Các Kịch Bản Dữ Liệu (Data Scenarios)

Cả 3 kịch bản đều dùng **chính xác cùng 15.000 ảnh** (bus: 2.628 box, car: 155.250 box, truck: 6.421 box, motorcycle: 1.963 box), chỉ khác nhau ở **cách phân phối dữ liệu về 5 client (C0 -> C4)**:

| Ký hiệu kịch bản | Số client thiếu bus | Mô tả phân bổ cụ thể | Ý nghĩa khoa học |
|---|:---:|---|---|
| **`pooled`** | **0/5** | Dữ liệu chia đều ngẫu nhiên (IID) cho cả 5 client. Mọi client đều có đủ cả 4 lớp. | **Trần lý tưởng (Upper Bound):** Thước đo đối chứng khi không có hiện tượng Non-IID. |
| **`s5`** *(hay `S1b5`)* | **3/5** | **C0, C1, C2:** Hoàn toàn không có bus (`missing: [bus]`).<br>**C3:** Bị rút bớt truck (`missing: [truck]`).<br>**C4:** Giữ toàn bộ xe bus còn lại và các lớp khác. | **Kịch bản trọng tâm của đề tài:** Mô phỏng sát thực tế khi đa số client bị khuyết hẳn một lớp xe. |
| **`bus1`** | **4/5** | **C0, C1, C2, C3:** Đều không có bus.<br>**Chỉ duy nhất C4** sở hữu toàn bộ 2.628 box bus. | **Kịch bản cực đoan:** Kiểm tra độ bền vững (robustness) của mô hình khi mức độ thiếu lớp lên mức tối đa. |

---

## 4. Các Phương Pháp Huấn Luyện (Methods)

Khi đọc biểu đồ và bảng kết quả, bạn sẽ thấy 4 phương pháp so sánh chính:

| Tên chuẩn | Mã trong code | Bản chất thuật toán | Vai trò trong nghiên cứu |
|---|---|---|---|
| **`FedAvg`** | `A0c` | Thuật toán Federated Averaging kinh điển (McMahan et al.), có giữ COCO pre-trained head. | **Baseline đối chứng cơ bản.** Bất kỳ giải pháp nào cũng phải vượt qua mốc này. |
| **`Box`** | `A1c` | FedAvg với trọng số tổng hợp trên Server theo **số lượng bounding box** của từng lớp thay vì số ảnh. | **Đối chứng class-count:** AP50 bus 0,358 ở 3/5, thấp hơn FedAvg (0,387). |
| **`Prox`** | `FedProxc` | Thuật toán FedProx kinh điển (Li et al.), bổ sung số hạng phạt proximal term $\frac{\mu}{2}\|w - w^t\|^2$ ($\mu=0.01$). | **Baseline xử lý Non-IID:** AP50 bus 0,361 ở 3/5, không hơn FedAvg. |
| **`Ours`** | `P4noKD` | **Công thức cải tiến:**<br>1. **Cosine Round LR:** Tốc độ học suy giảm theo hàm Cosine qua các vòng (round).<br>2. **Client RFS ($t=0.5$):** Lặp lại ảnh chứa lớp hiếm/thiếu tại client.<br>3. **noKD:** Bỏ bước chưng cất tri thức để giảm thời gian (xem ablation). | **Cao nhất ở cả 3 kịch bản**, nhưng vẫn suy giảm khi thiếu lớp (AP50 bus 0,541 → 0,492 → 0,469). |

---

## 5. Kết Quả Cốt Lõi & Phát Hiện Khoa Học

Số liệu lấy từ `research/results_summary.csv` (chỉ số chính = trung bình vòng 10 và 20; n = số seed, ghi kèm từng số).

1. **Thiếu lớp có làm giảm AP50 bus (H1)?** Có, nhưng mức giảm AP50 nhỏ: FedAvg 0,415 (0/5, n=3) → 0,387 (3/5, n=3) → 0,367 (4/5, n=2). Mức suy giảm ghép cặp theo seed: +0,029 (3/5) và +0,047 (4/5).
2. **AP50 che giấu mức mất mát ở ngưỡng vận hành.** Ở điểm tin cậy 0,25, số bus phát hiện đúng (TP) của FedAvg chỉ còn 36 % (3/5) và 10 % (4/5) so với kịch bản lý tưởng, trong khi AP50 bus vẫn giữ 93 % và 89 %.
3. **Công thức cải tiến (`Ours`) trên kịch bản 3/5 (n=3):**
   * Thắng **18/18** so sánh ghép cặp theo seed với 3 baseline.
   * AP50 bus **0,492** so với FedAvg 0,387 (**+0,105** điểm AP), FedProx 0,361, Box 0,358.
   * mAP50 **0,531** so với FedAvg 0,434 (+0,097). Các lớp còn lại cũng tăng: car +0,039, truck +0,102, motorcycle +0,143.
   * Vẫn suy giảm khi thiếu lớp (+0,050 ở 3/5, +0,074 ở 4/5), nên phương pháp chưa loại bỏ được hiện tượng.
4. **Ablation (seed 42, AP50 bus):** FedAvg 0,382 → +KD 0,428 → +Cosine LR 0,454 → +RFS 0,496 → bỏ KD 0,487. KD có đóng góp (+0,046) khi thêm vào, nhưng khi đã có Cosine LR và RFS thì bỏ KD chỉ mất 0,008 AP50 bus, đồng thời giảm thời gian từ 8,9 xuống 6,7 phút/vòng (FedAvg 5,6 phút/vòng, Tesla T4).
5. **Kết cục của 1.597 bus thật (D1, ngưỡng 0,25, seed 42, n=1):** FedAvg ở 3/5 đúng 20,6 %, nhầm sang car/truck 38,0 %, bỏ sót 33,8 %, điểm thấp 7,6 % (lý tưởng: đúng 40,3 %, nhầm 28,1 %). `Ours` ở 3/5: đúng 35,4 %, nhầm 27,7 %. Lỗi chủ yếu là nhầm lớp và bỏ sót, không phải điểm tin cậy thấp.

---

## 6. Bộ Hình Báo Cáo Luận Văn (Figures)

Mỗi hình có PNG, PDF và CSV chứa đúng số liệu được vẽ, nằm trong **`research/figures/report/`**. Chú thích có số liệu nằm ở `docs/DAN_Y_BAO_CAO.md`.

| File hình | Nội dung |
|---|---|
| [`fig1_fl_architecture`](research/figures/report/fig1_fl_architecture.pdf) | Kiến trúc FL: 1 server, 5 client |
| [`hinh02_ba_kich_ban`](research/figures/report/hinh02_ba_kich_ban.png) | Ba kịch bản dữ liệu (client × lớp) |
| [`hinh03_phan_bo_box`](research/figures/report/hinh03_phan_bo_box.png) | Heatmap số box theo client và lớp |
| [`hinh04_ap50_bus_theo_muc_thieu`](research/figures/report/hinh04_ap50_bus_theo_muc_thieu.png) | AP50 bus theo mức thiếu 0/5 → 3/5 → 4/5 |
| [`hinh05_muc_suy_giam_theo_phuong_phap`](research/figures/report/hinh05_muc_suy_giam_theo_phuong_phap.png) | Mức suy giảm so với kịch bản lý tưởng |
| [`hinh06_ap50_theo_lop_thieu_3_5`](research/figures/report/hinh06_ap50_theo_lop_thieu_3_5.png) | AP50 theo 4 lớp ở 3/5 |
| [`hinh08_chenh_lech_ghep_cap`](research/figures/report/hinh08_chenh_lech_ghep_cap.png) | Chênh lệch ghép cặp theo seed (18/18) |
| [`hinh11_ablation`](research/figures/report/hinh11_ablation.png) | Ablation từng thành phần |
| [`hinh13_giu_lai_ap50_va_tp`](research/figures/report/hinh13_giu_lai_ap50_va_tp.png) | Tỷ lệ giữ lại AP50 so với TP |
| [`hinh14_thoi_gian_huan_luyen`](research/figures/report/hinh14_thoi_gian_huan_luyen.png) | Thời gian mỗi vòng và mỗi run |

Hình 7 (kết cục bus), 9 (đường PR), 10 (confusion matrix) và 12 (histogram điểm) đang chờ đánh giá lại checkpoint (`scripts/eval_checkpoints_local.py`).

---

## 7. Hướng Dẫn Tái Hiện & Chạy Thử (Quickstart)

### Cài đặt môi trường
Phiên bản ghim: xem `research/decisions/ADR-001-environment-versions.md` (Ultralytics 8.3.253, Flower 1.21.0).

### Xuất lại số liệu và hình báo cáo
```bash
python scripts/export_results_csv.py      # research/results_summary.csv
python scripts/extract_report_data.py     # số box theo client, thời gian mỗi vòng
python scripts/plot_report_figures.py     # research/figures/report/hinhNN_*
```

### Đọc số liệu thực nghiệm
```python
import pandas as pd
df = pd.read_csv("research/results_summary.csv")
print(df[(df["row_type"] == "mean") & (df["scenario"] == "s5")][["method", "mAP50", "AP50_bus", "AP50_car", "AP50_truck", "AP50_motorcycle"]])
```
