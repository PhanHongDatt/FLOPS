# Kế hoạch thực thi: FedTraff-KP (Missing-Class Non-IID) trên Kaggle

Phạm vi: phần của Snow trong khóa luận "FLOps cho Federated Learning giám sát giao thông" — xử lý Missing-Class Non-IID cho YOLOv8, dữ liệu BDD100K, huấn luyện trên Kaggle.

Quy ước độ tin cậy trong tài liệu này:
- **[ĐÃ TEST]**: logic đã chạy kiểm thử (numpy, dữ liệu giả) trong môi trường của mình.
- **[CHƯA TEST]**: code mẫu, bắt buộc smoke test trên Kaggle trước khi tin. Phụ thuộc phiên bản Ultralytics.
- **[CẦN XÁC MINH]**: thông tin có thể đã đổi, kiểm tra lại trực tiếp.

---

## 0. Tóm tắt kế hoạch

| Giai đoạn | Mục tiêu | Đầu ra | Cổng qua (go/no-go) |
|---|---|---|---|
| P0 | Chuẩn bị dữ liệu + baseline centralized + thí nghiệm chẩn đoán quên lớp | Dữ liệu YOLO, mAP centralized, hình "AP lớp bus trước/sau local update" | Quên lớp vacant xuất hiện rõ (AP lớp vacant giảm mạnh sau local update) |
| P1 | Partition + baseline FedAvg/FedProx | Bảng S0 vs S1 của FedAvg/FedProx | S1 tụt rõ so với S0 ở AP lớp vacant |
| P2 | Phương pháp: client (ρ-loss) + server (class-aware agg) | Code + kết quả sơ bộ 1 seed | Ít nhất một biến thể hơn FedAvg ở AP lớp vacant |
| P3 | Ablation + ρ sweep + baseline FedRS/FedLC/FedVLS chuyển sang BCE | Bảng ablation | Phân tách được đóng góp client vs server |
| P4 | Chạy chính 3 seed + viết báo cáo | Bảng mean ± std, biểu đồ AP theo lớp | — |

Nguyên tắc: **không viết phương pháp hoàn chỉnh trước khi qua cổng P0**. Nếu quên lớp không xuất hiện ở YOLOv8 (ví dụ do khởi tạo từ COCO), phải đổi thiết kế thí nghiệm, không đổi kết luận.

---

## 1. Prompt đã tối ưu

### 1.1 Prompt tìm phương pháp nghiên cứu (dán cho Claude/LLM có web search)

```text
VAI TRÒ: Bạn là trợ lý nghiên cứu về Federated Learning cho object detection.

BỐI CẢNH:
- Khóa luận: FL cho giám sát giao thông, YOLOv8 (loss cls = BCE sigmoid theo từng lớp, box = CIoU + DFL), dữ liệu BDD100K, 4 lớp: car, bus, truck, motor.
- Vấn đề: Missing-Class Non-IID — một số client hoàn toàn không có dữ liệu (cả ảnh lẫn bounding box) của một số lớp.
- Phương pháp dự kiến (FedTraff-KP): (a) client: hệ số ρ ∈ {0, 0.25, 1} điều chỉnh đóng góp loss/gradient của kênh lớp vacant; (b) server: tổng hợp tham số đầu ra theo lớp bằng trọng số riêng từng lớp (số bbox), phần backbone/neck dùng FedAvg/FedProx; nếu không client nào đủ dữ liệu cho một lớp thì giữ nguyên tham số global.
- Tài nguyên: Kaggle (GPU T4/P100, ~30 giờ GPU/tuần), 4–6 client mô phỏng tuần tự.

RÀNG BUỘC BẮT BUỘC:
1. Chỉ trích dẫn bài đã đọc được nội dung (abstract tối thiểu); ghi rõ "đã đọc toàn văn / chỉ abstract / chỉ snippet".
2. Venue phải từ hạng C trở lên (CORE/ICORE) hoặc journal Q1–Q2; preprint chỉ được dùng làm tham khảo phụ và phải ghi rõ là preprint.
3. Không tự suy ra số liệu, năm, trang, hạng venue. Không chắc thì ghi "CHƯA XÁC MINH".
4. Không so sánh số giữa các bài nếu giao thức khác nhau.

NHIỆM VỤ:
A. Tìm và phân tích 8–12 công trình về: (i) label skew / vacant class / missing class; (ii) federated object detection với heterogeneity; (iii) class-wise / classifier-head aggregation; (iv) forgetting khi local update.
B. Với mỗi bài, trả lời 5 câu: (1) giả định bài toán, (2) cơ chế cốt lõi, (3) độ tin cậy thực nghiệm (số client, seed, báo vòng cuối hay max, cùng giao thức hay không), (4) hạn chế, (5) chuyển sang detector sigmoid/BCE được không và cần sửa gì.
C. Kiểm tra rủi ro tính mới: có công trình nào đã làm đồng thời client-side vacant-class protection VÀ class-wise aggregation cho object detection chưa? Nếu có, nêu khác biệt cụ thể.
D. Đề xuất tối đa 3 hướng phương pháp, mỗi hướng gồm: giả thuyết kiểm chứng được, thay đổi loss/aggregation (công thức), baseline phải so sánh, ablation, chỉ số (per-class AP, mAP50, mAP50-95, FP/FN lớp vacant), rủi ro và điều kiện bác bỏ giả thuyết.

ĐỊNH DẠNG ĐẦU RA:
- Bảng bài báo (tên, venue, năm, mức đã đọc, cơ chế, hạn chế, khả năng chuyển sang BCE).
- Mỗi hướng đề xuất: 1 đoạn giả thuyết + công thức + thiết kế thí nghiệm tối thiểu.
- Cuối cùng: danh sách điều CHƯA XÁC MINH.
```

### 1.2 Prompt viết code (dán kèm file kế hoạch này)

```text
Viết code chạy được trên Kaggle (Python, Ultralytics YOLOv8n, PyTorch) cho mô phỏng FL tuần tự (không Flower ở giai đoạn này).
Yêu cầu:
- Tuân thủ cấu trúc repo và hàm ở mục 6 của kế hoạch; không đổi chữ ký hàm.
- Mọi bước phải resumable: lưu checkpoint global + log mỗi round vào /kaggle/working.
- Không dùng API không có trong phiên bản Ultralytics đã pin; in version ở đầu notebook.
- Mỗi hàm mới kèm test nhỏ (assert) chạy được trên CPU với dữ liệu giả.
- Mỗi round ghi CSV: round, client_id, n_img, n_box_per_class, local_time_s, bytes_up; và global metrics: mAP50, mAP50-95, AP từng lớp, FP/FN từng lớp.
- Báo rõ phần nào chưa kiểm thử được.
```

---

## 2. Môi trường Kaggle

Các điểm [CẦN XÁC MINH] trong tài khoản của bạn:
- GPU: một P100 (16 GB) hoặc hai T4 (2×16 GB). Mô phỏng tuần tự chỉ cần 1 GPU; bật 2×T4 không tăng tốc trừ khi bạn tự chạy 2 client song song.
- Hạn mức GPU hàng tuần: thường được nhắc là khoảng 30 giờ/tuần nhưng Kaggle không công bố con số cố định và có thể thay đổi; xem trong phần Settings của tài khoản.
- Một phiên GPU chạy tối đa 12 giờ (theo tài liệu Kaggle được tổng hợp tháng 9/2026).
- Dung lượng lưu trữ /kaggle/working khoảng 20 GB.

Hệ quả thiết kế:
1. **Mọi thí nghiệm phải chia nhỏ theo round và resume được.** Mỗi phiên chạy N round rồi lưu `global_round_XXX.pt` + `log.csv`; phiên sau đọc lại và chạy tiếp.
2. **Mỗi (phương pháp, kịch bản, seed) là một "run" có ID**, ví dụ `S1b_full_rho0_seed1`. Ghi kết quả vào dataset Kaggle riêng (đẩy output của notebook thành dataset) để không mất.
3. **Giới hạn kích thước:** YOLOv8n, imgsz 512 (hoặc 416 nếu quá chậm), mỗi client 2.000–3.000 ảnh, 20–30 round, 1–2 epoch cục bộ. Chốt số cụ thể sau khi đo ở P0.
4. Cài đặt đầu notebook (pin phiên bản):

```bash
pip install -q "ultralytics==<VERSION_ĐÃ_TEST>"   # thay bằng bản bạn smoke test thành công
python -c "import ultralytics, torch; print(ultralytics.__version__, torch.__version__)"
```

5. **Giai đoạn FLOps (Flower + MLflow + Docker) làm ngoài Kaggle** (máy local hoặc VM), dùng lại đúng hàm `aggregate`, `partition`, loss từ repo. Kaggle chỉ để chạy thí nghiệm nặng, vì chạy Flower simulation (Ray) trên notebook dễ tốn tài nguyên và khó resume.

---

## 3. Dữ liệu BDD100K

### 3.1 Nguồn
- Dataset Kaggle thường được dùng: `solesensei/solesensei_bdd100k` [CẦN XÁC MINH còn tồn tại và cấu trúc]. Cấu trúc thường gặp: `images/100k/{train,val}` (70.000 train, 10.000 val) và nhãn `labels/*.json` (một số bản có tên `det_20/det_train.json`).
- Bộ 100K ảnh có **một frame được gán nhãn cho mỗi video** (frame giây thứ 10). Vì vậy chia theo train/val chính thức đã tránh rò rỉ giữa các frame cùng video. Phần "chia theo video sequence" trong đề cương vẫn nên giữ như một câu giải thích, nhưng không cần cơ chế chia riêng.
- BDD100K có thuộc tính `weather`, `scene`, `timeofday` trong nhãn, dùng được để tạo heterogeneity tự nhiên (FedPylot khuyến nghị chia theo ranh giới tự nhiên với detection).

### 3.2 Ánh xạ lớp
BDD100K gọi lớp "motor" cho xe máy. Ánh xạ: `car→0, bus→1, truck→2, motor→3` (đúng 4 lớp đề cương, đặt tên hiển thị là motorcycle). Các lớp khác bị bỏ.

Lưu ý quan trọng: **việc bỏ các lớp khác (person, rider, bike, train, traffic light/sign)** làm các đối tượng đó thành "background" trong mọi client giống nhau, nên không gây lệch giữa client, nhưng sẽ làm giảm mAP tuyệt đối. Ghi rõ trong báo cáo.

### 3.3 Convert sang YOLO format [CHƯA TEST]
Ảnh 1280×720; nhãn BDD: `labels[i].category`, `labels[i].box2d{x1,y1,x2,y2}`.

```python
import json, os
MAP = {"car":0, "bus":1, "truck":2, "motor":3}
W, H = 1280, 720

def convert(det_json, out_lbl_dir):
    os.makedirs(out_lbl_dir, exist_ok=True)
    img_classes = {}
    for item in json.load(open(det_json)):
        lines, cls = [], set()
        for lb in item.get("labels", []):
            c = lb.get("category")
            if c not in MAP or "box2d" not in lb: continue
            b = lb["box2d"]
            x, y = (b["x1"]+b["x2"])/2/W, (b["y1"]+b["y2"])/2/H
            w, h = (b["x2"]-b["x1"])/W, (b["y2"]-b["y1"])/H
            if w <= 0 or h <= 0: continue
            lines.append(f"{MAP[c]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            cls.add(MAP[c])
        stem = os.path.splitext(item["name"])[0]
        open(f"{out_lbl_dir}/{stem}.txt", "w").write("\n".join(lines))
        img_classes[stem] = cls
    return img_classes     # dùng cho partition
```

Smoke test bắt buộc: đếm số ảnh, số bbox mỗi lớp; vẽ 5 ảnh với bbox để kiểm tra tọa độ.

### 3.4 Thống kê cần làm trước khi thiết kế kịch bản
Đếm số ảnh và bbox của từng lớp ở train. Lớp motor và bus thường hiếm, nên có thể không đủ ảnh để chia thành 4 client với một lớp vắng mặt ở một số client. Quyết định số client và lớp vacant **dựa trên số đếm thật**, không dựa trên giả định.

---

## 4. Thiết kế partition

Nguyên tắc cho missing-class trong detection: client "thiếu lớp bus" phải **không chứa ảnh nào có bus** (loại ảnh), không phải xóa nhãn bus. Xóa nhãn biến bus thành background giả (hard negative sai), đó là bài toán khác (partially labeled).

### 4.1 Kịch bản (4 client, lớp: car0, bus1, truck2, motor3)

| Kịch bản | Mô tả | Mục đích |
|---|---|---|
| S0 (IID) | Chia ngẫu nhiên đều, không client nào thiếu lớp | Kiểm tra phương pháp không làm hại khi không có missing class |
| S1b (chính) | C1 thiếu bus, C2 thiếu truck, C3 thiếu motor, C4 đủ cả 4 | Missing-class mức vừa, giống Bảng 1 của đề cương |
| S1a (nhẹ) | Chỉ C1 thiếu bus | Mức nhẹ |
| S1c (nặng, mở rộng) | Bus chỉ có ở 1 client | Chỉ làm nếu còn thời gian (đúng phạm vi đề cương) |
| Control pair | Hai nhóm cùng số ảnh và gần cùng số bbox, khác nhau chỉ ở có/không có bus | Tách riêng tác động của missing class |

### 4.2 Code partition [ĐÃ TEST] (dữ liệu giả, 20.000 ảnh, 4 client)

Kết quả kiểm thử: không client nào chứa lớp vacant của nó; kích thước các client cân bằng (4.592–4.593 ảnh); không ảnh nào bị gán hai lần.

```python
import random
from collections import defaultdict

def partition(img_classes, vacant, seed=0, per_client=None):
    """img_classes: {img_id: set(class_ids)} (only the K target classes).
    vacant: list[set] one per client = classes the client must NOT contain.
    An image is eligible for a client only if it contains NO vacant class of that client
    (so a vacant class has zero images AND zero boxes -> true missing class, no fake background).
    Returns list[list[img_id]]."""
    rng = random.Random(seed)
    ids = [i for i, s in img_classes.items() if s]          # drop images w/o target class
    rng.shuffle(ids)
    C = len(vacant); out = [[] for _ in range(C)]
    for i in ids:
        elig = [c for c in range(C) if not (img_classes[i] & vacant[c])]
        if not elig: continue                                # image fits nobody -> dropped
        # pick the least-filled eligible client -> balanced sizes
        c = min(elig, key=lambda k: len(out[k]))
        if per_client is None or len(out[c]) < per_client:
            out[c].append(i)
    return out
```

Cách dùng:
```python
img_classes = convert("det_train.json", "labels_yolo/train")        # {stem: set(class_ids)}
vacant = [{1}, {2}, {3}, set()]                                      # S1b
parts = partition(img_classes, vacant, seed=0, per_client=2500)
# lưu manifest: partition_S1b_seed0.json = {"vacant": [...], "clients": [[stem,...],...]}
```
Mỗi partition phải lưu **manifest JSON** (đúng yêu cầu partition manifest trong nội dung 3 của đề cương) và dùng cùng `partition seed` giữa các phương pháp.

Lưu ý hạn chế: hàm gán ảnh vào client ít đầy nhất trong số client hợp lệ, nên ảnh chứa lớp hiếm sẽ dồn vào các client không vacant lớp đó. Điều này là hệ quả tự nhiên của ràng buộc, cần báo cáo phân bố bbox/lớp/client thực tế (bảng) thay vì chỉ mô tả ý định.

### 4.3 Validation
Dùng toàn bộ tập val chính thức (đã lọc theo 4 lớp) làm global test set tại server. Không client nào được dùng val để chọn model trừ khi nêu rõ trong báo cáo.

---

## 5. Thí nghiệm chẩn đoán (P0) — làm trước tiên

Mục đích: kiểm chứng giả thuyết "local update làm quên lớp vacant ở YOLOv8" và xác định **quên xảy ra ở đâu** (bias/trọng số lớp cls cuối, hay các lớp dùng chung). Đây tái hiện, trong detection, hình mà FedLC và FedVLS vẽ cho classification.

Quy trình:
1. Huấn luyện một global model trên dữ liệu đủ 4 lớp (centralized hoặc FedAvg S0 vài round). Đo AP từng lớp trên val: `AP_before[c]`.
2. Lấy client C1 (thiếu bus). Từ global model, huấn luyện cục bộ 1–2 epoch bằng loss mặc định. Đo `AP_after[c]` trên val.
3. Với kết quả tách theo nhóm tham số, tính `‖Δθ‖` (hoặc `‖Δθ‖/‖θ‖`) cho: (a) hàng bus của conv cuối nhánh cls và bias bus, ở cả 3 scale, (b) các hàng lớp khác của conv cuối, (c) các conv trước đó trong nhánh cls, (d) neck, (e) backbone.
4. Lặp 3 lần với seed khác nhau, và lặp với hai kiểu khởi tạo: **COCO-pretrained** và **khởi tạo ngẫu nhiên hoặc đã train trên 4 lớp**. FedPylot nêu rằng việc dùng trọng số pretrained trên COCO có thể góp phần giữ ổn định, nên nếu khởi tạo từ COCO, hiện tượng quên có thể bị che.

Tiêu chí qua cổng (đặt trước khi chạy): `AP_after[bus] < AP_before[bus] − max(2 · std qua seed, ngưỡng tuyệt đối bạn chọn)`. Kết luận chỉ đưa ra khi sự thay đổi tham số và suy giảm AP theo lớp có cùng xu hướng (đúng ràng buộc trong đề cương).

Ba kết cục và việc cần làm:
- Quên rõ, tập trung ở hàng/bias lớp bus → hướng ρ-loss + class-wise aggregation là hợp lý.
- Quên rõ, lan sang các lớp conv trước trong nhánh cls → ρ phải áp ở mức loss (không chỉ chặn gradient hàng cuối), cân nhắc distillation.
- Không quên rõ → xem lại khởi tạo, số epoch cục bộ, mức thiếu lớp; nếu vẫn không có, chuyển sang kịch bản S1c hoặc báo cáo kết quả âm một cách trung thực.

---

## 6. Cấu trúc code (Kaggle, mô phỏng tuần tự)

```text
repo/
  data/convert.py           # mục 3.3
  data/partition.py         # mục 4.2  [ĐÃ TEST]
  fl/aggregate.py           # mục 6.1  [ĐÃ TEST]
  fl/loss_rho.py            # mục 6.2  [CHƯA TEST]
  fl/client.py              # mục 6.3  [CHƯA TEST]
  fl/server.py              # vòng lặp FL, resume, ghi log
  eval/evaluate.py          # mục 6.4  [CHƯA TEST]
  configs/*.yaml            # scenario × method × seed
  notebooks/run_{P0,P1,P2}.ipynb
```

### 6.1 Class-aware aggregation [ĐÃ TEST]

Logic: tham số của lớp cuối nhánh cls (`model.<idx>.cv3.<scale>.2.weight/bias`, cả 3 scale) được tổng hợp theo từng hàng lớp, trọng số theo số bbox lớp đó của các client "đủ dữ liệu" (`n_box ≥ min_box`); mọi tham số khác dùng FedAvg theo số ảnh. Nếu không client nào đủ dữ liệu cho lớp c, hàng c giữ nguyên từ global.

Các kiểm thử đã qua (dữ liệu giả, 3 client, 4 lớp): lớp không client nào có → giữ nguyên global; lớp chỉ một client có → bằng đúng hàng của client đó; lớp có nhiều client → trọng số theo bbox, chỉ tính client đủ ngưỡng; tham số dùng chung và nhánh box (`cv2`) = FedAvg; khi mọi client cùng số bbox → trùng FedAvg.

```python
import re
CLS_HEAD = re.compile(r"model\.\d+\.cv3\.\d+\.2\.(weight|bias)$")

def aggregate(global_sd, client_sds, n_imgs, n_box, min_box=10):
    """global_sd: dict name->array; client_sds: list of dict; n_imgs: list[int];
    n_box: list of per-class bbox count arrays (len nc) per client."""
    nc = len(n_box[0]); tot = float(sum(n_imgs))
    out = {}
    for k, g in global_sd.items():
        if not hasattr(g, "shape") or g.dtype.kind != "f":
            out[k] = client_sds[0][k]; continue
        if CLS_HEAD.search(k):
            new = g * 1.0
            for c in range(nc):
                ok = [i for i in range(len(client_sds)) if n_box[i][c] >= min_box]
                s = float(sum(n_box[i][c] for i in ok))
                if s == 0: continue            # keep global row
                acc = g[c] * 0.0
                for i in ok:
                    acc = acc + (n_box[i][c] / s) * (client_sds[i][k][c] - g[c])
                new[c] = g[c] + acc
            out[k] = new
        else:
            acc = g * 0.0
            for i, sd in enumerate(client_sds):
                acc = acc + (n_imgs[i] / tot) * (sd[k] - g)
            out[k] = g + acc
    return out
```

Ghi chú:
- Biểu thức regex `model\.\d+\.cv3\.\d+\.2\.(weight|bias)$` phụ thuộc cách đặt tên của Ultralytics YOLOv8. **Phải in `state_dict().keys()` và kiểm tra** trên bản bạn dùng (chỉ số lớp Detect khác nhau giữa các cỡ model). Nếu bản Ultralytics đổi tên lớp, kiểm thử sẽ vẫn qua nhưng không áp đúng tham số — hãy thêm assert đếm số khóa khớp (kỳ vọng: 3 scale × 2 = 6 khóa cho `weight` và `bias`).
- Với FedProx: phần proximal term nằm ở client (mục 6.3); aggregation vẫn dùng hàm này.
- Baseline `class-count weighted FedAvg` (đề cương): áp trọng số theo số bbox lớp c cho **toàn bộ** tham số của hàng c nhưng không dùng masking ở client — dùng đúng hàm trên với `ρ = 1` ở client.

### 6.2 ρ-loss tại client [CHƯA TEST]

Ý tưởng: nhân số hạng BCE của kênh lớp vacant với ρ. Khi `ρ = 0`, kênh vacant không sinh gradient nào, kể cả cho các lớp conv trước trong nhánh cls. Khi `ρ = 1`, huấn luyện bình thường.

Mã dưới đây dựa trên cấu trúc `v8DetectionLoss` (thuộc tính `self.bce` dạng `BCEWithLogitsLoss(reduction="none")` trả về tensor `[B, A, nc]`). **Phải đọc `ultralytics/utils/loss.py` của bản đã pin để xác nhận.**

```python
import torch
from ultralytics.utils.loss import v8DetectionLoss

class RhoDetectionLoss(v8DetectionLoss):
    """class_w: tensor (nc,), ví dụ [1,0.25,1,1] nghĩa là lớp 1 (bus) vacant với rho=0.25."""
    def __init__(self, model, class_w):
        super().__init__(model)
        base_bce, w = self.bce, class_w.to(self.device).view(1, 1, -1)
        class _WBCE(torch.nn.Module):
            def forward(self_, pred, target):
                return base_bce(pred, target) * w          # broadcast theo kênh lớp
        self.bce = _WBCE()
```

Gắn vào trainer:

```python
from ultralytics.models.yolo.detect import DetectionTrainer

class FLTrainer(DetectionTrainer):
    class_w = None                                   # đặt trước khi train()
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        cw = self.class_w
        model.init_criterion = lambda: RhoDetectionLoss(model, cw)
        return model
```

Kiểm thử bắt buộc trên Kaggle (CPU/GPU, 1 batch):
- `class_w = ones` cho loss bằng loss mặc định (sai số < 1e-6).
- `class_w[1] = 0`: gradient của hàng bus trong conv cuối và bias bus phải bằng 0 (kiểm tra `.grad`), các hàng khác khác 0.

Biến thể V2 (distillation cho kênh vacant, mở rộng sau P2): với kênh vacant v, thay target 0 bằng `sigmoid(logit_global[..., v]).detach()` (cần forward global model trên cùng batch). Chỉ làm khi V1 không đủ.

### 6.3 Client update và FedProx [CHƯA TEST]

```python
def client_update(global_sd, data_yaml, class_w, epochs=1, mu=0.0, imgsz=512, batch=16, seed=0):
    """Trả (state_dict_local, n_img, n_box_per_class). Optimizer không giữ trạng thái giữa các round."""
    trainer = FLTrainer(overrides=dict(
        model="yolov8n.yaml", data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch,
        device=0, workers=2, seed=seed, deterministic=True, plots=False, val=False,
        warmup_epochs=0, close_mosaic=0, project="runs_fl", name="tmp", exist_ok=True))
    trainer.class_w = class_w
    trainer.setup_model()
    trainer.model.load_state_dict(global_sd)
    trainer.train()
    ckpt = torch.load("runs_fl/tmp/weights/last.pt", weights_only=False)
    return {k: v.cpu() for k, v in ckpt["ema"].float().state_dict().items()}
```

Lưu ý:
- Cách nạp global model vào trainer và lấy trọng số sau train (dùng `ema` hay `model`) thay đổi theo phiên bản. Smoke test: với `epochs=0` hoặc lr rất nhỏ, trọng số trả về phải ≈ global.
- Tắt mosaic cuối (`close_mosaic=0`) và đặt `warmup_epochs=0` để mỗi round hành xử giống nhau; ghi cấu hình augmentation vào manifest.
- FedProx cần thêm số hạng `(mu/2)‖θ − θ_global‖²` vào loss; cách đơn giản và an toàn là thêm bằng callback `on_train_batch_end`/ghi đè `criterion`. Triển khai riêng sau khi FedAvg chạy ổn.
- Đếm `n_img`, `n_box_per_class` trực tiếp từ nhãn của client (không từ dữ liệu val).

### 6.4 Đánh giá global [CHƯA TEST]

```python
from ultralytics import YOLO
def evaluate(global_sd, val_yaml, imgsz=512):
    m = YOLO("yolov8n.yaml"); m.model.load_state_dict(global_sd)
    r = m.val(data=val_yaml, imgsz=imgsz, batch=32, plots=False, verbose=False)
    return dict(map50=r.box.map50, map=r.box.map, ap50_per_class=r.box.ap50.tolist(),
                ap_class_index=r.box.ap_class_index.tolist())
```
FP/FN theo lớp: lấy từ confusion matrix (`r.confusion_matrix.matrix`) tại ngưỡng cố định; ghi rõ ngưỡng conf/IoU.

### 6.5 Vòng lặp server (phác thảo)

```text
for round in range(R):
    for c in clients:  sd_c, n_img_c, n_box_c = client_update(global_sd, yaml_c, class_w_c, ...)
    global_sd = aggregate(global_sd, [sd_c...], [n_img_c...], [n_box_c...], min_box=THR)   # hoặc FedAvg
    log(evaluate(global_sd, val_yaml)); save checkpoint + CSV
```
`class_w_c`: client tự xác định lớp vacant từ `n_box_c` (lớp có `n_box < THR`) và đặt `class_w[c]=ρ`.

---

## 7. Ma trận thí nghiệm và ngân sách GPU

### 7.1 Các phương pháp
| ID | Mô tả |
|---|---|
| M1 | FedAvg |
| M2 | FedProx |
| M3 | class-count weighted FedAvg (đề cương) |
| M4 | chỉ client (ρ-loss), server FedAvg |
| M5 | chỉ server (class-aware aggregation), client loss mặc định |
| M6 | đầy đủ FedTraff-KP (ρ ∈ {0, 0.25, 1}) |
| M7 | FedRS/FedLC/FedVLS chuyển sang BCE (baseline học thuật, chọn 1–2) |

### 7.2 Ưu tiên chạy
- **Tier 1 (bắt buộc):** S1b × {M1, M3, M4, M5, M6(ρ=0)} × 3 seed = 15 run; S0 × {M1, M6} × 3 seed = 6 run.
- **Tier 2:** ρ sweep {0, 0.25, 1} × S1b × 3 seed; M2; control pair.
- **Tier 3:** M7, S1a, S1c.

### 7.3 Công thức ngân sách (điền số sau khi đo ở P0)
```text
GPU-giờ / run  ≈ rounds × clients × local_epochs × t_epoch(client) + rounds × t_eval
Tổng GPU-giờ   ≈ Σ_run (GPU-giờ / run)
Số tuần Kaggle ≈ Tổng GPU-giờ / hạn mức GPU mỗi tuần (xem tài khoản)
```
`t_epoch(client)` và `t_eval` **phải đo thật** ở P0 trên cấu hình cuối (imgsz, batch, số ảnh/client). Nếu tổng vượt ngân sách: giảm số ảnh/client, imgsz, rounds, hoặc cắt Tier 3, không giảm số seed của Tier 1 xuống dưới 3 (yêu cầu đề cương).

### 7.4 Giao thức báo cáo (khớp đề cương)
- Tối thiểu 3 seed; cùng partition seed giữa các phương pháp; báo mean ± std.
- Báo **kết quả vòng cuối** (hoặc trung bình k vòng cuối), không chỉ giá trị cao nhất, vì FedVLS báo mức cao nhất của global model và điều này dễ thiên lệch.
- Chỉ số: per-class AP50, mAP50, mAP50-95, precision, recall, FP/FN lớp vacant; thời gian local/round/aggregate; kích thước update.
- Ngưỡng "không suy giảm đáng kể" ở S0: đặt = 2 × std của FedAvg qua 3 seed (chốt trước khi chạy).
- Không so số của bạn với số trong các bài khác khi cấu hình khác; luôn chạy lại baseline.

---

## 8. Lịch 15 tuần (khớp bảng kế hoạch trong đề cương)

| Tuần | Công việc | Đầu ra |
|---|---|---|
| 1–2 | Khảo sát tài liệu theo prompt 1.1; tải dữ liệu; convert; thống kê lớp | Bảng related work (đã đọc), thống kê BDD100K |
| 3 | Baseline centralized YOLOv8n; đo `t_epoch` | mAP centralized, ngân sách GPU |
| 3–4 | **P0 chẩn đoán quên lớp** | Hình AP trước/sau + bảng ‖Δθ‖; quyết định đi tiếp |
| 5–6 | Partition S0/S1b/control; baseline FedAvg, FedProx | Manifest, bảng S0 vs S1b |
| 7–8 | ρ-loss (M4), class-aware aggregation (M5) | Code đã test, kết quả sơ bộ 1 seed |
| 9–10 | M6, ρ sweep, M3, ablation | Bảng ablation |
| 11–12 | Chạy chính 3 seed (Tier 1 → 2) | Bảng mean ± std |
| 13 | Tích hợp Flower/MLflow, quality gate (ngoài Kaggle) | Demo end-to-end |
| 14–15 | Viết báo cáo, hình AP theo lớp, rà soát tái lập | Bản thảo khóa luận |

Dự phòng: 1 tuần đệm giữa tuần 10–12 cho các run lỗi/hết quota.

---

## 9. Rủi ro và phương án

| Rủi ro | Dấu hiệu | Phương án |
|---|---|---|
| Quên lớp không xuất hiện (pretrained COCO che) | AP lớp vacant không giảm ở P0 | Dùng khởi tạo khác, tăng epoch cục bộ, tăng mức thiếu lớp |
| Lớp hiếm (motor/bus) quá ít ảnh | Không đủ ảnh chia client | Giảm số client, hoặc dùng lớp tần suất vừa; thêm lớp rider/bicycle và ghi rõ giới hạn |
| Hết quota GPU | Run dở dang | Resume theo round; cắt Tier 3; giảm imgsz |
| Ultralytics đổi API | Lỗi attribute/khóa state_dict | Pin phiên bản; assert số khóa cls head = 6 |
| Trùng ý với công trình khác (class-aware agg cho detection) | Tìm thấy ở bước C của prompt 1.1 | Nêu khác biệt (cơ chế client + ablation), hoặc chuyển trọng tâm sang phân tích + chuyển thể BCE |
| ρ=0 chưa đủ vì lớp conv trước vẫn lệch bởi lớp khác | AP lớp vacant vẫn giảm ở M4 | Thêm distillation (V2) hoặc FedProx |

---

## 10. Danh sách cần xác minh trước khi triển khai

1. Dataset BDD100K trên Kaggle: còn không, tên file nhãn, số ảnh train/val.
2. Phiên bản Ultralytics đã pin: tên khóa `cv3.*.2`, `init_criterion`, cách lấy `ema`.
3. Hạn mức GPU/tuần và giới hạn phiên trong tài khoản Kaggle của bạn.
4. Hạng CORE/ICORE của từng venue trích dẫn (FedRS, FedLC, FedVLS, FedNTD, NIID-Bench và các bài khác).
5. FedProx+LA (arXiv 2405.01108): mới đọc abstract; đọc phương pháp để đánh giá rủi ro trùng ý tưởng.
6. FedRS, FedNTD, FedOrbit, FLea, FedExIT: chưa đọc trực tiếp; đọc trước khi trích dẫn.