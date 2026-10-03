# Phân tích đồ án FLOPS: Federated Object Detection dưới Missing-Class Non-IID

> Cập nhật 2026-10-04. Mọi con số trong tài liệu này lấy từ **(a)** code trong repo, **(b)** source
> Ultralytics 8.3.253 / Flower 1.21.0 đã ghim phiên bản, hoặc **(c)** các lần chạy Kaggle đã ghi trong
> `research/experiment_registry/`. Nhãn bằng chứng theo CLAUDE.md §4:
> `[YOLO-DOC]` source Ultralytics · `[FL-DOC]` source Flower · `[LITERATURE]` bài báo đã kiểm trong
> `research/evidence/literature_registry.yaml` · `[ĐO]` số đo thực nghiệm · `[SUY LUẬN]` lập luận
> từ số liệu, chưa được kiểm chứng riêng · `[GIẢ THUYẾT]` giả thuyết luận văn, cần thực nghiệm.

---

## 0. Tóm tắt định lượng

| Câu hỏi | Trả lời ngắn | Căn cứ |
|---|---|---|
| Bài toán là gì? | Huấn luyện YOLOv8n liên kết (4 client) khi **một số client không có lớp nào đó** (bus/truck) | §1, §3 |
| Thiệt hại nằm ở đâu? | Lớp hiếm bị thiệt nặng **ngay cả khi dữ liệu IID**: FedAvg so với centralized giảm AP50 car −6,0 %, truck −21,5 %, bus −32,8 %, motorcycle −59,4 % | `[ĐO]` §5.2 |
| Cơ chế gây quên? | Với client không có bus, BCE chỉ sinh **gradient đẩy logit bus xuống**: 134.400 số hạng âm mỗi batch 16 ảnh, 0 số hạng dương | `[YOLO-DOC]` + tính toán §3.1 |
| Phần tham số "riêng của lớp"? | Chỉ **780 / 3.022.085 phần tử (0,026 %)**: 6 tensor `cv3.{0,1,2}.2` | `[ĐO]` F1 §2.3 |
| Phương pháp đề xuất làm gì? | A3: gộp hàng tham số của lớp *c* chỉ từ client có lớp *c*; A2b: nhân BCE của lớp vắng với ρ = 0,25 (giảm 4× áp lực đẩy xuống); A4b = A2b + A3 | §4 |
| Kết quả hiện có (S1b-2k, 30 round, 1 seed)? | Xem §5.3. **Thăm dò**: G4/G5 chưa pass, 1 seed | `[ĐO]` |

---

## 1. Thuật ngữ

| Thuật ngữ | Định nghĩa dùng trong đồ án |
|---|---|
| **Federated Learning (FL)** | Nhiều *client* huấn luyện cục bộ trên dữ liệu riêng, chỉ gửi tham số mô hình cho *server*; server gộp thành mô hình toàn cục. Dữ liệu không rời client. |
| **Round** | Một vòng: server gửi mô hình toàn cục → mỗi client train cục bộ → server gộp. Ở đây: 1 round = **1 epoch cục bộ** trên dữ liệu của client. |
| **IID / Non-IID** | IID: dữ liệu các client cùng phân phối. Non-IID: khác phân phối. Các dạng chính: *label distribution skew* (tỉ lệ lớp khác nhau), *quantity skew* (số mẫu khác nhau), và trường hợp cực đoan **missing-class** (một client có **0 mẫu dương** của một lớp). |
| **Missing-Class Non-IID** | Client có **0 box** của lớp *c*. Trong đồ án, client "thiếu bus" **không chứa ảnh nào có bus** (loại cả ảnh, không chỉ xóa nhãn), để bus không thành *hard-negative* sai (plan.md §5). |
| **Client drift** | Mô hình cục bộ trôi về tối ưu của dữ liệu riêng, lệch khỏi tối ưu toàn cục. Đây là mục tiêu của FedProx/SCAFFOLD. |
| **Catastrophic forgetting** | Kiến thức đã học (ở đây: phát hiện bus) bị xóa khi tiếp tục train trên dữ liệu không có nó. |
| **Class head / hàng lớp** | Lớp conv 1×1 cuối nhánh phân loại của YOLOv8 (`model.22.cv3.<s>.2`), mỗi lớp có **một hàng** trọng số (64 giá trị) + 1 bias ở mỗi scale *s* ∈ {P3, P4, P5}. |
| **Logit / BCE** | Logit *z* = đầu ra trước sigmoid; xác suất *p* = σ(*z*). YOLOv8 dùng **Binary Cross-Entropy** độc lập cho từng lớp, *không* softmax. |
| **Anchor point** | Mỗi ô lưới của 3 feature map dự đoán một box; ảnh 640 px có 80² + 40² + 20² = **8.400** anchor. |
| **TAL (Task-Aligned Assigner)** | Bộ gán của YOLOv8: mỗi box thật nhận tối đa `topk = 10` anchor dương; các anchor còn lại là âm cho mọi lớp. |
| **AP / mAP50 / mAP50-95** | AP = diện tích dưới đường precision-recall của một lớp. mAP50 = trung bình AP ở IoU 0,5; mAP50-95 = trung bình trên IoU 0,50…0,95. |
| **conf = 0,001 vs 0,25** | AP được tính ở ngưỡng **0,001** (chuẩn Ultralytics khi validate) để có toàn bộ đường PR; **FP/FN** đếm ở ngưỡng vận hành **0,25** (ADR-009). |
| **FP / FN** | False positive (dự đoán sai) / false negative (bỏ sót box thật), theo confusion matrix ở conf 0,25, IoU 0,45. |
| **EMA** | Exponential Moving Average của trọng số do Ultralytics duy trì; trọng số client gửi lên là EMA bản fp32 (ADR-008). |
| **AMP** | Automatic Mixed Precision (fp16/fp32) khi train trên GPU. |
| **Matched control** | Partition đối chứng giống hệt S1b về số ảnh và số box các lớp không-mục-tiêu, **khác duy nhất ở có/không bus**, để tách hiệu ứng thiếu lớp khỏi các yếu tố gây nhiễu (CLAUDE.md §10). |
| **Ablation A0–A4b** | Các biến thể bật/tắt từng thành phần của phương pháp (§4). |
| **Gate G0–G11** | Cổng kiểm soát tiến trình nghiên cứu (CLAUDE.md §7); không được nhảy cổng. |

---

## 2. Các thành phần của đồ án

### 2.1 Dữ liệu BDD100K (4 lớp mục tiêu) `[ĐO]`

| Lớp | bbox train | ảnh train chứa lớp | % ảnh train | bbox val | ảnh val chứa lớp |
|---|---:|---:|---:|---:|---:|
| car | 713.211 | 69.072 | 98,9 % | 102.506 | 9.879 |
| truck | 29.971 | 18.890 | 27,3 % | 4.245 | 2.689 |
| **bus** | 11.672 | 8.993 | 13,0 % | 1.597 | 1.242 |
| motorcycle | 3.002 | 2.284 | 3,3 % | 452 | 334 |

- 69.271 ảnh train dùng được (69.863 có nhãn, 592 không chứa lớp mục tiêu); 10.000 ảnh val.
- **Mất cân bằng cực mạnh**: car có số box gấp 61× bus và 238× motorcycle.
- **Kiểm chéo evaluator:** ở round 0 (mô hình chưa phát hiện gì), FN = 102.506 / 1.597 / 4.245 / 452,
  trùng khít số box val ở bảng trên. Như vậy evaluator đọc đúng toàn bộ nhãn val.
- **Hệ quả thiết kế:** car không thể là lớp vắng (98,9 % ảnh có car); motorcycle quá ít (452 box val)
  để đo hiệu ứng có ý nghĩa thống kê → **bus là lớp mục tiêu chính**, truck thứ cấp (plan.md quyết định D1).

### 2.2 Partition (cách chia dữ liệu cho client)

Ba kịch bản, đều do `src/data/partitioner.py` sinh tất định theo seed và lưu `manifest.yaml`.

| Kịch bản | Mô tả | Mục đích |
|---|---|---|
| **S0** (IID) | Chia đều ngẫu nhiên | Kiểm tra phương pháp không làm hại trường hợp bình thường |
| **S1b** | C0, C1: **0 bus**; C2: 0 truck; C3: đủ 4 lớp | Kịch bản Missing-Class chính |
| **S1-Control-Matched** | Từ S1b, **hoán đổi tối thiểu** ảnh không-bus của C0/C1 lấy ảnh có bus, giữ nguyên số ảnh và vector box các lớp khác | Đối chứng nhân quả cho H1 |

**S1b-2k thực tế** (2.000 ảnh/client, dùng trong phiên 5) `[ĐO]`:

| Client | car | **bus** | truck | motorcycle | lớp vắng |
|---|---:|---:|---:|---:|---|
| C0 | 20.664 | **0** | 1.012 | 89 | bus |
| C1 | 20.524 | **0** | 841 | 71 | bus |
| C2 | 20.944 | 732 | **0** | 109 | truck |
| C3 | 20.700 | 635 | 1.586 | 78 | — |
| **Tổng** | 82.832 | **1.367** | 3.439 | 347 | |

**Đối chứng matched-2k** `[ĐO]`: bus = 683 / 683 / 732 / 683 (tổng **2.781**); car của C0 lệch
20.664 → 20.654 (−0,05 %), các lớp không-mục-tiêu khác giữ nguyên → `match_report: matched = true`.

> ⚠️ **Lưu ý định lượng khi đọc H1:** tổng box bus của đối chứng gấp **2,03×** S1b (2.781 so với 1.367).
> So sánh S1b với đối chứng vì vậy đo hiệu ứng gộp của **(i) bus vắng ở 2/4 client** và
> **(ii) ít hơn một nửa lượng giám sát bus**. Đây là bản chất của việc "thiếu lớp" (bỏ client có
> lớp đi thì mất cả dữ liệu lớp đó), nhưng phải được nêu rõ khi kết luận.

### 2.3 Mô hình YOLOv8n (4 lớp) `[ĐO]` `[YOLO-DOC]`

Đếm bằng `research/feasibility/F1/verify_map.py` trên stack Kaggle đã ghim (torch 2.7.1+cu128):

| Nhóm | Phần tử | Tỉ lệ | Vai trò |
|---|---:|---:|---|
| backbone (`model.0–9`) | 1.277.707 | 42,28 % | Trích đặc trưng, dùng chung mọi lớp |
| neck (`model.10–21`) | 990.738 | 32,78 % | Kết hợp đa tỉ lệ (PAN-FPN), dùng chung |
| detect_reg (`cv2`) | 382.662 | 12,66 % | Hồi quy box, **không phụ thuộc lớp** |
| detect_cls (`cv3`) | 370.962 | 12,28 % | Nhánh phân loại |
| └─ **class head** (`cv3.<s>.2`) | **780** | **0,026 %** | **Phần duy nhất có hàng riêng cho từng lớp** |
| detect_dfl | 16 | ~0 % | Tích phân DFL cố định |

- Class head = 3 scale × (weight 4×64 + bias 4) = 780 phần tử; mỗi lớp sở hữu đúng **195 phần tử**.
- Bias khởi tạo `[YOLO-DOC]` `bias_init = ln(5/nc/(640/s)²)`: −8,541 / −7,155 / −5,768 cho stride 8/16/32
  (xác suất 1,95·10⁻⁴ / 7,8·10⁻⁴ / 3,1·10⁻³), khớp giá trị quan sát trong checkpoint.
- Từ COCO-pretrained chỉ truyền được **319/355** key: toàn bộ 36 key của `cv3` bị khởi tạo lại, vì
  kênh ẩn của cv3 phụ thuộc nc (`c3 = max(ch[0], min(nc,100))`: 64 với nc=4, 80 với nc=80) (ADR-008).

### 2.4 Huấn luyện cục bộ (mỗi client, mỗi round) `[YOLO-DOC]`

| Thiết lập | Giá trị | Ghi chú |
|---|---|---|
| Optimizer | SGD, lr0 = 0,01, momentum 0,937, weight decay 5·10⁻⁴ | mặc định Ultralytics |
| Batch / nbs | 16 / 64 → gộp gradient 4 batch cho mỗi bước | |
| Số bước tối ưu mỗi round | ⌈2000/16⌉ / 4 ≈ **31** (S1b-2k) | |
| Loss | `7,5·box(CIoU) + 0,5·cls(BCE) + 1,5·DFL` | hệ số mặc định `box/cls/dfl` |
| AMP | bật | |
| EMA | decay = 0,9999·(1−e^(−u/2000)); với u ≈ 31 bước/round → **0,015**, tức EMA ≈ trọng số thô | |
| Warm-up / close_mosaic | 0 / 0 cho round FL (1 epoch); 3 / 10 cho centralized G2 | ADR-009 |

### 2.5 Server và chiến lược gộp (Flower 1.21) `[FL-DOC]`

Server chạy `flwr.simulation` với 4 client (Ray actor), mỗi GPU chạy một client (`client_num_gpus: 1.0`).
Sau mỗi round, server lưu checkpoint và, theo `eval_every`, đánh giá mô hình toàn cục trên **toàn bộ
10.000 ảnh val** (cùng một evaluator cho mọi phương pháp).

### 2.6 Đánh giá

- AP/mAP ở conf 0,001; FP/FN ở 0,25 (một lượt validate duy nhất nhờ cơ chế có sẵn của Ultralytics,
  `utils/metrics.py: conf = 0.25 if conf in {None, 0.001}`) `[YOLO-DOC]`.
- Thống kê confidence theo lớp ở ngưỡng 0,25 (F2/F3).

### 2.7 Hạ tầng (không phải đóng góp khoa học nhưng quyết định tính đúng)

Các lỗi đã tìm và sửa, mỗi lỗi đều làm sai hoặc làm mất kết quả: model nc=80 so với nc=4; trọng số client
bị làm tròn fp16; `val()` fuse model tại chỗ; stdout pipe đầy gây treo; giới hạn 500 file output của Kaggle;
tên metric MLflow không hợp lệ; seed G2 không được truyền; FedProx/SCAFFOLD/A2b từng là vỏ rỗng
(ADR-008 … ADR-011).

---

## 3. Vì sao Missing-Class làm hỏng mô hình: phân tích định lượng

### 3.1 Gradient của BCE trên client thiếu bus `[YOLO-DOC]` + tính toán

Với một anchor và lớp *c*: ℓ = −[t·log σ(z) + (1−t)·log(1−σ(z))] ⇒ **∂ℓ/∂z = σ(z) − t**.

- Trên client **không có bus**, mọi anchor đều có t_bus = 0 ⇒ ∂ℓ/∂z_bus = σ(z_bus) **> 0 ở mọi anchor**:
  gradient luôn đẩy logit bus **xuống**, không bao giờ kéo lên.
- Số số hạng mỗi batch: 16 ảnh × 8.400 anchor = **134.400 số hạng âm** cho bus, **0 số hạng dương**.
- Trên C2 (có bus): 732 box / 2.000 ảnh = 0,366 bus/ảnh; với `topk = 10`, mỗi batch có tối đa
  ≈ 16 × 0,366 × 10 ≈ **59 anchor dương** cho bus so với ~134.341 anchor âm.
- Hệ quả `[SUY LUẬN]`: ở mô hình đã biết bus, σ(z_bus) trên các vùng "giống bus" lớn, nên gradient đẩy
  xuống lớn chính ở những vùng đó. Đây là cơ chế quên trực tiếp lên **hàng bus của class head** và gián
  tiếp lên **đặc trưng dùng chung** (qua backprop vào backbone/neck/cv3 chung).

### 3.2 Pha loãng khi gộp tham số

FedAvg gộp mỗi tham số theo số ảnh: w_i = n_i / Σn_j. Trong S1b-2k, mọi client có 2.000 ảnh ⇒ **w_i = 0,25**.

Với hàng bus của class head:
θ_bus^(t+1) = 0,25·(θ_bus,C0 + θ_bus,C1) **[2 client chỉ đẩy xuống]** + 0,25·(θ_bus,C2 + θ_bus,C3).

⇒ **50 % "khối lượng" cập nhật** của hàng bus đến từ client không thể học bus. Với truck: 25 % (C2).

### 3.3 Quên có thể nằm ngoài class head `[SUY LUẬN]`

Class head chỉ chiếm 0,026 % tham số; **99,97 %** còn lại là dùng chung. Nếu phần lớn thiệt hại cho bus
nằm ở đặc trưng dùng chung, thì mọi cơ chế **chỉ can thiệp vào hàng lớp** (A1, A3, A2a) sẽ bị giới hạn.
Đây chính là câu hỏi F2/F3 phải trả lời (G5, Gate A/B/C): chưa đo, không được giả định.

### 3.4 Thiệt hại cho lớp hiếm có sẵn kể cả khi IID `[ĐO]`

Phiên 2 (S0 IID, 1 seed): FedAvg đạt **77,2 %** mAP50 của centralized, và tỉ lệ thiệt tăng theo độ hiếm
(car −6,0 % → motorcycle −59,4 %). Vì vậy hiệu ứng Missing-Class phải được đo **so với đối chứng matched**,
không so với centralized.

---

## 4. Các phương pháp và công thức

Ký hiệu: θ^t là mô hình toàn cục ở round *t*; θ_i là mô hình sau khi client *i* train cục bộ;
n_i là số ảnh của client *i*; n_i^c là số box lớp *c* ở client *i*.

### 4.1 Baseline

| Phương pháp | Công thức / cơ chế | Nhắm vào | Trạng thái |
|---|---|---|---|
| **FedAvg** `[LITERATURE]` McMahan et al. 2017 | θ^(t+1) = Σ_i (n_i/N)·θ_i | — | ✅ |
| **FedProx** `[LITERATURE]` Li et al. 2020 | Client tối thiểu F_i(w) + (μ/2)‖w − θ^t‖²; gradient thêm μ(w − θ^t); μ = 0,01 | Client drift do dữ liệu dị biệt | ✅ (ADR-010) |
| **SCAFFOLD** `[LITERATURE]` Karimireddy et al. 2020 | Hiệu chỉnh gradient bằng biến điều khiển (c − c_i) | Client drift | ⛔ **chặn**: client cũ không hiệu chỉnh, sai công thức Δc_i, K sai, mất c_i giữa round |
| **FedNova** `[LITERATURE]` Wang et al. 2020 | Chuẩn hóa cập nhật theo số bước cục bộ τ_i | Không nhất quán mục tiêu | chưa chạy |

> FedProx và SCAFFOLD chống **trôi tham số nói chung**; không phương pháp nào trong chúng biết client nào
> **thiếu lớp nào**. Vì vậy chúng không nhắm trực tiếp vào cơ chế §3.1–§3.2 `[SUY LUẬN]`.

### 4.2 Phương pháp đề xuất (bài của đồ án) `[GIẢ THUYẾT]`

**H3 — gộp nhận biết lớp ở server (A3)**, chỉ áp cho 6 tensor class head:
- S_c = { i : n_i^c ≥ τ_elig }, τ_elig = 1
- w_i^c = n_i^c / Σ_{j∈S_c} n_j^c
- θ^(t+1)[c] = θ^t[c] + Σ_{i∈S_c} w_i^c·(θ_i[c] − θ^t[c])
- **S_c = ∅ ⇒ giữ nguyên θ^t[c]** (luật no-contributor). Phần tham số còn lại gộp theo FedAvg.

**A1 — đối chứng class-count**: w_i^c = n_i^c / Σ_j n_j^c trên **mọi** client, không có ngưỡng và không có luật no-contributor.

Trọng số thực tế cho **hàng bus** trong S1b-2k:

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| FedAvg | 0,250 | 0,250 | 0,250 | 0,250 |
| A1 = A3 | **0** | **0** | 0,535 | 0,465 |

**Định danh toán học A3 ≡ A1 trong S1b** `[SUY LUẬN, chứng minh được]`: vì τ_elig = 1, client có n_i^c = 0
đóng góp trọng số 0 ở cả hai công thức, nên mẫu số bằng nhau; dạng delta với Σw = 1 cho cùng tổ hợp lồi.
Hai công thức chỉ khác khi S_c = ∅ (không client nào có lớp *c*), mà điều này **không bao giờ xảy ra** trong
S1b (bus có ở C2 và C3; truck có ở C0, C1, C3). Hệ quả:
1. Trong S1b, **A3 và A1 phải cho kết quả như nhau**; mọi chênh lệch đo được giữa chúng chính là
   **nhiễu chạy lại** (GPU không tất định), tức một ước lượng sàn nhiễu cho 1 seed.
2. Đóng góp riêng của H3 so với A1 chỉ thể hiện được ở kịch bản có S_c = ∅ (S1d), hoặc với τ_elig > 1
   (plan.md §6.4).

**H2 — bảo toàn phía client**:
- *A2a* (che tham số sau khi train): giữ hàng của lớp vắng gần θ^t với hệ số ρ. Dưới A3 nó **không làm
  thay đổi gì** (A3 đã loại chính client đó khỏi S_c), nên A4a ≡ A3 (plan.md §6.3).
- ***A2b* (ADR-006)**: nhân BCE của lớp vắng với ρ ⇒ ∂(ρ·ℓ)/∂z = **ρ·σ(z)**. Với **ρ = 0,25**, áp lực
  "đẩy bus xuống" trên C0/C1 giảm **4 lần**; ρ = 0 thì xóa hẳn (đã kiểm trên trainer thật: bias hàng bus
  giữ nguyên từng bit). Khác A2a, A2b thay đổi **quỹ đạo của tham số dùng chung**, nên tách được khỏi A3.
- **Dự đoán có thể bác bỏ** (plan.md §6.1): giảm áp lực âm thì có thể làm **tăng FP bus**.

**A4b = A2b + A3**: phương pháp đầy đủ.

---

## 5. Thực nghiệm và kết quả

### 5.1 Ngân sách đo được (Kaggle T4 ×2) `[ĐO]`

| Hạng mục | Thời gian |
|---|---|
| Cài môi trường + convert dataset | ~9 phút |
| G2 centralized, 1 epoch (69k ảnh, batch 16, 1 GPU) | ~13 phút |
| FL round S0 full (4 × 17,3k ảnh) + eval | ~20 phút (FedProx ~23,5 phút) |
| Eval toàn bộ val (conf 0,001) | ~4–5 phút |
| **1 arm S1b-2k, 30 round, eval mỗi 5 round** | **~64–66 phút** |

### 5.2 Phiên 2: S0 IID, 5 round, toàn bộ dữ liệu, seed 42 `[ĐO]`

| | Centralized (10 epoch) | FedAvg | FedProx |
|---|---:|---:|---:|
| mAP50 | 0,488 | 0,377 | 0,376 |
| mAP50-95 | 0,323 | 0,244 | 0,244 |
| AP50 car / bus / truck / motorcycle | 0,720 / 0,494 / 0,525 / 0,215 | 0,676 / 0,332 / 0,412 / 0,087 | 0,678 / 0,332 / 0,409 / 0,083 |

- FedProx − FedAvg nằm trong ±0,004 AP50 ở mọi lớp ⇒ **không có hiệu ứng đo được trên IID** (đúng kỳ vọng).
- FedAvg chưa hội tụ ở round 5 (mAP50 +0,030/round). FL chỉ đi qua dữ liệu 5 lượt, so với 10 lượt của G2.

### 5.3 Phiên 5: S1b-2k, 30 round, seed 42 (thăm dò) `[ĐO]`

**s5b (đã xong):**

| | A0 @ đối chứng | A2b (ρ 0,25) @ S1b | A4b @ S1b |
|---|---:|---:|---:|
| mAP50 | 0,2911 | 0,2839 | 0,2846 |
| mAP50-95 | 0,1821 | 0,1743 | 0,1746 |
| AP50 car | 0,6217 | 0,6269 | 0,6268 |
| **AP50 bus** | **0,2556** | 0,2219 | 0,2160 |
| AP50 truck | 0,2807 | 0,2822 | 0,2906 |
| AP50 motorcycle | 0,0064 | 0,0045 | 0,0049 |
| FP bus (conf 0,25) | 3.035 | 546 | 777 |
| FN bus (conf 0,25) | 942 | 1.376 | 1.371 |
| Recall bus @0,25 = 1 − FN/1.597 | 41,0 % | 13,8 % | 14,2 % |

**s5a (A0, FedProx, A1, A3 trên S1b):** ⏳ *đang chạy, sẽ được điền vào bảng tổng hợp ở §5.4.*

**Quan sát từ s5b** (chưa phải kết luận về phương pháp, vì thiếu A0@S1b để so):
1. So với đối chứng, cả A2b và A4b trên S1b đều có AP50 bus thấp hơn 13–15 % và **ít FP bus hơn 74–82 %**,
   nhưng **nhiều FN bus hơn ~46 %**, tức mô hình trên S1b "dè dặt" với bus hơn hẳn. Đây là chiều hướng của
   việc đối chứng có gấp 2,03× số box bus.
2. A4b − A2b: AP50 bus −0,006, FP bus +231. Muốn biết chênh lệch này có ý nghĩa hay không, cần sàn nhiễu
   (|A3 − A1| trong s5a, §4.2).
3. motorcycle ≈ 0,005 AP50 ở mọi arm: với 347 box train toàn federation, lớp này **không học được** ở quy
   mô 2k, nên không dùng để kết luận.

### 5.4 Bảng tổng hợp S1b-2k

⏳ Được điền sau khi s5a hoàn tất.

---

## 6. Hạn chế và mức độ tin cậy

| Hạn chế | Hệ quả định lượng | Cách khắc phục |
|---|---|---|
| **1 seed** | Không có std; chênh lệch < sàn nhiễu (ước lượng bằng \|A3 − A1\|) là không phân biệt được | 3 seed (§14): 7 arm × 3 seed × ~65 phút ≈ 23 giờ GPU |
| G4/G5 chưa pass | Kết quả của phương pháp là thăm dò, chưa đủ để kết luận "hiệu quả" | F2 + F3 (~2 giờ) → ADR-003 |
| Đối chứng có 2,03× box bus | H1 đo hiệu ứng gộp của "vắng" + "ít dữ liệu" | Nêu rõ khi báo cáo; có thể thêm đối chứng cân số box |
| A3 ≡ A1 trong S1b | H3 chưa có cơ hội khác A1 | Kịch bản S1d (bus vắng ở mọi client có mặt trong một số round) hoặc τ_elig = 50 |
| ρ = 0,25 chọn trước, chưa quét | Chưa biết ρ tối ưu | Quét ρ ∈ {0; 0,25; 1}, 1 seed (~3 × 65 phút) |
| 2.000 ảnh/client | mAP tuyệt đối thấp hơn (0,28–0,29 so với 0,38 ở S0 full) | Chấp nhận để đủ ngân sách; ghi rõ |
| Kênh cv3 khởi tạo lại | Không tận dụng kiến thức car/bus/truck/motorcycle sẵn có của COCO | ADR-009 mục 5 (giữ head 80 lớp, cần ADR riêng) |

---

## 7. Lộ trình giải quyết Non-IID (định lượng theo ngân sách)

| Bước | Mục tiêu | Chi phí GPU | Điều kiện để chuyển bước |
|---|---|---:|---|
| 1. Đọc s5a, tính sàn nhiễu \|A3 − A1\| | Biết chênh lệch nào là thật | 0 | — |
| 2. F2 + F3 (phiên 3) | G5: hàng lớp có tác động riêng không? Quên nằm ở đâu? | ~2 giờ | Chốt τ_AP |
| 3. Quét ρ (A2b), 1 seed | Chọn ρ* trước khi chạy main (không tune trên kết quả cuối) | ~3,3 giờ | — |
| 4. S1d hoặc τ_elig = 50 | Cho H3 khác A1 | ~1–2 giờ/arm | — |
| 5. Main: 3 seed × {A0, FedProx, A1, A3, A2b(ρ*), A4b, A0@control} | Kết quả báo cáo được (§14, §22) | ~23 giờ (chia 2 tuần quota) | G4 + G5 pass |

---

## 8. Nguồn

- Mã nguồn: `src/model/parameter_map.py`, `src/federated/strategies/class_aware_agg.py`,
  `src/preservation/rho_loss.py`, `src/federated/proximal.py`, `src/data/partitioner.py`.
- Quyết định: `research/decisions/ADR-006`, `-008`, `-009`, `-010`, `-011`; `research/plan/plan.md` §5–§7.
- Kết quả: `research/experiment_registry/registry.yaml`,
  `research/experiment_registry/reports/2026-10-03_s2_baselines_feasibility.md`; log Kaggle
  `flops-s2-baseline-g2-g3` v3, `flops-s5b-s1b-method` v1, `flops-s5a-s1b-baselines` v1.
- Văn liệu (đã kiểm trong `research/evidence/literature_registry.yaml`): McMahan et al. 2017 (AISTATS,
  arXiv:1602.05629); Li et al. 2020 (MLSys, arXiv:1812.06127); Karimireddy et al. 2020 (ICML,
  arXiv:1910.06378); Wang et al. 2020 (NeurIPS, arXiv:2007.07481); Zhao et al. 2018 (arXiv:1806.00582);
  Yu et al. 2020 BDD100K (CVPR, arXiv:1805.04687); Ultralytics YOLOv8 (phần mềm, 8.3.253).
