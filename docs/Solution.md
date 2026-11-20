# SPEC — Các phương pháp cải tiến cho lớp bus (thiếu lớp) và motorcycle (lớp hiếm, vật thể nhỏ)

> Tài liệu giao việc cho Claude Code. Ngày: 05/11/2026.
> Repo: FLOPS (YOLO11s + Flower 1.21.0 + Ultralytics 8.3.253, Kaggle 2 × T4).
> Đọc toàn bộ tài liệu trước khi sửa code. Làm theo đúng thứ tự các giai đoạn ở Mục 7.

---

## 0. Quy tắc bắt buộc

1. **Không bịa số liệu.** Mọi con số trong báo cáo hay ADR phải lấy từ run thật và được ghi vào `research/experiment_registry/registry.yaml`. Run lỗi hoặc kết quả âm cũng phải ghi.
2. **Mặc định tắt.** Mọi thành phần mới bật bằng cờ cấu hình. Khi tắt hết cờ, kết quả phải **giống hệt** cấu hình hiện tại (kiểm bằng unit test và một smoke run).
3. **Không sửa bộ đánh giá** (evaluator, cách tính AP50/mAP50, tập val) trong spec này. Chỉ được **thêm** chỉ số mới ở Mục 6.
4. **Viết ADR trước khi chạy.** Mỗi phương pháp có một ADR mới ghi giả thuyết, cấu hình và quy tắc quyết định **trước** khi chạy thí nghiệm chính. Không đổi quy tắc sau khi đã thấy kết quả.
5. **Không chọn siêu tham số trên tập dùng để báo cáo.** Nếu cần dò λ hay ngưỡng, chỉ dò trên seed 42 và ghi rõ là ablation.
6. **Thông tin còn nghi ngờ thì kiểm tra trong code** (Mục 1), không giả định.
7. **Mỗi thành phần một commit**, có unit test đi kèm.

---

## 1. Việc cần xác minh trong code trước khi bắt đầu

Ghi kết quả xác minh vào `research/notes/spec_verify_YYYYMMDD.md`.

| # | Cần xác minh | Vì sao cần |
|---|---|---|
| V1 | Chưng cất not-true hiện tại (thành phần ② của cấu hình tốt nhất) áp lên **toàn bộ anchor** hay chỉ một phần? Teacher là mô hình nào, chạy ở độ phân giải nào? | Phương pháp B1 có thể thay thế thành phần này |
| V2 | Trong `ultralytics/utils/loss.py` (`v8DetectionLoss.__call__`) bản 8.3.253: tên biến và shape của `pred_scores`, `target_scores`, `fg_mask`; vị trí tính `target_scores_sum`; assigner có trả `target_labels` không (hiện có thể đang bị bỏ bằng `_`) | Cần cho B1 |
| V3 | Teacher COCO ở chế độ `eval()` có trả về feature thô của Detect head không (`Detect.forward` ở eval thường trả `(y, x)`, trong đó `x` là list feature map) | Lấy điểm lớp của teacher mà không bật `train()`, vì `train()` làm BatchNorm cập nhật thống kê |
| V4 | Thứ tự anchor của teacher và học trò có khớp 1–1 không (cùng kiến trúc, cùng stride 8/16/32, cùng imgsz 960) | Cần cho B1 |
| V5 | Với partition hiện tại (`s5moto`, 5 client), mỗi client có bao nhiêu box của từng lớp? Motorcycle có mặt ở mọi client không? | Xác định `missing_classes` của từng client và phạm vi phương pháp M |
| V6 | Cách Flower truyền cấu hình theo vòng xuống client (`on_fit_config_fn` hoặc tương đương) | Cần cho M1 (tắt mosaic theo vòng) |
| V7 | Bộ đánh giá xử lý 76 lớp COCO không thuộc mục tiêu thế nào (lọc trước NMS, sau NMS, hay bỏ qua khi tính metric) | Ảnh hưởng tới diễn giải số báo nhầm |
| V8 | Các run baseline (FedAvg; kịch bản lý tưởng) đã có ở **cùng cấu hình 960 px, dữ liệu `s5moto`** chưa | Nếu chưa, phải chạy ở Giai đoạn 0 |

---

## 2. Cấu hình nền

"**Nền**" là cấu hình tốt nhất hiện tại: FedAvg ở server, head COCO 80 lớp, learning rate cosine theo vòng, repeat factor sampling t = 1.0, toàn bộ ảnh motorcycle (`s5moto`), huấn luyện và đánh giá ở 960 px. Mọi phương pháp dưới đây là **Nền + thành phần mới**, so với Nền trên cùng seed và cùng partition.

Chỉ số lớp trong head COCO: `car = 2`, `motorcycle = 3`, `bus = 5`, `truck = 7`.

---

## 3. Phương pháp B1 — Chưng cất lớp vắng có chọn lọc theo vùng dễ nhầm (ưu tiên 1)

### 3.1 Giả thuyết

Ở client không có lớp *c* (ví dụ bus), các vật thể thuộc lớp dễ nhầm với *c* (truck, car) nhận mục tiêu bằng 0 cho *c*. Điểm của *c* vì vậy bị đẩy xuống mạnh nhất đúng tại ranh giới phân loại, dẫn tới bus thật bị nhận thành truck/car. Phân tích lỗi trước đây cho thấy khoảng 74% số bus bị bỏ sót thêm là do nhầm lớp. Thay mục tiêu 0 tại **riêng các vùng này** bằng xác suất của teacher COCO sẽ giữ được ranh giới, trong khi mẫu âm ở vùng nền vẫn được giữ.

### 3.2 Quy tắc

Ký hiệu:
- `M_k`: tập lớp vắng ở client k (`n_k^c == 0`, client tự đếm từ nhãn cục bộ).
- `N(c)`: tập lớp dễ nhầm với c.
- `q[a,c] = sigmoid(teacher_logit[a,c])`.
- `s[a]`: điểm căn chỉnh TAL của anchor dương a.

```
Với mỗi c ∈ M_k, mỗi anchor a:
  nếu fg_mask[a] và target_labels[a] ∈ N(c):
      target_scores[a, c] = λ · min(q[a,c], q_max) · s[a]
  ngược lại:
      giữ nguyên (= 0)
```

- Mục tiêu của lớp được gán (`target_labels[a]`) **không đổi**.
- Client không có lớp vắng: loss **giống hệt** Nền.
- Loss khung và DFL không đổi.

### 3.3 Cài đặt

**File:** `ultralytics/utils/loss.py` (bản vá trong repo, hoặc subclass `v8DetectionLoss`, ví dụ `src/losses/absent_class_kd.py`; ưu tiên subclass).

```python
CAR, MOTO, BUS, TRUCK = 2, 3, 5, 7
CONFUSABLE = {BUS: (TRUCK, CAR), TRUCK: (BUS, CAR)}

# ... trong __call__, sau assigner:
target_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(...)
target_scores_sum = max(target_scores.sum(), 1)       # PHẢI tính TRƯỚC khi sửa

if self.cfg.absent_kd.enabled and self.missing_classes:
    with torch.no_grad():
        t_scores = self.teacher_cls_scores(batch["img"])   # (B, A, 80), đã sigmoid
    align = target_scores.amax(-1)                          # s[a]
    for c in self.missing_classes:
        confus = torch.zeros_like(fg_mask)
        for d in CONFUSABLE.get(c, ()):
            confus |= fg_mask & (target_labels == d)
        soft = self.cfg.absent_kd.lam * t_scores[..., c].clamp(max=self.cfg.absent_kd.q_max) * align
        target_scores[..., c] = torch.where(confus, soft.to(target_scores.dtype), target_scores[..., c])

loss_cls = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
```

**`teacher_cls_scores`:**
- Teacher là YOLO11s COCO gốc, `eval()`, `requires_grad=False`, cùng thiết bị và cùng AMP.
- Lấy feature thô của Detect head (xem V3), tách phần phân loại đúng như học trò làm, rồi `sigmoid`.
- Không được gọi `teacher.train()`.
- Nếu pipeline đã có teacher forward cho chưng cất not-true, dùng lại kết quả đó, không forward hai lần.

**Truyền `missing_classes` qua Flower:** trong `client_fn`, đọc manifest của client, tính `missing_classes = [c for c in (CAR, MOTO, BUS, TRUCK) if count[c] == 0]`, rồi gắn vào loss/trainer. Thông tin này không gửi lên server.

**Quan hệ với chưng cất not-true hiện tại (theo V1):**
- Nếu bản hiện tại chưng cất trên mọi anchor: tạo cấu hình `B1_replace` (tắt bản cũ, bật B1) **và** `B1_add` (giữ bản cũ, thêm B1).
- Run chính dùng `B1_replace`.

### 3.4 Cấu hình

```yaml
absent_kd:
  enabled: false        # mặc định tắt
  lam: 1.0              # ablation: 0.5
  q_max: 0.5            # ablation: 1.0 (= không trần)
  confusable:           # chỉ số head COCO
    5: [7, 2]           # bus  -> truck, car
    7: [5, 2]           # truck -> bus, car
  scope: confusable     # ablation: all_anchors (tương đương chưng cất dày đặc)
  teacher_weights: yolo11s.pt
```

### 3.5 Unit test

1. Client không có lớp vắng: `target_scores` sau hàm **bằng hệt** trước hàm.
2. Chỉ các phần tử `(a, c)` với `fg_mask[a]`, `target_labels[a] ∈ N(c)`, `c ∈ M_k` bị thay đổi. Mọi phần tử khác giữ nguyên.
3. `target_scores[a, target_labels[a]]` không đổi.
4. Mọi giá trị mới ≤ `lam · q_max · s[a]`.
5. `target_scores_sum` dùng giá trị **trước** khi sửa.
6. Khi `enabled: false`: loss của một batch cố định bằng loss của Nền (sai số ≤ 1e-6).
7. Teacher không bị cập nhật: kiểm tra tham số và thống kê BatchNorm trước và sau 1 bước.

---

## 4. Các phương pháp cho motorcycle (lớp hiếm, vật thể nhỏ)

Kiểm tra V5 trước. Nếu motorcycle có ở mọi client thì đây là cải tiến cho lớp hiếm và vật thể nhỏ, không phải giải pháp cho thiếu lớp. Báo cáo phải trình bày tách riêng.

### M1 — Giảm thu nhỏ vật thể khi augmentation (ưu tiên 2)

- Đặt `scale` (Ultralytics hyp) = 0.25 (mặc định 0.5). Ablation: 0.2, 0.3.
- **Tắt mosaic ở K vòng FL cuối** (K = 5 trên 20 vòng). Lưu ý: `close_mosaic` của Ultralytics tính theo epoch, mà mỗi vòng FL chỉ có 1 epoch, nên tham số này **không** có tác dụng. Phải để server gửi `mosaic: 0.0` trong fit config cho các vòng `r > R − K`, và client đặt `trainer.args.mosaic` tương ứng trước khi train.

```yaml
small_obj_aug:
  enabled: false
  scale: 0.25
  mosaic_off_last_rounds: 5
```

Test: log `mosaic` và `scale` thực tế mỗi vòng ở client; xác nhận 5 vòng cuối có `mosaic = 0`.

### M2 — Dán motorcycle trong phạm vi dữ liệu của chính client (ưu tiên 2)

- **Cơ sở:** Kisantal et al. 2019, "Augmentation for small object detection" (arXiv:1902.07296); Ghiasi et al., CVPR 2021, Simple Copy-Paste.
- **Ngân hàng:** mỗi client, trước vòng 1, cắt toàn bộ box motorcycle từ ảnh của chính nó (BDD100K không có mask, nên cắt theo khung, mở rộng 2 px mỗi cạnh). Lưu trong RAM hoặc cache cục bộ, **không gửi đi đâu**.
- **Transform** (chèn vào pipeline augmentation của client, **trước** mosaic):
  - Với xác suất `p = 0.5` mỗi ảnh, dán `n ~ U{1..3}` motorcycle lấy ngẫu nhiên từ ngân hàng.
  - Giữ kích thước gốc (không phóng to).
  - Vị trí: tâm box nằm ở nửa dưới ảnh (`y > 0.5·H`); IoU với mọi box hiện có (kể cả box vừa dán) ≤ 0.05; thử tối đa 20 lần, không được thì bỏ.
  - Làm mờ mép bằng mask có viền mềm (feather 3 px).
  - Thêm nhãn box motorcycle tương ứng.
- Không dùng `copy_paste` có sẵn của Ultralytics, vì nó cần nhãn segmentation.

```yaml
moto_copy_paste:
  enabled: false
  p: 0.5
  n_max: 3
  min_y_frac: 0.5
  max_iou: 0.05
  feather_px: 3
  class_id: 3
```

Test: (1) mỗi box dán nằm trong ảnh và có nhãn đúng `class_id`; (2) IoU với box cũ ≤ `max_iou`; (3) khi `enabled: false`, ảnh và nhãn không đổi; (4) lưu 20 ảnh mẫu ra `research/figures/debug_copy_paste/` để kiểm tra bằng mắt.

### M3 — Kết hợp NWD vào bước gán nhãn TAL (ưu tiên 3)

- **Cơ sở:** Wang et al. 2021, "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection" (arXiv:2110.13389).
- Với hai box (cx, cy, w, h):

```
W2² = (cx1−cx2)² + (cy1−cy2)² + ((w1−w2)/2)² + ((h1−h2)/2)²
NWD = exp(−sqrt(W2²) / C)
```

- Trong `TaskAlignedAssigner`, thay độ đo trùng khớp dùng để tính `align_metric` bằng `α·IoU + (1−α)·NWD`. Bắt đầu với α = 0.5.
- C = trung bình `sqrt(w·h)` của box motorcycle ở độ phân giải huấn luyện (960 px), tính từ dữ liệu và ghi vào ADR. Không lấy hằng số từ bài báo.
- Không sửa loss khung trong bước này.

```yaml
nwd_assign:
  enabled: false
  alpha: 0.5
  C: null      # tính tự động từ dữ liệu nếu null, ghi log giá trị dùng
```

Test: NWD ∈ (0, 1]; NWD = 1 khi hai box trùng; khi `alpha: 1.0` kết quả gán nhãn bằng hệt Nền.

### M4 — Tầng phát hiện P2 (tùy chọn, CẦN NHÓM DUYỆT TRƯỚC)

- Thêm tầng phát hiện stride 4, dựa theo `yolov8-p2.yaml` của Ultralytics.
- **Hệ quả:** tầng mới phải khởi tạo ngẫu nhiên nên mất một phần lợi ích của head COCO; tính toán, bộ nhớ và chi phí truyền thông tăng.
- **Không triển khai nếu chưa có xác nhận của nhóm.**

### M5 — SAHI khi suy luận (tùy chọn, nhánh báo cáo riêng)

- Dùng thư viện `sahi` với mô hình cuối của run, ô 640×640, chồng 20%.
- **Chỉ đánh giá**, không đổi huấn luyện. Kết quả báo cáo trong bảng riêng, không trộn với kết quả đánh giá trên ảnh nguyên.

---

## 5. Phương pháp tùy chọn cho bus (CẦN NHÓM DUYỆT TRƯỚC)

Chỉ triển khai khi nhóm xác nhận.

- **B2 — Neo backbone về COCO (L2-SP)** (Li et al., ICML 2018): `L += β · Σ ||w − w_COCO||²` trên backbone và neck, với β ∈ {1e-4, 1e-3}, chỉ ở client có `missing_classes`. Có thể kết hợp bắt chước feature map của teacher trong vùng box xe (Wang et al., CVPR 2019).
- **B3 — Dán bus công khai từ COCO vào client thiếu bus** (Ghiasi et al., CVPR 2021). **Thay đổi giả định bài toán** (client dùng dữ liệu công khai), nên phải có xác nhận của giảng viên hướng dẫn trước.

---

## 6. Chỉ số cần thêm vào mỗi lần đánh giá

Ghi vào registry (trường `results`) cho **mọi** run của spec này, kể cả Nền:

| Chỉ số | Định nghĩa | Dùng cho |
|---|---|---|
| AP50 theo lớp, mAP50 | Như hiện tại | Tất cả |
| Phân rã kết cục bus thật (1.597 box) | Số bus thật: phát hiện đúng / nhầm thành truck / nhầm thành car / điểm bus thấp / không phát hiện, ở conf 0.25. Dùng lại script phân tích lỗi bus hiện có | B1 (kiểm cơ chế) |
| FP bus ở vùng nền | Dự đoán bus ở conf 0.25 không khớp box thật nào thuộc bất kỳ lớp mục tiêu nào | B1 (kiểm an toàn) |
| AP50 motorcycle theo kích thước | Chia box GT theo diện tích ở ảnh gốc: nhỏ < 32², vừa 32²–96², lớn > 96² | M1–M3 |
| Thời gian mỗi vòng, thời gian mỗi run, VRAM đỉnh | Log từ client và server | Bảng chi phí trong báo cáo |

Mọi hình phải sinh từ registry bằng script, kèm CSV chứa đúng số liệu đã vẽ.

---

## 7. Thứ tự thực hiện và ma trận thí nghiệm

Seed chính: 42, 123, 2024. Kịch bản: Lý tưởng (chia đều), Thiếu bus 3/5, Thiếu bus 4/5.

### Giai đoạn 0 — Chuẩn bị
1. Hoàn thành V1–V8 (Mục 1).
2. Thêm chỉ số ở Mục 6 vào bộ đánh giá (chỉ thêm, không sửa cách tính cũ).
3. Nếu V8 cho thấy thiếu: chạy **Nền** và **FedAvg** ở 960 px, dữ liệu `s5moto`, trên 3 kịch bản × 3 seed.

### Giai đoạn 1 — B1
| Run | Kịch bản | Seed | Mục đích |
|---|---|---|---|
| Smoke: Nền + B1, 2 vòng | Thiếu bus 3/5 | 42 | Kiểm code chạy, loss hợp lệ, teacher không đổi |
| Ablation: `scope: all_anchors` | Thiếu bus 3/5 | 42 | Chứng minh cần giới hạn vùng |
| Ablation: `q_max: 1.0` | Thiếu bus 3/5 | 42 | Vai trò của trần |
| Ablation: `lam: 0.5` | Thiếu bus 3/5 | 42 | Độ nhạy λ |
| **Chính: Nền + B1** (λ, q_max chốt trong ADR trước) | Cả 3 kịch bản | 42, 123, 2024 | Kiểm định |

### Giai đoạn 2 — M1 + M2
| Run | Kịch bản | Seed |
|---|---|---|
| Smoke M2 (ảnh debug) | Thiếu bus 3/5 | 42 |
| Ablation: chỉ M1; chỉ M2 | Thiếu bus 3/5 | 42 |
| **Chính: Nền + M1 + M2** | Lý tưởng, Thiếu bus 3/5 | 42, 123, 2024 |

### Giai đoạn 3 — M3
| Run | Kịch bản | Seed |
|---|---|---|
| Nền + M1 + M2 + M3 | Thiếu bus 3/5 | 42 → nếu đạt quy tắc thì chạy thêm 123, 2024 |

### Giai đoạn 4 — Tổ hợp cuối
- Nền + B1 + (M1 + M2 [+ M3] nếu đạt) trên 3 kịch bản × 3 seed.
- M4, M5, B2, B3: chỉ khi nhóm duyệt.

---

## 8. Quy tắc quyết định đề xuất (nhóm chốt trong ADR trước khi chạy)

**B1 đạt khi đủ cả 4 điều kiện:**
1. Mức suy giảm do thiếu lớp, tức `AP50_bus(lý tưởng) − AP50_bus(thiếu bus 3/5)` ghép cặp theo seed, **nhỏ hơn** của Nền ở **3/3 seed**.
2. Tổng số bus thật bị nhầm thành truck + car **giảm** so với Nền (trung bình 3 seed).
3. FP bus ở vùng nền tăng **không quá 20%** so với Nền.
4. mAP50 không giảm quá 0,01 so với Nền ở mọi seed.

**M1 + M2 đạt khi:**
1. AP50 motorcycle tăng **≥ 0,010** so với Nền ở **3/3 seed**.
2. AP50 motorcycle nhóm vật thể nhỏ tăng (trung bình 3 seed).
3. AP50 car, bus, truck giảm không quá 0,01.

**M3 đạt khi:** điều kiện như M1 + M2, so với Nền + M1 + M2.

Không đạt thì vẫn ghi kết quả vào registry, kèm ADR kết luận "bác bỏ".

---

## 9. Đầu ra mong đợi

1. Code các thành phần (mặc định tắt) cùng unit test, mỗi thành phần một commit.
2. ADR mới cho B1, M1+M2, M3 (giả thuyết, cấu hình, quy tắc quyết định, kết luận).
3. Các run trong `registry.yaml`, có đủ chỉ số ở Mục 6.
4. File `research/notes/spec_verify_YYYYMMDD.md` (kết quả V1–V8).
5. Hình, sinh từ registry, có CSV đi kèm:
   - Mức suy giảm do thiếu lớp: Nền so với Nền + B1, từng seed.
   - Phân rã kết cục bus thật: Nền so với Nền + B1.
   - AP50 motorcycle theo kích thước: Nền so với Nền + M1 + M2.
6. Bảng tóm tắt cuối: phương pháp | kịch bản | AP50 bus | AP50 motorcycle | mAP50 | mức suy giảm do thiếu lớp | đạt/không đạt theo ADR.

---

## 10. Tài liệu tham khảo (đối chiếu và thêm vào `literature_registry.yaml` trước khi trích dẫn)

- Lee et al., "Preservation of the Global Knowledge by Not-True Distillation in Federated Learning" (FedNTD), NeurIPS 2022.
- Shmelkov, Schmid, Alahari, "Incremental Learning of Object Detectors without Catastrophic Forgetting", ICCV 2017.
- Kisantal et al., "Augmentation for small object detection", arXiv:1902.07296, 2019.
- Ghiasi et al., "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation", CVPR 2021.
- Wang et al., "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection", arXiv:2110.13389, 2021.
- Akyon, Altinuc, Temizel, "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection", ICIP 2022.
- Li, Grandvalet, Davoine, "Explicit Inductive Bias for Transfer Learning with Convolutional Networks" (L2-SP), ICML 2018.
- Wang et al., "Distilling Object Detectors with Fine-grained Feature Imitation", CVPR 2019.
- Gupta, Dollár, Girshick, "LVIS: A Dataset for Large Vocabulary Instance Segmentation" (repeat factor sampling), CVPR 2019.