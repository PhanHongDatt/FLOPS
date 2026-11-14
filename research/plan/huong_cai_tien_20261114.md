# Tổng hợp hướng cải tiến – FLOPS / FedTraff-KP

Cập nhật: 14/11/2026 (đã đối chiếu số liệu trong repo cùng ngày). Văn bản này thay cho các danh sách hướng rời rạc
trước đó. Spec triển khai chi tiết cho lớp motorcycle nằm trong `claude/SPEC_phuong_phap_cai_tien.md` (ngoài repo).

---

## 1. Hai mục tiêu cần tách riêng

| | Mục tiêu nghiên cứu | Mục tiêu kỹ thuật |
|---|---|---|
| Nội dung | Thu hẹp **mức suy giảm do thiếu lớp** của lớp bus | Đạt AP khoảng 0,70 |
| Con số hiện tại | ≈ 0,036 AP50 bus (YOLO11m, 960px, **1 seed**) | mAP50 0,642 (11m), 0,610 (11s) |
| Giới hạn | Đóng hết mức suy giảm thì bus chỉ đạt ≈ 0,638 (bằng trường hợp lý tưởng) | Phần lớn khoảng cách tới 0,70 **không** do thiếu lớp |

Hệ quả: một phương pháp có thể đóng góp cho luận văn mà không đưa AP lên 0,70, và ngược lại. Báo cáo phải nói rõ
mỗi thay đổi nhắm vào mục tiêu nào.

## 2. Những điều đã biết (làm căn cứ)

- Các phương pháp nhắm riêng vào lớp thiếu (11 lần thử, gồm chưng cất not-true, focal/EFL, CCVR, đóng băng head,
  chưng cất trên anchor dễ nhầm, tăng tần suất tổng hợp) **không** thu hẹp mức suy giảm.
- Công thức huấn luyện (LR cosine, RFS, độ phân giải 960px) cải thiện **mọi lớp** nhưng không thu hẹp mức suy giảm.
- FL chỉ kém huấn luyện tập trung khoảng 0,006 mAP50.
- Kết quả dương duy nhất cho bus là teacher học trên **dữ liệu có nhãn ở server**. Kết quả này vượt ngoài giả định
  FL hiện tại.
- Phân rã lỗi bus (D1, YOLO11s P4noKD vòng 20, thiếu lớp so với lý tưởng): số bus bị nhầm sang car/truck tăng
  thêm **164**, số bus bị bỏ sót hoàn toàn tăng thêm **140**.
- **Chưa biết** suy giảm nằm ở đặc trưng hay ở head. CCVR thất bại do cách cài đặt, và đóng băng head chỉ đóng băng
  conv cuối, nên cả hai **không** phải bằng chứng cho câu hỏi này.

## 3. Lộ trình đề xuất

### Bước 0 – Chẩn đoán vị trí suy giảm (ADR-036, quy tắc đã chốt)

**Mục đích:** xác định suy giảm nằm ở đặc trưng, ở thân head, ở phần đọc ra cuối, hay không nằm ở phân loại.
Kết quả nào cũng đưa thẳng được vào luận văn và là bằng chứng mà cổng G5 đang thiếu.

**Thiết kế** (chi tiết và quy tắc quyết định: `research/decisions/ADR-036-locate-missing-class-degradation.md`):
- **Checkpoint:** COCO ban đầu, S26 (thiếu lớp), S27 (lý tưởng) trên YOLO11m; lặp lại trên cặp YOLO11s (S21b/S28).
- **Lấy đặc trưng** tại ô ở tâm box ground-truth của bus/truck/car trên **cả ba tầng P3/P4/P5**, báo riêng từng
  tầng. Tầng chính chốt theo kích thước box (cạnh căn bậc hai diện tích ở 960px: < 64 → P3, < 192 → P4, còn lại
  P5). Không dùng ROI-align để tránh thêm tham số tự chọn.
- **Ba vị trí đo:** đầu ra neck và `cv3[l][1]` đo bằng linear probe; tầng cuối `cv3[l][2]` không train probe, đo
  điểm sigmoid bus của chính mô hình và tỷ lệ bus thắng car/truck theo logit gốc.
- **Probe:** hồi quy logistic, dữ liệu train cân bằng ba lớp, chia nửa tập val theo ảnh; báo ma trận nhầm lẫn 3×3,
  macro-accuracy, recall bus; bootstrap theo ảnh (1.000 lần, ghép cặp).
- **Ngưỡng 0,03** (đơn vị recall của probe, không so với 0,036 AP50): bằng khoảng nửa độ rộng khoảng tin cậy 95 %
  với khoảng 800 box bus ở nửa đánh giá, cộng thêm điều kiện khoảng tin cậy không chứa 0, và phải lặp lại trên
  cả 11m và 11s.

**Giới hạn:** mỗi kịch bản có 1 seed, nên bootstrap theo ảnh không bao được nhiễu giữa các lần train.

**Kiểm tra trước khi chạy: đã đạt (14/11).**
- S26/S27: config giống hệt, chỉ khác partition; hai commit không khác dòng nào trong `src/`.
- S21b/S28: config giống hệt. Code khác 545 dòng nhưng chỉ là tính năng mới đều tắt bằng cờ (B1, M1–M3). Smoke S24
  (ADR-028) cho loss vòng 1 của client C4 trùng S21b đến từng chữ số.
- Cả bốn run có checkpoint vòng 20.

**Chi phí:** khoảng 1–2 giờ GPU Kaggle để trích đặc trưng; probe chạy trên CPU.

### Bước 1 – Hướng B: chưng cất theo anchor ở server bằng ảnh không nhãn (cần thầy duyệt)

**Ý tưởng:** sau mỗi vòng, server chưng cất vào mô hình toàn cục bằng bản đồ điểm theo từng anchor, gộp **có nhận
biết lớp**:
- kênh bus: trung bình sigmoid của các client **có** bus;
- kênh car/truck/…: trung bình từ các client có lớp đó;
- nhánh hồi quy box: lấy từ mô hình toàn cục, hoặc trung bình mọi client.

Các mô hình cùng kiến trúc và cùng độ phân giải nên anchor khớp 1–1, không cần gộp box (WBF).

**Dữ liệu không nhãn (đã xác minh 14/11):** 54.863 ảnh BDD100K train chưa client nào dùng (tổng 69.863; partition
dùng 15.000, bản pooled dùng đúng cùng 15.000 ảnh). Trong phần này có **9.015** box bus (6.931 ảnh), gấp khoảng
3,4 lần tổng số bus của mọi client (2.657), và 23.521 box truck.
- Về dữ liệu, tín hiệu bus không yếu.
- **Gây nhiễu mới:** kho ảnh ở server giàu bus hơn client. **Bắt buộc có nhánh đối chứng:** cùng kho ảnh, cùng vòng
  chưng cất, nhưng lấy trung bình mọi client cho mọi lớp. Chỉ phần chênh so với nhánh này mới được quy cho quy tắc
  gộp nhận biết lớp (H3).
- Đã đối chiếu tên ảnh: không trùng ảnh nào với tập val (0 trên 10.000).

**Phần việc chính:** viết vòng train ở server cho ảnh không nhãn (trainer Ultralytics cần nhãn), gồm nạp ảnh, chạy
các mô hình client, tính loss chưng cất và cập nhật. Hàm loss chưng cất có thể dùng lại; vòng train là phần mới.
Phải chọn rõ cách xử lý BatchNorm (đóng băng hay cập nhật trên dữ liệu server).

**Kỳ vọng và giới hạn:**
- Hướng này **không tương đương** teacher có nhãn: kiến thức bus đến từ mô hình của 2 client có bus, nên đừng kỳ vọng
  mức cải thiện như kết quả dùng dữ liệu có nhãn.
- Ảnh không nhãn cùng miền với tập đánh giá, nên kết quả có thể lạc quan. Nếu có thể, thêm một bản dùng ảnh giao
  thông từ bộ dữ liệu khác.
- Chi phí: server chạy suy luận 5 mô hình mỗi vòng; cần giới hạn số ảnh mỗi vòng.
- Giả định "server có dữ liệu không nhãn" **phải được thầy duyệt** và đánh dấu là phương pháp ngoài đề cương. Có thể
  chuẩn bị code song song, nhưng chưa chạy.

### Bước 2 – Rẽ nhánh theo kết quả chẩn đoán

- **Hướng A (chỉ khi đặc trưng bị suy giảm):** ràng buộc ở mức đặc trưng giữa các client, ví dụ prototype hoặc học
  tương phản theo lớp. Hướng này chỉ nhắm phần bus bị **nhầm**, không chắc giúp phần bị **bỏ sót**, và có thể làm
  điểm bus thấp thêm; phải theo dõi recall bus.
- **Hiệu chỉnh head (khi thân head hoặc phần đọc ra cuối bị lệch):** hiệu chỉnh bias hoặc logit riêng cho lớp bus ở
  server, hoặc làm lại CCVR đúng cách (giữ bias, chỉ hiệu chỉnh trọng số, tập ảo giữ đúng tỷ lệ mẫu nền). **Không
  dùng FedLC** (mục 4). Phải đối chiếu với các lần thử trước để không lặp lại cấu hình đã thất bại.
- **Không khác biệt ở phân loại:** phân tích lỗi định vị (IoU của bus phát hiện được) và ảnh hưởng của ngưỡng
  confidence.

### Song song – Mục tiêu kỹ thuật và lớp motorcycle

Các thay đổi này **không** nhằm thu hẹp mức suy giảm. Báo cáo phải ghi rõ chúng phục vụ mục tiêu kỹ thuật.
- Tắt mosaic và giảm scale augmentation ở các vòng cuối.
- Copy-paste box trong phạm vi từng client.
- NWD trộn vào TAL cho vật thể nhỏ.
- Tùy chọn (tốn tài nguyên): head P2, suy luận SAHI.
- Lưu ý từ smoke M1–M3 (S24, ADR-029): số phát hiện sai tăng mạnh và crop bị dán lên nắp capo. Trước khi chạy cần
  **sửa vùng dán**, và chạy **từng phương pháp riêng** (M1, M2, M3) để quy được nguyên nhân.

## 4. Đã bỏ

- **Hướng C (phạt logit car/truck tại anchor bus):** BCE vốn đã làm việc này, và cách phạt này không tác động lên
  client thiếu bus.
- **FedLC:** thiết kế cho softmax, mất tác dụng khi lớp vắng hoàn toàn.
- **Lặp lại các phương pháp nhắm lớp thiếu đã thất bại** mà không có giả thuyết mới từ bước chẩn đoán.

## 5. Cần thầy quyết định

1. Có chấp nhận giả định server có ảnh không nhãn cùng miền (Hướng B) không?
2. Luận văn ưu tiên mục tiêu nào: thu hẹp mức suy giảm (nghiên cứu) hay đạt 0,70 (kỹ thuật)?
3. Có duyệt ngân sách GPU cho ma trận 3 seed của ADR-033 (A0c và P4t1, hai kịch bản, khoảng 136 GPU-giờ, khoảng
   2,5 tuần trên hai tài khoản) không? Hiện mức suy giảm 0,036 mới có 1 seed.

## 6. Còn chưa xác minh

- Mức suy giảm 0,036 mới có 1 seed.
- FedREPA: chưa đọc bản đầy đủ. FedVLS, FedMR, FedSeg và FedDF cần đọc lại phần phương pháp trước khi cài đặt.

**Đã xác minh (14/11):**
- Precision/recall và TP/FP trong registry **không mâu thuẫn**: hai điểm vận hành khác nhau. TP/FP/FN lấy từ ma
  trận nhầm lẫn ở ngưỡng confidence 0,25 (`src/model/yolo_wrapper.py`, `_per_class_fp_fn`); precision/recall ở
  ngưỡng cho F1 cao nhất. Không trộn hai loại số trong cùng một phép tính. Trong ma trận nhầm lẫn, một bus bị đoán
  thành truck được đếm cả vào FN_bus lẫn FP_truck.
- Số ảnh train chưa dùng, số bus trong đó, và việc không trùng tập val: xem Bước 1.
- Hai cặp run của Bước 0 chỉ khác nhau ở cách chia dữ liệu: xem Bước 0.
