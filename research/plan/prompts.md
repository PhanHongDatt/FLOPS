# Prompt phụ trợ cho workstream Missing-Class Non-IID

Tách khỏi `plan.md` v2 (kế hoạch chỉ chứa quyết định và thiết kế). Bản gốc của hai prompt này nằm trong `plan_v1_archive.md` §1.

---

## P1 — Khảo sát literature, đầu ra **nạp trực tiếp vào registry**

Khác bản v1: đầu ra phải đúng schema `research/evidence/literature_registry.yaml` để không phát sinh bước chuyển thủ công, và danh sách bài cần tìm gắn thẳng vào 7 mục còn mở ở `plan.md` §11.1.

```text
VAI TRÒ: Trợ lý nghiên cứu về Federated Learning cho object detection.

BỐI CẢNH:
- Khóa luận: FL giám sát giao thông, YOLOv8 (cls loss = BCE sigmoid từng lớp,
  box = CIoU + DFL), BDD100K, 4 lớp: car, bus, truck, motorcycle.
- Bài toán: Missing-Class Non-IID — một số client không có ảnh lẫn bbox của một số lớp.
- Lớp vacant chính đã chốt: bus (13.0% ảnh train, 11.672 bbox). Thứ cấp: truck.
- Phương pháp dự kiến: (a) client: hệ số rho ∈ {0, 0.25, 1} điều chỉnh đóng góp
  loss của kênh lớp vacant (loss-level); (b) server: tổng hợp tham số lớp cuối
  nhánh cls theo từng lớp với tập eligible + luật giữ nguyên global khi không có
  client đóng góp; backbone/neck dùng FedAvg.
- Giao thức: 4 client, 10 round, 1 local epoch, fraction_fit = 1.0, YOLOv8n, imgsz 640.

RÀNG BUỘC BẮT BUỘC:
1. Chỉ trích bài đã đọc được nội dung (tối thiểu abstract); ghi rõ mức đã đọc:
   "toàn văn" | "chỉ abstract" | "chỉ snippet".
2. Venue từ hạng C trở lên (CORE/ICORE) hoặc journal Q1–Q2. Preprint chỉ làm tham
   khảo phụ và phải ghi rõ là preprint.
3. KHÔNG tự suy ra số liệu, năm, trang, hạng venue. Không chắc → ghi "CHƯA XÁC MINH".
4. KHÔNG so sánh số giữa các bài nếu giao thức khác nhau.
5. KHÔNG dùng blog, bản tóm tắt do AI sinh, snippet GitHub ngẫu nhiên, StackOverflow
   làm bằng chứng phương pháp.

NHIỆM VỤ:
A. Tìm 8–12 công trình về: (i) label skew / vacant class / missing class;
   (ii) federated object detection có heterogeneity; (iii) class-wise hoặc
   classifier-head aggregation; (iv) forgetting khi local update.
   Ưu tiên xác minh đúng 7 bài đang nợ: FedPylot, FedVLS, FedLC, FedRS, FedNTD,
   NIID-Bench, FedProx+LA (arXiv 2405.01108).
B. Mỗi bài trả lời 5 câu: (1) giả định bài toán; (2) cơ chế cốt lõi; (3) độ tin cậy
   thực nghiệm (số client, số seed, báo vòng cuối hay max, cùng giao thức hay không);
   (4) hạn chế; (5) chuyển sang detector sigmoid/BCE được không, cần sửa gì.
C. RỦI RO TÍNH MỚI: có công trình nào đã làm ĐỒNG THỜI client-side vacant-class
   protection VÀ class-wise aggregation cho object detection chưa? Nếu có, nêu
   khác biệt cụ thể.
D. Câu hỏi riêng cho thiết kế hiện tại: có bài nào cho thấy class-count weighted
   aggregation và per-class eligibility filtering KHÁC nhau khi mọi lớp đều có ít
   nhất một client nắm giữ? (liên quan plan.md §6.4)

ĐỊNH DẠNG ĐẦU RA:
1. Khối YAML đúng schema registry cho từng bài:
     - id: <slug>
       title: ""
       authors: []
       year:
       venue: ""
       source: ""            # URL trang proceedings hoặc arXiv abs
       read_level: ""        # toàn văn | chỉ abstract | chỉ snippet
       reused_concept: ""
       relation: ""          # reproduction | adaptation | extension
       evidence_tag: ""      # [LITERATURE] | [NEEDS-VERIFICATION]
2. Bảng tóm tắt: tên | venue | năm | mức đã đọc | cơ chế | hạn chế | chuyển BCE được?
3. Mục C, D trả lời riêng.
4. Cuối cùng: danh sách CHƯA XÁC MINH.
```

---

## P2 — Prompt viết code (ràng buộc theo repo, không theo cây thư mục tưởng tượng)

Khác bản v1: v1 yêu cầu tuân theo cây `fl/` không tồn tại và nói "không dùng Flower ở giai đoạn này" — trái `ADR-002`.

```text
Viết code cho repo D:\FLOPS, chạy được trên Kaggle (Python 3.11, PyTorch 2.7.1,
ultralytics 8.3.253, flwr 1.21.0).

RÀNG BUỘC:
- Đọc CLAUDE.md, ADR-001, ADR-002, research/gates.yaml, research/plan/plan.md TRƯỚC khi sửa.
- Chế độ FL: Flower simulation mode (fl.simulation.start_simulation) — ADR-002. KHÔNG
  viết vòng lặp FL thứ hai, KHÔNG tạo thư mục fl/.
- Dùng đúng các file ở bảng map plan.md §6.6. File mới duy nhất được phép tạo trong
  giai đoạn này: src/preservation/rho_loss.py (và test của nó).
- KHÔNG đổi public contract của module khác mà không có ADR (CLAUDE.md §17).
- Siêu tham số lấy từ configs/base_config.yaml; KHÔNG hard-code số rải rác.
- KHÔNG dùng API không có trong ultralytics 8.3.253; in version ở đầu notebook.
- Mọi bước resumable: checkpoint global + log mỗi round vào /kaggle/working.
- Mỗi round ghi CSV: round, client_id, n_img, n_box_per_class, local_time_s, bytes_up;
  và global: mAP50, mAP50-95, AP/lớp, FP/lớp, FN/lớp.
- Mỗi hàm mới kèm test chạy được trên CPU với dữ liệu giả (CLAUDE.md §18).
- Nêu rõ phần nào chưa kiểm thử được, gắn [NEEDS-VERIFICATION].

THỨ TỰ ƯU TIÊN (không làm bước sau trước bước trước):
1. R1/R2/R3 ở plan.md §6.5 (3 lỗi chặn ablation).
2. F1 runtime verify parameter map.
3. Partition: luật "ít đầy nhất" §5.4 + matched-pair §5.3.
4. Chỉ sau khi 1–3 xanh: src/preservation/rho_loss.py (A2b).
```
