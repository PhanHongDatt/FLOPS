# Giải thích ngắn các hình (báo cáo 18/11/2026)

Ours = phương pháp chính (giữ head COCO + chưng cất KD từ mô hình COCO + LR cosine + lặp lại ảnh lớp hiếm RFS).
FedAvg = mốc so sánh. Số liệu YOLO11m lấy trung bình vòng 10 và 20; FedAvg 3 seed, Ours 2 seed.

## Hình 1 – Phân phối dữ liệu
- Hai kịch bản dùng cùng 15.000 ảnh, 5 client × 3.000 ảnh; tổng số hộp mỗi lớp như nhau.
- Lý tưởng: mỗi client giữ khoảng 20 % mỗi lớp. Thiếu lớp: client 0–2 không có bus, client 3 không có truck.
- Vì dữ liệu tổng giống hệt, mọi chênh lệch giữa hai kịch bản là do thiếu lớp.

## Hình 2 – Kết quả chính (thiếu lớp)
- Ours hơn FedAvg ở mọi lớp: mAP50 0,527 → 0,643; bus 0,492 → 0,602; truck 0,513 → 0,622; moto 0,321 → 0,525.
- Moto tăng nhiều nhất (+0,20). Kết quả ổn định giữa các seed (lệch < 0,004).
- Car đã vượt mục tiêu 0,70; bus, truck, moto còn thiếu 0,08–0,18.

## Hình 3 – Mức tụt do thiếu lớp
- Mức tụt = AP lý tưởng − AP thiếu lớp. Ours gần như xóa mức tụt ở car, truck (0,010) và moto (0,004).
- Recall bus: FedAvg tụt từ 0,45 xuống 0,18 khi thiếu lớp; Ours chỉ tụt từ 0,55 xuống 0,46.
- Riêng AP50 bus của Ours vẫn tụt 0,038: đây là phần chưa giải được.

## Hình 4 – Đổi mô hình YOLO11s → YOLO11m (cùng dữ liệu, 960 px)
- Mô hình lớn hơn tăng mọi lớp khoảng +0,02 đến +0,05 (mAP50 0,61 → 0,64).
- Huấn luyện FL bằng Ours mang lại nhiều hơn đổi mô hình: khoảng +0,15 mAP50 so với mô hình COCO ban đầu.

## Hình 5 – Mức tụt theo mô hình
- Mức tụt bus của Ours giữ nguyên 0,038 trên cả YOLO11s và YOLO11m: mô hình lớn hơn không giải quyết thiếu lớp.
- Ours giảm mức tụt truck và moto so với FedAvg. Mức tụt bus của FedAvg nhỏ hơn (0,017) chỉ vì AP bus của FedAvg vốn thấp.

## Hình 6 – Chẩn đoán nguyên nhân
- Khi thiếu lớp, điểm tin cậy bus của FedAvg sụt mạnh (0,076, lý tưởng 0,290). KD trong Ours giữ điểm này ở mức
  mô hình COCO ban đầu (0,267), nên Ours ít bỏ sót bus hơn.
- Chỉnh riêng phần phân loại không đủ: thay hàng bus còn 76–86 % mức tụt; huấn luyện lại toàn bộ nhánh phân loại
  trên 15.000 ảnh có nhãn vẫn còn 64–66 % (ngưỡng đạt là 50 %).
- Kết luận: nguyên nhân không chỉ nằm ở head; hướng tiếp theo phải tác động lên cả mô hình.

## Hình 7 – Tiến độ tới mục tiêu 0,70
- Tăng độ phân giải 640 → 960 px và đổi sang YOLO11m đưa mAP50 của Ours từ 0,545 lên 0,643.
- Còn cách mục tiêu khoảng 0,06; moto (0,525) là lớp yếu nhất.
- Bước 640 px dùng cách chia dữ liệu khác, chỉ đọc xu hướng.

## Hình 8 – Precision và recall
- Thiếu lớp làm bus bị **bỏ sót**, không bị báo nhầm: FedAvg có precision bus 0,90 nhưng recall chỉ 0,18.
- Ours cân bằng lại: recall bus 0,46, precision 0,76. Moto: precision tăng mạnh 0,29 → 0,62. Truck: recall 0,40 → 0,58.
- Mỗi mô hình tự chọn ngưỡng tin cậy (F1 trung bình cao nhất), nên precision và recall có thể đổi chiều nhau; so sánh
  chất lượng tổng thể vẫn dùng AP50.

## Hình 9 – Lý tưởng (đủ lớp) so với thiếu lớp, AP50 tuyệt đối
- Kịch bản lý tưởng là kịch bản chuẩn để so: cùng 15.000 ảnh, nhưng client nào cũng có đủ 4 lớp.
- FedAvg: mAP50 0,543 → 0,527; bus 0,509 → 0,492; truck 0,532 → 0,513; moto 0,343 → 0,321.
- Ours: mAP50 0,656 → 0,643; bus 0,640 → 0,602; truck 0,633 → 0,622; moto 0,529 → 0,525; car gần như không đổi.
- Ours ở kịch bản thiếu lớp vẫn cao hơn FedAvg ở kịch bản đủ lớp khoảng 0,10 mAP50 (0,643 so với 0,543).
- Bus là lớp tụt nhiều nhất ở Ours (0,038); car hầu như không bị ảnh hưởng vì mọi client đều có car.

## Hình 10 – Phát hiện đúng (TP), báo nhầm (FP), bỏ sót (FN)
- Đếm ở ngưỡng tin cậy 0,25, trung bình vòng 10 và 20. Một bus bị đoán thành truck tính là bỏ sót của bus và
  báo nhầm của truck.
- Bus (1.597 xe): FedAvg thiếu lớp chỉ tìm đúng 226 xe, bỏ sót 1.371 (86 %). Ours thiếu lớp tìm đúng 606, bỏ sót 992.
  So với đủ lớp, Ours vẫn bỏ sót thêm khoảng 280 xe bus: đây là phần mức tụt còn lại.
- Car: Ours giảm báo nhầm từ khoảng 28.800 xuống 16.400 (−43 %), đổi lại bỏ sót tăng nhẹ.
- Motorcycle: Ours giảm báo nhầm từ 740 (FedAvg đủ lớp) xuống khoảng 210, và tìm đúng nhiều hơn.
- Truck: Ours tìm đúng nhiều hơn (khoảng 2.400 so với 1.400–1.900) nhưng báo nhầm cũng nhiều hơn (khoảng 2.700);
  chưa tách được bao nhiêu trong số đó là bus hoặc car bị đoán thành truck.

## Hình 11 – So sánh với các baseline: AP50
- Cùng kịch bản thiếu lớp, cùng 15.000 ảnh, cùng YOLO11m 960 px và 20 vòng; mọi phương pháp dùng seed 42 nên so
  được trực tiếp. Mỗi baseline mới có 1 seed nên đây là so sánh mô tả, chưa kiểm định thống kê.
- Baseline:
  - FedAvg theo số box: gộp mô hình theo số box từng lớp thay vì số ảnh.
  - FedProx (Li et al., 2020): thêm số hạng kéo mô hình client về gần mô hình toàn cục.
  - FedNTD (Lee et al., NeurIPS 2022): phương pháp gốc mà Ours cải biên, KD lấy "thầy" là mô hình toàn cục vòng trước.
- mAP50: FedAvg 0,524 · theo số box 0,528 · FedProx 0,531 · FedNTD 0,547 · **Ours 0,642**.
- Ours cao nhất ở cả 4 lớp. Hơn baseline tốt nhất (FedNTD) +0,095 mAP50; bus +0,106, truck +0,118, moto +0,121.
- FedAvg theo số box và FedProx gần như không khác FedAvg (≤ +0,02 ở mọi lớp): đổi cách gộp hay kéo mô hình về gần
  nhau không xử lý được thiếu lớp.
- FedNTD giúp moto rõ nhất (+0,08), nhưng bus và truck gần như không đổi. Ours đổi "thầy" thành mô hình COCO cố định
  và thêm LR cosine và RFS, nên tăng thêm ở mọi lớp.
- Tập trung (một máy giữ đủ 15.000 ảnh, huấn luyện như FedAvg) lại thấp nhất (0,459), thấp hơn cả mô hình COCO ban đầu
  (0,486). Huấn luyện thường 20 epoch làm quên kiến thức COCO, rõ nhất ở moto (0,38 → 0,22). Vì vậy cột này
  **không phải mốc trần** của Ours. Mốc trần đúng là huấn luyện tập trung bằng chính công thức Ours: trên YOLO11s
  (ADR-026), Ours FL chỉ kém mốc đó 0,006 mAP50.

## Hình 12 – So sánh với các baseline: precision và recall
- Bus: FedAvg, theo số box và FedProx có precision cao (0,85–0,90) nhưng recall rất thấp (0,18–0,25), nghĩa là gần như
  không dám đoán bus. FedNTD có recall 0,35, Ours 0,47 – cao nhất trong các phương pháp FL.
- Moto: Ours có cả precision (0,63) lẫn recall (0,53) cao nhất.
- Truck: Ours có recall cao nhất (0,59), precision ở mức trung bình (0,61).
- Car: Ours có precision cao nhất (0,87), recall thấp hơn một chút (0,70 so với 0,74–0,79).

## Hình 13 – So sánh với các baseline: TP, FP, FN
- Bus (1.597 xe thật): FedAvg và FedProx chỉ tìm đúng khoảng 170–190 xe; FedNTD khoảng 310; Ours khoảng 610.
  Ours bỏ sót ít nhất trong các phương pháp FL (990 so với 1.290–1.430).
- Car: Ours báo nhầm ít nhất (16.000 so với 24.900–26.600 ở các baseline).
- Moto: Ours tìm đúng nhiều nhất (256) và báo nhầm ít nhất (202).
- Truck: Ours tìm đúng nhiều nhất (2.388) nhưng cũng báo nhầm nhiều nhất (2.707). Đây là điểm cần chú ý khi trình bày.

## Tóm lại
1. Ours là cấu hình FL tốt nhất hiện có: +0,116 mAP50 so với FedAvg, ổn định giữa các seed. So với 4 baseline (FedAvg,
   FedAvg theo số box, FedProx, FedNTD), Ours cao nhất ở mọi lớp (Hình 11).
2. Ours giảm bỏ sót bus và xóa gần hết mức tụt ở truck và moto, nhưng còn tụt 0,038 AP50 ở bus.
3. Đổi mô hình hay chỉnh riêng phần phân loại đều không đóng được mức tụt này; đây là hướng nghiên cứu tiếp theo.

---

# Mô tả phương pháp Ours

Ours dựa trên FedAvg: không đổi kiến trúc YOLO11m và không đổi cách server gộp mô hình. Ours chỉ thay đổi
**cách khởi tạo** và **cách mỗi client huấn luyện cục bộ**, gồm 4 thành phần.

## Quy trình một vòng
1. Server gửi mô hình toàn cục cho 5 client.
2. Mỗi client huấn luyện 1 epoch trên dữ liệu riêng của mình. Trong lúc này, client áp dụng thành phần ② (KD),
   ③ (learning rate theo vòng) và ④ (RFS).
3. Client gửi trọng số về server. Server lấy trung bình có trọng số theo số ảnh (FedAvg).
4. Lặp lại 20 vòng. Mô hình được đánh giá ở vòng 10 và 20, kết quả báo cáo là trung bình của hai vòng.

## Bốn thành phần

**① Khởi tạo từ YOLO11m đã huấn luyện trên COCO và giữ nguyên head 80 lớp**
- Không tạo head 4 lớp mới. Bốn lớp mục tiêu dùng đúng đầu ra COCO tương ứng: car, bus, truck, motorcycle.
- Lý do: mô hình nhận ra bus ngay từ vòng 0 (AP50 bus 0,48 khi chưa huấn luyện). Client thiếu bus vì vậy không
  phải học bus từ đầu.

**② Chưng cất tri thức (KD) từ mô hình COCO ban đầu**
- Mô hình COCO ban đầu được giữ đóng băng làm "mô hình thầy". Ở mỗi vị trí dự đoán và với mỗi lớp **không phải
  nhãn đúng** tại vị trí đó, loss thêm một số hạng giữ điểm tin cậy của mô hình học gần với điểm của mô hình thầy.
  Hai hệ số đều bằng 1 (trọng số β = 1, nhiệt độ τ = 1).
- Lý do: client không có bus luôn coi bus là "không có" ở mọi vị trí, nên đẩy điểm bus xuống. KD giữ điểm bus
  ở mức của mô hình COCO thay vì để nó bị kéo về 0. Hình 6 cho thấy: không có KD, điểm bus rơi xuống 0,076;
  có KD, điểm bus giữ ở 0,267.
- Căn cứ: FedNTD (Lee và cộng sự, NeurIPS 2022), chưng cất trên các lớp không phải nhãn đúng.
- Hai điểm điều chỉnh của đề tài: (a) chuyển từ softmax sang sigmoid, áp dụng riêng cho từng lớp tại từng vị trí;
  (b) mô hình thầy là mô hình COCO cố định, không phải mô hình toàn cục của vòng trước như trong bài gốc.

**③ Learning rate giảm theo đường cosine qua các vòng**
- Learning rate của client giảm dần từ 0,01 (vòng 1) xuống 0,0005 (vòng 20), thay vì giữ cố định.
- Lý do: các vòng cuối cập nhật nhỏ hơn, nên mô hình của các client ít kéo lệch nhau khi gộp và mô hình hội tụ ổn định hơn.

**④ Lặp lại ảnh chứa lớp hiếm (RFS) bên trong mỗi client**
- Mỗi client tự tính tỷ lệ ảnh có chứa mỗi lớp trong dữ liệu của mình. Ảnh chứa lớp hiếm (bus, moto, truck)
  được lặp lại nhiều lần hơn trong mỗi epoch. Hệ số lặp của lớp c là max(1, √(1 / tỷ lệ ảnh có lớp c)).
  Mỗi client tăng từ 3.000 lên khoảng 4.300–4.600 lượt ảnh mỗi vòng.
- Không chia sẻ dữ liệu hay thống kê nào ra ngoài client. Nhãn và loss giữ nguyên; chỉ tần suất xuất hiện của
  ảnh thay đổi.
- Lý do: car chiếm khoảng 93 % số hộp, nên lớp hiếm ít được học. Moto là lớp hưởng lợi nhiều nhất.
- Căn cứ: Repeat Factor Sampling của bộ dữ liệu LVIS (Gupta và cộng sự, CVPR 2019).

## Thông số huấn luyện

| Thông số | Giá trị |
|---|---|
| Mô hình | YOLO11m, khởi tạo từ COCO, head 80 lớp |
| Ảnh đầu vào | 960 px |
| Số client / số vòng / epoch mỗi vòng | 5 / 20 / 1 |
| Bộ tối ưu | SGD, momentum 0,937, weight decay 0,0005, batch 8 |
| Learning rate | cosine theo vòng, 0,01 → 0,0005 |
| KD | mô hình thầy COCO cố định, β = 1, τ = 1 |
| RFS | ngưỡng t = 1,0, tính riêng trong mỗi client |
| Gộp mô hình | FedAvg, trọng số theo số ảnh |
| Đánh giá | 10.000 ảnh val BDD100K, trung bình vòng 10 và 20 |
| Phần mềm | Ultralytics 8.3.253, PyTorch 2.7.1, Flower 1.21.0, GPU Kaggle T4 × 2 |

## Đóng góp của từng thành phần
Đo trên YOLO11s, 640 px, kịch bản thiếu lớp; xem `giai_doan_truoc/fig4`.
- Thêm KD vào FedAvg: mAP50 0,434 → 0,466.
- Thêm learning rate cosine: → 0,493.
- Kết hợp cả bốn thành phần (Ours): 0,545; moto 0,223 → 0,423.
- Khi đã có learning rate cosine và RFS, thêm KD chỉ tăng mAP50 0,002 nhưng tăng moto 0,027. Hạ ngưỡng RFS xuống
  t = 1,0 tăng thêm mAP50 0,012 và moto 0,030.

## Giới hạn
- Ours là một công thức huấn luyện FL tốt hơn nói chung: nó tăng AP ở cả kịch bản lý tưởng. Ours không phải
  phương pháp chữa riêng cho hiện tượng thiếu lớp, vì AP50 bus vẫn tụt 0,038.
- Mỗi client phải chạy thêm mô hình thầy, nên thời gian huấn luyện khoảng 10 giờ mỗi lần chạy, so với khoảng 5–6 giờ
  của FedAvg.
