# Kế hoạch thực thi v2 — FedTraff-KP (Missing-Class Non-IID, YOLOv8 + Flower, Kaggle)

**Phiên bản:** v2 — 2026-10-01
**Thay thế:** `plan_v1_archive.md` (giữ nguyên văn, không xóa — CLAUDE.md §16)
**Phạm vi:** phần Missing-Class Non-IID cho YOLOv8 trong khóa luận "FLOps cho Federated Learning giám sát giao thông".

## 0. Tài liệu này quan hệ thế nào với constitution

Đây là **kế hoạch thực thi**, không phải nguồn chân lý. Thứ tự ưu tiên khi xung đột:

```
CLAUDE.md  >  ADR đã accepted  >  research/gates.yaml  >  configs/*.yaml  >  tài liệu này
```

| Thứ | Nguồn chân lý | Tài liệu này được phép làm gì |
|---|---|---|
| Phiên bản môi trường | `ADR-001`, `environment.lock` | chỉ tham chiếu, **không** đặt version mới |
| Siêu tham số huấn luyện | `configs/base_config.yaml` (luận văn Ch.3) | chỉ tham chiếu; lệch → phải mở ADR |
| Chế độ chạy FL | `ADR-002` | chỉ tham chiếu |
| Tiến độ gate | `research/gates.yaml` | chỉ tham chiếu |
| Bằng chứng literature | `research/evidence/literature_registry.yaml` | chỉ trích bài **đã** có trong registry |
| Cấu trúc code | `src/` hiện có (CLAUDE.md §15) | chỉ map, **không** đề xuất cây thư mục mới |

Nhãn bằng chứng dùng đúng CLAUDE.md §4 (`[LITERATURE]` / `[YOLO-DOC]` / `[FL-DOC]` / `[THESIS-HYPOTHESIS]` / `[ENGINEERING]` / `[NEEDS-VERIFICATION]`).
Nhãn **độ đã-test của code** là một trục riêng, độc lập: `{code: tested | untested | runtime-unverified}`. Hai trục này không thay thế nhau — đây là lỗi của v1.

---

## 1. Quyết định đã chốt (2026-10-01)

| # | Quyết định | Căn cứ | ADR cần mở |
|---|---|---|---|
| D1 | **Lớp vacant chính = `bus`**, thứ cấp = `truck`. `motorcycle` giữ trong 4 lớp target nhưng **không** làm missing-class target; báo cáo như "rare, hiện diện mọi client". | §4 (số đếm thật) | ADR-005 |
| D2 | **Giữ Flower simulation mode trên Kaggle** (`ADR-002` không đổi). Bổ sung resume-theo-round vào `src/experiments/runner.py`; **không** viết vòng lặp FL thứ hai. | ADR-002 accepted | — (không đổi) |
| D3 | **H2 có hai biến thể tách rõ**: `A2a` = param-level (`src/preservation/mask.py`, đã có), `A2b` = loss-level (mới). A2b là cơ chế **chính**; A2a chuyển thành **test định danh** (§6.3). | §6.3 | ADR-006 |
| D4 | **SCAFFOLD/FedNova ở Tier 2, 1 seed trước**, nâng 3 seed nếu quota cho phép; nếu không → ghi resource limitation theo CLAUDE.md §11. | code đã có trong repo | ghi trong experiment registry |
| D5 | **Mọi sửa code phải giữ/cải thiện đường chạy Kaggle.** Hợp đồng kiểm được + 9 blocker riêng của Kaggle (K1–K9) ở **§13**. Không tạo đường chạy chỉ hoạt động ở local. | yêu cầu người dùng 2026-10-02 | ADR-004 (môi trường main) |

ADR-003 **giữ chỗ cho quyết định Gate A/B/C của G5** (CLAUDE.md §8) — không dùng số này cho việc khác.
ADR-004 giữ chỗ cho môi trường chạy main experiment (đã nêu trong ADR-002 §Related).

---

## 2. Ánh xạ giai đoạn ↔ gate (v1 thiếu hoàn toàn phần này)

`research/gates.yaml` là tracker sống duy nhất. Nhãn P0–P4 chỉ là tên gọi tiện dụng.

| Nhãn | Việc | Gate chính thức | Trạng thái hiện tại |
|---|---|---|---|
| — | Registry bằng chứng | **G0** | `passed` (8 entry) |
| P-env | `pip freeze` phiên Kaggle đầu → điền `environment.lock` | **G1** | `in_progress` ← **việc kế tiếp** |
| P-base | Baseline centralized; đo `t_epoch`, `t_eval` | **G2** | `code_ready`, chờ dataset + GPU |
| P-fedavg | Baseline FedAvg (+FedProx) | **G3** | `code_ready` |
| P0-a | **F1 runtime verify** parameter map | **G5 / F1** | `runtime_confirmed: true` (local CPU, 2026-10-02) — chạy lại `verify_map.py` trên Kaggle stack đã pin |
| P0-b | **F2 controlled perturbation** — v1 **bỏ sót**, CLAUDE.md §8 bắt buộc | **G5 / F2** | `not_started` |
| P0-c | **F3 matched missing-class local training** (≈ "chẩn đoán quên lớp" của v1) | **G5 / F3**, đồng thời là bằng chứng cho **G4** | `code_ready` (2026-10-02); luật pre-register DRAFT chờ duyệt `τ_AP` |
| P0-d | Quyết định Gate **A / B / C** → **ADR-003** | **G5** | `not_started` |
| P1 | S1 vs S1-Control: bằng chứng tham số **và** dự đoán | **G4** | `not_started` |
| P2 | A2b (client), A3 (server) | **G6**, **G7** | `not_started` |
| P3 | Full method + ablation A0–A4b | **G8**, **G9** | `not_started` |
| P4 | Main 3 seed + kết luận | **G10**, **G11** | `not_started` |

**Không** implement phương pháp đầy đủ trước khi G4 **và** G5 pass (CLAUDE.md §7).

---

## 3. Môi trường — tham chiếu, không đặt mới

Theo `ADR-001` (accepted 2026-08-08), khớp stack trong `notebooks/01_smoke_G1_G3.py`:

| Thành phần | Giá trị chốt |
|---|---|
| Python | 3.11 |
| PyTorch / torchvision | 2.7.1 / 0.22.0 (cu128) |
| Ultralytics | **8.3.253** |
| Flower | **1.21.0** |
| MLflow | `>=2.0,<3.0` (ADR-002-A1: giữ protobuf<5 cho flwr 1.21.0) |

Việc còn lại của **G1**: `pip freeze` trong phiên Kaggle đầu → điền `environment.lock` (hiện **DRAFT rỗng**).

`[ENGINEERING]` Ghi nhận sai lệch tại máy local: `flwr 1.29.0` (≠ 1.21.0); `ultralytics` **chưa cài**. Do đó mọi giả định về tên khóa `state_dict`, nội thất `v8DetectionLoss`, `init_criterion`, `ckpt["ema"]` đều `[NEEDS-VERIFICATION]`, và **không được** implement thành research logic trước khi F1 runtime pass (CLAUDE.md §4).

### 3.1 Siêu tham số — canonical từ `configs/base_config.yaml`

| Tham số | Giá trị chốt | Lệch so với plan v1 |
|---|---|---|
| Model / weights | YOLOv8n / `yolov8n.pt` (COCO pretrained) | v1 dùng `yolov8n.yaml` (scratch) → **sai canonical** |
| Image size | **640** | v1 đề xuất 512 → cần ADR nếu muốn giảm vì compute |
| Batch / optimizer / lr0 | 16 / SGD / 0.01 | — |
| FL rounds | **10** | v1 đề xuất 20–30 → cần ADR |
| Local epochs/round | **1** | — |
| num_clients / fraction_fit / fraction_evaluate | 4 / 1.0 / 1.0 | **xem §6.4 — fraction_fit = 1.0 có hệ quả nghiêm trọng cho H3** |
| conf / iou (eval) | 0.25 / 0.70 | — |
| Seeds | **42, 123, 2024** (`min_seeds: 3`) | v1 ghi "seed1/2/3" → dùng đúng 3 số này |

Nếu G2 đo được 10 round × 1 epoch chưa hội tụ, **đó là một phát hiện cần ADR** (kèm đường cong hội tụ làm bằng chứng), không phải cớ để sửa số im lặng (CLAUDE.md §5).

---

## 4. Dữ liệu BDD100K — số đếm thật (đã thực hiện 2026-10-01)

`[ENGINEERING]` Dữ liệu đã có tại `bdd100k_kaggle/`: `images/100k/{train,val}`, nhãn `labels/det_20/{det_train,det_val}.json`.
Schema xác nhận: mỗi item có `name / attributes / timestamp / labels[]`; mỗi label có `category` + `box2d{x1,y1,x2,y2}`; tên lớp xe máy trong BDD là **`motor`**. → Mục [CẦN XÁC MINH] #1 của v1 **đã giải quyết**.

| Lớp | bbox train | ảnh train chứa lớp | % ảnh train | bbox val | ảnh val chứa lớp |
|---|---:|---:|---:|---:|---:|
| car | 713,211 | 69,072 | 98.9% | 102,506 | 9,879 |
| truck | 29,971 | 18,890 | 27.3% | 4,245 | 2,689 |
| **bus** | 11,672 | 8,993 | 13.0% | 1,597 | 1,242 |
| motor | 3,002 | 2,284 | **3.3%** | **452** | **334** |

Train: **69,863** ảnh có nhãn (không phải 70.000 như v1 ghi); **592** ảnh không chứa lớp target nào → **69,271** ảnh dùng được. Val: 10.000 ảnh.

### 4.1 Bốn hệ quả thiết kế rút ra từ số thật

1. **`motor` không làm lớp vacant chính** → cơ sở của D1. Với 4 client × 4.000 ảnh (≈23% pool), tổng bbox motor toàn federation ≈ 694 → ≈170/client; val chỉ **452** bbox motor. `std` của AP_motor qua seed sẽ lớn hơn hiệu ứng cần đo → tiêu chí `2·std` **không kiểm được**. AP_motor báo cáo như quan sát thứ cấp kèm cảnh báo nhiễu.
2. **`car` không thể là lớp vacant** (98.9% ảnh chứa car): client loại car chỉ còn ~800 ảnh. Ghi thành ràng buộc suy từ dữ liệu, không phải lựa chọn.
3. **Loại ảnh dịch chuyển cả class-composition, không chỉ mất lớp.** Client loại bus mất 13.0% ảnh ứng viên, loại truck mất 27.3%; phần bị loại dồn sang client khác, nên client giữ bus có tỷ lệ ảnh-chứa-bus ≈2× base rate. → S1 **tự sinh confound**; S1-Control matched-pair (§5.3) là **điều kiện cần** để claim G4, không phải tùy chọn.
4. **Dữ liệu không phải nút cổ chai**: 69.271/4 = 17.317 ảnh/client tối đa. "2.500 ảnh/client" của v1 là lựa chọn **compute**, không phải giới hạn dữ liệu. Đề xuất **4.000 ảnh/client** làm điểm khởi đầu, chốt sau khi đo `t_epoch` ở G2; mục tiêu tối thiểu **≥1.000 bbox bus mỗi client-có-bus**.

### 4.2 Convert sang YOLO — dùng code đã có

**Không** viết `data/convert.py` mới. Dùng `scripts/prepare_bdd100k.py` + `src/data/bdd100k.py` (`TARGET_CLASSES`, `load_bdd100k_annotations`, `filter_target_classes`).
Ánh xạ `car→0, bus→1, truck→2, motor→3`; **key trong manifest/config giữ tên `motorcycle`** (khớp `TARGET_CLASSES`) — chỉ tên lớp nguồn BDD là `motor`.

`[ENGINEERING]` Bỏ các lớp khác (person/rider/bike/train/traffic light/sign) biến chúng thành background **giống nhau ở mọi client** → không gây lệch giữa client, nhưng hạ mAP tuyệt đối. Phải ghi trong báo cáo.

Kiểm tra bắt buộc sau convert: đếm lại 4 con số bbox ở bảng trên từ các file `.txt` đã sinh và **assert khớp**; vẽ 5 ảnh có bbox để xác nhận toạ độ.

---

## 5. Thiết kế partition

Nguyên tắc (giữ từ v1 — **đúng**): client "thiếu bus" phải **không chứa ảnh nào có bus** (loại ảnh), không phải xóa nhãn bus. Xóa nhãn biến bus thành hard-negative sai → đó là bài toán partially-labeled, một giả thuyết khác.

### 5.1 Định nghĩa missing / rare / eligible — thống nhất một lần

v1 để **ba** ngưỡng chạy song song (`0` theo CLAUDE.md, `rare_threshold: 50` trong config, `min_box=10` trong code plan). Chốt lại:

| Khái niệm | Quy tắc | Dùng ở đâu | Bất biến? |
|---|---|---|---|
| `missing` | `n_box == 0` | định nghĩa kịch bản; **điều kiện áp ρ ở client** | CLAUDE.md §9 — **không được đổi** |
| `rare` | `0 < n_box < 50` (`rare_rule_id: pre_registered_rule_v1`) | phân tích, báo cáo | pre-register, khóa trước khi xem AP |
| `eligible` (H3) | `τ_elig = 1`, tức `n_box ≥ 1` ⇔ "không missing" | tập `S_c` trong aggregation | **primary**; `τ_elig = 50` là biến thể sensitivity đã khai báo |

Chọn `τ_elig = 1` làm primary **có lý do**: nó khiến "eligible" trùng khít "không missing", nên phương pháp **không thêm** siêu tham số tự do mới, và luật no-contributor chỉ kích hoạt theo đúng nghĩa gốc. `τ_elig = 50` chạy như phân tích độ nhạy đã công bố trước, **không** phải một núm điều chỉnh ẩn.

ρ ở client **chỉ áp cho lớp `missing` (`n_box == 0`)**, không áp cho `rare`. (v1 dùng `n_box < THR` nên trộn lẫn hai khái niệm.)

### 5.2 Kịch bản

4 client, lớp `car0, bus1, truck2, motorcycle3`. Lớp target chính = **bus**.

| ID | Mô tả | bus có ở | Mục đích | Tier |
|---|---|---|---|---|
| **S0** | IID, chia ngẫu nhiên đều | 4/4 | kiểm phương pháp không làm hại khi không có missing class | 1 |
| **S1b** (chính) | C0 −bus, C1 −bus, C2 −truck, C3 đủ | 2/4 | missing-class mức vừa; truck là target thứ cấp | 1 |
| **S1a** (nhẹ) | C0 −bus; còn lại đủ | 3/4 | độ nặng thấp | 3 |
| **S1c** (nặng) | C0,C1,C2 −bus; C3 đủ | 1/4 | độ nặng cao | 3 |
| **S1d** (no-contributor) | **cả 4 client −bus** | 0/4 | **chứng minh dứt điểm luật §12** `θ_c^{t+1}=θ_c^t`; xem §6.4 | 2, smoke/feasibility |
| **S1-Control** | matched pair, khác duy nhất ở có/không bus | — | tách tác động missing-class khỏi data-size/composition | 1 (bắt buộc cho G4) |

`[ENGINEERING]` Kiểm tra khả thi S1b với 4.000 ảnh/client: C0,C1 chỉ nhận trong 60.278 ảnh không-bus; C2 trong 50.381 ảnh không-truck; C3 toàn bộ. Tổng tiêu thụ 16.000/69.271 ≈ 23% pool → kỳ vọng ≈2.780 bbox bus chia cho C2+C3 (≈1.390/client, đạt ngưỡng ≥1.000) và ≈6.900 bbox truck chia cho C0,C1,C3 (≈2.300/client). **Phải xác nhận lại bằng manifest thật**, con số trên chỉ là ước lượng tỷ lệ.

### 5.3 S1-Control: thuật toán matched-pair (v1 chỉ nêu mục tiêu, không có thuật toán)

Mục tiêu CLAUDE.md §10: hai nhóm giống nhau nhất có thể về số ảnh, tổng bbox, nguồn ghi, tần suất lớp không-target; khác **duy nhất** ở có/không bus.

```
INPUT : pool ảnh (đã convert), N ảnh/client, dung sai tol = 5%
OUTPUT: cặp (P_without, P_with) + bảng match đạt được

1. P_without ← lấy N ảnh từ tập ảnh KHÔNG chứa bus
     (đây chính là một client vacant của S1b → dùng lại để liên kết hai thí nghiệm)
2. Tính profile mục tiêu của P_without:
     B_total, n_car, n_truck, n_motor, phân bố timeofday / weather
3. P_with ← copy(P_without)
4. k ← số ảnh cần thay để bbox bus của P_with đạt mức trung vị
     của các client CÓ bus trong S1b
5. Lặp k lần:
     - chọn ảnh a ∈ P_with để loại (ưu tiên a có tổng bbox gần ảnh thay thế)
     - chọn ảnh b ∉ P_without, b CHỨA bus, cùng timeofday với a,
       |boxes(b) − boxes(a)| ≤ 2
     - thay a → b
6. ACCEPT chỉ khi:  |n_car, n_truck, n_motor lệch| ≤ tol
                    |B_total lệch| ≤ tol
                    |phân bố timeofday/weather lệch| ≤ 5 điểm phần trăm
7. Ghi bảng match đạt được: cột nào khớp, cột nào KHÔNG. Không khớp → báo cáo,
   KHÔNG claim nhân quả (CLAUDE.md §10).
```

Cặp này dùng trực tiếp cho **F3** (§6.2): cùng checkpoint global, cùng seed, cùng hyperparams, khác duy nhất bus.

**Trạng thái — ĐÃ IMPLEMENT (2026-10-02):** `src/data/partitioner.partition_matched_control` + `MatchReport`, scenario `S1-Control-Matched`, config `configs/partition/s1_control_matched_seed42.yaml`. Đã validate trên dữ liệu tổng hợp đúng tỷ lệ BDD100K (§4): `matched = True`, số ảnh **giống hệt**, lệch box car/truck/motorcycle **≈0.0%**, và ~490 swap/client để đưa bus từ 0 lên mức trung vị của client có bus.

Hai điều chỉnh so với bản phác thảo trên, phát hiện khi chạy thử:

1. **Cost function phải khớp theo *vector* lớp non-target, không phải tổng box.** Chọn ảnh thay thế chỉ theo tổng box làm lệch `truck`/`motorcycle` vượt ngưỡng (ảnh donor mang truck/motor theo tần suất dataset). Cost hiện tại là drift chuẩn hóa theo từng lớp non-target, nên các cột đó về ≈0%.
2. **Không được gate trên *tổng* box.** Control **phải** nhiều box hơn đúng bằng số box lớp target nó vừa nhận — gate tổng là sai về khái niệm. Tiêu chí thật là **số ảnh** + **box non-target**; tổng box vẫn được báo cáo nhưng không dùng để pass/fail.

`configs/partition/s1_control.yaml` cũ (chỉ `missing_map: {}`, tức IID, kèm note *tuyên bố* đã matched) **không** đạt yêu cầu và không được dùng cho C1/G4.

### 5.4 Thuật toán gán ảnh: sửa khiếm khuyết trong repo

`[ENGINEERING]` `src/data/partitioner.py` (`partition_missing_class`) gán mỗi ảnh cho **client eligible đầu tiên** → C0 hút gần hết ảnh nó đủ điều kiện → client lệch kích thước nặng, tức **trộn missing-class với data-size**, đúng điều §10 cấm.

Luật thay thế (v1 đã test trên dữ liệu giả, cho 4.592–4.593 ảnh/client):

```
cho mỗi ảnh i (theo thứ tự đã shuffle với seed):
    elig ← [c : img_classes[i] ∩ vacant[c] == ∅]
    nếu elig rỗng            → drop (ghi log số lượng)
    c ← argmin_{c ∈ elig} len(out[c])        # ít đầy nhất → cân bằng kích thước
    nếu len(out[c]) < per_client → gán; ngược lại → drop
```

**Trạng thái — ĐÃ IMPLEMENT (2026-10-02):** luật "ít đầy nhất" + **pass cân bằng lại** (di chuyển ảnh từ client lớn nhất sang client nhỏ nhất khi còn nước đi hợp lệ). Kết quả: chênh lệch kích thước **≤ 1 ảnh**, và trên dữ liệu thử đúng tỷ lệ BDD là **0 ảnh**. Phần lệch còn lại (nếu có) bị ràng buộc bởi `missing_map` chứ không do thứ tự gán, và được log rõ.

Test kèm theo (CLAUDE.md §18, `tests/test_matched_control.py`): lớp vacant có **đúng 0 box**; chênh lệch kích thước ≤ 1 ảnh; không ảnh nào gán hai lần; tiền định theo seed; `per_client` được tôn trọng; số ảnh drop được log.

Hệ quả phải báo cáo (v1 nêu đúng, giữ): ảnh lớp hiếm dồn về client không-vacant lớp đó → **in bảng phân bố bbox/lớp/client thực tế** từ manifest, không mô tả ý định.

### 5.5 Validation

Toàn bộ val chính thức (10.000 ảnh, lọc 4 lớp) làm **global test set** tại server.
Luật chọn model pre-register: **báo cáo vòng cuối** (hoặc trung bình 2 vòng cuối), **không** dùng val để chọn round/model. Nếu sau này cần chọn model, phải tách val-monitor / val-test và khai báo — không dùng cùng một tập cho cả hai (v1 chỉ nói nửa vời).

Về leakage: `[ENGINEERING]` BDD100K 100K có **một frame được gán nhãn cho mỗi video**, nên split train/val chính thức đã tránh rò rỉ giữa các frame cùng video; không cần cơ chế chia theo sequence riêng. Vẫn ghi `source_sequence` vào manifest theo schema §9 CLAUDE.md.

---

## 6. Phương pháp — hai phát hiện phải xử lý trước khi code

### 6.1 Hai cơ chế ρ là hai giả thuyết khác nhau (D3)

| | **A2a** param-level (`src/preservation/mask.py:93`, đã có) | **A2b** loss-level (mới) |
|---|---|---|
| Can thiệp ở | sau local train, trước upload | trong local train, ở số hạng BCE |
| ρ=0 nghĩa là | hoàn trả hàng cls của lớp missing về global | kênh lớp missing không sinh gradient |
| Ảnh hưởng tham số dùng chung | **không** | **có** (cv3[0:2], neck, backbone bớt tín hiệu từ kênh đó) |
| Test §18 "shared params unaffected" | áp dụng được | **không** áp dụng — phải viết lại tiêu chí (§8.3) |

v1 chỉ mô tả A2b và không hề biết `mask.py` tồn tại. Cả hai đều giữ, nhưng vai trò khác nhau — xem §6.3.

**Hai lỗ hổng của A2b mà v1 chưa lường:**
- `[ENGINEERING]` **weight decay + EMA vẫn sửa hàng được bảo vệ** dù gradient = 0. `client_update` của v1 còn trả `ckpt["ema"]` (trung bình trượt có trộn trọng số trước train). → Test "ρ=1 ≡ update thường" và "ρ=0 giữ nguyên" sẽ fail vì lý do không liên quan tới giả thuyết. Xử lý: loại nhóm tham số đó khỏi weight decay, và so sánh trên `model` thay vì `ema`; ghi rõ lựa chọn vào manifest.
- `[THESIS-HYPOTHESIS]` ρ=0 **xóa áp lực âm** ("ở đây không có bus") cho kênh vacant → logit lớp đó không bị ép xuống → dự đoán **FP lớp vacant tăng**. Vì vậy **FP/lớp là chỉ số bắt buộc**, không chỉ AP. Đây là một dự đoán khả bác bỏ của A2b, nên ghi vào §7.

`[YOLO-DOC]` **Đã kiểm 2026-10-02 (ultralytics 8.3.253, `utils/loss.py`):** `v8DetectionLoss.bce = nn.BCEWithLogitsLoss(reduction="none")`, áp lên `pred_scores` sau `permute(0, 2, 1)` ⇒ tensor `[B, anchors, nc]`, rồi `.sum() / target_scores_sum`; `DetectionModel.init_criterion` trả `v8DetectionLoss(self)` (hoặc `E2EDetectLoss` nếu `end2end`). Tiền đề của A2b đúng với bản pin; phần còn mở là cơ chế chèn trọng số theo lớp vào loss (chưa implement, vẫn chặn bởi G4/G5).

### 6.2 F1 / F2 / F3 — thứ tự bắt buộc trước khi viết phương pháp

| | Việc | Điều kiện pass | Đầu ra |
|---|---|---|---|
| **F1** | Chạy `src/model/parameter_map.build_parameter_map(model)` trên YOLOv8n thật, diff với `research/feasibility/F1/parameter_map.yaml`; **assert đúng 6 khóa** cls head (3 scale × {weight, bias}) | tên + shape khớp; `runtime_confirmed: true` | cập nhật `parameter_map.yaml` |
| **F2** | Nhiễu **chỉ** nhóm tham số ứng viên (hàng cls của lớp bus), đo AP target vs AP non-target, confidence, FP/FN | quy tắc quyết định **pre-register trước khi đo**; so hiệu ứng target với non-target/control | `research/feasibility/F2/` |
| **F3** | Từ cùng checkpoint global, train cục bộ trên cặp matched (§5.3); theo dõi `Δθ` theo backbone / neck / cls / box / nhóm class-associated + ΔAP/lớp + confidence + FP/FN | **đồng thời** có bằng chứng tham số **và** bằng chứng dự đoán cùng xu hướng | `research/feasibility/F3/` |
| **Gate** | A (supported) / B (partial) / C (unsupported) | → **ADR-003** | — |

**F2 là phần v1 bỏ sót.** Không có F2 thì không ra được quyết định A/B/C, và G5 không thể pass.

Biến bắt buộc trong F3: **hai kiểu checkpoint khởi đầu, cả hai phải đã nhận diện được bus.** `[YOLO-DOC]` **Sửa 2026-10-02 (F1 runtime):** `yolov8n.pt` thô **không** hợp lệ làm điểm khởi đầu F3 — ở nc=4 toàn bộ nhánh cv3 (36 key, gồm cả kênh ẩn 64 vs 80) lệch shape với COCO nên được **khởi tạo mới** (`DetectionModel.load` → `intersect_dicts`), tức head không có kiến thức "bus" nào để quên. Thay bằng: (1) checkpoint G2 centralized 4 lớp; (2) global FedAvg sau vài round warm-up từ COCO backbone (giữ được biến "pretrained backbone" của v1). `summary.yaml: target_known_before` ghi lại điều này; `false` ⇒ inconclusive, không phải kết quả âm.

Tiêu chí pass của F3/G4, pre-register trước khi chạy:
`AP_after[bus] < AP_before[bus] − max(2 · std qua 3 seed, ngưỡng tuyệt đối đã khai báo)`
**và** `Δθ` ở nhóm class-associated lệch rõ so với nhóm control. Chỉ có AP (hoặc chỉ có L2) thì **không** đủ (CLAUDE.md §8).

### 6.3 A2a dưới A3 là một **định danh toán học**, không phải câu hỏi thực nghiệm

Phát hiện quan trọng khi đối chiếu `mask.py` với `class_aware_agg.py`:

- A2a bảo vệ **đúng** các hàng cls của lớp mà client đó `missing`.
- A3 (với `τ_elig = 1`) **đã loại** chính client đó khỏi `S_c` cho lớp đó.

⇒ Dưới A3, A2a **không thay đổi bất cứ gì**: `A4a ≡ A3` về mặt số học, không phải "có thể bằng". Hệ quả:

1. **Không** chạy A4a 3 seed — đó là 3 run vô nghĩa.
2. Dùng A4a làm **test định danh** ở smoke scale: chạy A3 và A4a cùng seed, assert tham số global sau mỗi round **bằng nhau trong sai số số học**. Nếu khác → có bug ở eligibility hoặc ở mask.
3. Vì A2a không tách được khỏi A3, **A2b (loss-level) là cơ chế client-side duy nhất tách được khỏi server-side** — nó thay đổi quỹ đạo của tham số **dùng chung**, thứ mà aggregation phía server không chạm tới. Đây là lý do khoa học cho D3.

### 6.4 A1 vs A3: rủi ro tính mới phải xử lý **trước** khi chạy main

Công thức A3 (đã sửa theo §12, dạng delta, giữ `θ^t`):

```
Với nhóm class-associated P_cls = { model.*.cv3.*.2.{weight,bias} }  (assert đúng 6 khóa):
    S_c = { i : n_i^c ≥ τ_elig }
    w_i^c = n_i^c / Σ_{j∈S_c} n_j^c
    θ^{t+1}[c] = θ^t[c] + Σ_{i∈S_c} w_i^c · ( θ_i[c] − θ^t[c] )
    S_c = ∅  →  θ^{t+1}[c] = θ^t[c]        # log action: keep_global
Tham số còn lại: FedAvg (hoặc FedProx/SCAFFOLD/FedNova theo phương pháp), trọng số n_img.
```

Baseline A1 (class-count weighted FedAvg) dùng `w_i^c = n_i^c / Σ_{j} n_j^c` trên **toàn bộ** client, không có ngưỡng eligibility, không có luật no-contributor.

**Vấn đề:** client có `n_i^c = 0` thì trọng số của nó trong A1 **vốn đã bằng 0**. Với `τ_elig = 1`, tập `S_c` của A3 trùng khít tập client có trọng số khác 0 của A1. Thêm `fraction_fit = 1.0` (canonical) và S1b có bus ở 2/4 client ⇒

> **A3 ≡ A1 về mặt số học trong S1b.** Khác biệt duy nhất nằm ở trường hợp `S_c = ∅`, mà trong S1b không bao giờ xảy ra.

Nếu không xử lý, tiêu chí CLAUDE.md §22.6 ("class-count baseline không giải thích hết phần tăng") bị vi phạm **theo cấu trúc**, không phải vì may rủi — tức đóng góp H3 sẽ không bảo vệ được. Ba cách xử lý, **chọn trước khi chạy main**:

| Phương án | Nội dung | Chi phí / rủi ro |
|---|---|---|
| **P-a** (đề xuất) | Khai báo thẳng: đóng góp riêng của A3 là **bảo đảm no-contributor**, và chứng minh nó bằng **S1d** (bus vắng ở cả 4 client) — ở đó A1 suy biến/không xác định trong khi A3 chứng minh được giữ `θ^t`. A2b là đóng góp cơ chế chính. | Rẻ, trung thực, chạy được ngay ở feasibility scale. Thu hẹp phạm vi claim của H3. |
| **P-b** | Dùng `τ_elig = 50` làm primary → A3 loại cả client "rare" mà A1 vẫn tính ⇒ A3 ≠ A1. | Thêm một siêu tham số cần biện minh; phải giải thích vì sao 50. |
| **P-c** | Thêm `fraction_fit < 1.0` làm biến kịch bản → có round không client nào được chọn mà có bus ⇒ luật no-contributor kích hoạt trong S1b/S1c. | Lệch khỏi `fraction_fit: 1.0` của luận văn → **cần ADR + phê duyệt scope**. |

Khuyến nghị: **P-a làm chính, P-b chạy như phân tích độ nhạy đã công bố trước.** P-c chỉ mở nếu giáo viên hướng dẫn đồng ý sửa giao thức.

### 6.5 Ba lỗi trong repo phải sửa **trước** mọi ablation

| # | Vị trí | Lỗi | Sửa |
|---|---|---|---|
| **R1** | `src/federated/strategies/class_aware_agg.py` | Khi `S_c = ∅`, trace ghi `action: keep_global` nhưng giá trị để lại là **FedAvg của các client**, không phải `θ^t` (strategy không lưu global round trước). Vừa sai §12, vừa là trace **nói sai sự thật** (§19/§20). | Cache `θ^t` từ `configure_fit` / `initial_parameters`; dùng công thức dạng delta ở §6.4 |
| **R2** | cùng file | `counts is None → treat as eligible`: biến lỗi mất metadata thành "eligible toàn bộ" một cách im lặng | **fail loudly** |
| **R3** | `src/federated/client.py` vs `client_variants.py:229` | `class_counts_json` **chỉ** được phát bởi `PreservationClient`. Vậy cấu hình A3 (server class-aware + client mặc định) → server nhận `counts=None` → rơi vào R2 → **H3 suy biến thành FedAvg** trong khi trace báo `class_count_weighted`. Ô A3 của ablation hiện **không chạy được như mô tả**. | Đưa `class_counts_json` xuống `YOLOFlowerClient.fit` (client cơ sở) |

Thêm `R4` (không chặn nhưng nên làm): `mask.py` và `class_aware_agg.py` hard-code `^model\.22\.`; dùng `model\.\d+\.cv3\.\d+\.2\.(weight|bias)$` + **assert đếm đúng 6 khóa** (ý tưởng đúng của v1) để không âm thầm khớp 0 khóa nếu Ultralytics đổi chỉ số layer.

#### Trạng thái sửa — 2026-10-02 (ADR-007)

| ID | Trạng thái | Ghi chú |
|---|---|---|
| B1 đường chạy ablation | ✅ đã sửa | `--ablation {A0,A1,A2a,A2b,A3,A4a,A4b}` + `--client-mechanism` + `--rho` (bắt buộc khi có mechanism) |
| B2 strategy crash | ✅ đã sửa | builder đồng nhất `build_*(num_rounds, **kw)`; thêm `build_scaffold`/`build_fednova`/`build_class_count_fedavg` |
| B3 seed không tới local train | ✅ đã sửa | `derive_local_seed(base, round, client)` truyền vào trainer |
| B4 keep_global sai | ✅ đã sửa | `configure_fit` cache `θ^t`; dạng delta; test `test_no_contributor_keeps_exact_global_row` |
| B5 per-client config | ✅ đã sửa | `class_counts`/`missing_classes`/`rho` là tham số constructor, lấy từ manifest |
| R2 counts=None im lặng | ✅ đã sửa | raise mặc định (`require_class_counts`) |
| R3 `class_counts_json` | ✅ đã sửa | chuyển vào `YOLOFlowerClient._fit_metrics` |
| R4 regex layer index | ✅ đã sửa | `find_cls_head_indices` raise khi khớp 0 khóa, warn khi ≠ 6 |
| H1 val lặp 9 lượt/round | ✅ đã sửa | `client_run_val: false` + `fraction_evaluate: 0.0` ở smoke/feasibility |
| H2 rò rỉ test set vào best.pt | ✅ đã sửa | hệ quả của `val=False` trong `model.train()` |
| H3 warmup chiếm cả round | ✅ đã sửa | `warmup_epochs: 0.0`, `close_mosaic: 0` trong config |
| H4 FP/FN theo lớp | ✅ đã sửa | `yolo_wrapper._per_class_fp_fn` — orientation confusion matrix vẫn `[NEEDS-VERIFICATION]` |
| H5/K4 resume | ✅ đã sửa | `src/experiments/checkpoint.py` + `--resume` + run_id tiền định; từ chối ghi đè khi thiếu `--resume` |
| H6 aggregation_trace | ✅ đã sửa | `trace_dir=run_dir` được truyền; `verify_artifacts` yêu cầu cho cả `ClassCountFedAvg` |
| H7 test cho 2 module đóng góp | ✅ đã sửa | 40 → **107 test**, chạy không cần GPU/torch-CUDA/ultralytics |
| H8 SCAFFOLD = FedAvg | ✅ đã sửa | server step Algorithm 1 Option II, dùng `eta_global` |
| M1 `make_client_fn` | ✅ đã xóa | code chết + `num_examples=1` |
| M3 `set_parameters` | ✅ đã sửa | raise khi lệch số phần tử (chặn `zip()` cắt im lặng) |
| M7 requirements | ✅ đã sửa | `mlflow>=2.0,<3.0`, `numpy>=2.0,<3.0`, thêm `protobuf<5` — trước đó trái stack Kaggle đã verify |
| K1/K2/K3/K5 | ✅ đã sửa | `workers`/`client_num_cpus`/`client_num_gpus=1.0`/`keep_last_checkpoints`/`prune_client_weights` vào config |
| K7 `YOLO_CONFIG_DIR` | ✅ đã sửa | đặt trong `notebooks/04_ablation_smoke.py` |
| K8 Kaggle chỉ chạy A0 | ✅ đã sửa | `notebooks/04_ablation_smoke.py` chạy A0/A1/A3/A2a/A4a + xác nhận A2b/A4b bị chặn |
| **A2b (`rho_loss.py`)** | ⛔ **cố ý chưa làm** | `LossPreservationClient.fit()` raise; chặn bởi F1 runtime + ADR-006 (CLAUDE.md §4) |
| **M2** buffer int | ⬜ chưa sửa | giữ trong vector để không lệch index; đã ghi chú trong `get_parameters` |
| **M4** `plots/` rỗng | ⬜ chưa sửa | §21 yêu cầu plots — làm cùng bước vẽ hình báo cáo |
| **M5** `environment.lock` rỗng | ⬜ chưa sửa | đây chính là G1, cần phiên Kaggle |
| **Confidence stats** | ⬜ chưa làm | §13 yêu cầu; cần một pass predict riêng, làm cùng F2/F3 |
| **K6** Internet/AMP cache | ⬜ chưa xác minh | kiểm ở phiên Kaggle đầu |
| **K9** mâu thuẫn KAGGLE_SETUP | ⬜ chưa giải | cần ADR-004 sau khi K4 đã chạy thật |

### 6.6 Map plan → đường dẫn repo (thay cho cây `fl/` của v1)

| Chức năng | **Dùng file đã có** | Ghi chú |
|---|---|---|
| Convert BDD → YOLO | `scripts/prepare_bdd100k.py`, `src/data/bdd100k.py` | — |
| Partition + manifest | `src/data/partitioner.py`, `scripts/generate_partition.py` | sửa theo §5.4 + thêm matched-pair §5.3 |
| Thống kê lớp/client | `src/data/class_stats.py` | nguồn của `class_counts_json` |
| Parameter map / F1 | `src/model/parameter_map.py` | F1 runtime verify |
| A2a param-level ρ | `src/preservation/mask.py` | đã có, chuyển thành test định danh §6.3 |
| **A2b loss-level ρ** | **`src/preservation/rho_loss.py` (FILE MỚI DUY NHẤT)** | sau khi F1 pass |
| A3 class-aware agg | `src/federated/strategies/class_aware_agg.py` | sửa R1/R2 |
| A1 class-count baseline | cùng file, cờ riêng (không eligibility, không no-contrib) | phải tách rõ khỏi A3 |
| FedAvg/FedProx/SCAFFOLD/FedNova | `src/federated/strategies/*.py` | đã có đủ 4 |
| Client | `src/federated/client.py`, `client_variants.py` | sửa R3 |
| Vòng lặp FL + resume | `src/federated/server.py` (`fl.simulation.start_simulation`), `src/experiments/runner.py` | thêm resume-theo-round, **không** viết `fl/server.py` |
| Đánh giá | `src/evaluation/metrics.py`, `drift.py` | `drift.py` dùng cho Δθ của F3 |
| Artifact/run dir | `src/utils/artifacts.py` | đã có `verify_artifacts` |
| Notebook Kaggle | `notebooks/01_smoke_G1_G3.py`, `02_baseline_G2_G3.py`, `03_missing_class_G4.py` | mở rộng, không tạo mới trùng |

---

## 7. Ma trận thí nghiệm

### 7.1 Ablation — map thẳng vào A0–A4 của CLAUDE.md §11

| ID | Client | Server | Vai trò | Seed |
|---|---|---|---|---|
| **A0** | — | FedAvg | baseline bắt buộc | 3 |
| **A1** | — | class-count weighted (không eligibility, không no-contrib) | control "class-count giải thích được không?" | 3 |
| **A2a** | ρ param-level | FedAvg | client-only, bản param | 1 (+ test định danh §6.3) |
| **A2b** | ρ loss-level | FedAvg | **client-only chính** | 3 |
| **A3** | — | class-aware (τ_elig, no-contrib) | server-only | 3 |
| **A4a** | ρ param-level | class-aware | **≡ A3 (định danh)** → chỉ smoke để verify | 0 (không báo cáo như run độc lập) |
| **A4b** | ρ loss-level | class-aware | **full method** | 3 |
| ref | — | FedProx / SCAFFOLD / FedNova | reference baseline | 1 (Tier 2, D4) |

### 7.2 Ưu tiên chạy

- **Tier 1 (bắt buộc):** `S1b × {A0, A1, A2b, A3, A4b} × 3 seed` = **15 run**; `S0 × {A0, A4b} × 3 seed` = **6 run**. Tổng **21 run**.
- **Tier 2:** ρ sweep; `S1d` (chứng minh no-contributor, §6.4 P-a); S1-Control/F3; `τ_elig = 50` sensitivity; FedProx/SCAFFOLD/FedNova 1 seed.
- **Tier 3:** S1a, S1c, port FedRS/FedLC/FedVLS sang BCE — **chỉ sau khi** các bài đó vào `literature_registry.yaml`.

### 7.3 Chọn ρ: quy tắc đề bạt pre-register (chống cherry-picking)

Không có bài nào trong registry hiện tại biện minh được một giá trị ρ cụ thể ⇒ ρ là `[THESIS-HYPOTHESIS]`, phải xác định bằng thực nghiệm, **không** bằng phán đoán.

```
Bước 1 (feasibility, 1 seed = 42): chạy A2b với ρ ∈ {0, 0.25, 1} trên S1b.
Bước 2 (quy tắc đề bạt, khóa TRƯỚC khi xem kết quả):
        chọn ρ* = argmax  AP_bus(vòng cuối)
        hòa trong 1 điểm AP → chọn ρ LỚN hơn (can thiệp nhẹ hơn)
        nếu FP_bus của ρ* > 1.5 × FP_bus của ρ=1 → loại ρ* đó, lấy ứng viên kế tiếp
Bước 3 (main, 3 seed): chỉ ρ* vào A2b/A4b.
Bước 4: BÁO CÁO cả 3 giá trị ở bảng sweep, kể cả giá trị xấu (CLAUDE.md §20).
```

ρ=1 đồng thời là **control nội bộ**: A2b với ρ=1 phải trùng A0 trong sai số số học — nếu không, có bug.

### 7.4 Ngân sách GPU

```
GPU-giờ/run ≈ rounds × clients × local_epochs × t_epoch(client) + rounds × t_eval
            = 10 × 4 × 1 × t_epoch + 10 × t_eval
Tổng Tier 1 ≈ 21 × (GPU-giờ/run)
```

`[RESOURCE-BUDGET-UNRESOLVED]` `t_epoch` và `t_eval` **chưa đo** (chưa có GPU + ultralytics trong môi trường này). Phải đo ở **G2** với cấu hình cuối (imgsz 640, batch 16, 4.000 ảnh/client) rồi điền. **Không** khởi động `main` trước khi báo chi phí dự kiến (CLAUDE.md §14).

Nếu vượt ngân sách, thứ tự cắt: Tier 3 → Tier 2 → giảm ảnh/client → (cần ADR) giảm imgsz. **Không** giảm số seed của Tier 1 xuống dưới 3.

---

## 8. Chỉ số, test, artifact

### 8.1 Chỉ số (khớp `configs/base_config.yaml` + CLAUDE.md §13)

- Detection: `mAP50`, `mAP50-95`, `AP/lớp`, `precision/lớp`, `recall/lớp`, `FP/lớp`, `FN/lớp`, thống kê confidence.
- FL: metric theo round, hội tụ, round tốt nhất/cuối, thời gian local/round/aggregate, bytes update.
- Missing-class: `ΔAP_c = AP_method,c − AP_FedAvg,c`.
- Tham số: norm theo module, cosine similarity, drift của nhóm class-associated (`src/evaluation/drift.py`).

`FP/FN` phải định nghĩa **một lần** trong metrics contract: lấy từ confusion matrix tại `conf=0.25, iou=0.70` (canonical), có hàng/cột background — **không** đồng nhất với FP/FN ở mức AP. Đối chiếu định nghĩa này với `src/evaluation/metrics.py` hiện có trước khi báo cáo, để tránh hai định nghĩa song song.

### 8.2 Giao thức báo cáo

Tối thiểu 3 seed (42, 123, 2024); cùng partition seed giữa các phương pháp; báo `mean ± std`; **vòng cuối** (hoặc trung bình 2 vòng cuối) chứ không phải giá trị cao nhất; cùng evaluator cho mọi phương pháp. Ngưỡng "không suy giảm đáng kể" ở S0 = `2 × std` của A0 qua 3 seed, **chốt trước khi chạy**. Không so số với bài khác khi cấu hình khác — luôn chạy lại baseline.

### 8.3 Test bắt buộc (CLAUDE.md §18), có sửa cho A2b

| Nhóm | Test | Ghi chú |
|---|---|---|
| Dataset | lớp missing có **đúng 0** box; đếm khớp §4; partition tiền định theo seed; không ảnh trùng; chênh kích thước client ≤1% | mở rộng `tests/test_partitioner.py` |
| Parameter map | tên tồn tại; shape khớp 8.3.253; **assert đúng 6 khóa** cls head; nc khớp config | `tests/test_parameter_map.py` |
| A2a | `ρ=1` ≡ update thường; `ρ=0` chỉ giữ nhóm protected; shared params không đổi | giữ nguyên §18 |
| **A2b** | `ρ=1` ≡ loss mặc định (sai số <1e-6); `ρ=0` → **gradient hàng cls lớp vacant = 0**, hàng khác ≠ 0 | tiêu chí §18 "shared params unaffected" **không áp dụng** — A2b cố ý đổi shared params. Phải nêu rõ trong test docstring, kèm kiểm tra weight-decay/EMA không sửa hàng protected (§6.1) |
| A3 | 0 / 1 / nhiều client eligible; chuẩn hóa trọng số; **no-contributor giữ đúng `θ^t`** (test R1); tương thích khi mọi lớp đều có mặt | `counts=None` phải **raise**, không im lặng |
| Định danh | `A4a ≡ A3` từng round, cùng seed | test mới theo §6.3 |
| Integration | 1 round smoke xong; artifact + metadata sinh ra; cùng evaluator giữa các phương pháp | `verify_artifacts` |

### 8.4 Artifact (CLAUDE.md §21) — dùng `src/utils/artifacts.py`

Mỗi run phải có: `config.yaml`, `environment.json`, `partition_manifest.*`, `metrics.csv`, `per_class_metrics.csv`, `round_metrics.csv`, `aggregation_trace.*`, `run.log`, `checkpoint/`, `plots/`, `README.md`; kèm `git commit | versions | dataset | partition ID | seed | run class`.

Mỗi run **phải** đăng ký một entry trong `research/experiment_registry/registry.yaml` (hiện `experiments: []`) với status `planned | running | completed | failed-valid | failed-technical | rejected`. **Không xóa kết quả âm.**

Run ID: `<scenario>_<ablation>_<rho>_seed<seed>_<runclass>`, ví dụ `S1b_A4b_rho0.25_seed42_main`.

Resume: mỗi round lưu `global_round_XXX.pt` + append CSV vào `/kaggle/working`; cuối phiên đẩy output thành Kaggle dataset (ADR-002). Phiên sau đọc round cuối và chạy tiếp.

---

## 9. Lịch còn lại (tính từ 2026-10-01, điều chỉnh theo phần repo đã có)

Khác v1: tuần 1–3 của v1 (literature, download, convert, code baseline) **phần lớn đã xong trong repo** — `prepare_bdd100k.py`, `train_centralized.py`, `run_fl_experiment.py`, 4 strategy, notebooks 01–03 đều `code_ready`. Nút thắt thật là **GPU + G1**.

| Tuần | Việc | Gate | Đầu ra |
|---|---|---|---|
| 1 | Phiên Kaggle đầu: pip freeze → `environment.lock`; sửa **R1/R2/R3** + test | G1 | `environment.lock`, test xanh |
| 2 | Convert + assert §4.2; partition S0/S1b + matched-pair §5.3; bảng phân bố thật | — | manifest + bảng phân bố |
| 3 | Centralized baseline; đo `t_epoch`/`t_eval` → điền ngân sách §7.4 | **G2** | mAP centralized, ngân sách |
| 4 | FedAvg (+FedProx) baseline; S0 vs S1b | **G3** | bảng baseline |
| 5 | **F1 runtime** + **F2 perturbation** | G5 phần 1 | `F2/`, quyết định sơ bộ |
| 6 | **F3** matched pair, 3 seed, 2 kiểu khởi tạo → **ADR-003** Gate A/B/C | **G5**, bằng chứng **G4** | `F3/`, ADR-003 |
| 7 | S1 vs S1-Control đầy đủ (tham số + dự đoán) | **G4** | kết luận H1 |
| 8–9 | A2b (`rho_loss.py`) + A3 (sau khi R1–R3 xanh); ρ sweep 1 seed → ρ* | **G6**, **G7** | ρ*, kết quả sơ bộ |
| 10 | A1, S1d, test định danh A4a≡A3, ablation đủ A0–A4b 1 seed | **G8**, **G9** | bảng ablation |
| 11–12 | Tier 1 main 3 seed (21 run), resume theo quota | **G10** | bảng `mean ± std` |
| 13 | Tier 2 (SCAFFOLD/FedNova 1 seed, τ_elig=50, FLOps/MLflow demo) | — | bảng phụ |
| 14–15 | Viết báo cáo, hình AP theo lớp, rà soát tái lập | **G11** | bản thảo |

Dự phòng: 1 tuần đệm giữa tuần 11–12 cho run lỗi/hết quota.

---

## 10. Rủi ro

| Rủi ro | Dấu hiệu | Phương án |
|---|---|---|
| **A3 ≡ A1 ⇒ H3 mất tính mới** (§6.4) | ablation cho A3 = A1 tới chữ số cuối | P-a: khai báo đóng góp là no-contributor + chứng minh bằng S1d; P-b: τ_elig=50; P-c: fraction_fit<1 (cần ADR) |
| **A4a ≡ A3** (§6.3) | hai run trùng khít | Không chạy A4a 3 seed; dùng làm test định danh; full method = A4b |
| **Client gửi trọng số fp16 / server crash nc=80 / eval fuse model** (ADR-008, đã sửa 2026-10-02) | Δθ hàng lớp giống hệt nhau giữa các phía/seed; lỗi size/count mismatch | Snapshot EMA fp32 ở `on_train_epoch_end`; `build_model` luôn nc=4 có seed; `evaluate` chạy trên bản sao. Xác nhận lại trên GPU (AMP) ở phiên Kaggle đầu |
| Không quan sát được quên lớp (COCO pretrained che) | AP_bus không giảm ở F3 | Dùng checkpoint G2 (4 lớp) làm khởi tạo thứ hai; tăng local epoch; chuyển S1c; nếu vẫn không → **báo cáo kết quả âm** |
| ρ=0 làm FP lớp vacant tăng | FP_bus tăng trong khi AP không tăng | Quy tắc đề bạt §7.3 đã loại trường hợp này; cân nhắc biến thể distillation (chỉ sau G6) |
| weight decay/EMA phá bảo vệ ρ=0 | test "ρ=0 giữ nguyên" fail | Loại nhóm protected khỏi weight decay; dùng `model` thay `ema`; ghi vào manifest |
| AP_motor quá nhiễu | std qua seed > hiệu ứng | Đã xử lý bằng D1: motor là quan sát thứ cấp, không phải target |
| Hết quota GPU | run dở | Resume theo round; cắt Tier 3 → Tier 2; (cần ADR) giảm imgsz |
| Ultralytics đổi API/tên khóa | lỗi attribute, hoặc regex khớp 0 khóa | Pin 8.3.253; assert đúng 6 khóa (R4); F1 runtime là điều kiện tiên quyết |
| 10 round chưa hội tụ | đường cong còn dốc ở round 10 | Đây là **phát hiện cần ADR**, không sửa số im lặng |
| Trùng ý với công trình đã có | thấy khi khảo sát literature | Nêu khác biệt cụ thể (A2b tác động shared params + ablation tách được), hoặc chuyển trọng tâm sang phân tích |

---

## 11. Danh sách `[NEEDS-VERIFICATION]` còn mở

1. **Literature chưa vào registry.** v1 dùng **FedPylot**, **FedVLS**, **FedLC**, **FedRS**, **FedNTD**, **NIID-Bench**, **FedProx+LA (arXiv 2405.01108)** như sự thật (ví dụ "FedPylot khuyến nghị chia theo ranh giới tự nhiên", "FedVLS báo mức cao nhất"). `literature_registry.yaml` hiện chỉ có **8 entry**: FedAvg, FedProx, SCAFFOLD, FedNova, Flower, Zhao2018, YOLOv8, BDD100K. → Phải đăng ký (title | authors | year | venue | source | reused concept | reproduction/adaptation/extension) **trước khi** dùng để biện minh thiết kế, hoặc hạ xuống `[NEEDS-VERIFICATION]` và **không** dùng làm căn cứ. **Mọi câu dẫn 7 bài trên đã được bỏ khỏi v2.**
2. Hạng CORE/ICORE của từng venue sẽ trích dẫn.
3. `ultralytics 8.3.253`: tên khóa `cv3.*.2.*`, `self.bce` reduction/shape, `init_criterion`, `ema` vs `model` khi lấy trọng số sau train. → **F1 runtime**.
4. Hạn mức GPU/tuần và giới hạn phiên của tài khoản Kaggle (v1 ghi "~30 giờ/tuần" là con số Kaggle không công bố chính thức).
5. `t_epoch`, `t_eval` ở cấu hình cuối → điền §7.4.
6. Ước lượng bbox/client ở §5.2 — phải xác nhận lại bằng manifest thật sau khi partition.
7. Có công trình nào đã làm **đồng thời** client-side vacant-class protection **và** class-wise aggregation cho object detection chưa? → kết hợp với mục 1.

---

## 12. Những gì v1 làm đúng và v2 giữ nguyên

Ghi lại để không mất khi so hai bản: loại ảnh thay vì xóa nhãn; chạy chẩn đoán trước khi viết phương pháp; ba nhánh kết cục kể cả "không quên → báo kết quả âm"; COCO-pretrained là confound của H1; ngưỡng go/no-go pre-register `2·std`; báo vòng cuối thay vì vòng tốt nhất; cùng partition seed; ≥3 seed; không so số với bài khác; resume theo round + run ID + đẩy output thành Kaggle dataset; bắt đo `t_epoch` thật trước khi lập ngân sách; bắt in bảng phân bố bbox/lớp/client **thực tế**; A1 class-count là control bắt buộc; regex không hard-code chỉ số layer + assert đếm 6 khóa; luật gán "ít đầy nhất" cho partition; dạng delta giữ `θ^t` trong aggregation.

---

## 13. Hợp đồng chạy được trên Kaggle (D5 — ràng buộc cho MỌI lần sửa code)

Quyết định **D5 (2026-10-02):** mọi sửa code trong workstream này phải **giữ hoặc cải thiện** đường chạy Kaggle hiện có (`notebooks/01_smoke_G1_G3.py`, `02_baseline_G2_G3.py`, `03_missing_class_G4.py` → gọi `scripts/*.py` qua `subprocess`). Không được tạo đường chạy chỉ hoạt động ở local.

### 13.1 Những gì đường Kaggle hiện tại đã làm đúng — không được phá

| Đã có | Vị trí |
|---|---|
| Partition **sinh trên Kaggle**, nên đường dẫn tuyệt đối trong `C*_train.txt` là path Kaggle | `02:104`, `01:255` |
| `--global-data-yaml` được truyền → `evaluate_fn` tập trung hoạt động → `per_class_metrics.csv` + `final_params.npz` | `01:305`, `02:179`, `03:157` |
| MLflow file store cục bộ `--mlflow-uri` | `01:303`, `02:177`, `03:155` |
| Cài pin đúng ADR-001 + ADR-002-A1 mỗi phiên | `01`, `03` cell cài đặt |
| `environment.lock` điền từ `pip freeze` của phiên | `01:161-164` |
| Export artifact sang `/kaggle/working/flops_export/` để lưu thành Output Dataset | `01:347-366` |

⇒ Khi sửa `scripts/run_fl_experiment.py` **không được đổi tên/bỏ** các cờ `--partition --algorithm --data-yaml-dir --exp-config --run-class --seed --mlflow-uri --mlflow-experiment --global-data-yaml`; chỉ được **thêm** cờ mới có giá trị mặc định an toàn.

### 13.2 Blocker riêng của Kaggle — phải sửa cùng lúc với B/H ở §6.5

| ID | Vấn đề | Yêu cầu |
|---|---|---|
| **K1** | Kaggle GPU notebook chỉ có ~4 vCPU `[NEEDS-VERIFICATION]`. `client_num_cpus=1` × 4 client = chiếm hết CPU, driver/server không còn slot → Ray có thể treo. | Đưa `client_num_cpus` / `client_num_gpus` vào **config** (`configs/experiments/*.yaml`), không để hard-code mặc định trong `server.py:182-183`. Kaggle: ghi rõ giá trị đã test. |
| **K2** | `train_one_round` không đặt `workers` → Ultralytics mặc định 8 worker **mỗi client** × 4 client = 32 worker trên 4 vCPU → thrash / lỗi shared memory. | Truyền `workers` từ config (Kaggle: 2, local: 2). Cùng lúc sửa H3 (`warmup_epochs`) và H1 (`val=False`) trong **một** chỗ. |
| **K3** | `client_num_gpus=0.25` → 4 client đồng thời. Trên T4/P100 16 GB thì ~4×3–4 GB là **sát mép OOM**; trên GPU 4 GB local thì **chắc chắn OOM**. Nếu chọn 2×T4, Ray thấy 2 GPU → placement khác. | Mặc định `1.0` (tuần tự hóa) cho cả Kaggle và local; chỉ hạ xuống sau khi đo VRAM thật. Ghi vào `environment.json` của run. |
| **K4** | Giới hạn **12 h/phiên**. Hiện **không có resume** (H5): `run_id` có timestamp nên không tìm lại được run; chỉ lưu `final_params.npz` ở round cuối. | Resume là **bắt buộc**, không phải tùy chọn: thêm `--resume-run-dir` (hoặc `run_id` tiền định không timestamp), lưu `checkpoint/global_round_XXX.npz` **mỗi round**, và `round_metrics.csv` append (đã có). Không có K4 thì Tier 1 không thể chạy trên Kaggle. |
| **K5** | `/kaggle/working` ~20 GB. Mỗi run sinh `run_dir/clients/C*/round_N/weights/{last,best}.pt` ≈ 4 client × 10 round × ~12 MB ≈ **480 MB/run**, chưa kể plot của Ultralytics. | Thêm bước dọn: sau khi aggregate xong round N, xóa weights round N của client (giữ log/metrics). Ghi policy vào `README.md` của run. |
| **K6** | Ultralytics tải `yolov8n.pt` và (ở một số bản) tải model phụ cho **AMP check** `[NEEDS-VERIFICATION]` → cần **Internet = ON** trong notebook. | Hoặc bật Internet, hoặc pre-cache weights trong Kaggle Dataset và trỏ `weights` vào đó. Kiểm ở smoke, ghi vào `docs/KAGGLE_SETUP.md`. |
| **K7** | `YOLO_CONFIG_DIR` chưa đặt → Ultralytics ghi settings vào `~/.config/Ultralytics`, mất sau mỗi phiên và có thể cảnh báo quyền. | Đặt `YOLO_CONFIG_DIR=/kaggle/working/.ultralytics` ở cell đầu notebook. |
| **K8** | Cả 3 notebook chỉ gọi `--algorithm FedAvg` → kể cả sau khi sửa B1/B2, **Kaggle vẫn chỉ chạy A0**. | Sau B1/B2: thêm notebook (hoặc tham số hóa 03) để chạy `A1, A2b, A3, A4b`; mỗi ô ablation phải có một lệnh chạy được trên Kaggle. |
| **K9** | `docs/KAGGLE_SETUP.md:6` ghi "**Không dùng Kaggle cho G10 main experiments** (12h không đủ)" — trái với khuyến nghị dùng Kaggle cho Tier 1 ở §7. | Sau khi K4 xong, mâu thuẫn này phải được giải quyết bằng **ADR-004** (môi trường chạy main): resume theo round làm Kaggle khả thi cho main; nếu không làm K4 thì phải chọn môi trường khác. |

### 13.3 Tiêu chí "đã chạy được trên Kaggle" (definition of done cho mỗi lần sửa)

Một thay đổi code chỉ được coi là xong khi:

1. `pytest tests/ -q` xanh **trên máy local không có GPU/ultralytics** (toàn bộ test mới phải là numpy/mock-level).
2. Smoke trên Kaggle: `--run-class smoke` (2 round, batch 4) hoàn tất **không lỗi**, với **mỗi** algorithm bị ảnh hưởng bởi thay đổi đó.
3. `verify_artifacts` không báo thiếu — kể cả `aggregation_trace.yaml` khi algorithm là `ClassAwareAgg` (H6).
4. Dừng phiên giữa run rồi chạy lại phải **tiếp tục đúng round** (kiểm thử K4 bằng cách kill sau round 1).
5. `environment.json` của run có đầy đủ version (⇒ `environment.lock` phải được điền trước — đó là G1).
6. Dung lượng `/kaggle/working` sau run ≤ ngưỡng đã khai báo (K5).
