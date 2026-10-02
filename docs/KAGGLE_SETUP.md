# Hướng dẫn chạy FLOPS trên Kaggle (G1 → G5 feasibility)

> Cập nhật 2026-10-02, khớp với commit hiện tại trên `main`. Áp dụng cho **G1–G5 ở
> mức smoke/feasibility** (ADR-002). **Không** dùng Kaggle cho main experiment G10
> (phiên tối đa 12h, quota GPU ~30h/tuần).

---

## 0. Tổng quan — 4 phiên Kaggle

| Phiên | Notebook | Mục tiêu (gate) | Cần mang vào | Mang ra | Thời gian GPU (ước tính) |
|---|---|---|---|---|---|
| **1** | `01_smoke_G1_G3.py` | G1 môi trường · F1 trên stack đã pin · kiểm fp32 trên GPU (ADR-008) · smoke FL | dataset BDD100K | `environment.lock`, `F1_runtime_map.yaml`, run smoke | ~0,5–1h |
| **2** | `02_baseline_G2_G3.py` | G2 centralized · G3 FedAvg S0 | dataset | run G2 (**có checkpoint G2**), run G3 | ~3–5h |
| **3** | `03_missing_class_G4.py` (Cell 1–3, 5b, 6; `RUN_C1 = False`) | **F2** + **F3** (G5, bằng chứng G4) | dataset + output phiên 2 | `flops_export/F2`, `flops_export/F3` | ~1,5–2,5h |
| **4** | `03_missing_class_G4.py` (Cell 1–5, 7–8; `RUN_C1 = True`) | **C1**: S1b vs S1-Control-Matched, 3 seed (bằng chứng dự đoán G4) | dataset | 6 run FL + bảng ΔAP | ~6–9h → **chia 2 phiên** theo seed |

Thời gian là ước tính cho GPU T4 và chưa được đo (`[RESOURCE-BUDGET-UNRESOLVED]`).
Phiên 2 sẽ đo `t_epoch`/`t_eval` thật. **Ghi lại thời gian thực tế của từng phiên** vào
`research/plan/plan.md` §7.4 trước khi lên kế hoạch các phiên sau.

### Việc phải chốt trước khi chạy

| Trước phiên | Việc | Ở đâu |
|---|---|---|
| 2 | Xem lại `warmup_epochs=0` / `close_mosaic=0` của G2 centralized (100 epoch dùng chung giá trị đặt cho round FL 1 epoch; mặc định của Ultralytics là 3 / 10). Đổi thì cần ADR. | `scripts/train_centralized.py`, `configs/base_config.yaml` |
| 3 | **Chốt `τ_AP`** (đề xuất 0.01) cho luật quyết định F2/F3. Phải chốt **trước khi xem** bất kỳ output thật nào (CLAUDE.md §8). | `research/feasibility/F2/README.md`, `F3/README.md` |
| 3 | Duyệt ADR-008 (trọng số client: EMA fp32) | `research/decisions/ADR-008-*.md` |

---

## 1. Chuẩn bị một lần

### 1.1 Dataset BDD100K → Kaggle Dataset (private)

1. Tải từ https://bdd-data.berkeley.edu/portal.html:
   - `bdd100k_images_100k.zip` (~5,3 GB)
   - `bdd100k_det_20_labels_trainval.zip` (~55 MB)
2. Giải nén, **giữ đúng cấu trúc** (có thể xóa `test/` để giảm còn ~4,5 GB):
   ```
   bdd100k_kaggle/
   ├── images/100k/{train,val}/*.jpg
   └── labels/det_20/{det_train.json, det_val.json}
   ```
3. Kaggle → **Datasets → New Dataset**, title `bdd100k-flops`, **Private**, kéo thả thư mục `bdd100k_kaggle/`.
4. Các notebook mặc định đọc dataset tại:
   ```
   /kaggle/input/datasets/phdatt/bdd100k-flops/bdd100k_kaggle
   ```
   Nếu owner/slug của bạn khác, sửa biến `BDD100K_RAW` ở Cell 3 (notebook 01) và Cell 2 (notebook 02, 03).

### 1.2 Code

Repo `https://github.com/PhanHongDatt/FLOPS` đang **public**, nên Kaggle clone thẳng được, không cần token.
Trước mỗi phiên, kiểm tra trên máy local rằng mọi thay đổi đã được push (`git status` sạch, `git log origin/main` chứa commit mới nhất).

### 1.3 Đưa notebook lên Kaggle

Các notebook trong repo ở dạng script `# %%` (mỗi khối `# %%` là một cell). Có hai cách:

- **Cách A (khuyến nghị): chuyển sang `.ipynb` rồi import**
  ```powershell
  pip install jupytext
  jupytext --to ipynb notebooks/01_smoke_G1_G3.py notebooks/02_baseline_G2_G3.py notebooks/03_missing_class_G4.py
  ```
  Kaggle → **Code → New Notebook → File → Import Notebook** → chọn file `.ipynb`.
  Các file `.ipynb` sinh ra này không cần commit.
- **Cách B:** tạo notebook trống rồi dán từng khối `# %%` vào từng cell.

---

## 2. Quy trình chung cho mỗi phiên

1. **Settings** (panel phải): Accelerator **GPU T4 x2** (hoặc P100) · Internet **On** · Persistence **No**.
2. **Add Data**: `bdd100k-flops`. Từ phiên 3 trở đi, thêm **output của phiên trước**: Add Data → *Your Work* → chọn notebook → version đã lưu.
   Output sẽ được mount dưới `/kaggle/input/<tên-notebook>/...`.
3. **Cell 0 (thêm vào đầu mọi notebook)**: clone repo.
   ```python
   import subprocess, pathlib
   if not pathlib.Path("/kaggle/working/FLOPS").exists():
       subprocess.check_call(["git", "clone", "--depth", "1",
                              "https://github.com/PhanHongDatt/FLOPS.git", "/kaggle/working/FLOPS"])
   print(subprocess.check_output(["git", "-C", "/kaggle/working/FLOPS", "log", "--oneline", "-1"], text=True))
   ```
   Ghi lại commit hash được in ra. Hash này cũng nằm trong `environment.json` của mọi run.
4. Chạy các cell theo thứ tự (Run All, hoặc từng cell với notebook 03).
5. **Lưu kết quả**: **Save Version → Save & Run All (Commit)**, hoặc **Quick Save** kèm *Save output*.
   Mọi thứ cần giữ phải nằm trong `/kaggle/working/flops_export/` (các notebook đã tự copy vào đó).
   `/kaggle/working` **mất hết** khi phiên kết thúc.

---

## 2b. Chạy từ VS Code bằng Kaggle CLI (không cần mở trình duyệt)

Đã cài đặt sẵn trên máy (2026-10-02): `kaggle` CLI 2.2.4, `jupytext`; token ở
`C:\Users\hogda\.kaggle\access_token` (tài khoản `phdatt`). File token nằm **ngoài repo** và
bị chặn bởi `.gitignore`. **Không bao giờ commit token**, vì repo đang public. Nếu token lộ: Kaggle →
Settings → API → tạo token mới rồi ghi đè file trên.

Các phiên được khai báo trong `kaggle/sessions.yaml`:

| Phiên | Kernel trên Kaggle | Notebook | Ghi chú |
|---|---|---|---|
| `s1` | `phdatt/flops-s1-smoke-g1` | 01 | G1 + F1 + kiểm fp32 + smoke |
| `s2` | `phdatt/flops-s2-baseline-g2-g3` | 02 | sinh checkpoint G2 |
| `s3` | `phdatt/flops-s3-f2-f3` | 03, `RUN_C1=False` | tự gắn output của `s2` (`kernel_sources`) để lấy checkpoint G2 |
| `s4a` / `s4b` | `phdatt/flops-s4a-c1-seed42` / `...-s4b-c1-seed123-2024` | 03 | C1 chia seed |

Mỗi phiên dùng 3 lệnh, gõ ở terminal VS Code tại `D:\FLOPS`:

```powershell
python scripts/kaggle_run.py s1 push     # đẩy lên, Kaggle chạy nền trên GPU T4 x2
python scripts/kaggle_run.py s1 status   # queued / running / complete / error
python scripts/kaggle_run.py s1 output   # tải output + log về kaggle/output/s1/
```

- `push` chỉ chạy khi working tree **sạch** và commit đã **push lên GitHub**, vì notebook trên Kaggle
  clone repo và `checkout` đúng commit đó. Commit hash nằm ở ô đầu notebook và trong `environment.json` của mọi run.
- `python scripts/kaggle_run.py s1 build` chỉ tạo `kaggle/build/s1/` (notebook + `kernel-metadata.json`)
  để xem trước, không đẩy lên.
- Notebook tự dò đường mount dataset (`src/utils/kaggle_paths.py`), vì kernel đẩy qua API có thể thấy
  `/kaggle/input/bdd100k-flops/` thay vì `/kaggle/input/datasets/phdatt/bdd100k-flops/`.
- Ô cuối tự xóa `data/bdd100k_yolo` để output nhẹ; giữ lại `flops_export/` và `FLOPS/artifacts/`.
- Theo dõi log trực tiếp: mở `https://www.kaggle.com/code/phdatt/<slug>` (link được in ra sau `push`).
- Chạy `s3` **sau khi `s2` đã complete**, vì `kernel_sources` lấy output của version mới nhất của `s2`.

---

## 3. Phiên 1: notebook 01 (G1, F1, ADR-008, smoke)

| Cell | Việc | Kết quả mong đợi |
|---|---|---|
| 1 / 1b | Gỡ TensorFlow; cài torch 2.7.1+cu128, ultralytics 8.3.253, flwr 1.21.0, mlflow<3; `pip install -e` repo | in `✅ Repo installed, environment intact.` |
| 2 | Audit version + ghi `environment.lock` (pip freeze) | `✅ All pinned versions match ADR-001` (nếu lệch: ghi addendum ADR-002) |
| 3 | Đường dẫn; kiểm dataset đã mount | không lỗi `BDD100K dataset not mounted` |
| **3b** | `verify_map.py` + `check_fp32_upload.py --device 0` | `OK: 6 class-head keys, 780 class-specific elements` **và** `OK: trained client weights are fp32` (dòng trước đó có `amp=True`) |
| 4 | Convert BDD100K → YOLO | `✅ YOLO dataset at /kaggle/working/data/bdd100k_yolo` |
| 5 | Partition S0 IID | in class counts của 4 client |
| 6 | Smoke FedAvg (2 round) | `✅ Smoke FL run complete.` Qua được round 1 là xác nhận các bản sửa ADR-008 (nc=4, không fuse) chạy đúng trên GPU |
| 7 | Kiểm artifact §21 | không thiếu file |
| 8 | Export → `flops_export/smoke_*` (gồm `environment.lock`, `F1_runtime_map.yaml`) | |

**Nếu Cell 3b báo FAIL thì dừng.** Gửi lại log cho mình xem; đừng chạy tiếp phiên 2, vì mọi Δθ của F2/F3 sẽ sai.

**Sau phiên 1** (trên máy local):
1. Tải output → chép `environment.lock` vào gốc repo, và chép `F1_runtime_map.yaml` thành `research/feasibility/F1/runtime_map.yaml`.
2. Cập nhật `research/gates.yaml`: G1 → `passed` (kèm ngày); G5 notes: F1 đã xác nhận trên stack đã pin.
3. Commit + push. Bạn có thể nhờ mình làm bước này: chỉ cần đưa file đã tải về.

---

## 4. Phiên 2: notebook 02 (G2 centralized, G3 FedAvg)

Notebook tự cài lại môi trường (Cell 1). Nếu chưa có dữ liệu YOLO và partition S0 thì Cell 2 tự convert lại (~5 phút).

| Cell | Việc | Ghi chú |
|---|---|---|
| 3 | **G2** `train_centralized.py`, feasibility (10 epoch, batch 8) | **Checkpoint G2** nằm ở `artifacts/runs/G2-centralized_feasibility_seed42_centralized/checkpoint/train/weights/best.pt` |
| 4 | **G3** FedAvg S0, 5 round | `round_metrics.csv` có AP theo lớp từng round |
| 5–6 | Kiểm §21, ghi registry | |
| 7 | Export → `flops_export/baseline_*/G2`, `/G3` | thư mục G2 chứa luôn checkpoint |

**Ghi lại** thời gian một epoch (G2) và một lượt eval để điền ngân sách §7.4.
**Lưu version có output.** Phiên 3 cần output này để lấy checkpoint G2.

Kiểm nhanh trước khi rời phiên: AP50 của **bus** trong `metrics.csv` của G2 phải **> 0**. Nếu bằng 0 thì F2/F3 không đo được gì (script sẽ cảnh báo).

---

## 5. Phiên 3: notebook 03, chỉ F2 + F3 (G5)

**Điều kiện:** đã chốt `τ_AP`; đã Add Data **output của phiên 2**.

1. Chạy Cell 1 (môi trường), Cell 2 (đường dẫn), Cell 3 (partition S1b + matched control).
   - Cell 3 in `S1-Control-Matched matched = True/False`. `False` vẫn chạy được, nhưng mọi kết luận chỉ là *quan sát*. F3 tự chép `match_report.yaml` vào thư mục output của nó.
2. Ở Cell 4, đặt **`RUN_C1 = False`**. Cell 4 sẽ in `[SKIP]` và Cell 5 cho bảng rỗng. Có thể bỏ qua cả hai.
3. **Cell 5b (F2)**: dòng đầu in `G2_WEIGHTS: ... (exists)`. Notebook tự tìm checkpoint trong `artifacts/runs`, `flops_export` và `/kaggle/input/**/G2/...`.
   Nếu in `(NOT FOUND)`, gán tay `G2_WEIGHTS = Path("/kaggle/input/<...>/G2/checkpoint/train/weights/best.pt")`.
   - 35 lần đánh giá trên cùng tập val con 2000 ảnh → `flops_export/F2/<run>/summary.yaml` (ma trận hiệu ứng).
4. **Cell 6 (F3)**: cặp C0 S1b vs C0 matched, 3 seed → `flops_export/F3/<run>/summary.yaml`.
   - Log có `target_known_before=True` là đúng. `False` nghĩa là checkpoint chưa nhận ra bus, khi đó kết quả là *inconclusive*, không phải kết quả âm.
5. Cell 8: export. Lưu version có output.

**Sau phiên 3:** gửi mình `F2/.../summary.yaml`, `F3/.../summary.yaml`, `metrics.csv` và log. Mình sẽ áp luật đã pre-register và soạn **ADR-003** (Gate A/B/C).

---

## 6. Phiên 4 (có thể chia 2): notebook 03, C1 (bằng chứng dự đoán G4)

1. Cell 1–3 như phiên 3.
2. Cell 4: `RUN_C1 = True`. Để vừa trong giới hạn 12h, **chia seed**:
   - Phiên 4a: `SEEDS = [42]`, khoảng 2–3h
   - Phiên 4b: `SEEDS = [123, 2024]`, khoảng 4–6h
3. Cell 5: bảng ΔAP theo lớp (S1b − Control). Lớp bị thiếu: **bus** (C0, C1) và **truck** (C2).
4. Cell 7–8: registry + export. Mỗi phiên lưu version riêng.

Nếu kernel bị ngắt giữa một run nhưng phiên vẫn còn, chỉ cần chạy lại Cell 4. `run_one` truyền `--resume`, nên run đã xong sẽ được bỏ qua, run dở dang chạy tiếp từ checkpoint round cuối. Sang phiên mới thì `/kaggle/working` đã bị xóa, nên chạy lại seed đó từ đầu.

Chạy C1 bằng sweep (`scripts/sweep_experiments.py --sweep configs/sweeps/c1_h1_missing_class.yaml`) cũng được, nhưng phải trỏ `--base-dir /kaggle/working/data --global-data-yaml /kaggle/working/data/bdd100k_yolo/data.yaml`, vì notebook sinh partition vào `/kaggle/working/data/partitions`.

---

## 7. Sau toàn bộ: điều kiện để đóng gate

| Gate | Đóng khi | Bằng chứng |
|---|---|---|
| G1 | phiên 1 pass, `environment.lock` đã commit | `environment.lock` |
| G2 / G3 | phiên 2 pass, artifact §21 đầy đủ | `flops_export/baseline_*` |
| G5 | F1 trên stack đã pin + F2 + F3 → ADR-003 quyết định A/B/C | F1 `runtime_map.yaml`, F2/F3 `summary.yaml` |
| G4 | F3 (tham số + dự đoán) **và** C1 (ΔAP bus nhất quán qua 3 seed), control `matched: true` | F3 + C1 + `match_report.yaml` |

Không xóa kết quả âm. Không bỏ seed xấu (CLAUDE.md §20).

---

## 8. Xử lý sự cố

| Triệu chứng | Nguyên nhân / cách xử lý |
|---|---|
| Cell 3b: `FAIL: uploaded weights are not trained fp32 values` | Thứ tự callback của Ultralytics khác với bản đã kiểm. **Dừng**, gửi log (ADR-008). |
| Cell 3b: `Expected exactly 6 class-head params` | Version ultralytics khác 8.3.253 hoặc model không phải nc=4. Kiểm lại Cell 1/2. |
| `size mismatch for model.22.cv3...` | Code cũ: kiểm commit hash ở Cell 0 là bản mới nhất. |
| `Parameter count mismatch ... 127 state_dict entries` | Model bị fuse: cũng là dấu hiệu code cũ (đã sửa trong ADR-008). |
| `TypeError: 'str' object is not a mapping` ở `prepare_run` | Code cũ (lỗi đọc `environment.lock` dạng pip freeze, đã sửa 2026-10-02). |
| `control side ... must have positive target boxes` (F3) | Partition control chưa được sinh, hoặc không có ảnh donor. Chạy lại Cell 3 và xem `match_report.yaml`. |
| `Checkpoint has zero AP50 for 'bus'` (F2/F3) | Checkpoint G2 chưa học được bus. Kiểm `metrics.csv` của G2; có thể cần train lâu hơn (cần ADR). |
| `G2_WEIGHTS ... (NOT FOUND)` | Chưa Add Data output của phiên 2, hoặc đường dẫn khác. Gán tay `G2_WEIGHTS`. |
| CUDA out of memory | Giảm `train.batch_size` trong `configs/experiments/feasibility.yaml`, hoặc đặt `nbs` để giữ batch hiệu dụng (ghi vào config của run). |
| Cảnh báo version ở Cell 2 | Ghi addendum ADR-002 trước khi sang G2. |
| `pip` báo lỗi protobuf / mlflow | Cell 1 đã gỡ TensorFlow chưa? `mlflow` phải `<3.0`, `protobuf<5` (ADR-002-A1). |
| Notebook crash giữa chừng | Lưu version thường xuyên; restart rồi chạy lại từ cell đã pass gần nhất. |

---

## Tham chiếu

ADR-001 (version) · ADR-002 / ADR-002-A1 (Kaggle, pin mlflow/protobuf) · ADR-007 (cấu hình theo client)
· ADR-008 (model nc=4, trọng số fp32, val stub) · `research/feasibility/F1|F2|F3/README.md`
· `research/plan/comparison_design.md` (C1) · CLAUDE.md §7 (gate), §8 (F1–F3), §14 (ngân sách), §21 (artifact)
