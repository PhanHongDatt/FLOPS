# Kaggle Setup — FLOPS G1 → G4 Execution Environment

Hướng dẫn từ zero → chạy được `notebooks/01_smoke_G1_G3.py` trên Kaggle.
Theo **ADR-002** (Kaggle deployment). Áp dụng cho gates G1–G4 only.

**Không dùng Kaggle cho G10 main experiments** (12h session limit không đủ).

---

## 0. Prerequisites

- [ ] Tài khoản Kaggle đã verify phone (yêu cầu để bật GPU + Internet)
- [ ] Tài khoản BDD100K đã đăng ký tại https://bdd-data.berkeley.edu/ (để tải dataset)
- [ ] Repo FLOPS đã commit các thay đổi mới nhất (locally)
- [ ] Local disk trống ~10 GB (để giải nén BDD100K trước khi upload)

---

## 1. Tải BDD100K (local, ~1 giờ tùy mạng)

Truy cập https://bdd-data.berkeley.edu/portal.html → login → tải 2 file:

| File | Size | Mô tả |
|---|---|---|
| `bdd100k_images_100k.zip` | ~5.3 GB | 100k ảnh (train 70k + val 10k + test 20k) |
| `bdd100k_det_20_labels_trainval.zip` | ~55 MB | Detection labels JSON |

Giải nén giữ đúng structure (quan trọng — `scripts/prepare_bdd100k.py` expect đúng path này):

```
bdd100k/
├── images/
│   └── 100k/
│       ├── train/*.jpg
│       ├── val/*.jpg
│       └── test/*.jpg   ← có thể xóa để tiết kiệm dung lượng, không dùng
└── labels/
    └── det_20/
        ├── det_train.json
        └── det_val.json
```

**Optional — giảm dung lượng upload:**
```powershell
# Xóa test images (không dùng cho G1-G4)
Remove-Item -Recurse -Force bdd100k/images/100k/test
```

Sau khi xóa test: dataset còn ~4.5 GB.

---

## 2. Upload BDD100K lên Kaggle Dataset (private, ~30–60 phút)

**Option A — Kaggle UI (khuyến nghị):**

1. Vào https://www.kaggle.com/datasets → **New Dataset**
2. Title: `bdd100k-flops` (hoặc tên tùy chọn — nhớ để dùng ở cell notebook)
3. Visibility: **Private**
4. Kéo thả toàn bộ folder `bdd100k/` vào drop zone
5. Chờ upload xong (phụ thuộc mạng — thường 30–60 phút cho ~4.5 GB)
6. **Create**

**Option B — Kaggle CLI:**

```powershell
pip install kaggle
# Đặt kaggle.json vào ~/.kaggle/
kaggle datasets init -p ./bdd100k
# Edit ./bdd100k/dataset-metadata.json — set title, id, license
kaggle datasets create -p ./bdd100k -r zip
```

**Sau khi upload xong**, lưu lại slug dataset dạng `<username>/bdd100k-flops` (dùng ở step 5).

---

## 3. Upload FLOPS repo lên Kaggle (chọn 1 trong 2)

### Option 3A — GitHub private repo (khuyến nghị nếu bạn có repo GitHub)

1. Push repo lên GitHub private:
   ```powershell
   git remote add origin git@github.com:<user>/FLOPS.git
   git push -u origin main
   ```
2. Tạo GitHub Personal Access Token với scope `repo` (Settings → Developer settings → PATs)
3. Trong Kaggle notebook sẽ clone bằng: `git clone https://<TOKEN>@github.com/<user>/FLOPS.git`

### Option 3B — Upload as Kaggle Dataset

1. Zip repo local (loại bỏ `data/`, `artifacts/`, `.git/` để giảm size):
   ```powershell
   Compress-Archive -Path src, scripts, configs, tests, notebooks, research, docker, requirements.txt, environment.lock, CLAUDE.md, README.md -DestinationPath flops_repo.zip
   ```
2. Upload zip lên Kaggle Datasets tương tự Step 2 với title `flops-repo`
3. Trong notebook mount tại `/kaggle/input/flops-repo/` rồi copy vào `/kaggle/working/FLOPS/`

---

## 4. Tạo Kaggle Notebook

1. Vào https://www.kaggle.com/code → **New Notebook**
2. **Settings** (bên phải):
   - Accelerator: **GPU T4 x2** (hoặc **P100** nếu có)
   - Internet: **On** (bắt buộc — để pip install)
   - Persistence: **No** (không cần)
   - Environment: **Latest** (mặc định)
3. **Add Data** (bên phải):
   - Search dataset của bạn: `bdd100k-flops` → Add
   - Nếu dùng Option 3B: search `flops-repo` → Add
4. Notebook sẽ mount tại:
   - `/kaggle/input/bdd100k-flops/bdd100k/` (BDD100K)
   - `/kaggle/input/flops-repo/` (nếu Option 3B)

---

## 5. Cell setup ban đầu (paste vào cell đầu tiên)

### Nếu Option 3A (GitHub clone):

```python
import subprocess, os
# Thay <TOKEN> và <user> bằng giá trị thực
subprocess.check_call([
    "git", "clone",
    "https://<TOKEN>@github.com/<user>/FLOPS.git",
    "/kaggle/working/FLOPS"
])
print("Cloned.")
```

### Nếu Option 3B (dataset copy):

```python
import shutil
from pathlib import Path

src = Path("/kaggle/input/flops-repo")
dst = Path("/kaggle/working/FLOPS")
if not dst.exists():
    # Nếu upload là zip → cần unzip; nếu upload folder → chỉ copy
    if (src / "flops_repo.zip").exists():
        shutil.unpack_archive(src / "flops_repo.zip", dst)
    else:
        shutil.copytree(src, dst)
print("Repo ready at", dst)
```

### Verify dataset mount:

```python
from pathlib import Path
BDD = Path("/kaggle/input/bdd100k-flops/bdd100k")
assert (BDD / "images/100k/train").exists(), f"BDD100K structure wrong at {BDD}"
assert (BDD / "labels/det_20/det_train.json").exists(), "Missing det_train.json"
print("✅ BDD100K mounted correctly")
```

Nếu assertion fail → dataset upload sai structure. Quay lại Step 1–2 fix.

---

## 6. Chạy notebook 01

Copy nội dung `notebooks/01_smoke_G1_G3.py` vào các cell của Kaggle notebook (mỗi block `# %%` = 1 cell). Chạy tuần tự **Run All** hoặc từng cell một.

**Nếu path Kaggle dataset khác với notebook default:**

Notebook default expect `/kaggle/input/bdd100k/` (không có prefix `-flops`). Sửa Cell 3 của notebook 01:

```python
BDD100K_RAW = Path("/kaggle/input/bdd100k-flops/bdd100k")  # ← sửa theo tên dataset của bạn
```

**Thời gian ước tính (T4 GPU):**

| Cell | Nội dung | Thời gian |
|---|---|---|
| 1 | pip install torch + libs | ~5–8 phút |
| 2 | Environment audit + freeze | ~10 giây |
| 3 | Path setup | ~1 giây |
| 4 | prepare_bdd100k.py (100k ảnh, symlink) | ~3–5 phút |
| 5 | generate_partition.py (scan labels) | ~30 giây |
| 6 | Smoke FL run (2 rounds, 4 clients, 1 epoch, batch 4) | ~10–20 phút |
| 7 | Verify §21 artifacts | ~1 giây |
| 8 | Export to /kaggle/working/flops_export | ~30 giây |
| 9 | Summary | ~1 giây |

**Tổng: ~20–35 phút** — hoàn toàn nằm trong Kaggle 12h session.

---

## 7. Export artifacts (sau khi notebook 01 xong)

Kaggle session ephemeral — mọi thứ trong `/kaggle/working/` sẽ mất sau khi close notebook.
Để giữ artifacts + `environment.lock`:

1. Click **Save Version** góc trên phải notebook
2. Chọn **Quick Save** (Advanced options → Save output)
3. Kaggle sẽ snapshot `/kaggle/working/flops_export/` vào Output

**Hoặc** tạo Output Dataset:

1. Trong notebook thêm cell cuối:
   ```python
   # Nội dung /kaggle/working/flops_export/ sẽ tự thành Kaggle Output
   # Sau khi Save Version, download về local từ:
   # https://www.kaggle.com/<user>/notebook/output
   ```
2. Tải `flops_export.zip` về local
3. Extract → commit `environment.lock` vào repo:
   ```powershell
   Copy-Item flops_export/environment.lock D:\FLOPS\environment.lock
   git add environment.lock
   git commit -m "chore(env): populate environment.lock from Kaggle smoke run"
   ```

---

## 8. Update gate status (local, sau khi notebook 01 pass)

Sửa `research/gates.yaml`:

```yaml
G1:
  name: Environment reproducible
  status: passed                        # ← đổi từ in_progress
  passed_date: "2026-08-21"             # ← ngày pass
  notes: >
    environment.lock populated from Kaggle T4 smoke session (see ADR-002).
    All pinned versions per ADR-001 verified in Kaggle environment.
```

Commit:
```powershell
git add research/gates.yaml environment.lock
git commit -m "chore(gates): G1 passed via Kaggle smoke session (ADR-002)"
```

---

## 9. Troubleshooting

### "CUDA out of memory" trong FL run
- Giảm `batch_size` trong `configs/experiments/smoke.yaml` từ 4 → 2
- Hoặc giảm `client_resources={"num_gpus": 0.25}` → `0.5` trong `server.py` (nhưng phải giảm num_clients)

### "det_train.json not found"
- Verify path Kaggle dataset: `!ls /kaggle/input/bdd100k-flops/bdd100k/labels/det_20/`
- Nếu structure khác, sửa `BDD100K_RAW` trong notebook Cell 3

### `pip install ultralytics` bị lỗi dependency
- Kaggle preinstalled numpy có thể conflict. Thử:
  ```python
  !pip install --force-reinstall ultralytics==8.3.253
  ```

### FL run treo ở round 0
- Thường do `client_resources` conflict với GPU quota. Kiểm tra:
  ```python
  import torch
  print(torch.cuda.memory_allocated() / 1e9, "GB used")
  ```

### Version mismatch warning ở Cell 2
- Kaggle preinstall torch 2.x thấp hơn 2.7.1. Cell 1 reinstall nhưng có thể partial. Xem `environment.lock` → nếu version ≠ ADR-001, cần **ghi addendum vào ADR-002** trước khi tiến sang G2+.

### Notebook 01 pass nhưng `per_class_metrics.csv` missing
- Verify Cell 6 pass `--global-data-yaml` (đã có sẵn trong notebook mới nhất). Nếu vẫn missing, check log warning "run_fl_server called without global_data_yaml" trong `run.log`.

### Kaggle "This notebook has crashed" giữa chừng
- Save version thường xuyên. Restart kernel + run từ cell cuối cùng đã pass.
- Nếu do OOM: giảm batch/rounds.

---

## 10. Sau notebook 01 pass — tiếp theo

Theo thứ tự:

| # | Notebook | Gate | Thời gian ước tính (T4) |
|---|---|---|---|
| 1 | 01_smoke_G1_G3.py | G1 (env) | ~30 phút ✅ |
| 2 | 02_baseline_G2_G3.py | G2 + G3 feasibility | ~2–3 giờ |
| 3 | 03_missing_class_G4.py | G4 partial (3 seeds × 2 scenarios) | ~4–6 giờ |

Notebook 02 và 03 có thể chạy chung 1 session hoặc split. Notebook 03 cần commit `final_params.npz` về Kaggle Dataset để sau này notebook parameter analysis dùng lại (F3 evidence).

---

## Reference

- ADR-001: environment versions (`research/decisions/ADR-001-environment-versions.md`)
- ADR-002: Kaggle deployment (`research/decisions/ADR-002-kaggle-deployment.md`)
- CLAUDE.md §5: single source of truth (environment)
- CLAUDE.md §7: stage gates
- CLAUDE.md §14: compute budget + run classes
- CLAUDE.md §21: artifact contract
