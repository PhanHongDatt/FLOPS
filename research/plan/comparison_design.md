# Thiết kế so sánh & đánh giá — C0–C9

**Ngày:** 2026-10-02 · **Phụ thuộc:** `research/plan/plan.md` v2 (§5.2 kịch bản, §7.1 ma trận, §7.3 quy tắc chọ ρ, §8.2 giao thức báo cáo), ADR-007.

Tài liệu này biến ma trận thí nghiệm thành **các phép so sánh có quy tắc quyết định đặt trước**. Mỗi comparison `Cx` có một file sweep tương ứng trong `configs/sweeps/` để chạy, và được tổng hợp bằng `scripts/compare_runs.py`.

> **Quy tắc bất biến:** mọi quy tắc quyết định dưới đây phải được **chốt trước khi xem kết quả**. Sửa quy tắc sau khi đã thấy số là vi phạm CLAUDE.md §20 và làm mất giá trị của toàn bộ so sánh.

---

## 0. Cách chạy baseline **cùng** phương pháp

Một lần gọi `run_fl_experiment.py` = **một arm**. Baseline được chạy bởi **cùng một sweep**, trên **cùng partition**, với **cùng bộ seed** và **cùng evaluator** — đó mới là điều §14 đòi hỏi, chứ không phải chạy đồng thời trong một process.

```bash
# 1. Xem kế hoạch trước (không chạy gì)
python scripts/sweep_experiments.py --sweep configs/sweeps/c6_full_ablation.yaml \
    --base-dir /kaggle/working --dry-run

# 2. Chạy — tuần tự, bỏ qua cell đã xong, tự --resume cell dở
python scripts/sweep_experiments.py --sweep configs/sweeps/c6_full_ablation.yaml \
    --base-dir /kaggle/working

# 3. Tổng hợp — TỪ CHỐI nếu vi phạm §14/§20
python scripts/compare_runs.py --arms A0 A1 A3 A2b_rho0.25 A4b_rho0.25 \
    --baseline A0 --run-class main --partition-id s1b_bus_seed42 \
    --out artifacts/comparisons/c6
```

Cell chạy **tuần tự** là cố ý: với `client_num_gpus: 1.0` thì ngay cả client trong một round đã bị tuần tự hóa; chạy hai arm song song chỉ dẫn tới OOM (plan.md §13 K3). Tuần tự + resume được là thứ sống sót qua giới hạn 12 h của Kaggle.

### Guard tự động trong `compare_runs.py`
Mặc định **từ chối** xuất bảng nếu: các arm khác partition · khác giao thức (`num_rounds, local_epochs, num_clients, fraction_fit, batch_size, image_size, lr0, conf, iou, weights`) · không cùng bộ seed · một arm kết thúc ở round khác · `run_class=smoke`. Dùng `--allow-invalid` thì output bị đóng dấu `COMPARISON_INVALID.md`. (`fraction_evaluate` **không** nằm trong danh sách vì nó không ảnh hưởng đường báo cáo.)

### Giới hạn đã khai báo
Partition seed **giữ cố định 42**; 3 seed (42/123/2024) chỉ thay đổi ngẫu nhiên **huấn luyện**. Vậy `std` báo cáo phản ánh nhiễu huấn luyện, **không** phản ánh nhiễu phân hoạch. Đây là cách đọc chặt hơn của §14 ("same partition seed across compared methods") và phải ghi rõ trong báo cáo.

---

## 1. Bảng so sánh

| ID | Câu hỏi | Giả thuyết | Arms | Kịch bản | Seed | Sweep | Gate |
|---|---|---|---|---|---|---|---|
| **C0** | Code có chạy không? | — | A0, A1, A3, A2a, A4a | S1b | 1 | `c0_plumbing_smoke` | — |
| **C1** | Thiếu lớp có gây suy giảm? | **H1** | A0 | S1b **vs** S1-Control | 3 | `c1_h1_missing_class` | G4 |
| **C2** | Class-count có giải thích hết? | — | A0, A1 | S1b | 3 | ⊂ `c3_server_only` | — |
| **C3** | Server-only có khác baseline? | **H3** | A0, A1, A3 | S1b | 3 | `c3_server_only` | G7 |
| **C4** | Luật no-contributor có tác dụng? | **H3** | A0, A1, A3 | **S1d** | 3 | `c4_no_contributor` | G7 |
| **C5a** | ρ nào? | **H2** | A0, A2b×{0, 0.25, 1} | S1b | 1 | `c5_rho_sweep` | G6 |
| **C5** | Client-only có cải thiện? | **H2** | A0, A2b(ρ\*) | S1b | 3 | ⊂ `c6_full_ablation` | G6 |
| **C6** | Full method & ablation | H2+H3 | A0, A1, A3, A2b(ρ\*), A4b(ρ\*) | S1b | 3 | `c6_full_ablation` | G8, G9 |
| **C7** | Có làm hại IID? | — | A0, A4b(ρ\*) | S0 | 3 | `c7_iid_no_harm` | G10 |
| **C8** | So với baseline FL khác | — | A0, SCAFFOLD, FedNova | S1b | 1 | `c8_reference_baselines` | §11 |
| **C9** | Độ nhạy τ_elig | — | A0, A1, A3(τ=1), A3(τ=50) | S1b | 1 | `c9_tau_elig_sensitivity` | — |

Tổng Tier 1 = C6 (15) + C7 (6) = **21 run**, khớp plan.md §7.2.

---

## 2. Quy tắc quyết định (pre-register)

Ký hiệu: `AP_bus` = `AP50_bus` ở **vòng cuối**; `std` = sample std qua 3 seed; `δ_abs` = ngưỡng tuyệt đối tối thiểu, **chốt = 0.01 AP** (1 điểm AP) để một chênh lệch bé hơn sai số làm tròn không bị gọi là hiệu ứng.

### C1 — H1: hiệu ứng Missing-Class (G4)
- **Đo:** `scenario_delta` ghép theo seed: `ΔAP_bus = AP_bus(S1b) − AP_bus(S1-Control)`.
- **Ủng hộ H1 khi ĐỒNG THỜI:**
  1. `mean(ΔAP_bus) < −max(2·std(ΔAP_bus), δ_abs)`;
  2. bảng matched-pair cho thấy hai nhóm khớp trong ±5% về số ảnh, tổng bbox và số bbox của **các lớp không phải target**;
  3. các lớp không-target **không** suy giảm cùng mức — nếu car/truck cũng giảm tương đương thì đó là hiệu ứng data-size/composition, không phải hiệu ứng lớp;
  4. **có bằng chứng tham số từ F3** (‖Δθ‖ của nhóm class-associated lệch rõ so với nhóm control). CLAUDE.md §8: chỉ AP là **không đủ**.
- **Bác bỏ/làm yếu khi:** chênh lệch nằm trong `2·std`; hoặc matched-pair thất bại; hoặc (3) sai.
- **Nếu H1 không được ủng hộ:** báo cáo kết quả âm, và chuyển sang S1c/S1d như biến độ nặng — **không** sửa thí nghiệm để ép ra kết luận dương (§22).

### C2/C3 — H3 phần "class-count"
- **Kỳ vọng đặt trước:** `A3 ≡ A1` trên S1b, vì τ_elig=1 + fraction_fit=1.0 làm tập eligible trùng tập client có count khác 0 (plan.md §6.4).
- **Quy tắc:** nếu `max|A3 − A1|` trên mọi metric `> δ_abs` → **coi là bug**, đi điều tra `aggregation_trace.yaml`, **không** báo cáo như phát hiện.
- **C2:** nếu `ΔAP_bus(A1 − A0) ≥` phần lớn `ΔAP_bus(A4b − A0)` thì class-count baseline **giải thích được** phần tăng → vi phạm điều kiện §22.6, và đóng góp của phương pháp phải được phát biểu lại.

### C4 — H3 phần "no-contributor" (nơi A3 ≠ A1)
- **Đo:** `AP_bus` vòng cuối trên S1d, cộng với số đếm action trong `aggregation_trace.yaml`.
- **Ủng hộ khi ĐỒNG THỜI:**
  1. trace cho `action: keep_global, reason: no_valid_contributor` ở **mọi** round, **mọi** khóa cls head, cho lớp bus;
  2. `AP_bus(A3)` ở vòng cuối **không thấp hơn** `AP_bus` ở round 0 quá `max(2·std, δ_abs)` → kiến thức global được bảo toàn;
  3. `AP_bus(A0)` và `AP_bus(A1)` **suy giảm** so với round 0 quá `max(2·std, δ_abs)`.
- **Bác bỏ khi:** A0/A1 cũng không suy giảm (⇒ hàng cls của lớp vắng vốn không drift, luật no-contributor không giải quyết vấn đề gì thực), hoặc A3 cũng suy giảm (⇒ suy giảm đến từ tham số **dùng chung**, không từ hàng cls — đây là một trong các kết cục âm hợp lệ ở §22).

### C5a → C5 — H2, chọn ρ
- **C5a (1 seed, feasibility):** chạy ρ ∈ {0, 0.25, 1}. Quy tắc đề bạt **khóa trước**: `ρ* = argmax AP_bus(vòng cuối)`; hòa trong `δ_abs` → chọn ρ **lớn hơn** (can thiệp nhẹ hơn); nếu `FP_bus(ρ*) > 1.5 × FP_bus(ρ=1)` thì **loại** ρ đó và lấy ứng viên kế tiếp.
- **Control nội bộ:** `A2b(ρ=1)` phải trùng `A0` trong sai số số học. Lệch ⇒ bug.
- **C5 (3 seed, main):** ủng hộ H2 khi `mean(ΔAP_bus(A2b − A0)) > max(2·std, δ_abs)`, **và** `FP_bus` không tăng quá 1.5×, **và** suy giảm ở lớp không-target ≤ `max(2·std, δ_abs)`.
- **Báo cáo cả ba giá trị ρ** bất kể kết quả (§20).

### C6 — Full method (G9)
Phương pháp chỉ được gọi là **promising** (không phải "proven") khi **cả 6** điều kiện §22 thỏa:
1. `ΔAP_bus(A4b − A0) > max(2·std, δ_abs)`;
2. dấu của `ΔAP_bus` **nhất quán ở cả 3 seed** (không phải 2/3);
3. suy giảm ở lớp không-target và ở `mAP50` được báo cáo và ≤ `max(2·std, δ_abs)`;
4. C7 đạt (không hại IID);
5. ablation ủng hộ cơ chế: `A4b > A3` **và** `A4b > A2b`, tức tổ hợp hơn từng phần;
6. `A1` **không** giải thích hết (C2).

Nếu (5) cho `A4b ≈ A3` thì kết luận là "server-only bằng full method" — một kết cục âm hợp lệ, phải báo cáo đúng như vậy.

### C7 — Không làm hại IID
- **Đạt khi:** `mAP50(A4b) ≥ mAP50(A0) − 2·std(mAP50(A0))`, với `std` lấy từ 3 seed của A0 trên S0, **chốt trước khi chạy**.

### C8 — Baseline tham chiếu
Chỉ để đặt bối cảnh, **không** dùng để tuyên bố đóng góp. 1 seed theo D4 ⇒ **không** báo `mean ± std`, chỉ báo giá trị đơn lẻ kèm nhãn `n=1`. Nếu quota không cho 3 seed thì ghi resource limitation theo §11. FedProx bị chặn ở CLI (proximal term chưa inject) — ghi là giới hạn, không im lặng bỏ.

### C9 — Độ nhạy τ_elig
Báo song song τ=1 (primary) và τ=50, bất kể cái nào tốt hơn. τ=50 **không** được chọn làm primary sau khi thấy kết quả — đó chính là hành vi §20 cấm.

---

## 3. Đầu ra của mỗi comparison

`scripts/compare_runs.py` ghi vào `artifacts/comparisons/<id>/`:

```
comparison_per_run.csv     # 1 dòng/run: arm, seed, partition_id, final_round, git_commit + mọi metric
comparison_summary.csv     # 1 dòng/(arm, metric): mean, std, n_seeds, "mean ± std", delta_vs_baseline
COMPARISON_INVALID.md      # CHỈ xuất hiện khi so sánh vi phạm §14/§20
plots/per_class_ap50.png           # bar theo lớp, có error bar, nhãn n=<số seed>
plots/map50_over_rounds.png        # đường theo round
plots/ap50_bus_over_rounds.png     # đường AP lớp target theo round
plots/delta_ap50_vs_baseline.png   # bar ΔAP theo lớp; cột ÂM phải để nguyên cho thấy
```

Hình `delta` **giữ cả cột âm**: che lớp mà phương pháp kém hơn là điều §20 cấm.

---

## 4. Thứ tự thực hiện

```
C0 (smoke, local hoặc Kaggle)
  └─ G1 environment.lock + F1 runtime
       └─ C1  → G4 (cần thêm F3 parameter evidence)
            └─ C3 + C2 → xác nhận đồng nhất A3≡A1
                 └─ C4 → G7 (đóng góp thật của H3)
                      └─ C5a → ρ*  → C5 → G6     [chặn: A2b cần rho_loss.py + ADR-006]
                           └─ C6 → G8, G9
                                └─ C7 → G10
                                     └─ C8, C9 (Tier 2/3)
                                          └─ G11
```

Mỗi comparison hoàn thành phải: đăng ký run vào `research/experiment_registry/registry.yaml`, cập nhật `research/gates.yaml`, và **giữ lại kết quả âm** (§16).

---

## 5. Còn thiếu / chưa chạy

- **Chưa có run nào được thực thi.** `artifacts/` rỗng, registry rỗng. Toàn bộ số trong mọi ví dụ ở đây là minh họa định dạng, không phải kết quả.
- **S1-Control chưa đạt yêu cầu matched-pair.** `configs/partition/s1_control.yaml` hiện chỉ là `missing_map: {}` (tức IID) kèm ghi chú *tuyên bố* đã matched. Thuật toán ở plan.md §5.3 **chưa được implement** ⇒ **C1 chưa thể claim G4** cho tới khi có.
- **A2b chưa tồn tại** ⇒ C5a/C5/C6 chạy sẽ fail ở arm A2b/A4b (có chủ ý). Chặn bởi F1 runtime + `src/preservation/rho_loss.py` + ADR-006.
- **Confidence statistics** (§13) chưa có; cần một pass predict riêng, làm cùng F2/F3.
- **δ_abs = 0.01 AP** là lựa chọn `[ENGINEERING]` của tôi, chưa có căn cứ literature. Nếu bạn/GVHD muốn giá trị khác thì phải đổi **trước** khi chạy C1.
