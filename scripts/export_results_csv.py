"""Export every YOLO11s / COCO-head result in the experiment registry to one CSV, with a note per row.

Rows: one per completed run (row_type=run), one per method x scenario mean over seeds (row_type=mean), and one per
registered run that has no result yet (row_type=pending) or was excluded with a logged reason (row_type=excluded). Metrics follow the pre-registered rule (mean of rounds 10
and 20); recall/precision/TP/FP/FN are at the operating threshold conf 0.25 (exploratory, ``operating_r10_20``).
Written as UTF-8 with BOM so Excel shows the Vietnamese notes correctly.

Usage:
  python scripts/export_results_csv.py [--out research/results_summary.csv]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean, stdev

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml  # noqa: E402

from src.evaluation.registry_results import (  # noqa: E402
    MISSING_BUS_CLIENTS, Run, _method, _scenario, by_seed, load_runs,
)

METHOD_DESC = {
    "A0c": "FedAvg (đối chứng)",
    "A1c": "FedAvg + trọng số theo số box của lớp (class-count)",
    "FedProxc": "FedProx mu=0.01",
    "P3": "chưng cất từ teacher (KD)",
    "P3LR": "P3 + cosine LR theo vòng",
    "P4": "P3LR + RFS t=0.5",
    "P4noKD": "cosine LR + RFS, không KD (phương pháp chính)",
    "P5f": "P4noKD + đóng băng nhánh phân loại (FedBABU)",
}
SCENARIO_DESC = {
    "pooled": "pooled: mọi client đủ 4 lớp (phân phối lý tưởng)",
    "s5": "S1b5: bus vắng 3/5 client, truck vắng C3",
    "bus1": "bus1: chỉ C4 có bus, truck vắng C3",
}
METRICS = ("mAP50", "AP50_bus", "AP50_car", "AP50_truck", "AP50_motorcycle",
           "recall_bus", "precision_bus", "TP_bus", "FP_bus", "FN_bus")
HEADER = ("row_type", "exp_id", "method", "method_desc", "scenario", "scenario_desc", "clients_without_bus", "seed",
          "n_seeds", *METRICS, "AP50_bus_std", "gap_AP50_bus_vs_pooled", "AP50_bus_pct_of_pooled",
          "recall_bus_pct_of_pooled", "note")
MAIN = ("A0c", "A1c", "FedProxc", "P4noKD")


def _f(v: float | None, nd: int = 4) -> str:
    return "" if v is None else f"{v:.{nd}f}"


def _pct(runs: list[Run], m: str, sc: str, key: str, seeds: set[int]) -> float | None:
    pooled, other = by_seed(runs, m, "pooled", key), by_seed(runs, m, sc, key)
    vals = [100 * other[s] / pooled[s] for s in seeds if s in pooled and s in other and pooled[s] > 0]
    return mean(vals) if vals else None


def _gap(runs: list[Run], m: str, sc: str, seeds: set[int]) -> float | None:
    pooled, other = by_seed(runs, m, "pooled", "AP50_bus"), by_seed(runs, m, sc, "AP50_bus")
    vals = [pooled[s] - other[s] for s in seeds if s in pooled and s in other]
    return mean(vals) if vals else None


def run_note(r: Run, runs: list[Run]) -> str:
    notes = ["đối chứng FedAvg"] if r.method == "A0c" else []
    if r.method not in MAIN:
        notes.append("ablation/thăm dò, 1 seed")
    if r.scenario == "pooled":
        notes.append("mốc so sánh cho gap thiếu lớp")
    if r.scenario == "bus1":
        notes.append("mức thiếu nặng nhất (ADR-022)")
    a0 = by_seed(runs, "A0c", r.scenario, "AP50_bus").get(r.seed)
    if r.method != "A0c" and a0 is not None:
        notes.append(f"AP50 bus − A0c cùng seed = {r.metrics['AP50_bus'] - a0:+.3f}")
    if "recall_bus" not in r.metrics:
        notes.append("không có số liệu ngưỡng 0,25")
    return "; ".join(notes)


def mean_note(m: str, sc: str, n: int, gap: float | None, runs: list[Run]) -> str:
    notes = [f"TB {n} seed"]
    if sc != "pooled" and gap is not None:
        g0 = _gap(runs, "A0c", sc, set(by_seed(runs, m, sc, "AP50_bus")))
        if m != "A0c" and g0 is not None:
            notes.append(f"gap {'lớn hơn' if gap > g0 else 'nhỏ hơn'} A0c ({gap:+.3f} so với {g0:+.3f}, cùng seed)")
    if sc == "bus1" or (m == "FedProxc" and sc == "pooled"):
        notes.append("thiếu seed 2024 (phiên s18c chưa có kết quả)")
    if n == 1:
        notes.append("chỉ 1 seed, không kết luận")
    return "; ".join(notes)


def pending_rows(registry: Path, runs: list[Run]) -> list[list[str]]:
    data = yaml.safe_load(registry.read_text(encoding="utf-8"))
    done = {(r.method, r.scenario, r.seed): r.exp_id for r in runs}
    rows = []
    for e in data.get("experiments", []):
        sc = _scenario(str(e.get("id", "")), str(e.get("scenario", "")))
        if sc is None or e.get("status") == "completed" or e.get("seed") is None:
            continue
        m, status = _method(e.get("method", "")), e.get("status")
        if status in ("running", "planned"):
            row_type = "pending"
            note = f"trạng thái registry: {status}; phiên {e.get('kaggle_kernel', '?')} chưa trả kết quả"
        else:
            row_type = "excluded"
            reason = str(e.get("notes") or e.get("interpretation") or "").split(". ")[0]
            note = f"trạng thái registry: {status}; giữ lại để minh bạch, không đưa vào trung bình. Lý do (registry): {reason}"
            rerun = done.get((m, sc, int(e["seed"])))
            note += f"; đã chạy lại thành công: {rerun}" if rerun else ("; không chạy lại" if status == "failed-technical" else "")
        rows.append([row_type, e["id"], m, METHOD_DESC.get(m, ""), sc, SCENARIO_DESC[sc], MISSING_BUS_CLIENTS[sc],
                     e["seed"], "", *[""] * (len(METRICS) + 4), note])
    return rows


def build_rows(runs: list[Run], registry: Path) -> list[list]:
    rows = []
    order = {m: i for i, m in enumerate((*MAIN, "P3", "P3LR", "P4", "P5f"))}
    key = lambda r: (order.get(r.method, 99), MISSING_BUS_CLIENTS[r.scenario], r.seed)  # noqa: E731
    for r in sorted(runs, key=key):
        seeds = {r.seed}
        gap = _gap(runs, r.method, r.scenario, seeds) if r.scenario != "pooled" else None
        rows.append(["run", r.exp_id, r.method, METHOD_DESC.get(r.method, ""), r.scenario, SCENARIO_DESC[r.scenario],
                     MISSING_BUS_CLIENTS[r.scenario], r.seed, 1,
                     *[_f(r.metrics.get(k), 1 if k[:3] in ("TP_", "FP_", "FN_") else 4) for k in METRICS], "",
                     _f(gap), _f(_pct(runs, r.method, r.scenario, "AP50_bus", seeds) if r.scenario != "pooled" else None, 1),
                     _f(_pct(runs, r.method, r.scenario, "recall_bus", seeds) if r.scenario != "pooled" else None, 1),
                     run_note(r, runs)])
    groups = sorted({(r.method, r.scenario) for r in runs if r.method in MAIN},
                    key=lambda g: (order[g[0]], MISSING_BUS_CLIENTS[g[1]]))
    for m, sc in groups:
        sel = [r for r in runs if r.method == m and r.scenario == sc]
        seeds = {r.seed for r in sel}
        vals = {k: [r.metrics[k] for r in sel if k in r.metrics] for k in METRICS}
        gap = _gap(runs, m, sc, seeds) if sc != "pooled" else None
        bus = vals["AP50_bus"]
        rows.append(["mean", "", m, METHOD_DESC[m], sc, SCENARIO_DESC[sc], MISSING_BUS_CLIENTS[sc],
                     "/".join(str(s) for s in sorted(seeds)), len(sel),
                     *[_f(mean(v), 1 if k[:3] in ("TP_", "FP_", "FN_") else 4) if v else "" for k, v in vals.items()],
                     _f(stdev(bus) if len(bus) > 1 else None), _f(gap),
                     _f(_pct(runs, m, sc, "AP50_bus", seeds) if sc != "pooled" else None, 1),
                     _f(_pct(runs, m, sc, "recall_bus", seeds) if sc != "pooled" else None, 1),
                     mean_note(m, sc, len(sel), gap, runs)])
    return rows + pending_rows(registry, runs)


GLOSSARY = [
    "GHI CHÚ CỘT",
    "row_type: run = một lần chạy; mean = trung bình theo seed của phương pháp × kịch bản; pending = đã đăng ký, chưa có kết quả; excluded = lỗi kỹ thuật hoặc bị bác bỏ, có lý do ghi trong registry",
    "Chỉ số AP/mAP: quy tắc đăng ký trước = trung bình vòng 10 và 20 (ADR-019/021); YOLO11s giữ head COCO; 5 client; 20 vòng",
    "recall_bus / precision_bus / TP_bus / FP_bus / FN_bus: tại ngưỡng điểm 0,25 (operating_conf); phân tích khám phá, không phải tiêu chí đăng ký trước",
    "Tổng số bus thật trong tập đánh giá = 1.597 (TP_bus + FN_bus)",
    "gap_AP50_bus_vs_pooled = AP50 bus(pooled) − AP50 bus(kịch bản), cùng phương pháp, ghép cặp theo seed; > 0 nghĩa là bị hại do thiếu lớp",
    "*_pct_of_pooled = giá trị kịch bản / giá trị pooled × 100, ghép cặp theo seed",
    "Mọi kịch bản dùng cùng 15.000 ảnh, cùng tổng box (bus 2.628, car 155.250, truck 6.421, motorcycle 1.963), 3.000 ảnh/client; chỉ khác client nào giữ lớp",
    "Quy tắc đọc: chỉ khẳng định 'cùng dấu ở mọi seed'; n = 3 không đủ cho kiểm định thống kê",
    "Nguồn: research/experiment_registry/registry.yaml; tạo bằng scripts/export_results_csv.py",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", type=Path, default=_REPO_ROOT / "research" / "experiment_registry" / "registry.yaml")
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "research" / "results_summary.csv")
    args = ap.parse_args()
    rows = build_rows(load_runs(args.registry), args.registry)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
        w.writerow([])
        for line in GLOSSARY:
            w.writerow(["note", line])
    counts = {t: sum(r[0] == t for r in rows) for t in ("run", "mean", "pending", "excluded")}
    print(f"wrote {args.out}: {counts}")


if __name__ == "__main__":
    main()
