"""Catalogue of every finished run under kaggle/output, as a Markdown table (Vietnamese).

Numbers come straight from each run's round_metrics.csv:
  * 30-round S1b-2k runs: mean of the evals at rounds 20/25/30 (pre-declared rule, ADR-011)
  * shorter runs (S0, 5 rounds) and the teacher T: the last evaluation
Runs that did not finish (or were invalidated) are listed with their status — negative and
failed runs are never dropped (project rules §16, §20).

Usage:
  python scripts/build_run_catalog.py > /tmp/catalog.md
  python scripts/build_run_catalog.py --insert docs/PHAN_TICH_NON_IID.md   # replace the marked block
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "kaggle" / "output"
RULE = (20, 25, 30)
BEGIN, END = "<!-- RUN-CATALOG:BEGIN -->", "<!-- RUN-CATALOG:END -->"

SCENARIO = {
    "s0_iid_seed42": "S0 IID (toàn bộ dữ liệu)",
    "centralized": "Tập trung (toàn bộ dữ liệu)",
    "s1b_bus_2k_seed42": "S1b-2k (C0, C1 thiếu bus; C2 thiếu truck)",
    "s1_control_matched_2k_seed42": "Control matched (đủ bus, 2,03× box bus)",
    "s1b_pooled_iid_2k_seed42": "S1b gộp-chia đều (D3)",
    "server_sample_1k_seed42": "1.000 ảnh của server",
}
ARM = {
    "A0": "FedAvg", "FedAvg": "FedAvg", "FedProx": "FedProx (μ 0,01)", "A1": "A1 gộp theo số box lớp",
    "A3": "A3 gộp nhận biết lớp (server)", "A2b_rho0.25": "A2b ρ 0,25 (client)",
    "A4b_rho0.25": "A4b = A2b + A3 (phương pháp đầy đủ)", "A5": "A5 chưng cất not-true (FedNTD)",
    "A6": "A6 Equalized Focal Loss", "A6c": "A6c focal loss (đối chứng)",
    "G2-centralized": "G2 train tập trung (10 epoch)", "T-teacher": "T teacher (train tập trung)",
    "B1": "B1 FedAvg từ T", "B2": "B2 = B1 + fine-tune ở server", "P1": "P1 chưng cất từ T cố định",
    "P2_rho0.25": "P2 = P1 + ρ 0,25",
}
# Runs whose directory exists but whose numbers are not valid results.
INVALID = {
    ("s8a", "B2"): "lỗi kỹ thuật: dừng ở vòng 2 (YOLO.train gọi 2 lần)",
    ("s8c_g_v1_invalid", "B2"): "không hợp lệ: chỉ train vòng 1 (cache nhãn hỏng)",
}
SESSION_ORDER = ["s2", "s5a", "s5b", "s6a", "s6b", "s7a", "s7b", "s8t", "s8a", "s8b", "s8c_g", "s8c_g_v1_invalid"]


def _f(v: str | None) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


def summarise(run: Path) -> dict | None:
    env_p, rm_p = run / "environment.json", run / "round_metrics.csv"
    if not env_p.exists() or not (rm_p.exists() or (run / "metrics.csv").exists()):
        return None
    if not rm_p.exists():
        rm_p = run / "metrics.csv"
    env = json.loads(env_p.read_text(encoding="utf-8"))
    if rm_p.name == "metrics.csv":                          # G2: 10 centralized epochs, one final evaluation
        rows = {0: {r["metric"]: r["value"] for r in csv.DictReader(rm_p.open(encoding="utf-8"))}}
    else:
        rows = {int(float(r["round"])): r for r in csv.DictReader(rm_p.open(encoding="utf-8"))}
    if not rows:
        return None
    last = max(rows)
    if all(k in rows for k in RULE):
        pick, how = [rows[k] for k in RULE], "TB vòng 20/25/30"
    else:
        pick, how = [rows[last]], f"vòng {last}" if last else "1 lần đánh giá"
    mean = lambda m: (sum(_f(r.get(m)) or 0.0 for r in pick) / len(pick)) if all(_f(r.get(m)) is not None for r in pick) else None
    return {"exp": str(env.get("exp_id")), "partition": str(env.get("partition_id")), "seed": env.get("seed"),
            "rounds": last, "how": how, **{m: mean(m) for m in
            ("mAP50", "AP50_bus", "AP50_truck", "AP50_car", "AP50_motorcycle", "FP_bus", "FN_bus")}}


def catalogue() -> list[dict]:
    out = []
    sessions = sorted((p for p in OUT.iterdir() if (p / "artifacts" / "runs").is_dir()),
                      key=lambda p: (SESSION_ORDER.index(p.name) if p.name in SESSION_ORDER else 99, p.name))
    for sess in sessions:
        for run in sorted((sess / "artifacts" / "runs").iterdir()):
            s = summarise(run)
            if s is None:
                continue
            s["session"] = sess.name
            s["status"] = INVALID.get((sess.name, s["exp"]), "hoàn thành")
            out.append(s)
    return out


def to_markdown(rows: list[dict]) -> str:
    num = lambda v, d=3: "—" if v is None else (f"{v:,.0f}".replace(",", ".") if d == 0 else f"{v:.{d}f}".replace(".", ","))
    lines = [
        "| Phiên | Kịch bản (partition) | Arm | Seed | Số vòng | Cách lấy số | mAP50 | AP50 bus | AP50 truck | AP50 car | AP50 moto | FP bus | FN bus | Trạng thái |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        bad = r["status"] != "hoàn thành"
        vals = ["—"] * 7 if bad else [num(r["mAP50"]), num(r["AP50_bus"]), num(r["AP50_truck"]), num(r["AP50_car"]),
                                     num(r["AP50_motorcycle"]), num(r["FP_bus"], 0), num(r["FN_bus"], 0)]
        lines.append(f"| {r['session']} | {SCENARIO.get(r['partition'], r['partition'])} | {ARM.get(r['exp'], r['exp'])} | "
                     f"{r['seed']} | {r['rounds']} | {r['how']} | " + " | ".join(vals) + f" | {r['status']} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--insert", type=Path, help="Markdown file whose RUN-CATALOG block is replaced")
    args = ap.parse_args()
    table = to_markdown(catalogue())
    if not args.insert:
        sys.stdout.reconfigure(encoding="utf-8")
        print(table)
        return
    text = args.insert.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise SystemExit(f"{args.insert}: markers {BEGIN} / {END} not found")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    args.insert.write_text(f"{head}{BEGIN}\n{table}\n{END}{tail}", encoding="utf-8")
    print(f"catalogue ({table.count(chr(10)) - 1} runs) written into {args.insert}")


if __name__ == "__main__":
    main()
