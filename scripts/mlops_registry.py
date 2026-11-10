"""Model registry CLI (ADR-031).

    # register a finished run (also backfills runs downloaded from Kaggle)
    python scripts/mlops_registry.py register --run-dir artifacts/runs/<run>
    python scripts/mlops_registry.py register --run-dir kaggle/output/s28/artifacts/runs/<run> \\
        --eval-data-yaml <YOLO_ROOT>/data.yaml --weights coco:yolo11s.pt      # pre-ADR-031 runs

    python scripts/mlops_registry.py list
    python scripts/mlops_registry.py check   --version 3      # dry run of the promotion rule
    python scripts/mlops_registry.py promote --version 3
    python scripts/mlops_registry.py promote --version 3 --force-reason "..."   # audited override
    python scripts/mlops_registry.py download --ref champion --dst artifacts/deploy/champion

Tracking URI: --mlflow-uri, else $MLFLOW_TRACKING_URI, else http://localhost:5000
(docker compose -f docker/docker-compose.yml up -d mlflow).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mlops.model_card import build_model_card, read_summary_metrics  # noqa: E402
from src.mlops.registry import (  # noqa: E402
    POLICY_PATH,
    _client,
    current_champion,
    decide_promotion,
    download_package,
    load_policy,
    promote,
    register_run,
    tags_to_card,
)
from src.mlops.tracking import default_tracking_uri, start_tracking  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("mlops_registry")


def _cmd_register(args: argparse.Namespace, policy) -> int:
    run_dir = args.run_dir.resolve()
    card = build_model_card(run_dir, eval_data_yaml=args.eval_data_yaml, weights_override=args.weights)
    # Reopens the run's own MLflow run when it was tracked on this store; otherwise starts one.
    tracker = start_tracking(run_dir, args.experiment, args.mlflow_uri, run_name=run_dir.name,
                             tags={"source": "mlops_registry.register", "run_dir": run_dir.name},
                             required=True)
    try:
        import yaml
        config_path = run_dir / "config.yaml"
        if config_path.is_file() and not tracker.resumed:   # a reopened run already holds its params
            tracker.log_params(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})
        tracker.log_metrics(read_summary_metrics(run_dir / "metrics.csv"))
        tracker.log_run_artifacts(run_dir)
        version = register_run(tracker, run_dir, card, policy)
    except Exception:
        tracker.end("FAILED")
        raise
    tracker.end()
    print(f"registered {policy.model_name} version {version} (alias {policy.candidate_alias}); "
          f"protocol from {card['protocol_source']}: {json.dumps(card['protocol'])}")
    return 0


def _cmd_list(args: argparse.Namespace, policy) -> int:
    client = _client(args.mlflow_uri)
    versions = client.search_model_versions(f"name='{policy.model_name}'")
    aliases: dict[str, list[str]] = {}
    try:
        for alias, ver in (client.get_registered_model(policy.model_name).aliases or {}).items():
            aliases.setdefault(str(ver), []).append(alias)
    except Exception:
        pass
    pm = policy.primary_metric
    print(f"{'ver':>4}  {'aliases':<20} {'run_class':<12} {'seed':>5}  {pm:>8}  run")
    for v in sorted(versions, key=lambda v: int(v.version)):
        card = tags_to_card(v.tags)
        metric = card["metrics"].get(pm)
        print(f"{v.version:>4}  {','.join(aliases.get(str(v.version), [])):<20} "
              f"{str(card.get('run_class')):<12} {str(card.get('seed')):>5}  "
              f"{'' if metric is None else f'{metric:.4f}':>8}  {card.get('run_name')}")
    return 0


def _cmd_check(args: argparse.Namespace, policy) -> int:
    client = _client(args.mlflow_uri)
    cand = client.get_model_version(policy.model_name, str(args.version))
    champ = current_champion(policy, client)
    decision = decide_promotion(tags_to_card(cand.tags), tags_to_card(champ.tags) if champ else None, policy)
    print(f"version {args.version} vs champion {champ.version if champ else 'none'}: "
          f"{'PROMOTABLE' if decision.ok else 'REFUSED'}\n{decision.report()}")
    return 0 if decision.ok else 1


def _cmd_promote(args: argparse.Namespace, policy) -> int:
    decision = promote(str(args.version), policy, client=_client(args.mlflow_uri),
                       force_reason=args.force_reason)
    applied = decision.ok or bool(args.force_reason)
    print(f"version {args.version}: {'promoted' if decision.ok else ('FORCED' if applied else 'REFUSED')}"
          f"\n{decision.report()}")
    return 0 if applied else 1


def _cmd_download(args: argparse.Namespace, policy) -> int:
    path = download_package(args.ref, policy, args.dst, tracking_uri=args.mlflow_uri)
    print(f"package downloaded to {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mlflow-uri", default=default_tracking_uri())
    ap.add_argument("--policy", type=Path, default=POLICY_PATH)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("register", help="log + register the model of a finished run")
    r.add_argument("--run-dir", type=Path, required=True)
    r.add_argument("--experiment", default="registry-backfill",
                   help="MLflow experiment for runs that were not tracked on this store")
    r.add_argument("--eval-data-yaml", type=Path, default=None,
                   help="global val data.yaml, to fingerprint the eval set of pre-ADR-031 runs")
    r.add_argument("--weights", default=None,
                   help="effective model weights (e.g. coco:yolo11s.pt) when the run has no eval_protocol.json")

    sub.add_parser("list", help="list registered versions")
    c = sub.add_parser("check", help="evaluate the promotion rule without changing anything")
    c.add_argument("--version", required=True)
    p = sub.add_parser("promote", help="move the champion alias if the rule passes")
    p.add_argument("--version", required=True)
    p.add_argument("--force-reason", default=None,
                   help="promote despite failed checks; the reason and the failures are tagged")
    d = sub.add_parser("download", help="download a package by alias or version")
    d.add_argument("--ref", default="champion")
    d.add_argument("--dst", type=Path, required=True)

    args = ap.parse_args(argv)
    policy = load_policy(args.policy)
    handlers = {"register": _cmd_register, "list": _cmd_list, "check": _cmd_check,
                "promote": _cmd_promote, "download": _cmd_download}
    return handlers[args.cmd](args, policy)


if __name__ == "__main__":
    raise SystemExit(main())
