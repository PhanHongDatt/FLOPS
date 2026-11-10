"""MLflow model registry: register packaged runs, promote by a pre-configured rule (ADR-031).

[ENGINEERING] deployment policy. Every registration creates a model version and moves the
``candidate`` alias to it; ``champion`` only moves through ``promote``, which applies
``decide_promotion`` mechanically (or records the failed checks when forced with a reason).
Version tags hold the card (metrics, protocol, run identity), so decisions need no downloads.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.data.bdd100k import TARGET_CLASSES
from src.mlops.model_card import MODEL_META_FILE, PROTOCOL_KEYS, write_model_meta

logger = logging.getLogger(__name__)

POLICY_PATH = Path(__file__).resolve().parents[2] / "configs" / "mlops" / "registry.yaml"
PACKAGE_ARTIFACT_PATH = "model"
_IDENTITY_KEYS = ("run_name", "exp_id", "run_class", "seed", "partition_id", "git_commit",
                  "params_sha256", "num_arrays", "format", "protocol_source")


@dataclass(frozen=True)
class PromotionPolicy:
    model_name: str = "flops-yolo-detector"
    candidate_alias: str = "candidate"
    champion_alias: str = "champion"
    allowed_run_classes: tuple[str, ...] = ("main",)
    primary_metric: str = "mAP50"
    min_primary_gain: float = 0.0
    per_class_metric_prefix: str = "AP50_"
    classes: tuple[str, ...] = TARGET_CLASSES
    per_class_max_drop: float = 0.0
    protocol_keys: tuple[str, ...] = PROTOCOL_KEYS


def load_policy(path: Path = POLICY_PATH) -> PromotionPolicy:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    promo = data.get("promotion", {}) or {}
    d = PromotionPolicy()
    return PromotionPolicy(
        model_name=str(data.get("registered_model_name", d.model_name)),
        candidate_alias=str(data.get("candidate_alias", d.candidate_alias)),
        champion_alias=str(data.get("champion_alias", d.champion_alias)),
        allowed_run_classes=tuple(promo.get("allowed_run_classes", d.allowed_run_classes)),
        primary_metric=str(promo.get("primary_metric", d.primary_metric)),
        min_primary_gain=float(promo.get("min_primary_gain", d.min_primary_gain)),
        per_class_metric_prefix=str(promo.get("per_class_metric_prefix", d.per_class_metric_prefix)),
        classes=tuple(promo.get("classes", d.classes)),
        per_class_max_drop=float(promo.get("per_class_max_drop", d.per_class_max_drop)),
        protocol_keys=tuple(promo.get("protocol_keys", d.protocol_keys)),
    )


# ── card <-> version tags ─────────────────────────────────────────────────────
def _tag_value(v: Any) -> str:
    return json.dumps(v) if isinstance(v, (list, tuple, dict)) else str(v)


def card_to_tags(card: dict[str, Any]) -> dict[str, str]:
    tags = {k: _tag_value(card[k]) for k in _IDENTITY_KEYS if card.get(k) is not None}
    tags.update({f"metric.{k}": repr(float(v)) for k, v in (card.get("metrics") or {}).items()})
    tags.update({f"protocol.{k}": _tag_value(v) for k, v in (card.get("protocol") or {}).items()})
    return tags


def _parse(v: str) -> Any:
    try:
        return json.loads(v)
    except (json.JSONDecodeError, TypeError):
        return v


def tags_to_card(tags: dict[str, str]) -> dict[str, Any]:
    card: dict[str, Any] = {"metrics": {}, "protocol": {}}
    for k, v in (tags or {}).items():
        if k.startswith("metric."):
            try:
                card["metrics"][k[len("metric."):]] = float(v)
            except ValueError:
                continue
        elif k.startswith("protocol."):
            card["protocol"][k[len("protocol."):]] = _parse(v)
        elif k in _IDENTITY_KEYS:
            card[k] = v
    return card


# ── promotion rule ────────────────────────────────────────────────────────────
@dataclass
class PromotionDecision:
    ok: bool
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def failed(self) -> list[str]:
        return [f"{name}: {detail}" for name, passed, detail in self.checks if not passed]

    def report(self) -> str:
        return "\n".join(f"  [{'PASS' if p else 'FAIL'}] {n}: {d}" for n, p, d in self.checks)


def decide_promotion(candidate: dict[str, Any], champion: dict[str, Any] | None,
                     policy: PromotionPolicy) -> PromotionDecision:
    """Apply the policy. Pure function: cards in, decision out."""
    checks: list[tuple[str, bool, str]] = []
    run_class = candidate.get("run_class")
    checks.append(("run_class", run_class in policy.allowed_run_classes,
                   f"{run_class!r} in {list(policy.allowed_run_classes)}"))

    c_proto = candidate.get("protocol") or {}
    missing = [k for k in policy.protocol_keys if c_proto.get(k) in (None, "")]
    checks.append(("protocol_complete", not missing,
                   "all protocol keys present" if not missing else f"missing {missing}"))

    c_metrics = candidate.get("metrics") or {}
    primary = policy.primary_metric
    if primary not in c_metrics:
        checks.append(("primary_metric", False, f"candidate has no {primary}"))
        return PromotionDecision(False, checks)

    if champion is None:
        checks.append(("champion", True, "no current champion"))
        return PromotionDecision(all(p for _, p, _ in checks), checks)

    h_proto = champion.get("protocol") or {}
    diff = {k: (c_proto.get(k), h_proto.get(k)) for k in policy.protocol_keys
            if c_proto.get(k) != h_proto.get(k)}
    checks.append(("same_protocol", not diff,
                   "same evaluation protocol" if not diff else f"candidate vs champion differ: {diff}"))

    h_metrics = champion.get("metrics") or {}
    if primary not in h_metrics:
        checks.append(("primary_metric", False, f"champion has no {primary}"))
    else:
        gain = c_metrics[primary] - h_metrics[primary]
        checks.append(("primary_metric", gain > policy.min_primary_gain,
                       f"{primary} {c_metrics[primary]:.4f} vs {h_metrics[primary]:.4f} "
                       f"(gain {gain:+.4f}, needs > {policy.min_primary_gain:+.4f})"))

    for cls in policy.classes:
        key = f"{policy.per_class_metric_prefix}{cls}"
        if key not in c_metrics or key not in h_metrics:
            checks.append((f"class:{cls}", False, f"{key} missing on "
                           f"{'candidate' if key not in c_metrics else 'champion'}"))
            continue
        drop = h_metrics[key] - c_metrics[key]
        checks.append((f"class:{cls}", drop <= policy.per_class_max_drop,
                       f"{key} {c_metrics[key]:.4f} vs {h_metrics[key]:.4f} "
                       f"(drop {drop:+.4f}, max {policy.per_class_max_drop:+.4f})"))
    return PromotionDecision(all(p for _, p, _ in checks), checks)


# ── MLflow operations ─────────────────────────────────────────────────────────
def _client(tracking_uri: str | None = None) -> Any:
    from mlflow.tracking import MlflowClient
    return MlflowClient(tracking_uri=tracking_uri) if tracking_uri else MlflowClient()


def log_model_package(tracker: Any, run_dir: Path, card: dict[str, Any]) -> None:
    """Log ``final_params.npz`` + ``model_meta.json`` under artifact path ``model/``."""
    meta = write_model_meta(run_dir, card)
    tracker.log_artifact(card["params_file"], artifact_path=PACKAGE_ARTIFACT_PATH)
    tracker.log_artifact(meta, artifact_path=PACKAGE_ARTIFACT_PATH)


def register_card(run_id: str, card: dict[str, Any], policy: PromotionPolicy,
                  client: Any = None, description: str | None = None) -> str:
    """Create a model version for the package logged in ``run_id``; move ``candidate`` to it."""
    from mlflow.exceptions import MlflowException

    client = client or _client()
    try:
        client.get_registered_model(policy.model_name)
    except MlflowException:
        client.create_registered_model(
            policy.model_name, description="FLOPS YOLO detector (ADR-031); promotion by configs/mlops/registry.yaml")
    source = client.get_run(run_id).info.artifact_uri.rstrip("/") + "/" + PACKAGE_ARTIFACT_PATH
    mv = client.create_model_version(
        name=policy.model_name, source=source, run_id=run_id, tags=card_to_tags(card),
        description=description or f"{card.get('run_name')} ({card.get('run_class')}, seed {card.get('seed')})",
    )
    client.set_registered_model_alias(policy.model_name, policy.candidate_alias, mv.version)
    logger.info("Registered %s version %s (alias %s) from run %s",
                policy.model_name, mv.version, policy.candidate_alias, run_id)
    return str(mv.version)


def register_run(tracker: Any, run_dir: Path, card: dict[str, Any], policy: PromotionPolicy) -> str:
    """Log the package into the tracker's run, verify it arrived, register it.

    ``Tracker`` swallows logging errors (tracking must not end a training run), so the package
    is checked on the store before a version is created — a version must never point at
    missing files.
    """
    if not getattr(tracker, "enabled", False):
        raise RuntimeError("model registration needs an enabled MLflow tracker")
    log_model_package(tracker, run_dir, card)
    client = _client(tracker.uri)
    logged = {Path(a.path).name for a in client.list_artifacts(tracker.run_id, PACKAGE_ARTIFACT_PATH)}
    expected = {Path(card["params_file"]).name, MODEL_META_FILE}
    if not expected <= logged:
        raise RuntimeError(f"model package upload incomplete: expected {sorted(expected)}, "
                           f"store has {sorted(logged)}")
    return register_card(tracker.run_id, card, policy, client=client)


def current_champion(policy: PromotionPolicy, client: Any = None) -> Any | None:
    from mlflow.exceptions import MlflowException

    client = client or _client()
    try:
        return client.get_model_version_by_alias(policy.model_name, policy.champion_alias)
    except MlflowException:
        return None


def promote(version: str, policy: PromotionPolicy, client: Any = None,
            force_reason: str | None = None) -> PromotionDecision:
    """Move ``champion`` to ``version`` if the rule passes, or if forced with a reason.

    A forced promotion is recorded on the version (reason + failed checks); it is never silent.
    """
    client = client or _client()
    version = str(version)
    cand = client.get_model_version(policy.model_name, version)
    champ = current_champion(policy, client)
    if champ is not None and str(champ.version) == version:
        return PromotionDecision(True, [("champion", True, f"version {version} already is {policy.champion_alias}")])

    decision = decide_promotion(tags_to_card(cand.tags), tags_to_card(champ.tags) if champ else None, policy)
    if not decision.ok and not force_reason:
        logger.info("Promotion of version %s refused:\n%s", version, decision.report())
        return decision

    client.set_registered_model_alias(policy.model_name, policy.champion_alias, version)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    client.set_model_version_tag(policy.model_name, version, "promotion.at", stamp)
    client.set_model_version_tag(policy.model_name, version, "promotion.previous_champion",
                                 str(champ.version) if champ else "none")
    if not decision.ok:
        client.set_model_version_tag(policy.model_name, version, "promotion.forced", force_reason)
        client.set_model_version_tag(policy.model_name, version, "promotion.failed_checks",
                                     json.dumps(decision.failed)[:5000])
        logger.warning("Version %s FORCED to %s (%s); failed checks: %s",
                       version, policy.champion_alias, force_reason, decision.failed)
    else:
        logger.info("Version %s promoted to %s:\n%s", version, policy.champion_alias, decision.report())
    return decision


def download_package(alias_or_version: str, policy: PromotionPolicy, dst: Path,
                     tracking_uri: str | None = None) -> Path:
    """Download a registered package (``champion``, ``candidate`` or a version number) to ``dst``."""
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    ref = (f"models:/{policy.model_name}/{alias_or_version}" if str(alias_or_version).isdigit()
           else f"models:/{policy.model_name}@{alias_or_version}")
    Path(dst).mkdir(parents=True, exist_ok=True)
    local = mlflow.artifacts.download_artifacts(artifact_uri=ref, dst_path=str(dst))
    if not (Path(local) / MODEL_META_FILE).is_file():
        raise FileNotFoundError(f"{ref}: package has no {MODEL_META_FILE}")
    return Path(local)
