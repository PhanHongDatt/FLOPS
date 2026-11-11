"""ADR-031 registry: model card, eval-set fingerprint, promotion rule, MLflow registry round trip."""
from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from src.mlops.model_card import (
    EVAL_PROTOCOL_FILE,
    build_model_card,
    eval_set_fingerprint,
    write_eval_protocol,
)
from src.mlops.registry import (
    POLICY_PATH,
    PromotionPolicy,
    card_to_tags,
    decide_promotion,
    load_policy,
    tags_to_card,
)

CLASSES = ("car", "bus", "truck", "motorcycle")
PROTO = {"model_weights": "coco:yolo11s.pt", "image_size": 960, "eval_conf": 0.001, "eval_iou": 0.7,
         "eval_set_id": "3-abc", "class_names": list(CLASSES)}


def _card(map50=0.5, per_class=0.5, run_class="main", protocol=None, **over):
    metrics = {"mAP50": map50, **{f"AP50_{c}": per_class for c in CLASSES}}
    metrics.update(over)
    return {"run_class": run_class, "metrics": metrics, "protocol": dict(protocol or PROTO)}


# ── eval protocol ─────────────────────────────────────────────────────────────
def _val_dir(root, names):
    d = root / "images" / "val"
    d.mkdir(parents=True)
    for n in names:
        (d / n).write_bytes(b"")
    (d / "notes.txt").write_text("not an image", encoding="utf-8")
    return d


def test_fingerprint_is_path_independent_and_list_equivalent(tmp_path):
    a = tmp_path / "a"
    _val_dir(a, ["2.jpg", "1.jpg", "3.png"])
    (a / "data.yaml").write_text(yaml.safe_dump({"path": str(a), "val": "images/val"}), encoding="utf-8")
    b = tmp_path / "b"
    b.mkdir()
    (b / "val.txt").write_text("/elsewhere/1.jpg\n/elsewhere/3.png\n/elsewhere/2.jpg\n", encoding="utf-8")
    (b / "data.yaml").write_text(yaml.safe_dump({"path": str(b), "val": "val.txt"}), encoding="utf-8")

    fa, fb = eval_set_fingerprint(a / "data.yaml"), eval_set_fingerprint(b / "data.yaml")
    assert fa == fb and fa["eval_images"] == 3 and fa["eval_set_id"].startswith("3-")

    (b / "val.txt").write_text("/elsewhere/1.jpg\n/elsewhere/2.jpg\n", encoding="utf-8")
    assert eval_set_fingerprint(b / "data.yaml") != fa


def test_write_eval_protocol_survives_a_broken_data_yaml(tmp_path):
    bad = tmp_path / "data.yaml"
    bad.write_text(yaml.safe_dump({"path": str(tmp_path)}), encoding="utf-8")   # no val entry
    p = write_eval_protocol(tmp_path, model_weights="yolo11s.pt", image_size=640, eval_conf=0.001,
                            eval_iou=0.7, eval_data_yaml=bad)
    assert "eval_set_id" not in p and (tmp_path / EVAL_PROTOCOL_FILE).is_file()


# ── model card ────────────────────────────────────────────────────────────────
def _finished_run(tmp_path, with_protocol=True):
    run = tmp_path / "A0c_main_seed42_s5"
    (run / "checkpoint").mkdir(parents=True)
    np.savez(run / "checkpoint" / "final_params.npz", np.zeros(3), np.ones((2, 2)))
    (run / "config.yaml").write_text(yaml.safe_dump({
        "model": {"weights": "yolo11s.pt", "image_size": 960},
        "evaluation": {"conf": 0.001, "iou": 0.7}, "classes": list(CLASSES)}), encoding="utf-8")
    (run / "environment.json").write_text(json.dumps({
        "exp_id": "A0c", "run_class": "main", "seed": 42, "partition_id": "s5", "git_commit": "abc"}),
        encoding="utf-8")
    (run / "metrics.csv").write_text("metric,value\nAP50_bus,0.39\nmAP50,0.43\nnote,n/a\n", encoding="utf-8")
    if with_protocol:
        (run / EVAL_PROTOCOL_FILE).write_text(json.dumps(PROTO), encoding="utf-8")
    return run


def test_model_card_from_a_finished_run(tmp_path):
    card = build_model_card(_finished_run(tmp_path))
    assert card["metrics"] == {"AP50_bus": 0.39, "mAP50": 0.43}
    assert card["protocol"]["model_weights"] == "coco:yolo11s.pt"      # effective, not config.yaml
    assert card["protocol_source"] == EVAL_PROTOCOL_FILE
    assert card["run_class"] == "main" and card["seed"] == 42 and card["num_arrays"] == 2
    assert len(card["params_sha256"]) == 64


def test_backfill_card_falls_back_to_config_and_cli_override(tmp_path):
    card = build_model_card(_finished_run(tmp_path, with_protocol=False), weights_override="coco:yolo11s.pt")
    assert card["protocol"]["model_weights"] == "coco:yolo11s.pt"
    assert card["protocol"]["image_size"] == 960
    assert "eval_set_id" not in card["protocol"]                       # not promotable without it
    assert card["protocol_source"] == "config.yaml+cli"


def test_unfinished_run_cannot_be_registered(tmp_path):
    with pytest.raises(FileNotFoundError, match="final_params.npz"):
        build_model_card(tmp_path)


def test_card_tag_roundtrip():
    card = {**_card(), "run_name": "r", "seed": 42}
    back = tags_to_card(card_to_tags(card))
    assert back["metrics"] == card["metrics"]
    assert back["protocol"] == card["protocol"]
    assert back["run_class"] == "main" and back["seed"] == "42"


# ── promotion rule ────────────────────────────────────────────────────────────
POLICY = PromotionPolicy()


def test_shipped_policy_file_loads():
    p = load_policy(POLICY_PATH)
    assert p.champion_alias == "champion" and p.primary_metric == "mAP50"
    assert "eval_set_id" in p.protocol_keys and "model_weights" not in p.protocol_keys


def test_first_main_model_becomes_champion():
    assert decide_promotion(_card(), None, POLICY).ok


@pytest.mark.parametrize("run_class", ["smoke", "feasibility"])
def test_non_main_runs_are_refused(run_class):
    d = decide_promotion(_card(run_class=run_class), None, POLICY)
    assert not d.ok and any(f.startswith("run_class") for f in d.failed)


def test_better_model_on_every_class_is_promoted():
    assert decide_promotion(_card(0.55, 0.55), _card(0.5, 0.5), POLICY).ok


def test_equal_primary_metric_is_not_enough():
    assert not decide_promotion(_card(0.5, 0.5), _card(0.5, 0.5), POLICY).ok


def test_one_class_dropping_blocks_promotion():
    d = decide_promotion(_card(0.6, 0.6, AP50_bus=0.49), _card(0.5, 0.5), POLICY)
    assert not d.ok and len(d.failed) == 1 and d.failed[0].startswith("class:bus")


def test_different_eval_protocol_blocks_promotion():
    other = {**PROTO, "image_size": 640}
    d = decide_promotion(_card(0.9, 0.9, protocol=other), _card(0.5, 0.5), POLICY)
    assert not d.ok and any(f.startswith("same_protocol") for f in d.failed)


def test_architecture_change_alone_does_not_block_promotion():
    bigger = {**PROTO, "model_weights": "coco:yolo11m.pt"}
    assert decide_promotion(_card(0.6, 0.6, protocol=bigger), _card(0.5, 0.5), POLICY).ok


def test_missing_eval_set_id_blocks_promotion():
    proto = {k: v for k, v in PROTO.items() if k != "eval_set_id"}
    d = decide_promotion(_card(protocol=proto), None, POLICY)
    assert not d.ok and any(f.startswith("protocol_complete") for f in d.failed)


# ── MLflow registry round trip (needs mlflow) ─────────────────────────────────
def test_register_promote_refuse_and_force(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    from src.mlops.registry import current_champion, promote, register_run
    from src.mlops.tracking import start_tracking

    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment("reg", artifact_location=(tmp_path / "artifacts").as_uri())
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)

    def register(name, map50, per_class):
        run = _finished_run(tmp_path / name)
        lines = [f"mAP50,{map50}"] + [f"AP50_{c},{per_class}" for c in CLASSES]
        (run / "metrics.csv").write_text("metric,value\n" + "\n".join(lines) + "\n", encoding="utf-8")
        tracker = start_tracking(run, "reg", uri, required=True)
        version = register_run(tracker, run, build_model_card(run), POLICY)
        tracker.end()
        return version

    v1 = register("one", 0.50, 0.50)
    assert str(client.get_model_version_by_alias(POLICY.model_name, "candidate").version) == v1
    assert current_champion(POLICY, client) is None
    assert promote(v1, POLICY, client=client).ok
    assert str(current_champion(POLICY, client).version) == v1

    v2 = register("two", 0.45, 0.45)                                   # worse: refused
    assert not promote(v2, POLICY, client=client).ok
    assert str(current_champion(POLICY, client).version) == v1

    d = promote(v2, POLICY, client=client, force_reason="rollback drill")   # forced: audited
    assert not d.ok
    champ = current_champion(POLICY, client)
    assert str(champ.version) == v2
    assert champ.tags["promotion.forced"] == "rollback drill"
    assert "primary_metric" in champ.tags["promotion.failed_checks"]
    assert champ.tags["promotion.previous_champion"] == v1
