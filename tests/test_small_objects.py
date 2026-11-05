"""M1-M3 for motorcycle (ADR-029, docs/Solution.md §4)."""
from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.data.moto_paste import PasteConfig, _iou, build_bank, feather_mask, paste
from src.federated.client import small_object_overrides
from src.model.nwd_assign import NWDAssign, NWDTaskAlignedAssigner, mean_box_size, nwd
from src.utils.config import load_small_objects_config


# ── M1 ────────────────────────────────────────────────────────────────────
def test_m1_mosaic_off_only_in_last_k_rounds():
    cfg = {"enabled": True, "scale": 0.25, "mosaic_off_last_rounds": 5}
    got = [small_object_overrides(cfg, r, 20) for r in range(1, 21)]
    assert all(g["scale"] == 0.25 for g in got)
    assert [g["mosaic"] for g in got] == [1.0] * 15 + [0.0] * 5
    assert small_object_overrides({**cfg, "enabled": False}, 20, 20) == {}
    assert small_object_overrides(None, 20, 20) == {}
    with pytest.raises(ValueError):
        small_object_overrides(cfg, 1, None)


def test_config_blocks_off_by_default():
    assert load_small_objects_config({}) == {}
    cfg = load_small_objects_config({"small_obj_aug": {"enabled": True}, "nwd_assign": {"enabled": True}})
    assert cfg["small_obj_aug"] == {"enabled": True, "scale": 0.25, "mosaic_off_last_rounds": 5}
    assert cfg["nwd_assign"] == {"alpha": 0.5, "C": None, "class_id": 3} and "moto_copy_paste" not in cfg
    with pytest.raises(ValueError):
        load_small_objects_config({"mixup": {}})


# ── M2 ────────────────────────────────────────────────────────────────────
def _scene():
    img = np.full((100, 200, 3), 50, np.uint8)
    cls = np.array([[2.0]], np.float32)                                # one car, upper left
    boxes = np.array([[0.2, 0.3, 0.2, 0.2]], np.float32)
    bank = [np.full((10, 8, 3), 250, np.uint8), np.full((6, 12, 3), 200, np.uint8)]
    return img, cls, boxes, bank


def test_m2_pasted_boxes_valid_labelled_and_apart():
    img, cls, boxes, bank = _scene()
    cfg = PasteConfig(p=1.0, n_max=3)
    for seed in range(30):
        out, c, b = paste(img, cls, boxes, bank, cfg, random.Random(seed))
        assert np.array_equal(img, _scene()[0]) and len(cls) == 1        # inputs untouched
        new_c, new_b = c[1:].reshape(-1), b[1:]
        assert len(new_c) >= 1 and np.all(new_c == 3)
        h, w = img.shape[:2]
        xyxy = np.stack([(b[:, 0] - b[:, 2] / 2) * w, (b[:, 1] - b[:, 3] / 2) * h,
                         (b[:, 0] + b[:, 2] / 2) * w, (b[:, 1] + b[:, 3] / 2) * h], 1)
        assert np.all(xyxy[:, :2] >= -1e-6) and np.all(xyxy[:, 2] <= w + 1e-6) and np.all(xyxy[:, 3] <= h + 1e-6)
        assert np.all(new_b[:, 1] > 0.5)                                   # centres in the lower half
        for i in range(1, len(xyxy)):
            others = np.delete(xyxy, i, 0)
            assert np.all(_iou(xyxy[i], others) <= cfg.max_iou + 1e-9)


def test_m2_off_or_empty_bank_leaves_image_alone():
    img, cls, boxes, bank = _scene()
    out = paste(img, cls, boxes, bank, PasteConfig(p=0.0), random.Random(0))
    assert out[0] is img and out[1] is cls and out[2] is boxes
    assert paste(img, cls, boxes, [], PasteConfig(p=1.0), random.Random(0))[0] is img


def test_feather_mask_and_bank():
    m = feather_mask(9, 9, 3)
    assert m[4, 4, 0] == 1.0 and 0 < m[0, 0, 0] < 0.5
    img = np.zeros((100, 200, 3), np.uint8)
    img[40:60, 90:110] = 255
    lab = {"im_file": "a.jpg", "cls": np.array([[3.0], [2.0]]),
           "bboxes": np.array([[0.5, 0.5, 0.1, 0.2], [0.1, 0.1, 0.1, 0.1]])}
    ds = SimpleNamespace(labels=[lab, dict(lab)], load_image=lambda i: (img, None, None))   # RFS repeat
    bank = build_bank(ds, 3)
    assert len(bank) == 1 and bank[0].shape == (24, 24, 3)               # one box, +2 px each side, once


# ── M3 ────────────────────────────────────────────────────────────────────
def test_nwd_range_and_identity():
    a = torch.tensor([[10.0, 10.0, 20.0, 30.0]])
    assert torch.allclose(nwd(a, a, 12.0), torch.ones(1))
    far = torch.tensor([[200.0, 200.0, 220.0, 230.0]])
    v = nwd(a, far, 12.0)
    assert 0 < v.item() < 1e-3


def test_nwd_alpha_one_assigns_like_baseline():
    from ultralytics.utils.tal import TaskAlignedAssigner
    torch.manual_seed(0)
    base = TaskAlignedAssigner(topk=10, num_classes=80, alpha=0.5, beta=6.0)
    anchors = torch.stack(torch.meshgrid(torch.arange(4, 64, 8.0), torch.arange(4, 64, 8.0), indexing="xy"), -1).view(-1, 2)
    pd_scores = torch.rand(1, len(anchors), 80)
    pd_boxes = torch.cat([anchors - 6, anchors + 6], -1).unsqueeze(0)
    gt_labels = torch.tensor([[[3.0], [2.0]]])
    gt_boxes = torch.tensor([[[10.0, 10.0, 22.0, 26.0], [30.0, 30.0, 60.0, 60.0]]])
    mask = torch.ones(1, 2, 1)
    ref = base(pd_scores, pd_boxes, anchors, gt_labels, gt_boxes, mask)
    same = NWDTaskAlignedAssigner(base, 1.0, 12.0)(pd_scores, pd_boxes, anchors, gt_labels, gt_boxes, mask)
    for r, s in zip(ref, same):
        assert torch.equal(r, s)
    mixed = NWDTaskAlignedAssigner(base, 0.5, 12.0)(pd_scores, pd_boxes, anchors, gt_labels, gt_boxes, mask)
    assert not torch.equal(ref[2], mixed[2])                              # target scores move


def test_mean_box_size_and_callback_restores_assigner():
    labels = [{"im_file": "a", "shape": (720, 1280), "cls": np.array([[3.0], [2.0]]),
               "bboxes": np.array([[0.5, 0.5, 0.02, 0.04], [0.5, 0.5, 0.5, 0.5]])}]
    c = mean_box_size(labels + labels, 3, 960)                             # repeated image counted once
    assert c == pytest.approx(np.sqrt(0.02 * 1280 * 0.75 * 0.04 * 720 * 0.75))
    assigner = object()
    crit = SimpleNamespace(assigner=SimpleNamespace(topk=10, num_classes=80, alpha=0.5, beta=6.0, eps=1e-9))
    original = crit.assigner
    model = SimpleNamespace(criterion=crit)
    trainer = SimpleNamespace(model=model, args=SimpleNamespace(imgsz=960),
                              train_loader=SimpleNamespace(dataset=SimpleNamespace(labels=labels)))
    hook = NWDAssign(alpha=0.5)
    hook.on_train_start(trainer)
    assert isinstance(crit.assigner, NWDTaskAlignedAssigner) and hook.c_used == pytest.approx(c)
    hook.on_train_end(trainer)
    assert crit.assigner is original and assigner is not None


def test_m2_on_real_yolo_dataset_through_mosaic(tmp_path):
    import cv2
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset

    from src.data.moto_paste import MotoCopyPaste, MotoPasteDataset
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    rng = np.random.default_rng(0)
    for i in range(4):
        cv2.imwrite(str(tmp_path / "images" / f"{i}.jpg"), rng.integers(0, 255, (360, 640, 3), dtype=np.uint8))
        (tmp_path / "labels" / f"{i}.txt").write_text("3 0.5 0.7 0.05 0.1\n2 0.2 0.3 0.2 0.2\n")
    hyp = get_cfg()
    ds = YOLODataset(img_path=str(tmp_path / "images"), imgsz=320, augment=True, hyp=hyp,
                     data={"names": {i: str(i) for i in range(80)}, "channels": 3}, batch_size=2)
    trainer = SimpleNamespace(build_dataset=lambda p, mode="train", batch=None: ds)
    cb = MotoCopyPaste(PasteConfig(p=1.0, n_max=3, debug_dir=str(tmp_path / "dbg"), debug_n=2))
    cb.on_pretrain_routine_start(trainer)
    assert trainer.build_dataset("x", "train") is ds and isinstance(ds, MotoPasteDataset) and cb.bank_size == 4
    plain = ds.labels[0]["cls"].copy()
    single = ds.get_image_and_label(0)
    assert len(single["cls"]) > 2 and np.all(single["cls"][2:] == 3)      # pasted motorcycles, labelled
    assert len(single["instances"]) == len(single["cls"])
    assert np.array_equal(ds.labels[0]["cls"], plain)                       # stored labels not modified
    item = ds[0]                                                            # full pipeline incl. mosaic
    assert item["img"].shape[0] == 3 and len(item["cls"]) == len(item["bboxes"])
    assert len(list((tmp_path / "dbg").glob("*.jpg"))) == 2
    val = SimpleNamespace(build_dataset=lambda p, mode="val", batch=None: "val-ds")
    MotoCopyPaste(PasteConfig()).on_pretrain_routine_start(val)
    assert val.build_dataset("x", "val") == "val-ds"                        # val set never touched
