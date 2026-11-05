"""M2 — paste motorcycle crops from the client's own images (docs/Solution.md §4, ADR-029).

[LITERATURE] Kisantal et al., "Augmentation for small object detection", arXiv:1902.07296 (2019):
copy-pasting small objects several times per image raises small-object AP; Ghiasi et al., "Simple
Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation", CVPR 2021. BDD100K has no
masks, so the crop is the box (+2 px each side) with a feathered edge instead of a mask `[ENGINEERING]`.
Ultralytics' own ``copy_paste`` needs segments and is not used.

The bank holds only crops of the client's own training images and never leaves the client.

[YOLO-DOC] ultralytics 8.3.253: ``BaseDataset.get_image_and_label`` loads the (resized) image and then
calls ``update_labels_info``; Mosaic fetches each of its tiles through ``get_image_and_label``. Pasting in
``update_labels_info`` therefore happens per image, before mosaic, as the spec asks. The trainer builds
its dataset through ``trainer.build_dataset``; ``install`` wraps it from ``on_pretrain_routine_start``
(called in the trainer's ``__init__``, before the dataloaders and their workers exist).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics.data.dataset import YOLODataset

PAD_PX = 2


@dataclass(frozen=True)
class PasteConfig:
    p: float = 0.5
    n_max: int = 3
    min_y_frac: float = 0.5
    max_iou: float = 0.05
    feather_px: int = 3
    class_id: int = 3
    tries: int = 20
    debug_dir: str | None = None
    debug_n: int = 20

    def __post_init__(self) -> None:
        if not 0 <= self.p <= 1 or self.n_max < 1 or not 0 <= self.min_y_frac < 1 or not 0 <= self.max_iou <= 1:
            raise ValueError(f"bad moto_copy_paste config {self}")


def build_bank(dataset: Any, class_id: int) -> list[np.ndarray]:
    """Crops (H, W, 3, uint8) of every ``class_id`` box, at the scale ``load_image`` returns; each image once."""
    bank: list[np.ndarray] = []
    seen: set[str] = set()
    for i, lab in enumerate(dataset.labels):
        if lab["im_file"] in seen or not np.any(lab["cls"].reshape(-1) == class_id):
            continue
        seen.add(lab["im_file"])
        img = dataset.load_image(i)[0]
        h, w = img.shape[:2]
        for c, (cx, cy, bw, bh) in zip(lab["cls"].reshape(-1), lab["bboxes"]):
            if int(c) != class_id:
                continue
            x1, y1 = int(max((cx - bw / 2) * w - PAD_PX, 0)), int(max((cy - bh / 2) * h - PAD_PX, 0))
            x2, y2 = int(min((cx + bw / 2) * w + PAD_PX, w)), int(min((cy + bh / 2) * h + PAD_PX, h))
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                bank.append(img[y1:y2, x1:x2].copy())
    return bank


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one xyxy box with ``boxes`` [N, 4] (pixels)."""
    if len(boxes) == 0:
        return np.zeros(0)
    ix = np.clip(np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]), 0, None)
    iy = np.clip(np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]), 0, None)
    inter = ix * iy
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-9)


def feather_mask(h: int, w: int, px: int) -> np.ndarray:
    """[h, w, 1] weights: 1 inside, linear ramp to ~0 over ``px`` pixels at the border."""
    if px <= 0:
        return np.ones((h, w, 1), np.float32)
    ys = np.minimum(np.arange(h), np.arange(h)[::-1]) + 1
    xs = np.minimum(np.arange(w), np.arange(w)[::-1]) + 1
    m = np.minimum.outer(ys, xs).astype(np.float32) / (px + 1)
    return np.clip(m, 0, 1)[..., None]


def paste(img: np.ndarray, cls: np.ndarray, bboxes: np.ndarray, bank: list[np.ndarray], cfg: PasteConfig,
          rng: random.Random) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """New (img, cls [N,1], bboxes [N,4] normalised xywh); the inputs are not modified."""
    if not bank or rng.random() >= cfg.p:
        return img, cls, bboxes
    h, w = img.shape[:2]
    out = img.copy()
    xyxy = np.stack([(bboxes[:, 0] - bboxes[:, 2] / 2) * w, (bboxes[:, 1] - bboxes[:, 3] / 2) * h,
                     (bboxes[:, 0] + bboxes[:, 2] / 2) * w, (bboxes[:, 1] + bboxes[:, 3] / 2) * h], 1) \
        if len(bboxes) else np.zeros((0, 4))
    new_cls, new_box = [], []
    for _ in range(rng.randint(1, cfg.n_max)):
        crop = bank[rng.randrange(len(bank))]
        ch, cw = crop.shape[:2]
        if ch >= h or cw >= w:
            continue
        for _ in range(cfg.tries):
            cy = rng.uniform(max(cfg.min_y_frac * h, ch / 2), h - ch / 2)
            cx = rng.uniform(cw / 2, w - cw / 2)
            x1, y1 = int(round(cx - cw / 2)), int(round(cy - ch / 2))
            box = np.array([x1, y1, x1 + cw, y1 + ch], dtype=float)
            if y1 + ch / 2 <= cfg.min_y_frac * h or np.any(_iou(box, xyxy) > cfg.max_iou):
                continue
            a = feather_mask(ch, cw, cfg.feather_px)
            region = out[y1:y1 + ch, x1:x1 + cw].astype(np.float32)
            out[y1:y1 + ch, x1:x1 + cw] = (a * crop + (1 - a) * region).round().astype(out.dtype)
            xyxy = np.vstack([xyxy, box])
            new_cls.append([cfg.class_id])
            new_box.append([(x1 + cw / 2) / w, (y1 + ch / 2) / h, cw / w, ch / h])
            break
    if not new_cls:
        return img, cls, bboxes
    return (out, np.vstack([cls.reshape(-1, 1), np.array(new_cls, dtype=cls.dtype)]),
            np.vstack([bboxes.reshape(-1, 4), np.array(new_box, dtype=bboxes.dtype)]))


class MotoPasteDataset(YOLODataset):
    """YOLODataset whose training images get M2 pastes; set on an existing instance by ``install``."""

    paste_bank: list[np.ndarray]
    paste_cfg: PasteConfig
    _paste_rng: random.Random | None = None
    _debug_written = 0

    def update_labels_info(self, label: dict) -> dict:
        if self._paste_rng is None:                       # per process; Ultralytics seeds each worker's random
            self._paste_rng = random.Random(random.getrandbits(64))
        img, cls, boxes = paste(label["img"], label["cls"], label["bboxes"], self.paste_bank, self.paste_cfg,
                                self._paste_rng)
        if img is not label["img"]:
            label = {**label, "img": img, "cls": cls, "bboxes": boxes}
            self._write_debug(label)
        return super().update_labels_info(label)

    def _write_debug(self, label: dict) -> None:
        cfg = self.paste_cfg
        if not cfg.debug_dir or self._debug_written >= cfg.debug_n:
            return
        import os

        import cv2
        img = label["img"].copy()
        h, w = img.shape[:2]
        for c, (cx, cy, bw, bh) in zip(label["cls"].reshape(-1), label["bboxes"]):
            colour = (0, 0, 255) if int(c) == cfg.class_id else (0, 200, 0)
            cv2.rectangle(img, (int((cx - bw / 2) * w), int((cy - bh / 2) * h)),
                          (int((cx + bw / 2) * w), int((cy + bh / 2) * h)), colour, 1)
        Path(cfg.debug_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(Path(cfg.debug_dir) / f"paste_{os.getpid()}_{self._debug_written:02d}.jpg"), img)
        self._debug_written += 1


class MotoCopyPaste:
    """Ultralytics callback: wraps ``trainer.build_dataset`` so the TRAIN dataset becomes MotoPasteDataset."""

    def __init__(self, cfg: PasteConfig) -> None:
        self.cfg = cfg
        self.bank_size = 0

    def on_pretrain_routine_start(self, trainer: Any) -> None:
        original = trainer.build_dataset

        def build_dataset(img_path: str, mode: str = "train", batch: int | None = None) -> Any:
            ds = original(img_path, mode, batch)
            if mode == "train" and type(ds) is YOLODataset:
                ds.paste_cfg = self.cfg
                ds.paste_bank = build_bank(ds, self.cfg.class_id)
                ds.__class__ = MotoPasteDataset
                self.bank_size = len(ds.paste_bank)
            return ds

        trainer.build_dataset = build_dataset
