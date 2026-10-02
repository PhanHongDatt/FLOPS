from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from src.data.bdd100k import TARGET_CLASSES

if TYPE_CHECKING:  # avoid hard dependency at import time
    from ultralytics import YOLO


def build_model(weights: str = "yolov8n.pt", init_seed: int = 0) -> "YOLO":
    """Return a YOLO detector with exactly ``len(TARGET_CLASSES)`` classes.

    ``YOLO("yolov8n.pt")`` alone is the 80-class COCO model. Clients turn into
    nc=4 models inside their first ``model.train()``, so any model that never
    trains (the server's evaluate model) kept nc=80 and ``set_parameters``
    crashed with a cv3 size mismatch after round 1. The whole cv3 branch
    differs, not only the class rows: Detect uses ``c3 = max(ch[0], min(nc, 100))``
    hidden channels [YOLO-DOC: ultralytics 8.3.253 nn/modules/head.py].

    When the source has a different class count, rebuild at nc=4 and transfer
    weights with ``DetectionModel.load`` — the same shape-matched
    ``intersect_dicts`` rule Ultralytics applies itself, so backbone/neck/cv2/dfl
    come from the checkpoint and cv3 is freshly initialised. Construction runs
    under a fixed ``init_seed`` so every Ray actor and the server build the
    identical initial model; the caller's RNG state is left untouched.

    A checkpoint that is already nc=4 (e.g. the G2 centralized ``best.pt``) is
    returned as loaded.
    """
    # Lazy import — lets modules that only use evaluate()/get_parameters()
    # (e.g. unit tests with mock models) import this file without needing
    # ultralytics installed on the dev machine.
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    nc = len(TARGET_CLASSES)
    with torch.random.fork_rng():
        torch.manual_seed(init_seed)
        model = YOLO(weights)
        if model.model.model[-1].nc != nc:
            rebuilt = DetectionModel(model.model.yaml, nc=nc, verbose=False)
            rebuilt.load(model.model, verbose=False)
            model.model = rebuilt
    model.model.names = dict(enumerate(TARGET_CLASSES))
    return model


def get_parameters(model: "YOLO") -> list[np.ndarray]:
    """Full ordered state_dict as NDArrays.

    NOTE: this deliberately includes non-float buffers (e.g. BatchNorm
    ``num_batches_tracked``). They are averaged like weights by FedAvg and cast
    back to their original dtype in ``set_parameters``, which is harmless for
    these counters. They are kept in the list because ``param_names`` index
    alignment is what ``preservation.mask`` and ``ClassAwareAggregation`` rely
    on — dropping them would silently shift every index.
    """
    return [
        val.cpu().numpy()
        for val in model.model.state_dict().values()
    ]


def set_parameters(model: "YOLO", parameters: list[np.ndarray]) -> None:
    import torch  # lazy so tests without torch can still import evaluate
    state_dict = model.model.state_dict()
    if len(parameters) != len(state_dict):
        raise ValueError(
            f"Parameter count mismatch: received {len(parameters)} arrays for "
            f"{len(state_dict)} state_dict entries. Silent zip() truncation here "
            "would load a partially-updated model."
        )
    new_state = {
        k: torch.as_tensor(np.ascontiguousarray(v)).to(dtype=state_dict[k].dtype)
        for k, v in zip(state_dict.keys(), parameters)
    }
    model.model.load_state_dict(new_state, strict=True)


def train_one_round(
    model: "YOLO",
    data_yaml: Path,
    epochs: int,
    batch: int,
    img_size: int,
    lr0: float,
    device: int | str,
    project: Path,
    name: str,
    seed: int = 0,
    workers: int = 2,
    warmup_epochs: float = 0.0,
    close_mosaic: int = 0,
    run_val: bool = False,
    deterministic: bool = True,
    nbs: int | None = None,
    extra_overrides: dict[str, Any] | None = None,
    val_stub_images: int = 8,
    proximal_mu: float = 0.0,
) -> dict[str, float]:
    """One round of local training.

    Defaults encode four corrections over the previous implementation:

    * ``run_val=False`` → Ultralytics no longer validates after every local
      epoch. Previously every client's ``data_*.yaml`` pointed ``val:`` at the
      SHARED GLOBAL val set, so each client ran a full 10k-image validation per
      epoch (pure waste), and ``model.train()`` then reloaded ``best.pt`` chosen
      by fitness **on that global val set** — i.e. local model selection on the
      test set. Harmless only while ``epochs == 1`` (best == last); a latent
      leak as soon as local epochs > 1.
    * ``warmup_epochs=0.0`` → Ultralytics' default is 3.0, which with
      ``epochs=1`` meant every FL round consisted entirely of warm-up and the
      learning rate never reached ``lr0``.
    * ``workers=2`` → the default of 8 per client x 4 concurrent Ray actors
      oversubscribes a 4-vCPU Kaggle notebook (plan.md §13 K2).
    * ``seed`` is forwarded → previously Ultralytics used its default
      ``seed=0`` in every client of every round, so the experiment seeds
      (42/123/2024) never influenced local training and multi-seed std was
      artificially narrow (CLAUDE.md §14).

    ``nbs`` (nominal batch size) enables gradient accumulation so a small
    ``batch`` can keep the canonical effective batch on a low-VRAM GPU.
    """
    if not run_val:
        data_yaml = val_stub_data_yaml(data_yaml, Path(project) / name, n_images=val_stub_images)
    overrides: dict[str, Any] = dict(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=img_size,
        optimizer="SGD",
        lr0=lr0,
        device=device,
        project=str(project),
        name=name,
        exist_ok=True,
        verbose=False,
        seed=seed,
        deterministic=deterministic,
        workers=workers,
        warmup_epochs=warmup_epochs,
        close_mosaic=close_mosaic,
        val=run_val,
        plots=False,
    )
    if nbs is not None:
        overrides["nbs"] = nbs
    if extra_overrides:
        overrides.update(extra_overrides)

    snapshot: dict[str, Any] = {}

    def _snapshot_final_ema(trainer: Any) -> None:
        if trainer.epoch + 1 >= trainer.epochs:
            snapshot.update(
                {k: v.detach().clone() for k, v in trainer.ema.ema.state_dict().items()}
            )

    model.add_callback("on_train_epoch_end", _snapshot_final_ema)
    # FedProx: mu * (w - w_global) added to every gradient (src/federated/proximal.py)
    from src.federated.proximal import ProximalTerm
    prox = ProximalTerm(proximal_mu)
    if prox.active:
        model.add_callback("on_train_start", prox.on_train_start)
        model.add_callback("on_train_end", prox.on_train_end)
    try:
        results = model.train(**overrides)
    finally:
        # callbacks live on the YOLO object and would pile up across FL rounds
        model.callbacks["on_train_epoch_end"].remove(_snapshot_final_ema)
        if prox.active:
            model.callbacks["on_train_start"].remove(prox.on_train_start)
            model.callbacks["on_train_end"].remove(prox.on_train_end)
            prox.on_train_end(None)   # remove hooks even if training raised
    _load_fp32_ema(model, snapshot)

    # With val=False, results_dict is empty or partially populated — that is
    # expected, not an error. Local metrics are not used for reporting; the
    # reporting path is the server's centralized evaluate_fn.
    metrics: dict[str, float] = {}
    if results is not None and hasattr(results, "results_dict"):
        for k, v in (results.results_dict or {}).items():
            try:
                metrics[k] = float(v)
            except (TypeError, ValueError):
                continue
    return metrics


_IMG_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm"}


def _split_images(data: dict[str, Any], data_yaml: Path, split: str) -> list[str]:
    """Image paths of a split entry (``train``/``val``): an image-list .txt or a directory."""
    root = Path(data.get("path") or data_yaml.parent)
    entry = Path(data[split])
    entry = entry if entry.is_absolute() else root / entry
    if entry.is_dir():
        return [str(p) for p in sorted(entry.iterdir()) if p.suffix.lower() in _IMG_SUFFIXES]
    lines = [ln.strip() for ln in entry.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [ln if Path(ln).is_absolute() else str(entry.parent / ln) for ln in lines]


def _write_split_yaml(data: dict[str, Any], data_yaml: Path, out_dir: Path,
                      images: list[str], tag: str) -> Path:
    import yaml

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    listing = out_dir / f"{tag}.txt"
    listing.write_text("\n".join(images) + "\n", encoding="utf-8")
    out = out_dir / f"{Path(data_yaml).stem}_{tag}.yaml"
    out.write_text(yaml.safe_dump({**data, "val": str(listing.resolve())}), encoding="utf-8")
    return out


def val_stub_data_yaml(data_yaml: Path, out_dir: Path, n_images: int = 8) -> Path:
    """Copy of ``data_yaml`` whose ``val`` is the first ``n_images`` training images.

    Ultralytics validates on the final epoch even with ``val=False`` and then
    runs ``final_eval`` [YOLO-DOC: ultralytics 8.3.253 engine/trainer.py] —
    two passes over the GLOBAL val set per client per round. The weights
    train_one_round returns are the fp32 EMA snapshot, independent of that
    validation, so a tiny stub costs nothing scientifically. Reported metrics
    always come from ``evaluate()`` on the real val set.
    """
    import yaml

    data_yaml = Path(data_yaml)
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    images = _split_images(data, data_yaml, "train")[:n_images]
    if not images:
        raise ValueError(f"{data_yaml}: no training images found for the val stub")
    return _write_split_yaml(data, data_yaml, out_dir, images, "val_stub")


def val_subset_data_yaml(data_yaml: Path, out_dir: Path, n_images: int, seed: int) -> Path:
    """Copy of ``data_yaml`` whose ``val`` is a seeded subset of ``n_images`` val images.

    Used where many evaluations of one model must share the same, cheaper val
    set (F2 perturbations). Selection is seeded and kept in sorted order.
    """
    import yaml

    data_yaml = Path(data_yaml)
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    images = sorted(_split_images(data, data_yaml, "val"))
    if n_images < len(images):
        keep = np.random.default_rng(seed).choice(len(images), size=n_images, replace=False)
        images = [images[i] for i in sorted(keep)]
    return _write_split_yaml(data, data_yaml, out_dir, images, f"val_subset_{n_images}_s{seed}")


def _load_fp32_ema(model: "YOLO", snapshot: dict[str, Any]) -> None:
    """Load the fp32 EMA captured at the end of the final epoch's training.

    Two Ultralytics behaviours fp16-round the EMA before ``train()`` returns
    [YOLO-DOC: ultralytics 8.3.253]:

    * ``engine/model.py train()`` reloads best.pt/last.pt, whose EMA is saved
      with ``.half()`` (``engine/trainer.py save_model``);
    * on CUDA with AMP, the final-epoch validation runs ``trainer.ema.ema.half()``
      IN PLACE (``engine/validator.py``) and only casts back afterwards.

    Uploading those weights rounds every client update to fp16 each round; on
    short local runs that is the size of the class-row update F3 must measure.
    The snapshot is taken in ``on_train_epoch_end``, which runs before that
    validation. The EMA itself is kept (unchanged semantics).
    """
    if not snapshot:
        raise RuntimeError(
            "No fp32 EMA snapshot was captured during train(); refusing to return "
            "fp16-rounded weights. Re-verify the callback order for the pinned "
            "ultralytics version."
        )
    model.model.load_state_dict(snapshot, strict=True)


def _per_class_fp_fn(results: Any) -> dict[str, float]:
    """Per-class FP/FN from the Ultralytics confusion matrix.

    CLAUDE.md §13 and ``configs/base_config.yaml`` both require FP_per_class and
    FN_per_class; nothing produced them before.

    Convention (ultralytics ``ConfusionMatrix.process_batch``): the matrix is
    indexed ``[predicted, actual]`` with index ``nc`` representing background,
    so for class ``c``::

        TP_c = M[c, c]
        FP_c = sum_j M[c, j] - TP_c      # includes predictions on background
        FN_c = sum_i M[i, c] - TP_c      # includes missed objects

    ``[NEEDS-VERIFICATION]`` matrix orientation must be confirmed against the
    pinned Ultralytics version during F1 runtime verification. These numbers are
    threshold-dependent (conf/iou passed to ``val()``) and are NOT the same
    quantity as the FP/FN implied by AP — the evaluation protocol must state the
    thresholds used.
    """
    out: dict[str, float] = {}
    cm = getattr(results, "confusion_matrix", None)
    matrix = getattr(cm, "matrix", None) if cm is not None else None
    if matrix is None:
        return out
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] < len(TARGET_CLASSES):
        return out
    for cls_idx, cls_name in enumerate(TARGET_CLASSES):
        if cls_idx >= m.shape[0]:
            break
        tp = float(m[cls_idx, cls_idx])
        out[f"TP_{cls_name}"] = tp
        out[f"FP_{cls_name}"] = float(m[cls_idx, :].sum()) - tp
        out[f"FN_{cls_name}"] = float(m[:, cls_idx].sum()) - tp
    return out


def confidence_stats(
    model: "YOLO",
    images: list[str],
    img_size: int,
    conf: float,
    iou: float,
    device: int | str,
    batch: int = 16,
) -> dict[str, float]:
    """Per-class prediction count and mean confidence over ``images`` (CLAUDE.md §13).

    ``YOLO.predict`` caches a predictor holding the model it first saw, and its
    AutoBackend fuses that model in place; predicting on a fresh copy with the
    cache cleared keeps the stats tied to the current weights.
    """
    nc = len(TARGET_CLASSES)
    confs: list[list[float]] = [[] for _ in range(nc)]
    original = model.model
    model.model = copy.deepcopy(original)
    model.predictor = None
    try:
        for r in model.predict(source=list(images), imgsz=img_size, conf=conf, iou=iou,
                               device=device, stream=True, verbose=False, batch=batch):
            if r.boxes is None:
                continue
            for c, cf in zip(r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()):
                if 0 <= c < nc:
                    confs[c].append(float(cf))
    finally:
        model.model = original
        model.predictor = None
    out: dict[str, float] = {}
    for c, name in enumerate(TARGET_CLASSES):
        out[f"n_pred_{name}"] = float(len(confs[c]))
        out[f"mean_conf_{name}"] = float(np.mean(confs[c])) if confs[c] else float("nan")
    return out


def evaluate(
    model: "YOLO",
    data_yaml: Path,
    img_size: int,
    conf: float,
    iou: float,
    device: int | str,
    workers: int | None = None,
) -> dict[str, float]:
    """Global-val metrics. ``workers=0`` loads images in-process — use it inside
    the Ray driver, where forking dataloader workers can deadlock."""
    val_kwargs: dict[str, Any] = {} if workers is None else {"workers": workers}
    # val() fuses Conv+BN of model.model IN PLACE (355 → 127 state_dict entries
    # for yolov8n), after which set_parameters on the same object fails. The
    # server reuses one evaluate model across rounds, so validate a copy.
    original = model.model
    model.model = copy.deepcopy(original)
    try:
        results = model.val(
            data=str(data_yaml),
            imgsz=img_size,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
            **val_kwargs,
        )
    finally:
        model.model = original
    metrics: dict[str, float] = {}
    if not results:
        return metrics

    metrics["mAP50"] = float(results.box.map50)
    metrics["mAP50-95"] = float(results.box.map)

    # Ultralytics: results.box.{ap50, ap, p, r} arrays are aligned with
    # results.box.ap_class_index, NOT with TARGET_CLASSES. Classes with zero
    # ground-truth boxes (or zero detections) in val are OMITTED from
    # ap_class_index. Previous code iterated enumerate(TARGET_CLASSES) and
    # used positional index — this mis-assigned AP to wrong class names when
    # any target class was absent from val, which is exactly the scenario H1
    # aims to measure.
    ap_class_index = list(results.box.ap_class_index)
    for i, cls_idx in enumerate(ap_class_index):
        idx = int(cls_idx)
        if 0 <= idx < len(TARGET_CLASSES):
            cls_name = TARGET_CLASSES[idx]
            metrics[f"AP50_{cls_name}"] = float(results.box.ap50[i])
            metrics[f"AP_{cls_name}"] = float(results.box.ap[i])
            metrics[f"precision_{cls_name}"] = float(results.box.p[i])
            metrics[f"recall_{cls_name}"] = float(results.box.r[i])

    metrics.update(_per_class_fp_fn(results))
    return metrics
