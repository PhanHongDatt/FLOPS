"""ADR-034 evaluation protocol P1: only the four target classes reach NMS."""
from __future__ import annotations

import torch

from src.model.class_filter import keep_only_classes


def test_hook_zeroes_only_non_target_scores():
    from src.model.yolo_wrapper import build_model
    model = build_model("coco:yolo11n.pt")
    net = model.model.eval()
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        ref = net(x)[0]
        handle = keep_only_classes(model, [2, 3, 5, 7])
        out = net(x)[0]
        handle.remove()
        again = net(x)[0]
    keep = [4 + c for c in (2, 3, 5, 7)]
    drop = [i for i in range(4, ref.shape[1]) if i not in keep]
    assert torch.equal(out[:, :4], ref[:, :4]) and torch.equal(out[:, keep], ref[:, keep])
    assert out[:, drop].abs().max() == 0 and ref[:, drop].abs().max() > 0
    assert torch.equal(again, ref)                                     # removable


def test_hook_is_inert_in_training_mode():
    from src.model.yolo_wrapper import build_model
    model = build_model("coco:yolo11n.pt")
    handle = keep_only_classes(model, [5])
    net = model.model.train()
    out = net(torch.rand(1, 3, 64, 64))
    assert isinstance(out, list)                                       # raw training outputs, untouched
    handle.remove()
