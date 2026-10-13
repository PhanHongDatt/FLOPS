"""ADR-020 N1 — detection-adapted CCVR: statistics, merging, cell selection and row-only re-fitting."""
from __future__ import annotations

import numpy as np
import torch

from src.preservation.classifier_calibration import (
    BACKGROUND, FeatureStats, background_mask, box_cells, calibrate, collect_stats, level_for_box, merge_all,
)


def test_level_by_size():
    assert [level_for_box(v) for v in (10, 63.9, 64, 127, 128, 400)] == [0, 0, 1, 1, 2, 2]


def test_merged_statistics_equal_pooled_statistics():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(50, 3)), rng.normal(2.0, 1.0, size=(30, 3))
    s1, s2 = FeatureStats(3), FeatureStats(3)
    s1.add(0, 5, a)
    s2.add(0, 5, b)
    mean, cov, n = merge_all([s1, s2]).gaussian(0, 5)
    both = np.vstack([a, b])
    assert n == 80 and np.allclose(mean, both.mean(0)) and np.allclose(cov, np.cov(both.T))
    assert FeatureStats(3).gaussian(0, 5) is None
    assert s1.gaussian(1, 5) is None


def test_box_cells_and_background_mask():
    cells = box_cells(np.array([[20, 20, 10, 10], [300, 300, 100, 100], [320, 320, 400, 400]]), [8, 16, 32], (640, 640))
    assert cells == [(0, 2, 2), (1, 18, 18), (2, 10, 10)]
    m = background_mask(np.array([[16, 16, 16, 16]]), 8, 4, 4)
    assert not m[1, 1] and not m[1, 2] and m[3, 3]


class _Det(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        self.cv3 = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Identity(), torch.nn.Conv2d(4, 8, 1)) for _ in range(3)])

    def forward(self, xs):
        return [self.cv3[i](x) for i, x in enumerate(xs)]


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = torch.nn.Conv2d(3, 4, 1)
        self.model = torch.nn.ModuleList([self.stem, _Det()])

    def forward(self, img):
        f = self.stem(img)
        return self.model[-1]([torch.nn.functional.avg_pool2d(f, s) for s in (8, 16, 32)])


def test_collect_stats_hooks_the_class_conv_input():
    m = _Model()
    batch = {"img": torch.randint(0, 255, (2, 3, 64, 64), dtype=torch.uint8),
             "cls": torch.tensor([[5.0], [2.0], [0.0]]), "batch_idx": torch.tensor([0.0, 1.0, 1.0]),
             "bboxes": torch.tensor([[0.25, 0.25, 0.2, 0.2], [0.5, 0.5, 0.1, 0.1], [0.7, 0.7, 0.1, 0.1]])}
    st = collect_stats(m, [batch], target_ids=[2, 5], bg_per_image=3)
    assert st.sums[(0, 5)]["n"] == 1 and st.sums[(0, 2)]["n"] == 1
    assert (0, 0) not in st.sums                                  # class 0 is not a target
    assert st.sums[(0, BACKGROUND)]["n"] == 6                    # 3 per image on level 0
    assert not m.model[-1].cv3[0][-1]._forward_pre_hooks        # hooks removed


def test_calibrate_changes_only_target_rows_and_separates_classes():
    m = _Model()
    conv = m.model[-1].cv3[0][-1]
    before = conv.weight.data.clone()
    st = FeatureStats(4)
    rng = np.random.default_rng(1)
    st.add(0, 5, rng.normal([3, 0, 0, 0], 0.3, size=(200, 4)))
    st.add(0, 2, rng.normal([0, 3, 0, 0], 0.3, size=(200, 4)))
    st.add(0, BACKGROUND, rng.normal([0, 0, 0, 0], 0.3, size=(400, 4)))
    rep = calibrate(m, st, target_ids=[2, 5], n_pos=300, n_bg=600, steps=400, lr=0.05)
    assert rep[0]["loss_end"] < rep[0]["loss_start"] and rep[1] == {"skipped": 1.0}
    untouched = [i for i in range(8) if i not in (2, 5)]
    assert torch.equal(conv.weight.data[untouched], before[untouched])
    x = torch.tensor([[3.0, 0, 0, 0], [0, 3.0, 0, 0]])
    logits = x @ conv.weight.data[:, :, 0, 0].T + conv.bias.data
    assert logits[0, 5] > logits[0, 2] and logits[1, 2] > logits[1, 5]


def test_cls_head_freeze_list_names_the_class_conv_of_every_level():
    from ultralytics import YOLO

    from src.model.yolo_wrapper import cls_head_freeze_list
    m = YOLO("yolo11n.yaml")
    names = cls_head_freeze_list(m)
    assert names == ["23.cv3.0.2", "23.cv3.1.2", "23.cv3.2.2"]
    hit = [k for k, _ in m.model.named_parameters() if any(f"model.{n}." in k for n in names)]
    assert len(hit) == 6 and all(".cv3." in k for k in hit)          # weight + bias per level, nothing else


def test_p5f_preset_is_p4nokd_plus_frozen_classifier():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.run_fl_experiment import _ABLATIONS
    p5f = dict(_ABLATIONS["P5f"])
    assert p5f.pop("freeze_cls_head") is True and p5f == _ABLATIONS["P4noKD"]
