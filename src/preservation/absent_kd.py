"""B1 — absent-class distillation restricted to confusable foreground anchors (ADR-028).

[THESIS-HYPOTHESIS] On a client with no box of class c (bus on C0-C2), every truck/car
anchor gets the hard target 0 for c, so the c score is pushed down hardest exactly at
the c / truck / car boundary; the bus error analysis (ADR-014 D1) attributes most of
the extra missed bus to class confusion. B1 replaces that 0 by the frozen COCO
teacher's probability, on those anchors only:

    for c in M_k (classes with zero boxes on client k), for every anchor a:
        if fg[a] and label[a] in N(c):  t[a, c] = lam * min(q[a, c], q_max) * s[a]
        else:                            t[a, c] unchanged (0)

q = sigmoid(teacher logit), s[a] = task-aligned score of the anchor (the row maximum of
the TAL target). Background anchors keep target 0, so the negative signal against
false c detections on background is kept (A2b removed it and FP bus rose ~3x, ADR-006).

Related work: Shmelkov et al., ICCV 2017 (distil old-class outputs when their
annotations are absent); Lee et al., NeurIPS 2022 (FedNTD). The confusable-region
restriction and the cap q_max are this thesis's choices, not taken from either paper.

[YOLO-DOC] ultralytics 8.3.253 ``v8DetectionLoss.__call__``: ``target_scores_sum`` is
computed from the assigner output before ``self.bce(pred_scores, target_scores)``, and
the box/DFL losses receive the unmodified ``target_scores``. The edit therefore lives
inside a ``criterion.bce`` wrapper: only the classification term sees the new targets,
the normaliser and box weights stay those of the baseline. The assigner's
``target_labels`` is discarded there, but TAL targets are ``one_hot(label) * s``, so the
label of a foreground anchor is the argmax of its target row and s is its maximum.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

SCOPES = ("confusable", "all_anchors")

# Defaults of configs/methods/absent_kd.yaml (spec Solution.md §3.4); COCO head ids.
DEFAULT_CONFUSABLE: dict[int, tuple[int, ...]] = {5: (7, 2), 7: (5, 2)}   # bus -> truck, car; truck -> bus, car


def absent_class_targets(
    target: torch.Tensor,
    teacher_prob: torch.Tensor,
    missing: Sequence[int],
    confusable: Mapping[int, Sequence[int]],
    lam: float,
    q_max: float,
    scope: str = "confusable",
) -> torch.Tensor:
    """New ``[B, A, nc]`` classification targets; ``target`` itself is not modified.

    ``scope="all_anchors"`` is the ablation without the region restriction: every anchor
    not labelled c gets ``lam * min(q, q_max)``, scaled by s[a] on foreground anchors.
    """
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
    if not missing:
        return target
    align = target.amax(-1)                       # s[a]; 0 on background anchors
    fg = align > 0
    labels = target.argmax(-1)
    t = target.float()
    out = t.clone()
    q = teacher_prob.float().clamp(max=float(q_max))
    for c in missing:
        if scope == "confusable":
            region = torch.zeros_like(fg)
            for d in confusable.get(int(c), ()):
                region |= fg & (labels == int(d))
            weight = align.float()
        else:
            region = ~(fg & (labels == int(c)))
            weight = torch.where(fg, align.float(), torch.ones_like(align, dtype=torch.float32))
        soft = float(lam) * q[..., int(c)] * weight
        out[..., int(c)] = torch.where(region, soft, t[..., int(c)])
    return out.to(target.dtype)


class AbsentKDBCE(torch.nn.Module):
    """Drop-in for ``v8DetectionLoss.bce`` that rewrites the targets before ``inner``.

    ``inner`` may be the NTD wrapper (B1_add): it then sees the rewritten targets, so its
    not-true mask (target == 0) leaves the entries B1 now supervises to B1.
    """

    def __init__(self, inner: torch.nn.Module, missing: Sequence[int], confusable: Mapping[int, Sequence[int]],
                 lam: float, q_max: float, scope: str = "confusable") -> None:
        super().__init__()
        if lam < 0 or not 0 < q_max <= 1:
            raise ValueError(f"need lam >= 0 and 0 < q_max <= 1, got lam={lam}, q_max={q_max}")
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
        self.inner = inner
        self.missing = [int(c) for c in missing]
        self.confusable = {int(k): tuple(int(d) for d in v) for k, v in confusable.items()}
        self.lam, self.q_max, self.scope = float(lam), float(q_max), scope
        self._teacher_logits: torch.Tensor | None = None

    @property
    def teacher_logits(self) -> torch.Tensor | None:
        return self._teacher_logits

    @teacher_logits.setter
    def teacher_logits(self, value: torch.Tensor | None) -> None:
        self._teacher_logits = value
        if hasattr(self.inner, "teacher_logits"):      # one teacher forward feeds both wrappers
            self.inner.teacher_logits = value

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self._teacher_logits is not None and self.missing:
            if self._teacher_logits.shape != logits.shape:
                raise RuntimeError(f"teacher scores {tuple(self._teacher_logits.shape)} do not match "
                                   f"student scores {tuple(logits.shape)}")
            target = absent_class_targets(target, torch.sigmoid(self._teacher_logits.float()), self.missing,
                                          self.confusable, self.lam, self.q_max, self.scope)
        return self.inner(logits, target)


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "methods" / "absent_kd.yaml"


def load_absent_kd_config(overrides: Mapping[str, object] | None = None,
                          path: str | Path = CONFIG_PATH) -> dict[str, object]:
    """``absent_kd`` block of the config file, with an arm's ``overrides`` applied (a new dict)."""
    import yaml
    with open(path, encoding="utf-8") as f:
        cfg = dict(yaml.safe_load(f)["absent_kd"])
    cfg.update(overrides or {})
    cfg["confusable"] = {int(k): [int(d) for d in v] for k, v in dict(cfg["confusable"]).items()}
    if cfg["scope"] not in SCOPES:
        raise ValueError(f"absent_kd.scope must be one of {SCOPES}, got {cfg['scope']!r}")
    return cfg
