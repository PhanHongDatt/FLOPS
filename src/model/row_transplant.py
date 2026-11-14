"""Class-row transplant between two checkpoints of the same architecture (ADR-037).

Copies one class's row of the final classification conv (``Detect.cv3[l][2]``, weight [nc, c3, 1, 1] and bias
[nc]) on every level from a donor parameter list into a recipient one; nothing else changes. Parameters are the
flat lists of ``model.state_dict()`` values in key order, as saved by ``save_global_checkpoint``.
"""
from __future__ import annotations

import re

import numpy as np

_FINAL_CLS_CONV = re.compile(r"^model\.\d+\.cv3\.\d+\.2\.(weight|bias)$")


def final_cls_conv_keys(keys: list[str]) -> list[str]:
    """State-dict keys of the last classification conv of every Detect level."""
    found = [k for k in keys if _FINAL_CLS_CONV.match(k)]
    if not found:
        raise ValueError("no Detect cv3[l][2] parameters among the keys")
    return found


def transplant_row(keys: list[str], recipient: list[np.ndarray], donor: list[np.ndarray],
                   class_id: int) -> list[np.ndarray]:
    """New parameter list: ``recipient`` with row ``class_id`` of every cv3[l][2] taken from ``donor``."""
    if not (len(keys) == len(recipient) == len(donor)):
        raise ValueError(f"length mismatch: {len(keys)} keys, {len(recipient)} recipient, {len(donor)} donor")
    targets = set(final_cls_conv_keys(keys))
    out = []
    for k, r, d in zip(keys, recipient, donor):
        if r.shape != d.shape:
            raise ValueError(f"{k}: shape {r.shape} vs {d.shape}")
        if k in targets:
            new = np.array(r, copy=True)
            new[class_id] = d[class_id]
            out.append(new)
        else:
            out.append(r)
    return out
