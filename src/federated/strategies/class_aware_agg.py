"""H3 — Server-side class-aware aggregation strategy.

[THESIS-HYPOTHESIS] For detect_cls final-conv parameters (model.22.cv3.*.2),
aggregate each class channel ONLY from clients that have that class present
(eligible clients), weighted by their class-count.

No-contributor rule (CLAUDE.md §12):
  If no eligible client for class c → keep global params for that channel.

Aggregation rule — class-count weighted [THESIS-HYPOTHESIS]:
  Eligible set for class c: S_c = {i : class_count_i[c] > 0}
  w_i^c = class_count_i[c] / Σ_{j ∈ S_c} class_count_j[c]
  θ_c^{t+1} = Σ_{i ∈ S_c} w_i^c · θ_i[c_channel]

Shared params (backbone, neck, cv2, dfl, intermediate cv3):
  Standard FedAvg weighted by num_examples.

Aggregation trace logged per CLAUDE.md §12:
  round, param, class, eligible_clients, action, reason

Client requirement: must include 'class_counts_json' in fit() metrics,
  e.g. {"car": 150, "bus": 0, "truck": 80, "motorcycle": 0}
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import flwr as fl
from flwr.common import (
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from src.data.bdd100k import TARGET_CLASSES

# Only the final nn.Conv2d in each cv3 branch is class-specific (F1 analysis).
_FINAL_CV3_RE = re.compile(r"^model\.22\.cv3\.\d+\.2\.(weight|bias)$")
NUM_CLASSES: int = len(TARGET_CLASSES)
CLASS_INDEX: dict[str, int] = {c: i for i, c in enumerate(TARGET_CLASSES)}


def _is_final_cv3(name: str, shape: tuple[int, ...]) -> bool:
    return bool(_FINAL_CV3_RE.match(name)) and len(shape) >= 1 and shape[0] == NUM_CLASSES


class ClassAwareAggregation(FedAvg):
    """Server-side class-aware aggregation strategy (H3).

    Extends FedAvg. For detect_cls final-conv params, applies per-class
    eligibility filtering. For all other params, uses standard FedAvg.

    Args:
        param_names: ordered list of model state_dict keys (same order as NDArrays).
                     Obtain via: list(model.model.state_dict().keys())
        trace_dir:   if set, writes aggregation_trace.yaml there after each round.
    """

    def __init__(
        self,
        param_names: list[str],
        trace_dir: Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.param_names = param_names
        self.trace_dir = trace_dir
        self._trace: list[dict[str, Any]] = []

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list,
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        if not results:
            return None, {}

        # ── Parse results ─────────────────────────────────────────────────
        client_params: list[NDArrays] = []
        client_weights: list[float] = []
        client_class_counts: list[dict[str, int] | None] = []

        for _, res in results:
            client_params.append(parameters_to_ndarrays(res.parameters))
            client_weights.append(float(res.num_examples))
            counts_raw = (res.metrics or {}).get("class_counts_json")
            if counts_raw:
                try:
                    client_class_counts.append(json.loads(str(counts_raw)))
                except (json.JSONDecodeError, TypeError):
                    client_class_counts.append(None)
            else:
                client_class_counts.append(None)

        total_examples = sum(client_weights) or 1.0

        # ── Step 1: FedAvg over all params (baseline / shared params result) ──
        agg: NDArrays = [np.zeros_like(p) for p in client_params[0]]
        for i, params in enumerate(client_params):
            w = client_weights[i] / total_examples
            for j, p in enumerate(params):
                agg[j] = agg[j] + w * p

        # ── Step 2: Override detect_cls final-conv with class-aware agg ───
        round_trace: list[dict[str, Any]] = []

        for param_idx, name in enumerate(self.param_names):
            if param_idx >= len(agg):
                break
            if not _is_final_cv3(name, agg[param_idx].shape):
                continue

            for cls_name, cls_idx in CLASS_INDEX.items():
                eligible_i: list[int] = []
                eligible_cc: list[float] = []

                for i, counts in enumerate(client_class_counts):
                    if counts is None:
                        # No class info → conservative: treat as eligible
                        eligible_i.append(i)
                        eligible_cc.append(client_weights[i])
                    elif counts.get(cls_name, 0) > 0:
                        eligible_i.append(i)
                        eligible_cc.append(float(counts[cls_name]))

                entry: dict[str, Any] = {
                    "round": server_round,
                    "param": name,
                    "class": cls_name,
                    "eligible_clients": eligible_i,
                }

                if not eligible_i:
                    # No-contributor rule — keep global (agg is FedAvg global here)
                    entry["action"] = "keep_global"
                    entry["reason"] = "no_valid_contributor"
                else:
                    total_cc = sum(eligible_cc) or 1.0
                    cls_agg = np.zeros_like(agg[param_idx][cls_idx])
                    for ei, cc in zip(eligible_i, eligible_cc):
                        cls_agg = cls_agg + (cc / total_cc) * client_params[ei][param_idx][cls_idx]
                    agg[param_idx][cls_idx] = cls_agg
                    entry["action"] = "class_count_weighted"
                    entry["num_eligible"] = len(eligible_i)

                round_trace.append(entry)

        self._trace.extend(round_trace)
        if self.trace_dir is not None:
            self._save_trace()

        return ndarrays_to_parameters(agg), {"n_clients": len(results)}

    def _save_trace(self) -> None:
        assert self.trace_dir is not None
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        with (self.trace_dir / "aggregation_trace.yaml").open("w", encoding="utf-8") as f:
            yaml.dump(self._trace, f, default_flow_style=False, allow_unicode=True)


def build_class_aware_agg(
    param_names: list[str],
    trace_dir: Path | None = None,
    **kwargs: Any,
) -> ClassAwareAggregation:
    """Factory for ClassAwareAggregation, matching the strategy builder interface."""
    return ClassAwareAggregation(param_names=param_names, trace_dir=trace_dir, **kwargs)
