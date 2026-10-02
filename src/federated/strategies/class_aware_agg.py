"""H3 — Server-side class-aware aggregation strategy (+ the A1 class-count control).

[THESIS-HYPOTHESIS] For the final classification conv of the Detect head
(``model.<idx>.cv3.<scale>.2.{weight,bias}``), aggregate each class channel ONLY
from clients that actually hold that class (the eligible set), weighted by their
class box count.

Aggregation rule — delta form (research/plan/plan.md §6.4)::

    S_c   = { i : n_i^c >= tau_elig }
    w_i^c = n_i^c / sum_{j in S_c} n_j^c
    theta^{t+1}[c] = theta^t[c] + sum_{i in S_c} w_i^c * ( theta_i[c] - theta^t[c] )
    S_c = empty  ->  theta^{t+1}[c] = theta^t[c]      # no-contributor rule

Shared params (backbone, neck, cv2, dfl, intermediate cv3): standard FedAvg
weighted by ``num_examples``.

Two modes, selected by ``no_contributor_rule``:

* ``True``  → **A3**, the H3 strategy: the no-contributor rule above applies and
  the previous global ``theta^t`` is genuinely preserved.
* ``False`` → **A1**, the *class-count weighted FedAvg* control: identical
  per-class weighting, but with no no-contributor guarantee — an empty eligible
  set falls back to the plain FedAvg value.

That single branch is the ONLY numerical difference between A1 and A3 when
``tau_elig = 1`` and every class is held by at least one participating client
(see plan.md §6.4 — this is a known, deliberate property, not an oversight).

Aggregation trace logged per CLAUDE.md §12: round, param, class,
eligible_clients, action, reason.

Client requirement: ``class_counts_json`` in ``fit()`` metrics, e.g.
``{"car": 150, "bus": 0, "truck": 80, "motorcycle": 0}``. Missing counts are a
hard error by default (``require_class_counts=True``) — silently treating a
client with unknown counts as eligible would turn a wiring failure into
plausible-looking numbers.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from flwr.common import (
    FitIns,
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from src.data.bdd100k import TARGET_CLASSES
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Final 1x1 nn.Conv2d inside each cv3 branch. The Detect layer index is NOT
# hard-coded: it is 22 for yolov8n at nc=4 today, but it shifts with model
# scale/config. A hard-coded "model.22." silently matches ZERO keys if the
# index moves, which would make this strategy degrade to FedAvg while still
# logging class-aware actions.
_FINAL_CV3_RE = re.compile(r"^model\.\d+\.cv3\.\d+\.2\.(weight|bias)$")

# 3 detection scales (P3/P4/P5) x {weight, bias}
EXPECTED_CLS_HEAD_KEYS = 6

NUM_CLASSES: int = len(TARGET_CLASSES)
CLASS_INDEX: dict[str, int] = {c: i for i, c in enumerate(TARGET_CLASSES)}


def _is_final_cls_head(name: str, shape: tuple[int, ...]) -> bool:
    return bool(_FINAL_CV3_RE.match(name)) and len(shape) >= 1 and shape[0] == NUM_CLASSES


def find_cls_head_indices(param_names: list[str], shapes: list[tuple[int, ...]]) -> list[int]:
    """Return indices of the class-specific params, failing loudly if none match.

    Raises:
        ValueError: when no parameter matches — i.e. the parameter map is wrong
            for the pinned Ultralytics version and class-aware aggregation would
            silently be a no-op.
    """
    idx = [i for i, (n, s) in enumerate(zip(param_names, shapes)) if _is_final_cls_head(n, s)]
    if not idx:
        raise ValueError(
            "No class-specific parameters matched "
            f"{_FINAL_CV3_RE.pattern!r} with dim0 == {NUM_CLASSES}. "
            "Class-aware aggregation cannot run. Re-verify F1 "
            "(research/feasibility/F1/parameter_map.yaml) against the pinned "
            "Ultralytics version."
        )
    if len(idx) != EXPECTED_CLS_HEAD_KEYS:
        logger.warning(
            "Expected %d class-head params (3 scales x weight/bias) but matched %d: %s. "
            "Verify F1 parameter map for the pinned Ultralytics version.",
            EXPECTED_CLS_HEAD_KEYS, len(idx), [param_names[i] for i in idx],
        )
    return idx


class ClassAwareAggregation(FedAvg):
    """Server-side class-aware aggregation (A3), or the A1 class-count control.

    Args:
        param_names: ordered model ``state_dict`` keys, same order as the NDArrays
            exchanged with clients. Obtain via ``list(model.model.state_dict().keys())``.
        trace_dir: directory for ``aggregation_trace.yaml``. When None no trace is
            written, which violates CLAUDE.md §12/§21 — the server always passes it.
        eligibility_threshold: ``tau_elig``. Primary value is 1, i.e. "eligible"
            means "not missing" and the method adds no new free hyper-parameter
            (plan.md §5.1). ``tau_elig = 50`` is the pre-registered sensitivity
            variant.
        no_contributor_rule: True → A3 (keep ``theta^t``); False → A1 control.
        require_class_counts: True → raise when a client omits ``class_counts_json``.
    """

    def __init__(
        self,
        param_names: list[str],
        trace_dir: Path | None = None,
        eligibility_threshold: int = 1,
        no_contributor_rule: bool = True,
        require_class_counts: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.param_names = param_names
        self.trace_dir = trace_dir
        self.eligibility_threshold = int(eligibility_threshold)
        self.no_contributor_rule = bool(no_contributor_rule)
        self.require_class_counts = bool(require_class_counts)
        self._trace: list[dict[str, Any]] = []
        # theta^t — the global parameters broadcast THIS round. Required by the
        # §12 no-contributor rule. Without it the previous implementation logged
        # action=keep_global while actually leaving the FedAvg average in place.
        self._global_prev: NDArrays | None = None
        self._cls_head_indices: list[int] | None = None

    # ── theta^t capture ───────────────────────────────────────────────────
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        self._global_prev = parameters_to_ndarrays(parameters)
        return super().configure_fit(server_round, parameters, client_manager)

    # ── helpers ───────────────────────────────────────────────────────────
    def _parse_class_counts(
        self,
        results: list[tuple[ClientProxy, FitRes]],
    ) -> list[dict[str, int]]:
        counts: list[dict[str, int]] = []
        for proxy, res in results:
            raw = (res.metrics or {}).get("class_counts_json")
            if raw is None:
                if self.require_class_counts:
                    raise ValueError(
                        f"Client {getattr(proxy, 'cid', '?')} returned no "
                        "'class_counts_json' in fit metrics. Class-aware "
                        "aggregation cannot determine per-class eligibility. "
                        "Treating it as eligible for every class would silently "
                        "degrade this strategy to FedAvg while the trace claims "
                        "class-count weighting (CLAUDE.md §19, §20)."
                    )
                counts.append({})
                continue
            try:
                parsed = json.loads(str(raw))
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(
                    f"Client {getattr(proxy, 'cid', '?')} sent malformed "
                    f"class_counts_json: {raw!r}"
                ) from exc
            counts.append({str(k): int(v) for k, v in dict(parsed).items()})
        return counts

    def _eligible(
        self, class_counts: list[dict[str, int]], cls_name: str
    ) -> tuple[list[int], list[float]]:
        idx: list[int] = []
        w: list[float] = []
        for i, counts in enumerate(class_counts):
            n = int(counts.get(cls_name, 0))
            if n >= self.eligibility_threshold:
                idx.append(i)
                w.append(float(n))
        return idx, w

    # ── aggregation ───────────────────────────────────────────────────────
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list,
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        if not results:
            return None, {}

        client_params: list[NDArrays] = [
            parameters_to_ndarrays(res.parameters) for _, res in results
        ]
        client_weights: list[float] = [float(res.num_examples) for _, res in results]
        class_counts = self._parse_class_counts(results)
        total_examples = sum(client_weights) or 1.0

        # Step 1 — FedAvg over everything. This is the final value for shared
        # params and the A1 fallback for class rows.
        agg: NDArrays = [np.zeros_like(p) for p in client_params[0]]
        for i, params in enumerate(client_params):
            w = client_weights[i] / total_examples
            for j, p in enumerate(params):
                agg[j] = agg[j] + w * p

        if self._cls_head_indices is None:
            self._cls_head_indices = find_cls_head_indices(
                self.param_names, [tuple(a.shape) for a in agg]
            )

        theta_prev = self._global_prev
        if theta_prev is None and self.no_contributor_rule:
            raise RuntimeError(
                "theta^t unavailable: configure_fit did not run before "
                "aggregate_fit, so the §12 no-contributor rule cannot be "
                "honoured. Refusing to silently fall back to FedAvg."
            )

        # Step 2 — override class rows.
        round_trace: list[dict[str, Any]] = []
        for param_idx in self._cls_head_indices:
            if param_idx >= len(agg):
                continue
            name = self.param_names[param_idx]

            for cls_name, cls_idx in CLASS_INDEX.items():
                eligible_i, eligible_cc = self._eligible(class_counts, cls_name)
                entry: dict[str, Any] = {
                    "round": int(server_round),
                    "param": name,
                    "class": cls_name,
                    "eligible_clients": eligible_i,
                    "tau_elig": self.eligibility_threshold,
                }

                if not eligible_i:
                    if self.no_contributor_rule:
                        # A3: theta_c^{t+1} = theta_c^t
                        assert theta_prev is not None  # guarded above
                        agg[param_idx][cls_idx] = theta_prev[param_idx][cls_idx]
                        entry["action"] = "keep_global"
                        entry["reason"] = "no_valid_contributor"
                    else:
                        # A1 control: no guarantee — leave the FedAvg value.
                        entry["action"] = "fedavg_fallback"
                        entry["reason"] = "no_valid_contributor_and_rule_disabled"
                    round_trace.append(entry)
                    continue

                total_cc = sum(eligible_cc) or 1.0
                if theta_prev is not None:
                    # Delta form (plan.md §6.4). Weights sum to 1, so this is
                    # numerically the same as the weighted average below; it is
                    # written this way to mirror the documented formula.
                    base = theta_prev[param_idx][cls_idx]
                    acc = np.zeros_like(base)
                    for ei, cc in zip(eligible_i, eligible_cc):
                        acc = acc + (cc / total_cc) * (
                            client_params[ei][param_idx][cls_idx] - base
                        )
                    agg[param_idx][cls_idx] = base + acc
                else:
                    cls_agg = np.zeros_like(agg[param_idx][cls_idx])
                    for ei, cc in zip(eligible_i, eligible_cc):
                        cls_agg = cls_agg + (cc / total_cc) * (
                            client_params[ei][param_idx][cls_idx]
                        )
                    agg[param_idx][cls_idx] = cls_agg

                entry["action"] = "class_count_weighted"
                entry["num_eligible"] = len(eligible_i)
                entry["weights"] = [round(cc / total_cc, 6) for cc in eligible_cc]
                round_trace.append(entry)

        self._trace.extend(round_trace)
        if self.trace_dir is not None:
            self._save_trace()
        else:
            logger.warning(
                "trace_dir is None — aggregation_trace.yaml will NOT be written "
                "(violates CLAUDE.md §12/§21)."
            )

        return ndarrays_to_parameters(agg), {
            "n_clients": len(results),
            "no_contributor_rule": self.no_contributor_rule,
        }

    def _save_trace(self) -> None:
        assert self.trace_dir is not None
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        with (self.trace_dir / "aggregation_trace.yaml").open("w", encoding="utf-8") as f:
            yaml.dump(self._trace, f, default_flow_style=False, allow_unicode=True)


def build_class_aware_agg(
    num_rounds: int,
    param_names: list[str],
    trace_dir: Path | None = None,
    eligibility_threshold: int = 1,
    no_contributor_rule: bool = True,
    require_class_counts: bool = True,
    **kwargs: Any,
) -> ClassAwareAggregation:
    """A3 factory. ``num_rounds`` is consumed here and NOT forwarded to FedAvg."""
    return ClassAwareAggregation(
        param_names=param_names,
        trace_dir=trace_dir,
        eligibility_threshold=eligibility_threshold,
        no_contributor_rule=no_contributor_rule,
        require_class_counts=require_class_counts,
        **kwargs,
    )


def build_class_count_fedavg(
    num_rounds: int,
    param_names: list[str],
    trace_dir: Path | None = None,
    eligibility_threshold: int = 1,
    require_class_counts: bool = True,
    **kwargs: Any,
) -> ClassAwareAggregation:
    """A1 control factory — class-count weighting WITHOUT the no-contributor rule."""
    return ClassAwareAggregation(
        param_names=param_names,
        trace_dir=trace_dir,
        eligibility_threshold=eligibility_threshold,
        no_contributor_rule=False,
        require_class_counts=require_class_counts,
        **kwargs,
    )
