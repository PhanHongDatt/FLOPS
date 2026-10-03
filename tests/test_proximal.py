"""FedProx proximal term (Li et al., 2020): grad += mu * (w - w_global).

FedProxClient used to fall back to plain FedAvg (TODO in client_variants.py), so a
"FedProx" result would have been FedAvg under another name.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.federated.proximal import ProximalTerm


def _trainer(scale: float = 1.0):
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 2)
    scaler = SimpleNamespace(get_scale=lambda: scale, is_enabled=lambda: scale != 1.0)
    return SimpleNamespace(model=model, scaler=scaler)


def _grads_after_step(trainer, mu: float, move: float = 0.5):
    prox = ProximalTerm(mu)
    prox.on_train_start(trainer)                 # snapshot w_global
    with torch.no_grad():
        for p in trainer.model.parameters():
            p.add_(move)                         # local drift w - w_global = move
    x = torch.ones(4, 3)
    loss = trainer.model(x).sum() * trainer.scaler.get_scale()
    loss.backward()
    grads = [p.grad.clone() for p in trainer.model.parameters()]
    prox.on_train_end(trainer)
    return grads


def test_gradient_includes_proximal_term():
    plain = _grads_after_step(_trainer(), mu=0.0)
    prox = _grads_after_step(_trainer(), mu=0.1)
    for g0, g1 in zip(plain, prox):
        torch.testing.assert_close(g1, g0 + 0.1 * 0.5)


def test_proximal_term_follows_amp_scale():
    """Under AMP the hook sees scaled gradients; the proximal part must be scaled too."""
    plain = _grads_after_step(_trainer(scale=1024.0), mu=0.0)
    prox = _grads_after_step(_trainer(scale=1024.0), mu=0.1)
    for g0, g1 in zip(plain, prox):
        torch.testing.assert_close(g1, g0 + 1024.0 * 0.1 * 0.5)


def test_hooks_are_removed_after_training():
    trainer = _trainer()
    prox = ProximalTerm(0.1)
    prox.on_train_start(trainer)
    prox.on_train_end(trainer)
    trainer.model.zero_grad()
    trainer.model(torch.ones(1, 3)).sum().backward()
    ref = _trainer()
    ref.model(torch.ones(1, 3)).sum().backward()
    for p, q in zip(trainer.model.parameters(), ref.model.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_zero_mu_is_a_no_op_and_negative_mu_rejected():
    assert ProximalTerm(0.0).active is False
    with pytest.raises(ValueError):
        ProximalTerm(-0.1)


def test_frozen_parameters_are_skipped():
    trainer = _trainer()
    trainer.model.bias.requires_grad_(False)
    prox = ProximalTerm(0.1)
    prox.on_train_start(trainer)
    assert len(prox._handles) == 1
    prox.on_train_end(trainer)


def test_fedprox_strategy_builds_and_sends_mu_to_clients():
    """build_fedprox used to forward num_rounds into flwr.FedProx (TypeError)."""
    from flwr.common import ndarrays_to_parameters
    from flwr.server.client_manager import SimpleClientManager

    from src.federated.server import _load_strategy_builder

    strategy = _load_strategy_builder("FedProx")(num_rounds=5, proximal_mu=0.1,
                                                 min_fit_clients=1, min_available_clients=1,
                                                 min_evaluate_clients=1)
    assert strategy.proximal_mu == 0.1

    class _Proxy:   # minimal ClientProxy stand-in for configure_fit
        cid = "0"

    cm = SimpleClientManager()
    cm.register(_Proxy())
    (_, fit_ins), = strategy.configure_fit(1, ndarrays_to_parameters([]), cm)
    assert fit_ins.config["proximal_mu"] == 0.1


def test_launcher_no_longer_blocks_fedprox():
    from scripts.run_fl_experiment import _DISABLED_ALGORITHMS, _STRATEGY_CLIENTS
    from src.federated.client_variants import FedProxClient

    assert "FedProx" not in _DISABLED_ALGORITHMS
    assert _STRATEGY_CLIENTS["FedProx"] is FedProxClient


def test_launcher_blocks_unfaithful_scaffold():
    """ScaffoldClient applies no (c - c_i) correction, uses -c_i instead of -c, K =
    epochs instead of local steps, and loses c_i between rounds: results would be
    FedAvg-like runs labelled SCAFFOLD (CLAUDE.md §22)."""
    from scripts.run_fl_experiment import _DISABLED_ALGORITHMS

    assert "SCAFFOLD" in _DISABLED_ALGORITHMS
    assert "c_i" in _DISABLED_ALGORITHMS["SCAFFOLD"]
