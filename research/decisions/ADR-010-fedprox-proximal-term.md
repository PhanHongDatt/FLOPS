# ADR-010: FedProx proximal term in Ultralytics local training

**Date:** 2026-10-03
**Status:** accepted (implemented + unit-tested; GPU run in session 2)
**Scope:** FL baseline (CLAUDE.md §11 reference baselines)

## Context

`FedProxClient` fell back to FedAvg (TODO), and `run_fl_experiment.py` blocked FedProx in
`_DISABLED_ALGORITHMS` until a proximal-term implementation existed. `build_fedprox` also
forwarded `num_rounds` into `flwr.FedProx`, which would have raised `TypeError`.

## Decision

- `[LITERATURE]` Li et al., 2020, MLSys (arXiv:1812.06127): client objective
  `h_k(w) = F_k(w) + (mu/2)||w - w^t||^2`, gradient term `mu (w - w^t)`.
- `[ENGINEERING]` `src/federated/proximal.py::ProximalTerm` registers a gradient hook on every
  trainable parameter of the Ultralytics trainer's model at `on_train_start` (after the model is
  built from the received global weights, so `w^t` is snapshotted there) and removes the hooks at
  `on_train_end` / in `finally`. Under AMP the proximal part is multiplied by the GradScaler
  scale, matching the scaled loss gradients. It fires on every backward, so it accumulates over
  Ultralytics' gradient-accumulation window like the loss gradient.
- `[FL-DOC]` `mu` reaches clients through Flower's `FedProx.configure_fit` (`proximal_mu` in the
  fit config); value from `federated.proximal_mu` = 0.01, identical for every FedProx run.
  Li et al. sweep mu ∈ {0.001, 0.01, 0.1, 1}; a sweep is optional and must be reported if done.
- `build_fedprox` consumes `num_rounds`; FedProx removed from `_DISABLED_ALGORITHMS`.

## Caveat

Ultralytics' detection loss is scaled by the batch size, so `mu` is relative to that loss
scale, not to a per-sample mean loss as in the paper's notation. Record `mu` with results;
compare FedProx only against FedAvg run with the same loss convention (true here).

## Tests

`tests/test_proximal.py`: gradient = loss gradient + mu·drift; AMP scale; hooks removed;
mu = 0 no-op; frozen params skipped; strategy builds and sends `proximal_mu`; launcher enabled.

## Not covered

SCAFFOLD (reviewed 2026-10-03) is **blocked in `_DISABLED_ALGORITHMS`**: `ScaffoldClient`
(1) applies no `(c − c_i)` correction during local training, (2) computes `Δc_i` with `−c_i`
instead of `−c` (Karimireddy et al. 2020, Alg. 1 Option II: `c_i⁺ = c_i − c + (x − y_i)/(K η_l)`),
(3) uses `K = local_epochs` instead of the number of local optimizer steps, and (4) loses `c_i`
between rounds because Flower recreates clients. Ultralytics' SGD momentum (0.937), weight decay,
EMA and gradient accumulation also change the Option II scaling (≈1/(1−β) ≈ 16× for the
update-to-gradient ratio). A faithful version needs its own ADR. The server strategy itself
follows Algorithm 1. FedNova not reviewed here.
