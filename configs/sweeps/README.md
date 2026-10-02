# Comparison sweeps

One file per **comparison** defined in `research/plan/comparison_design.md`
(C0–C9). A sweep file is the machine-readable form of "which cells form this
comparison"; the design doc holds the hypothesis and the pre-registered decision
rule for each one.

Run a sweep, then aggregate it:

```bash
python scripts/sweep_experiments.py --sweep configs/sweeps/c1_h1_missing_class.yaml \
    --base-dir /kaggle/working --dry-run          # inspect the plan first
python scripts/sweep_experiments.py --sweep configs/sweeps/c1_h1_missing_class.yaml \
    --base-dir /kaggle/working

python scripts/compare_runs.py --arms A0 --run-class feasibility \
    --scenario-delta s1_mc_seed42 s1_control_seed42 \
    --out artifacts/comparisons/c1_h1
```

Cells run sequentially and are skipped when already complete, so re-running a
sweep after a Kaggle session reset continues where it stopped.

## Schema

```yaml
sweep_id: <id>            # also the default MLflow experiment name
comparison: C1            # id in comparison_design.md
description: >            # free text
run_class: feasibility    # smoke | feasibility | main — picks configs/experiments/<x>.yaml
baseline_arm: A0          # the arm compare_runs.py uses for delta columns
defaults:
  seeds: [42, 123, 2024]
  tau_elig: 1
  global_data_yaml: data/bdd100k_yolo/data.yaml   # relative to --base-dir
  mlflow_uri: file:///kaggle/working/mlruns
cells:
  - scenario: S1b
    partition: partitions/s1_mc/manifest.yaml     # relative to --base-dir
    data_yaml_dir: partitions/s1_mc
    seeds: [42, 123, 2024]                        # optional, overrides defaults
    arms:
      - A0                                        # string form
      - {id: A2b, rho: 0.25}                      # mapping form when options are needed
      - {id: A3, tau_elig: 50}
```

An arm that applies a client mechanism (`A2a`, `A2b`, `A4a`, `A4b`) **must** state
`rho`: there is no literature-supported default (plan.md §7.3), so the sweep
refuses to guess one.

## Paths

Paths are relative to `--base-dir` unless absolute. On Kaggle use
`--base-dir /kaggle/working`; locally, whatever directory holds `partitions/` and
`data/`. Partitions must be generated in the **same environment** that runs the
sweep, because the per-client `C*_train.txt` files hold absolute image paths.
