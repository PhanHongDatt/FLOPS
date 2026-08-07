# MISSING_CLASS_NON_IID_AGENT.md
> Compact research + engineering constitution for the Missing-Class Non-IID / YOLOv8 workstream.

## 0. Response Contract — MUST
- **Every response MUST start with exactly `Snow` on the first line.**
- If this file was not read, do not act.
- Do not claim an experiment, test, source check, or result was completed unless it was actually performed.
- Keep answers concise; expand only when implementation/review requires detail.

---

## 1. Mission

Build and evaluate a **research-grounded Federated Object Detection method for Missing-Class Non-IID using YOLOv8 + Flower**.

Primary question:

> When some FL clients have zero positive samples for one or more traffic classes, can client-side knowledge preservation and server-side class-aware aggregation reduce class-level degradation without materially harming the global detector?

Target classes: `car`, `bus`, `truck`, `motorcycle`.

Core research path:

`prove problem -> locate cause -> validate intervention -> implement method -> ablate -> multi-seed evaluate -> conclude`

This is a scientific project, not a feature-delivery project.

---

## 2. Scope

### IN
- YOLOv8 object detection
- Flower-based FL
- IID vs Missing-Class Non-IID
- BDD100K or approved controlled subset
- 4–6 simulated clients
- FedAvg, FedProx, SCAFFOLD, FedNova
- class-count weighted aggregation
- client-side preservation
- server-side class-aware aggregation
- per-class metrics, parameter/update analysis
- reproducible experiments

### OUT unless explicitly approved
- tracking / segmentation
- Byzantine defense
- Differential Privacy / Secure Aggregation
- RL
- client selection / network scheduling
- large-scale resource optimization
- new YOLO architecture
- production serving / production rollback

Do not expand scope because an idea seems interesting.

---

## 3. Operating Modes

### MODE A — GUIDE / DESIGN
Allowed:
- inspect repo/docs/source;
- verify literature;
- define experiments, schemas, architecture, tests;
- create/update research documentation.

Not allowed:
- implement the full proposed method;
- run expensive main experiments;
- claim method effectiveness.

### MODE B — EXECUTION
Allowed only after relevant gate is passed:
- implement;
- test;
- run approved experiments;
- record results.

If the task does not explicitly authorize execution, default to **MODE A**.

---

## 4. Evidence Policy

Tag every non-trivial research decision with one of:

| Tag | Meaning |
|---|---|
| `[LITERATURE]` | supported by a research paper |
| `[YOLO-DOC]` | verified from pinned Ultralytics source/docs |
| `[FL-DOC]` | verified from Flower/original FL source |
| `[THESIS-HYPOTHESIS]` | thesis proposal; must be experimentally validated |
| `[ENGINEERING]` | implementation choice with no scientific claim |
| `[NEEDS-VERIFICATION]` | source cannot currently be verified |

### Source priority
1. peer-reviewed/original paper;
2. official Ultralytics source/docs;
3. official Flower source/docs;
4. official author implementation.

Do not use blogs, generated summaries, random GitHub snippets, or StackOverflow as methodological evidence.

### Offline/no-search rule
If authoritative sources cannot be accessed:
- never reconstruct DOI, venue, year, equations, tensor mapping, or implementation details from memory;
- mark them `[NEEDS-VERIFICATION]`;
- record what must be verified;
- **do not implement research logic whose justification remains unverified**.

For each reused method record:
`title | authors | year | venue | source | reused concept | reproduction/adaptation/extension`

---

## 5. Single Source of Truth

### Version resolution order
1. lockfile;
2. `requirements*` / `pyproject.toml` / environment file;
3. Docker image;
4. saved experiment metadata;
5. if absent: propose a version and request approval before pinning.

Never choose an Ultralytics/Flower/PyTorch version silently.

Record the resolved environment in one canonical file, e.g.:
`environment.lock`, `requirements.txt`, or `environment.yml`.

Every experiment must store:
- Python;
- PyTorch;
- Ultralytics;
- Flower;
- CUDA if used;
- dataset version;
- git commit;
- config;
- seed;
- partition ID.

YOLO parameter assumptions are valid only for the pinned version.

---

## 6. Research Hypotheses

### H1 — Missing-Class degradation `[THESIS-HYPOTHESIS]`
Clients with zero positives for class `c` may cause degradation in global detection of `c`.

**Support requires:** controlled S1 vs matched control showing consistent class-level degradation and compatible parameter/prediction evidence.

**Reject/weaken if:** effect is absent, unstable, or explained by data-size/source confounds.

---

### H2 — Client preservation `[THESIS-HYPOTHESIS]`
Reducing updates to **validated class-associated parameters** for a locally missing class may preserve prior global knowledge.

Candidate factor:

\[
\rho \in \{0,\;0.25,\;1.0\}
\]

- `rho=1`: ordinary update
- `rho=0`: maximum preservation
- `rho=0.25`: partial preservation

Do **not** apply this before class-parameter feasibility is established.

**Support requires:** target-class benefit across seeds with acceptable non-target/global impact.

**Reject/weaken if:** class parameters cannot be isolated, cross-class damage is substantial, or gains are not reproducible.

**Fallback:** select only a literature-supported alternative after evidence review; do not invent one.

---

### H3 — Server class-aware aggregation `[THESIS-HYPOTHESIS]`
For validated class-associated parameters, per-class client eligibility/weighting may outperform a single client-level weight under Missing-Class Non-IID.

**Support requires:** server-only/full ablation outperforming appropriate baselines in missing/rare-class behavior.

**Reject/weaken if:** gains are explained by simple class-count weighting or are inconsistent.

---

## 7. Stage Gates — DO NOT SKIP

```text
G0  Evidence/source registry ready
G1  Environment reproducible
G2  Centralized YOLO baseline reproducible
G3  FedAvg baseline reproducible
G4  Missing-Class effect established
G5  YOLO class-intervention feasibility established
      ├─ A: supported
      ├─ B: partially supported
      └─ C: unsupported -> literature-backed fallback review
G6  Client preservation validated
G7  Server class-aware aggregation validated
G8  Full method implemented
G9  Ablation completed
G10 Main multi-seed experiments completed
G11 Research conclusion approved
```

Never implement the full method before `G4` and `G5`.

---

## 8. YOLOv8 Feasibility Protocol

The guide may **specify** F1–F3 in MODE A. Running them requires MODE B.

### F1 — Parameter mapping
Inspect the pinned YOLOv8 source and runtime model.

Produce:

| Module | Parameter | Shape | Class-specific? | Shared? | Evidence | Mask candidate? |
|---|---|---:|---|---|---|---|

Inspect at least:
- backbone;
- neck;
- Detect head;
- classification branch;
- regression branch;
- class-related outputs;
- shared parameters.

Never assume "one weight vector per class".

### F2 — Controlled perturbation
Perturb only a candidate class-associated parameter group.

Measure:
- target-class AP;
- non-target AP;
- confidence;
- FP/FN;
- prediction changes.

Do not invent a universal threshold. Pre-register the decision rule before the final evaluation and compare target effects against non-target/control effects.

### F3 — Matched missing-class local training
Start from the same global checkpoint.

Compare:
- client with target class;
- matched client/partition without target class.

Track:

\[
\Delta\theta = \theta_{local}-\theta_{global}
\]

By:
- backbone;
- neck;
- classification branch;
- regression branch;
- validated class-associated group.

Measure:
- L2/update norm;
- cosine similarity/direction;
- per-class AP change;
- confidence change;
- FP/FN change.

A Missing-Class claim requires **parameter/update evidence + prediction-level evidence**, not L2 alone.

### Gate decision
- **A Supported:** isolated class intervention is technically meaningful.
- **B Partial:** use only demonstrably class-associated subset.
- **C Unsupported:** stop parameter-masking path; review evidence-backed fallback.

Record decision as an ADR.

---

## 9. Dataset & Partition Contract

Prefer BDD100K or approved controlled subset.

Split by **video sequence / recording segment** where possible to limit leakage.

Every partition must be deterministic and saved as `partition_manifest`.

Required fields:

```yaml
partition_id: mc_v1_seed42
seed: 42
sample_id: ...
source_sequence: ...
client_id: C1
split: train
class_counts:
  car: 0
  bus: 0
  truck: 0
  motorcycle: 0
bbox_total: 0
```

### Class-status schema

```yaml
client_id: C1
round_id: 1
num_images: 0
total_boxes: 0
boxes_per_class:
  car: 0
  bus: 0
  truck: 0
  motorcycle: 0
present_classes: []
missing_classes: []
rare_classes: []
definition:
  missing_threshold: 0
  rare_rule_id: pre_registered_rule_v1
```

`missing` normally means **zero positive boxes**.

Define `rare` from training/partition statistics **before model outcomes are inspected**. Never redefine rare classes after seeing AP.

---

## 10. Experimental Scenarios

### S0 — IID
Approximate class balance across clients.

Purpose: verify the proposed method does not materially degrade normal IID performance.

### S1 — Missing-Class Non-IID
One or more clients have zero positives for one or more classes.

Severity variables:
- missing classes/client;
- clients still containing the target class.

### S1-Control — Matched causal control
Keep groups as similar as feasible in:
- image count;
- total bbox count;
- recording source;
- non-target class frequencies.

Primary difference: target class present vs absent.

Do not attribute degradation to Missing-Class if major confounds remain.

---

## 11. Baselines & Ablation

### Reference baselines
- `Centralized YOLOv8`
- `FedAvg`
- `FedProx`
- `SCAFFOLD`
- `FedNova`

If compute prevents full SCAFFOLD/FedNova runs, document the resource limitation and obtain scope approval; do not silently omit them.

### Contribution controls
- `Class-count weighted FedAvg`
- `Client preservation only`
- `Server class-aware aggregation only`
- `Full method`

Mandatory ablation:

| ID | Client preservation | Server class-aware |
|---|---|---|
| A0 | No | No |
| A1 | No | class-count baseline |
| A2 | Yes | No |
| A3 | No | Yes |
| A4 | Yes | Yes |

The full method is not validated unless simpler controls are reported.

---

## 12. Aggregation Rules

### Shared components
Backbone/neck/shared detection parameters:
- use the approved standard FL aggregation;
- never silently treat them as class-specific.

### Validated class-associated parameters
For class `c`, aggregate only according to an evidence-backed or explicitly hypothesized per-class rule.

Any formula must be tagged:
`[LITERATURE]` or `[THESIS-HYPOTHESIS]`.

### No-contributor rule
If no participating client has valid contribution for class `c`:

\[
\theta_c^{t+1}=\theta_c^t
\]

Apply only to parameter groups validated by G5.

Trace:

```yaml
round: 1
class: bus
eligible_clients: []
action: keep_global
reason: no_valid_contributor
```

---

## 13. Metrics

### Detection
- mAP@0.5
- mAP@0.5:0.95
- AP/class
- Precision/class
- Recall/class
- FP/class
- FN/class
- confidence statistics

### FL
- metric/global round
- convergence
- best/final round
- local training time
- round time
- aggregation time
- update/model bytes

### Missing-Class analysis
For class `c`:

\[
\Delta AP_c=AP_{method,c}-AP_{FedAvg,c}
\]

### Parameter/update analysis
- norm by module;
- cosine similarity/direction;
- validated class-group drift.

Do not conclude from global mAP alone.

---

## 14. Reproducibility & Compute

Main configurations:
- minimum **3 seeds**;
- same partition seed/data split/client population/model init/rounds/local epochs/batch/image size across compared methods unless that variable is under study;
- report `mean ± std`.

Never compare methods on different partitions and call the difference algorithmic.

### Compute budget
Before a non-trivial run, record:
- GPU/VRAM if available;
- CPU/RAM;
- estimated wall time;
- estimated storage;
- client concurrency.

If unknown, mark `[RESOURCE-BUDGET-UNRESOLVED]`.

Use three run classes:
1. `smoke` — correctness only;
2. `feasibility` — reduced data/rounds;
3. `main` — approved full experiment.

Do not launch `main` before reporting estimated cost.

---

## 15. Repository Contract

Prefer existing repo structure. Do not refactor working code for aesthetics.

Suggested separation:

```text
src/
  data/
  model/
  federated/
    strategies/
  preservation/
  evaluation/
  experiments/
  utils/
configs/
tests/
research/
  evidence/
  decisions/
  feasibility/
  experiment_registry/
artifacts/
```

Research pipeline:

`partition -> class stats -> local train -> preservation -> local update -> aggregation -> global model -> evaluation`

Keep research logic out of Flower networking code where possible.

Use configuration files; no scattered magic numbers.

---

## 16. Research Records

### `research/evidence/literature_registry.yaml`
Store verified sources and reused concepts.

### `research/decisions/ADR-XXX-*.md`
Required for:
- pinned YOLO version;
- class-parameter mapping decision;
- feasibility Gate A/B/C;
- changing research equations;
- changing module contracts;
- selecting fallback.

### `research/feasibility/F1|F2|F3/`
Store config, logs, metrics, plots, notes.

### `research/experiment_registry/`
One entry per experiment with status:
`planned | running | completed | failed-valid | failed-technical | rejected`

Never delete negative results.

---

## 17. Multi-Agent Rules

Before editing research code, each agent must read:
1. this file;
2. current environment/version source;
3. relevant ADR;
4. relevant feasibility result;
5. experiment config/contract.

Path ownership should be explicit per task, e.g.:
- data/partition;
- YOLO inspection;
- FL strategy;
- evaluation.

Do not modify another module's public contract without an ADR.

Recommended branch:
`research/<scope>-<short-task>`

Recommended commit:
`research(scope): concise change`

Never run two agents that edit the same research contract concurrently without coordination.

---

## 18. Tests

Minimum:

### Dataset
- target missing class truly has zero boxes;
- counts are correct;
- partition deterministic;
- no forbidden sequence leakage.

### Parameter map
- expected names exist;
- shapes match pinned version;
- class count matches config.

### Preservation
- `rho=1` equals normal update;
- `rho=0` preserves only validated protected group;
- shared parameters unaffected unless explicitly intended.

### Aggregation
- 0 / 1 / many eligible clients;
- weight normalization;
- no-contributor behavior;
- all-classes-present compatibility.

### Integration
- smoke FL round completes;
- artifacts and metadata are produced;
- same evaluator is used across methods.

---

## 19. Review Rules

Reject a research PR if any answer is missing:

### Research
- Which hypothesis?
- Which verified source?
- Any heuristic?
- Why scientifically justified?
- What experiment can falsify it?

### YOLO
- Verified against pinned version?
- Parameter names/shapes documented?
- Cross-class effect considered?
- Regression/shared parameters protected?

### FL
- Correct weighting/normalization?
- Zero-eligible behavior?
- Exact parameter mapping?
- Baseline behavior preserved where expected?

### Reproducibility
- Seeds/config/partition saved?
- Same data and evaluator across methods?
- Artifacts complete?

### Evaluation
- per-class metrics recorded?
- negative results retained?
- no best-seed cherry-picking?

---

## 20. Anti-Cherry-Picking

Never:
- report only the best seed;
- silently drop failed/negative runs;
- change partitions between methods;
- hide classes where the method is worse;
- tune the proposal much more than baselines without reporting it;
- select different best rounds per method without declaring the rule;
- redefine rare classes after seeing results.

Technical failures may be excluded only with logged reasons.

---

## 21. Experiment Artifact Contract

A completed run must produce:

```text
config.yaml
environment.json
partition_manifest.*
metrics.csv
per_class_metrics.csv
round_metrics.csv
aggregation_trace.*
run.log
checkpoint/
plots/
README.md
```

Also record:
`git commit | versions | dataset | partition ID | seed | run class`

---

## 22. Scientific Success / Failure

A method is **promising**, not "proven", only when:
1. missing/rare-class AP improves vs FedAvg;
2. improvement is reasonably consistent across seeds;
3. non-target/global degradation is reported and acceptable;
4. IID behavior is not materially degraded;
5. ablation supports the claimed mechanism;
6. class-count baseline does not fully explain the gain.

Valid negative outcomes include:
- Missing-Class effect is weak;
- shared representations dominate degradation;
- masking fails;
- server-only equals full method;
- class-count weighting explains gains;
- `rho=0` harms adaptation;
- rare classes improve while common classes degrade.

Never modify the experiment merely to force a positive conclusion.

---

## 23. Definition of Done — Guide

This guide/spec is acceptable only if:
- [ ] scope and research questions are explicit;
- [ ] evidence/offline rules exist;
- [ ] canonical version resolution exists;
- [ ] H1–H3 include support/reject/fallback logic;
- [ ] G0–G11 are defined;
- [ ] F1–F3 and Gate A/B/C are specified;
- [ ] S0/S1/S1-Control are defined;
- [ ] baseline + ablation matrix is complete;
- [ ] class-stat/partition schemas exist;
- [ ] metrics include per-class + parameter/update evidence;
- [ ] 3-seed reproducibility protocol exists;
- [ ] tests/review/anti-cherry-picking rules exist;
- [ ] multi-agent/ADR rules exist;
- [ ] no unverified methodological claim is presented as fact.

---

## 24. Definition of Done — Experiment

An experiment is complete only when:
- code/tests pass;
- config + environment + partition are frozen;
- required artifacts exist;
- run status is registered;
- metrics are evaluated with the same protocol;
- interpretation distinguishes observation from causal claim;
- unresolved uncertainty is documented.

---

## 25. Required Agent Response Template

Every answer starts with `Snow`, then use only sections needed:

```markdown
Snow

## Objective
## Evidence
## Assumptions
## Files Inspected
## Change / Plan
## Tests
## Validation Experiment
## Results
## Risks / Open Questions
```

Rules:
- omit empty sections;
- never fabricate `Results`;
- if evidence is insufficient, write:

`[NEEDS-VERIFICATION] Insufficient evidence to implement this as research logic.`

Then state the exact source or feasibility test required.

---

## 26. First Actions for a New Agent

1. Start response with `Snow`.
2. Determine MODE A/B.
3. Inspect repository + environment version sources.
4. Read current ADR/evidence/feasibility records.
5. Identify current gate `Gx`.
6. Do only work allowed by that gate.
7. Before finishing: run applicable tests/checks and record unresolved risks.

---

## 27. Core Principle

**Hypothesis -> evidence -> controlled implementation -> controlled experiment -> result -> conclusion**

Never:

**idea -> code -> higher mAP -> claim contribution**
