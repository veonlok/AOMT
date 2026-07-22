# D-Flex Project Plan

Last updated: 2026-07-22

The task-level source of truth is `PROJECT_BACKLOG.md`; completed and failed experiment
outcomes belong in `EXPERIMENTS.md`.

Detailed improvements to the evaluation methodology and their recommended implementation
order are maintained in `EVALUATION_ENHANCEMENTS.md`.

## Current execution state

The evaluation engineering layer is now ready for checkpoint execution: causal fixtures,
training/evaluator parity, immutable provenance, multi-checkpoint paired statistics, iterative
and semi-autoregressive decoding, offline feedback rollouts, novelty manifests, corruption and
counterfactual generators, readiness guards, activation collection, and controlled probes are
implemented and covered by tests. The refreshed data and generated artifacts are frozen under
`data/`.

Production-tokenizer parity now passes 148/148 ScienceWorld validation trajectories, and all 13
AOMT adapters are hash-frozen locally; the matched seed-0 triad is identified in the manifest.
The next boundary is full base-model inference:
the seven LLaDA2.0-mini shards total 30.28 GiB and should be loaded on the GPU cluster. Installed
environment simulators and step-aligned state extraction are still required before state
validity, causal counterfactual, or task-success results can be claimed. The code marks offline
rollouts and unvalidated counterfactuals accordingly.

## Objective

Determine whether structured flexible trajectory masking produces better implicit text-world
models than causal diffusion, random masking, and autoregressive baselines. The result must be
supported by simulator-grounded dynamics tests, not training loss alone.

The project starts with a ScienceWorld minimum publishable result, then expands to ALFWorld
and WebShop. All three datasets are downloaded, but the current training runs and masking
validation cover ScienceWorld only; the other environments still need data audits, step-aligned
state extraction, and environment adapters.

## Project outcomes

The minimum publishable result contains:

1. A fair, replicated D-Flex versus D-AR comparison on the same diffusion checkpoint.
2. D-Random as the structured-masking control.
3. Action-conditioned next-state prediction and closed-loop horizons 3, 5, and 10.
4. Absolute ID/OOD task success and the OOD gap.
5. An observation-target ablation that connects any gain to dynamics supervision.
6. Three seeds, uncertainty, per-task results, compute accounting, and leakage controls.

Counterfactual consistency, corruption recovery, latent-state probes, autoregressive baselines,
and additional environments strengthen the result but do not precede validation of the core
ScienceWorld pipeline.

## Workstreams

| Workstream | Purpose | Main artifacts |
|---|---|---|
| Governance and reproducibility | Make runs comparable and auditable | frozen configs, result schema, preflight checks |
| Data and simulator | Supply leakage-controlled trajectories and ground-truth state | manifests, state schema, replay adapter |
| Training | Produce matched model conditions | checkpoints, logs, supervision audits |
| Evaluation | Test behaviour and dynamics under identical conditions | evaluation harnesses, per-example predictions, metrics |
| Ablations | Identify the source of any improvement | controlled experiment matrices |
| Statistics and reporting | Turn runs into defensible claims | analysis specification, tables, figures, experiment log |

## Dependency map

```text
Freeze configs and ScienceWorld split
                |
                v
Validate masks + freeze next-observation fixture
                |
                v
Seed-0 D-AR / D-Random / D-Flex triad
                |
                v
Supervision audit + shared next-observation comparison
                |
              Gate 1
             /      \
            v        v
 State schema/replay  Seeds 1 and 2 + analysis specification
            \        /
             v      v
 Next-state + closed-loop + ID/OOD evaluation
                |
              Gate 2
             /      \
            v        v
 Observation ablation  Counterfactuals/probes/corruption
             \      /
              v    v
       ScienceWorld result package
                |
              Gate 3
                |
     ALFWorld + WebShop expansion
                |
       Full ablations and report
```

## Phased execution

### Phase 0 — Freeze the protocol

Goal: make it impossible for nominally identical runs to differ silently.

Tasks: `GOV-01`, `GOV-02`, `DATA-01`, `TRAIN-01`, `EVAL-01`.

Exit criteria:

- ScienceWorld split and evaluation examples have stable IDs.
- Run metadata captures every comparison-relevant setting.
- Deterministic fixtures demonstrate causal visibility for D-AR/next-observation and
  bidirectional visibility only where D-Flex permits it.
- One command can validate each condition without allocating a full training run.

Estimate: 2–4 focused engineering days; no full GPU jobs.

### Phase 1 — Establish the seed-0 signal

Goal: determine whether the current implementation warrants replication and evaluation work.

Tasks: `TRAIN-02`, `TRAIN-03`, `EVAL-02`, with `REP-01` updated after completion.

Execution:

- Launch D-AR, D-Random, and D-Flex as a matched three-job batch.
- While jobs run, finish the machine-readable results schema and comparison notebook/script.
- Compare only the shared causal next-observation task; training-objective loss remains a
  convergence diagnostic.

Gate 1 — proceed when:

- all three conditions load and score on exactly the same evaluation fixture;
- supervision/context differences are quantified and no unexplained material mismatch remains;
- D-Flex is competitive enough on shared next-observation loss to justify the expensive suite;
- failure traces do not reveal leakage or a broken decoding path.

If D-Flex is clearly worse, first debug the objective mixture and inference implementation.
Do not launch the ablation matrix to search for a favourable variant.

Estimate: 1 engineering day plus 3 GPU jobs and queue time.

### Phase 2 — Build simulator-grounded evaluation

Goal: move from token loss to actual environment dynamics.

Tasks: `DATA-02`, `DATA-03`, `EVAL-03`, `EVAL-04`, `EVAL-05`, and `STAT-01`.

Implementation order:

1. Define state labels and a common transition record.
2. Replay trajectories through ScienceWorld and verify state alignment.
3. Implement one-step action-conditioned prediction.
4. Reuse the same transition/scoring interface for closed-loop rollouts.
5. Add task execution and the ID/OOD split.
6. Freeze metric versions and the statistical analysis specification.

Gate 2 — proceed when:

- a hand-audited fixture passes state alignment and transition scoring;
- a known-good simulator transition scores correctly and deliberate invalid outputs fail;
- closed-loop evaluation uses no future observation and saves replayable traces;
- ID/OOD task IDs and aggregation rules are fixed before the multi-seed comparison is read.

Estimate: 6–10 engineering days, dominated by simulator integration and decoding.

### Phase 3 — Replicate the core result

Goal: obtain the minimum statistical evidence for the central claim.

Tasks: `TRAIN-04`, `STAT-02`, plus evaluation of all nine checkpoints from three conditions and
three seeds.

Run organization:

- Use run names of the form `<condition>-<dataset>-s<seed>-<protocol_version>`.
- Treat a seed as a paired block: all conditions use the same split, evaluation examples, and
  where applicable sampling seeds.
- Launch seeds 1 and 2 as matched condition batches; never replace only an unfavourable seed.

Gate 3 — the core result is complete when:

- next-state, closed-loop, and ID/OOD metrics exist for all three seeds and conditions;
- means, intervals, effect sizes, per-task distributions, and compute are reported;
- conclusions survive the prespecified supervision and leakage checks;
- negative or mixed results are recorded without changing the primary metric.

Estimate: 6 GPU training jobs, evaluation compute, and 2–3 analysis days.

### Phase 4 — Explain the result

Goal: determine which aspect of flexible masking causes any gain.

First task: `ABL-01` observation-target ablation. Then prioritize `ABL-06` supervision/context
controls, `ABL-03` objective mixture, `ABL-02` curriculum, `ABL-04` granularity, and `ABL-05`
Think-block treatment.

Launch rule: each ablation gets one prespecified question and primary metric. Start with seed 0;
replicate only the informative comparisons with seeds 1 and 2. This staged rule limits the GPU
matrix without treating an unreplicated seed-0 result as confirmatory.

Estimate: at least 9 GPU jobs for the required observation ablation; the full matrix should be
budgeted only after observed effect sizes are available.

### Phase 5 — Add mechanistic and robustness evidence

Goal: test intervention response, internal state tracking, and recovery.

Tasks: `EVAL-06`, `EVAL-07`, `EVAL-08`, and `EVAL-09`.

Counterfactuals should be implemented before probes because they test functional behaviour.
Probes remain diagnostic and must retain shuffled-label, pretrained, bag-of-words, layer-wise,
and target-visibility controls. Corrective decoding is reported separately from training gains.

Estimate: 6–10 engineering days plus activation storage and evaluation compute.

### Phase 6 — Expand environments and finalize

Goal: test whether conclusions generalize beyond ScienceWorld.

Tasks: `DATA-04`, `DATA-05`, `ABL-07`, `STAT-03`, `REP-02`, `REP-03`, `REP-04`.

Apply the frozen common protocol to ALFWorld and WebShop, adding only documented
environment-specific state metrics. Always report each environment separately before any
normalized aggregate.

Estimate: schedule after ScienceWorld Gate 3; size depends on simulator and dataset access.

## Task organization and operating rhythm

Use a small Kanban board with these columns:

`Backlog -> Ready -> In progress -> Review -> Done`

Rules:

- Keep at most two engineering tasks in progress and one analysis task in review.
- GPU jobs do not consume engineering WIP, but cap each launch wave to one matched experiment
  batch so failures are detected before propagating across the matrix.
- Pull tasks by dependency and priority, not by ease. The recommended order is maintained in
  `PROJECT_BACKLOG.md`.
- Every task has one accountable owner even when implementation or review is shared.
- Review data/evaluation changes with saved fixtures; review experiment tasks with configs,
  logs, metrics, and artifact checks.

Suggested weekly rhythm:

- Start of week: choose the gate outcome and no more than five ready task IDs.
- Before a run wave: record hypotheses, configs, primary metrics, and stop/failure rules.
- Mid-week: review fixtures and failed runs before submitting more jobs.
- End of week: update statuses, `EXPERIMENTS.md`, compute use, decisions, and next dependencies.

## Experiment naming and artifact layout

Use this naming convention:

```text
<condition>-<dataset>-s<seed>-<protocol_version>
```

Each run directory should contain:

```text
runs/<run_id>/
  config.json
  environment.json
  metrics.json
  predictions.jsonl
  failures.jsonl
  charts/
  lora_adapter/
```

Evaluation outputs should reference the run ID and metric version rather than modifying the
training metrics in place.

## Risk controls

| Risk | Control |
|---|---|
| Lower D-Flex reconstruction loss is mistaken for better dynamics | Use shared next-observation and simulator-grounded metrics as primary evidence |
| Future information leaks through Think blocks or context | Positional audits, target-visibility fixtures, and Think-block ablations |
| D-Flex receives more or easier supervision | Audit scored tokens, target types, entropy proxies, contexts, and compute |
| Closed-loop evaluation accidentally inpaints future truth | Assert that future observations are absent and save replayable contexts |
| Large ablation matrix consumes compute before the method works | Gate seed-0, core evaluation, replication, and ablations in that order |
| One dataset or aggregate hides failures | Report per-task and per-environment distributions before aggregates |
| Probes exploit textual shortcuts | Anti-visibility histories and multiple probe baselines |
| Partial replication overstates certainty | Label exploratory runs; require three seeds for confirmatory comparisons |

## Immediate action list

1. Stage the exact LLaDA2.0-mini base revision on a GPU node and validate adapter loading.
2. Execute `run-suite` for the downloaded D-AR, D-Random, and D-Flex seed-0 adapters under all
   three decoders (`EVAL-02`).
3. Audit supervision and review the paired seed-0 comparison (`TRAIN-03`, `STAT-01`).
4. In parallel, install ScienceWorld and implement replay/state extraction (`DATA-02`,
   `DATA-03`); do not promote offline metrics to simulator-grounded claims.
5. Hold Gate 1 before beginning multi-seed runs or ablations.
