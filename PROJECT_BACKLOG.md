# D-Flex Project Backlog

Last updated: 2026-07-22

This backlog translates the report's evaluation plan into trackable work. `EXPERIMENTS.md`
remains the experiment record; this file describes work that is done, ready, or still required.
Execution phases and decision gates are defined in `PROJECT_PLAN.md`.

## How to use this backlog

- Priority: **P0** blocks the main scientific claim, **P1** is required for the full report, and
  **P2** is useful extension work.
- Status: `Done`, `Ready`, `In progress`, `Blocked`, or `Backlog`. Move an item to `Ready` only when all of
  its dependencies are complete.
- An experiment is not done when a cluster job finishes. It is done when its configuration,
  logs, metrics, artifacts, failure status, and interpretation are recorded in `EXPERIMENTS.md`.
- Use the task ID in branch names, run names, commits, and experiment-log entries.

## Current position

The repository has refreshed and hash-frozen ScienceWorld, ALFWorld, and WebShop splits, causal
ScienceWorld/WebShop fixtures, versioned novelty manifests, and a machine-readable readiness
audit. Training/evaluator parity, paired multi-run statistics, three decoding modes, offline
closed-loop generation, robustness generators, and controlled probe tooling are implemented and
tested. Production-tokenizer parity passes 148/148 ScienceWorld validation trajectories, and the
complete 13-adapter collection is downloaded and hash-frozen, with the matched seed-0 triad
identified separately. The fair checkpoint matrix has not been
run locally because its separate base model is 30.28 GiB and no NVIDIA runtime is exposed.
Simulator replay, step-aligned
state extraction, validated counterfactuals, activation extraction, and task execution remain
runtime-gated by checkpoints, labels, and simulator installations. The refreshed ALFWorld
validation data is plan-only and cannot support next-observation scoring as downloaded.

## Backlog

| ID | Priority | Workstream | Task and acceptance criteria | Depends on | Size | Status |
|---|---|---|---|---|---|---|
| INFRA-00 | P0 | Infrastructure | Replace global 8192-token padding with dynamic batch padding and record the resolved performance failure in the experiment log. | — | M | Done |
| DATA-00 | P0 | Data | Normalize the ScienceWorld ETO MVP into typed Goal/Think/Action/Observation blocks, create the initial manifest, and audit Think-block leakage. | — | L | Done |
| TRAIN-00 | P0 | Training | Implement D-AR suffix masking, D-Random block masking, and the four-objective D-Flex mixture including inverse dynamics. | DATA-00 | L | Done |
| EVAL-00 | P0 | Evaluation | Implement the common causal `val_nextobs` loss in the training script so all three diffusion conditions can be compared on one target. | TRAIN-00 | M | Done |
| GOV-01 | P0 | Reproducibility | Create a frozen experiment configuration containing model revision, code commit, dataset revision, objective mixture, context length, mask policy, seed, and decoding settings. Every new run writes this metadata beside `metrics.json`. | — | S | Ready |
| GOV-02 | P0 | Reproducibility | Define one machine-readable result schema for training and evaluation. It must include dataset, split, condition, seed, metric version, per-task values, aggregate value, compute, and failure flags. | GOV-01 | S | In progress |
| GOV-03 | P1 | Quality | Add a preflight checklist for data leakage, target visibility, checkpoint loading, deterministic sampling, and output paths; make failed checks stop a run. | GOV-01 | S | Backlog |
| DATA-01 | P0 | Data | Freeze ScienceWorld train/validation/test and ID/OOD task IDs in a versioned manifest. Verify no trajectory or task overlap and document the split rule. | — | M | In progress |
| DATA-02 | P0 | Data | Define the structured state-label schema needed by evaluation: inventory, location, subgoals, prerequisites, valid actions, and task flags. Include missing-value rules and label provenance. | DATA-01 | M | Backlog |
| DATA-03 | P0 | Data | Build simulator replay/state extraction for ScienceWorld. On a sampled audit set, reconstructed transitions must match simulator state and invalid trajectories must be flagged rather than silently scored. | DATA-02 | L | Backlog |
| DATA-04 | P1 | Data | Audit the downloaded ALFWorld trajectories and trajectory-level labels, add step-aligned simulator state, run leakage checks, and define an ID/OOD manifest using the common schema. | DATA-02 | L | In progress |
| DATA-05 | P1 | Data | Audit the downloaded WebShop trajectories and partial trajectory-level labels, add step-aligned constraint state, run leakage checks, and define an ID/OOD manifest using the common schema. | DATA-02 | L | In progress |
| TRAIN-01 | P0 | Training | Validate `d_ar`, `d_flex`, and `random_block` with deterministic dry runs. Check target blocks, causal visibility, achieved mask rate, loss-bearing token counts, and leakage assertions. Save one fixture per condition. | GOV-01 | S | Ready |
| TRAIN-02 | P0 | Training | Run the current seed-0 triad: D-AR, D-Random, and D-Flex for 400 steps with identical model/data/compute settings. All three runs must finish and produce adapters, configs, logs, `metrics.json`, and charts. | TRAIN-01, DATA-01 | M / 3 GPU jobs | Ready |
| TRAIN-03 | P0 | Training | Audit supervision comparability across the seed-0 triad: unique trajectories, gradient steps, batch tokens, scored tokens by block type, achieved mask rate, target entropy proxy, and context length. Explain material mismatches. | TRAIN-02, GOV-02 | M | Backlog |
| TRAIN-04 | P0 | Training | Run seeds 1 and 2 for the three main conditions only after Gate 1 passes. Record failed or pre-empted jobs and rerun with unchanged configuration. | TRAIN-03, EVAL-02 | M / 6 GPU jobs | Backlog |
| TRAIN-05 | P1 | Training | Implement and train AR-SFT and AR-FullSeq size-matched baselines, with the same data split and documented differences in substrate and supervision. | GOV-01, DATA-01 | L | Backlog |
| EVAL-01 | P0 | Evaluation | Freeze a deterministic `val_nextobs` evaluation set and add a metric test proving that every condition receives the same causal context, target tokens, and scoring mask. | DATA-01 | S | Done |
| EVAL-02 | P0 | Evaluation | Produce the seed-0 next-observation comparison for D-AR, D-Random, and D-Flex. Report per-task loss, bootstrap uncertainty across examples, context/target statistics, and qualitative failure cases; do not compare training-task losses as evidence. | TRAIN-02, EVAL-01 | M | Ready |
| EVAL-03 | P0 | Evaluation | Implement action-conditioned next-state evaluation against simulator observations and structured state. Report exact/structured correctness, precondition validity, object changes, and rule-consistent consequences separately from text similarity. | DATA-03, GOV-02 | L | Backlog |
| EVAL-04 | P0 | Evaluation | Implement closed-loop rollouts at horizons 3, 5, and 10. Feed predictions back as context, prohibit future ground truth, report error accumulation and invalid-state rate, and save replayable traces. | EVAL-03 | L | In progress |
| EVAL-05 | P0 | Evaluation | Implement ID/OOD task success and the OOD gap. Report absolute ID and OOD scores with uncertainty so a low gap caused by uniformly poor performance is not rewarded. | DATA-01, DATA-03 | L | In progress |
| EVAL-06 | P1 | Evaluation | Implement environment-valid ScienceWorld counterfactuals with simulator validation. Report consequence accuracy by edit type and exclude ambiguous/undefined edits using logged reasons. | DATA-03, EVAL-03 | L | In progress |
| EVAL-07 | P1 | Evaluation | Implement corruption recovery for impossible observations, invalid actions, deleted steps, and inconsistent states. Report recovery rate, steps to recovery, and post-recovery rollout quality. | EVAL-04 | L | In progress |
| EVAL-08 | P1 | Evaluation | Implement frozen-activation linear probes for state variables, including bag-of-words, pretrained-checkpoint, shuffled-label, layer-wise, and final-observation anti-leakage controls. | DATA-03 | L | In progress |
| EVAL-09 | P1 | Evaluation | Add iterative diffusion, one-shot mask prediction, and ProSeCo-style correction as decoding variants. Keep training-mask and inference-time correction effects separate. | EVAL-04, EVAL-07 | L | In progress |
| ABL-01 | P0 | Ablations | Run observation-target ablation: full D-Flex, action-only, and observation-only. Use the core world-model metrics and test whether gains shrink without observation reconstruction. | TRAIN-04, EVAL-03 | L / 9 condition-seed results (6 new jobs if full D-Flex is reused) | Backlog |
| ABL-02 | P1 | Ablations | Compare incremental curriculum with fixed low/medium/high ratios and 30/50/70% endpoints. Report convergence, reconstruction failure, next-state performance, and OOD performance. | TRAIN-04, EVAL-05 | L | Backlog |
| ABL-03 | P1 | Ablations | Remove inverse dynamics, retrodiction, imputation, and denoising one at a time. Predefine the primary metric before launching the matrix. | TRAIN-04, EVAL-03 | L | Backlog |
| ABL-04 | P1 | Ablations | Compare semantic-block, span, and token masking under matched loss-bearing target counts. | TRAIN-04, TRAIN-03 | L | Backlog |
| ABL-05 | P1 | Ablations | Compare keeping, masking, removing, and evaluation-time hiding of Think blocks. Re-run leakage checks for every treatment. | DATA-01, EVAL-03 | L | Backlog |
| ABL-06 | P1 | Ablations | Match supervision budget and context length explicitly; test whether gains survive both controls. | TRAIN-03, EVAL-03 | M | Backlog |
| ABL-07 | P2 | Ablations | Compare per-dataset, joint, and cross-dataset training after all three data pipelines pass their audits. | DATA-04, DATA-05 | XL | Blocked |
| STAT-01 | P0 | Analysis | Write the analysis specification before multi-seed runs: primary comparison, primary metric, aggregation, confidence intervals, paired tests, missing-run policy, and multiple-comparison handling. | EVAL-02 | M | Backlog |
| STAT-02 | P0 | Analysis | Produce the main three-seed ScienceWorld result table with means, uncertainty, per-task distributions, effect sizes, compute cost, and seed-level values. | TRAIN-04, EVAL-03, EVAL-04, EVAL-05, STAT-01 | M | Backlog |
| STAT-03 | P1 | Analysis | Add normalized cross-environment aggregates without hiding per-environment failures. Document the normalization formula and include a leave-one-environment-out sensitivity check. | DATA-04, DATA-05, GOV-02 | M | Blocked |
| REP-01 | P0 | Reporting | Keep `EXPERIMENTS.md` current after every run batch, including negative and failed runs. Link configs and artifacts rather than copying undocumented values. | Ongoing | S / recurring | Ready |
| REP-02 | P1 | Reporting | Build final figures and tables for task success, next-state prediction, closed-loop rollouts, counterfactuals, probes, corruption recovery, ablations, and compute. | STAT-02, required P1 evaluations | L | Backlog |
| REP-03 | P1 | Reporting | Write limitations and claim audit. Every paper claim must point to a metric, control, and result; exploratory or under-replicated results must be labelled. | REP-02 | M | Backlog |
| REP-04 | P1 | Release | Package manifests, evaluation fixtures, run configurations, metric code, environment instructions, and a clean reproduction command. Verify from a fresh checkout. | REP-02, GOV-03 | L | Backlog |

## Recommended next queue

Do these in order unless a cluster queue permits independent work in parallel:

1. `GOV-01`, `DATA-01`, `TRAIN-01`, and `EVAL-01`.
2. Launch `TRAIN-02` only after the dry-run and evaluation fixtures pass.
3. Complete `GOV-02` while the three training jobs run.
4. Run `TRAIN-03` and `EVAL-02`; review Gate 1 in `PROJECT_PLAN.md`.
5. If Gate 1 passes, start `DATA-02`/`DATA-03`, write `STAT-01`, and launch `TRAIN-04`.

## Definition of done for an experiment

- The hypothesis and primary metric were written before inspecting the result.
- The exact command/configuration, code revision, dataset revision, seed, and checkpoint are saved.
- Leakage, context, target, and supervision-budget checks pass.
- Raw per-example outputs and aggregate metrics use the versioned result schema.
- Failures and exclusions are explicit and reproducible.
- Results include uncertainty and seed-level values where replication is required.
- The outcome and interpretation are added to `EXPERIMENTS.md`, including negative results.
