# Evaluation Pipeline

This package implements the evaluation protocol separately from training. It freezes exactly
which context and target each checkpoint sees, emits per-example predictions, and then scores
those predictions without loading the model again.

## Implemented now

- deterministic last-observation or all-transition fixtures;
- removal of Think blocks flagged as temporally leaky;
- one-shot, iterative-diffusion, and semi-autoregressive LLaDA decoding with a common
  fully-masked target NLL;
- exact match, token precision/recall/F1, character similarity, and number consistency;
- structured state-key, state-change, and state-stability metrics when labels are present;
- bootstrap confidence intervals and per-environment, ID/OOD, and task groupings;
- absolute ID/OOD task scores and OOD-gap aggregation from environment episode outputs;
- simulator-validated counterfactual consistency and typed corruption-recovery scoring;
- strict prediction coverage and duplicate-ID checks;
- closed-loop trace validation at arbitrary horizons, including assertions that future ground
  truth was not used and that predicted observations were fed back after step one;
- machine-readable manifests, predictions, failures, and result files.
- exact training/evaluator token-parity checks, immutable dataset provenance, and paired
  hierarchical bootstrap comparisons across conditions and seeds;
- deterministic ID/OOD novelty manifests, corruption fixtures, simulator-validation-gated
  counterfactual candidates, activation collection, and controlled layer-wise probes.

Text metrics are diagnostics. They are not a substitute for structured simulator state metrics.
The scorer reports structured-state coverage and marks those metrics unavailable when the data
does not contain labels.

## 1. Freeze fixtures

The `last` mode mirrors the shared `val_nextobs` task in `train_llada_rb.py`:

```bash
python -m evaluations.cli prepare \
  --input cp2107-textworld-trajectories/scienceworld-compact-v2/validation.jsonl \
  --output runs/evaluation-data/scienceworld-compact-v2-validation-last.jsonl \
  --mode last --model-dir models/LLaDA2.0-mini --max-tokens 8192
```

Use `--mode all` for every action-to-observation transition. The adjacent manifest records the
source-file hash, ordered example-ID hash, coverage, and group counts. Commit the fixture and
manifest before comparing checkpoints.

## 2. Run a LLaDA checkpoint

```bash
python -m evaluations.cli run-llada \
  --fixtures runs/evaluation-data/scienceworld-compact-v2-validation-last.jsonl \
  --model-dir models/LLaDA2.0-mini \
  --adapter-dir models/AOMT/dflex_v2 \
  --condition dflex --run-seed 0 \
  --output runs/dflex_v2/eval/nextobs-predictions-v1.jsonl
```

The runner tokenizes every stored block independently, as training does, masks only the target
observation, and gives the model no future text. Select `--decode one_shot_argmax`,
`iterative_diffusion`, or `semi_autoregressive`; the disclosed target length is fixed to the
reference token count for all three modes. Text is useful for failure analysis, while the
first-pass fully-masked target NLL is the common checkpoint-comparison metric.

Before a model run, prove that training and evaluation construct identical token sequences:

```bash
python -m evaluations.cli check-parity \
  --input cp2107-textworld-trajectories/scienceworld-compact-v2/validation.jsonl \
  --model-dir models/LLaDA2.0-mini \
  --output runs/evaluation-data/parity-scienceworld-compact-v2.json
```

## 3. Score predictions

```bash
python -m evaluations.cli score \
  --fixtures runs/evaluation-data/scienceworld-compact-v2-validation-last.jsonl \
  --predictions runs/dflex_v2/eval/nextobs-predictions-v1.jsonl \
  --condition dflex --run-seed 0 --checkpoint runs/dflex_v2/lora_adapter \
  --output runs/dflex_v2/eval/nextobs-results-v1.json \
  --failures runs/dflex_v2/eval/nextobs-failures-v1.jsonl
```

Run this against D-AR, D-Random, and D-Flex using the same committed fixture. Missing
predictions fail the command by default; `--allow-missing` is exploratory only and retains the
coverage deficit in the result.

For a matched multi-seed matrix, `run-suite` evaluates every adapter and immediately performs
the paired analysis. Its `--run` value is `CONDITION:SEED:ADAPTER_DIR` and may be repeated:

```bash
python -m evaluations.cli run-suite \
  --fixtures runs/evaluation-data/scienceworld-compact-v2-validation-last.jsonl \
  --model-dir models/LLaDA2.0-mini \
  --dataset-manifest cp2107-textworld-trajectories/scienceworld-compact-v2/manifest.json \
  --run d_ar:0:runs/d_ar_s0/lora_adapter \
  --run random_block:0:runs/random_s0/lora_adapter \
  --run d_flex:0:runs/dflex_s0/lora_adapter \
  --primary d_flex --decode iterative_diffusion \
  --output-dir runs/evaluation-suite-v1
```

The comparison uses only common fixture IDs and paired seeds, then resamples seeds and
task/trajectory clusters rather than treating tokens as independent observations. Existing
prediction files can be compared separately with `compare`.

For the matched seed-0 adapters in `AK2802/AOMT`, stage the seven base-model shards on the
cluster and submit `run_evaluation_suite.sbatch`. Its first argument selects the decoder and its
second argument selects the iterative pass budget, for example:

```bash
sbatch run_evaluation_suite.sbatch one_shot_argmax 1
sbatch run_evaluation_suite.sbatch iterative_diffusion 16
sbatch run_evaluation_suite.sbatch semi_autoregressive 16
```

External predictors can be evaluated by emitting JSONL with at least:

```json
{"example_id":"trajectory-1:obs:8","prediction":"predicted observation"}
```

Optional fields are `predicted_state`, `nll`, `target_tokens`, `valid_preconditions`,
`rule_consistent`, and `valid_action`.

## 4. Score closed-loop traces

The rollout generator or simulator adapter must emit one record per start state and horizon:

```json
{
  "rollout_id": "trajectory-1:start-2:h3",
  "horizon": 3,
  "steps": [
    {
      "prediction": "model observation 1",
      "target": "simulator observation 1",
      "observation_context_source": "ground_truth_initial",
      "future_ground_truth_used": false
    },
    {
      "prediction": "model observation 2",
      "target": "simulator observation 2",
      "observation_context_source": "model",
      "future_ground_truth_used": false
    }
  ]
}
```

Score it with:

```bash
python -m evaluations.cli score-rollouts \
  --rollouts runs/dflex_v2/eval/rollouts-v1.jsonl \
  --condition dflex --run-seed 0 --checkpoint runs/dflex_v2/lora_adapter \
  --output runs/dflex_v2/eval/rollout-results-v1.json
```

Use horizons 3, 5, and 10 for report results. A trace that does not explicitly prove its context
source and absence of future ground truth is rejected.

`generate-rollouts` creates leakage-controlled oracle-action traces from recorded trajectories
using a predictor exposed as `module:object`. Recorded future actions are retained, but future
observations and Think blocks never enter the model context; each model observation is fed into
the next step. These offline traces are explicitly marked `simulator_grounded=false` and cannot
support simulator-validity or task-success claims.

## 5. Aggregate ID/OOD task success

An environment runner should emit one JSON object per executed episode:

```json
{"episode_id":"task-1-seed-0","condition":"dflex","environment":"scienceworld","eval_split":"ood","seed":0,"task_id":"task-1","score":85,"max_score":100,"success":true}
```

Then run:

```bash
python -m evaluations.cli score-task-success \
  --episodes runs/task-success-episodes-v1.jsonl \
  --output runs/task-success-results-v1.json
```

The report always places the absolute OOD score beside `mean(ID) - mean(OOD)`. A condition
cannot be called transferable merely because it performs poorly on both splits.

## 6. Counterfactual and corruption contracts

`score-counterfactuals` accepts paired base and edited-state predictions. Every included case
must have `environment_valid=true` and a non-empty `validity_source`; invalid or ambiguous
edits are excluded with logged reasons. The scorer reports edited-consequence accuracy and
whether the model changes its prediction exactly when the simulator consequence changes.

`score-corruptions` accepts typed recovery traces. Each post-corruption step supplies a model
prediction, target observation, and environment-derived `coherent` flag. It reports recovery
rate, steps to recovery, post-recovery rollout quality, and results by corruption/decoder pair.

```bash
python -m evaluations.cli score-counterfactuals \
  --cases runs/dflex_v2/eval/counterfactual-cases-v1.jsonl \
  --condition dflex --run-seed 0 --checkpoint runs/dflex_v2/lora_adapter \
  --output runs/dflex_v2/eval/counterfactual-results-v1.json

python -m evaluations.cli score-corruptions \
  --cases runs/dflex_v2/eval/corruption-cases-v1.jsonl \
  --condition dflex --run-seed 0 --checkpoint runs/dflex_v2/lora_adapter \
  --output runs/dflex_v2/eval/corruption-results-v1.json
```

Deterministic source fixtures can be generated with `generate-corruptions` and
`generate-counterfactuals`. Counterfactual outputs deliberately carry
`requires_simulator_validation=true`, `environment_valid=false`; the scorer refuses to include
them until an environment adapter records a validity source. This prevents text edits from being
reported as causal interventions.

## 7. Splits, provenance, and dataset readiness

`dataset-manifest` records the immutable Hugging Face revision and hashes every split.
`create-splits` derives versioned ID/OOD assignments relative to the training set using goal
template, task family, scene, or trajectory-length novelty. These definitions are heuristics and
remain visible in every assignment; validation/test is never relabelled OOD merely by filename.

The generated artifacts for the refreshed dataset are:

- `runs/evaluation-data/dataset-readiness-v2.json`;
- `runs/evaluation-data/scienceworld-compact-v2-goal-template-splits.json`;
- `runs/evaluation-data/scienceworld-compact-v2-validation-last.jsonl`;
- `runs/evaluation-data/parity-scienceworld-compact-v2.json`;
- `runs/evaluation-data/adapter-analysis-v1.json`.

## 8. Representation probes

`collect-activations` saves causal-context hidden-state pools keyed by fixture ID, checkpoint,
layer, and provenance. `score-probes` fits group-separated logistic probes and reports majority,
bag-of-words, target-redacted bag-of-words, and shuffled-label controls. It rejects a split if a
task/scene group leaks between train and test. A pretrained-checkpoint control is produced by
collecting a second activation file without an adapter.

## Current runtime and data limitations

The readiness audit is frozen in `runs/evaluation-data/dataset-readiness-v2.json`.
ScienceWorld compact v2 supports 100,922 replayed validation transitions with complete state and
validity labels. ALFWorld success v2 supports 2,831 transitions with complete `facts` and
`gamefile` state labels. WebShop still supports 1,405 transitions from 158 of 377 validation
trajectories without step-aligned state. Consequently:

- text and NLL evaluation can run now;
- structured next-state and state-change scoring is data-ready for ScienceWorld and ALFWorld;
- novelty manifests can stratify offline results, but simulator task success requires an
  environment execution adapter;
- valid counterfactuals require simulator-backed transformations;
- latent-state probes require model checkpoints plus usable, group-separated target labels.

All 13 adapters from `AK2802/AOMT` are downloaded under `models/AOMT/`; the primary matched
seed-0 comparison uses `dar_v2`, `drandom_v2`, and `dflex_v2`. Per-adapter source commits,
configuration hashes and weight hashes are recorded by
`runs/evaluation-data/workspace-preflight.json`. Compatible `transformers` and `peft` runtimes
plus the base tokenizer/config are installed; production parity passes all 1,718 retained
ScienceWorld validation fixtures. The model index is present but all seven base weight shards are
absent, so behavioral inference remains fail-fast blocked. Simulator task execution and
counterfactual validation still require live environment adapters.

The evaluator deliberately reports these metrics as unavailable instead of silently replacing
them with text similarity.
