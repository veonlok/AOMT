# Model Evaluation Report

Updated: 2026-07-22

## Status

The refreshed data pipeline, model inventory, causal fixture, parity test, and all 13 adapter
integrity diagnostics are complete. Behavioral checkpoint inference is **not complete** because
`models/LLaDA2.0-mini` contains the tokenizer, config, code, and weight index but none of the
seven base weight shards. The preflight therefore reports `evaluation_ready=false`; no NLL,
generation, state-accuracy, or model-ranking claim is made.

## Evaluation data

| Environment | Validation trajectories | Transitions | Step-state coverage | Validity coverage |
|---|---:|---:|---:|---:|
| ScienceWorld compact v2 | 1,762 | 100,922 | 100% | 100% |
| ALFWorld success v2 | 111 | 2,831 | 100% | unavailable |
| WebShop | 377 | 1,405 | 0% | unavailable |

The frozen ScienceWorld last-transition fixture contains 1,718 examples after excluding 44
trajectories longer than 8,192 production-tokenizer tokens. It contains 929 ID and 789 OOD
examples, has 100% structured-state coverage, averages 1,619.7 tokens, and has a maximum length
of 8,113 tokens. Training/evaluator parity passed on all 1,718 retained examples with zero
mismatches.

## Adapter diagnostics

All 13 LoRA adapters load as valid safetensors and share the same schema: 80 tensors,
2,949,120 parameters, rank 16, and alpha 32. Their base-model paths now consistently reference
`inclusionAI/LLaDA2.0-mini`. Parameter L2 norms range from 15.05 (`dar`) to 16.39
(`dprog_med`); all are finite.

For the matched v2 triad, raw parameter cosine similarities are 0.954 for D-Flex/D-Random,
0.954 for D-Flex/D-AR, and 0.932 for D-AR/D-Random. These values show that the adapters are
distinct while remaining close to a common initialization. They are optimization diagnostics,
not behavioral performance scores.

## Blocking check

The base index expects:

`model-00001-of-00007.safetensors` through `model-00007-of-00007.safetensors`.

Present shards: **0/7**. Once those files are staged, `run_evaluation_suite.sbatch` will fail-fast
preflight the layout and evaluate all 13 adapters against the frozen fixture.

## Artifacts

- `runs/evaluation-data/workspace-preflight.json`
- `runs/evaluation-data/dataset-readiness-v2.json`
- `runs/evaluation-data/parity-scienceworld-compact-v2.json`
- `runs/evaluation-data/scienceworld-compact-v2-goal-template-splits.json`
- `runs/evaluation-data/scienceworld-compact-v2-validation-last.jsonl`
- `runs/evaluation-data/adapter-analysis-v1.json`
