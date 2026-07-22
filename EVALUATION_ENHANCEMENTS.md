# Evaluation Enhancement Plan

Last updated: 2026-07-22

## Objective

Strengthen the evaluation framework so that the central D-Flex claim is supported by paired,
simulator-grounded tests of learned transition dynamics rather than reconstruction loss or text
similarity alone.

## Execution status

The repository implementation is complete for every enhancement that can be made independent
of external checkpoints and simulators. The remaining work is experiment execution or genuine
environment integration; it is not silently approximated with text metrics.

| Area | Implemented artifact | Remaining runtime dependency |
|---|---|---|
| Training/evaluator parity | Exact token-ID and mask parity checker plus regression test; production tokenizer passed 148/148 trajectories | Complete |
| Multi-run statistics | Suite runner and paired hierarchical bootstrap by seed and task/trajectory | Matched D-AR, D-Random, and D-Flex adapters |
| Decoding | One-shot, iterative diffusion, and semi-autoregressive modes with latency/pass accounting | GPU checkpoint runs |
| Provenance | Immutable HF revision/file hashes, fixture hashes, code/checkpoint/decoder metadata | Preserve manifests with each cluster run |
| Closed loop | Oracle-action generator with model feedback and future-observation/Think leakage guards | Predictor checkpoint; simulator for grounded validity/task success |
| ID/OOD | Versioned novelty manifests for ScienceWorld, ALFWorld, and WebShop | Simulator execution for absolute task success |
| Robustness | Six deterministic corruption generators and minimal counterfactual candidates | Simulator validation for causal counterfactual claims |
| Probes | Causal activation collection and group-safe probes with required lexical/label controls | Checkpoints and step-aligned target labels |
| Environment interface | Strict common adapter protocol and metadata capability audit | Install and implement each simulator's reset/step/state extraction |

Current frozen outputs are under `runs/evaluation-data/`. The refreshed
`scienceworld-compact-v2` and `alfworld-success-v2` variants now contain step-aligned simulator
state; the readiness audit reports 100% state coverage for both. WebShop remains partially
transition-ready and has no step-aligned state labels.

## P0 — Make the core comparison defensible

| Enhancement | Why it matters | Acceptance criterion |
|---|---|---|
| Training/evaluator parity test | The standalone fixture builder and `train_llada_rb.py::next_obs_mask` independently define the same task and could diverge | Both paths produce identical context and target token IDs for every validation trajectory |
| Full-checkpoint comparison command | Results are currently produced one checkpoint at a time | One command evaluates D-AR, D-Random, and D-Flex across all seeds and creates a seed-level comparison table |
| Paired statistical analysis | Ordinary bootstrap intervals ignore that models evaluate the same trajectories | Report paired effect sizes and confidence intervals, resampling task families and seeds rather than individual tokens |
| Iterative diffusion decoding | `evaluations/llada_nextobs.py` currently uses one-shot token-wise argmax, which can produce locally plausible but globally incoherent observations | Compare one-shot, iterative diffusion, and matched-budget semi-autoregressive decoding |
| Provenance manifest | Evaluation results must identify their exact data and model origins | Every result records the HF repository revision, file hashes, fixture hash, code commit, checkpoint, decoder, seed, and metric version |

The comparison report should include:

- token-weighted NLL;
- trajectory-macro NLL;
- paired D-Flex minus baseline differences;
- target length and context length;
- results by task family and action type;
- inference time and decoding steps.

## P0 — Add simulator-grounded state evaluation

Text similarity remains diagnostic. The central evaluation should use structured environment
state through a common adapter:

```python
reset(task_id, seed) -> state, observation
step(action) -> state, observation, reward, done, info
extract_state(state) -> dict
validate_transition(before, action, after) -> validation
```

Implement adapters in this order:

1. **ALFWorld**, because the refreshed data includes task type, PDDL parameters, and scene
   metadata.
2. **WebShop**, focusing on product attributes and constraint satisfaction.
3. **ScienceWorld**, requiring simulator replay because its rows still have empty state labels.

State-level metrics should include:

- exact state-delta accuracy;
- changed-key and unchanged-key accuracy;
- entity precision and recall;
- object-location accuracy;
- inventory accuracy;
- precondition validity;
- invalid-transition rate;
- hallucinated entity and relation rate;
- numeric-value accuracy for temperature, quantity, and price-like fields.

Score the predicted change from `s_t` to `s_t+1`, not only the full resulting state. Otherwise,
large unchanged state descriptions can dominate the result.

## P0 — Generate real closed-loop rollouts

`evaluations/rollouts.py` validates and scores traces but does not yet generate them. Add two
rollout modes:

- **Oracle-action dynamics rollout:** future actions come from the recorded trajectory while
  observations come from the model. This isolates environment modelling.
- **Agent rollout:** the model generates both actions and observations, measuring the combined
  policy/world-model system.

For horizons 3, 5, and 10, report:

- time to first state error;
- probability of remaining valid through each horizon;
- error-accumulation curve;
- final-state accuracy;
- invalid-action rate;
- repeated or degenerate state rate;
- task success;
- recovery after the first error.

Exclude future Think blocks from oracle-action rollouts because they may reveal ground-truth
state.

## P1 — Strengthen counterfactual evaluation

Add a simulator-backed case generator for:

- removing a prerequisite;
- relocating an object;
- opening or closing a receptacle;
- changing inventory;
- modifying a WebShop attribute constraint;
- changing a numeric property;
- substituting an action while holding the initial state fixed.

Each case should have a matched control where the edit is irrelevant to the selected action.
This distinguishes genuine causal sensitivity from a model that changes its prediction whenever
any context changes.

Add these metrics:

- consequence accuracy;
- change-sensitivity accuracy;
- irrelevant-edit invariance;
- minimal-pair preference accuracy;
- state-delta direction accuracy;
- counterfactual calibration.

## P1 — Improve corruption recovery

Extend the existing corruption scorer with a deterministic injector for:

- impossible observations;
- invalid actions;
- deleted transitions;
- duplicated transitions;
- swapped observations;
- contradictory state descriptions;
- corrupted entity names or numeric values.

Evaluate each corruption under:

- one-shot prediction;
- iterative diffusion;
- causal semi-autoregressive decoding;
- corrective refinement.

In addition to recovery rate and steps to recovery, measure:

- false-correction rate on clean trajectories;
- whether the original error is actually removed;
- whether correction introduces a new inconsistency;
- post-recovery task success;
- recovery quality versus corruption severity.

The clean-trajectory false-correction control is required: an aggressive decoder may appear
good at recovery simply because it rewrites everything.

## P1 — Add representation probes

Build an activation collection pass keyed by fixture ID, checkpoint, layer, and token position.
Probe targets should include:

- inventory;
- room or object location;
- achieved subgoals;
- valid-action preconditions;
- receptacle relationships;
- task constraints;
- environment-specific state flags.

Required controls:

- bag-of-words;
- frozen pretrained checkpoint;
- shuffled labels;
- majority class;
- final-observation-only;
- history with the target phrase removed;
- layer-wise comparison.

Use train/validation/test partitions grouped by task or scene so that nearly identical
trajectories cannot cross probe splits.

## P1 — Define genuine ID/OOD splits

The refreshed files do not contain explicit ID/OOD membership. Introduce versioned split
manifests based on meaningful novelty:

- unseen task templates;
- unseen object combinations;
- unseen rule or prerequisite combinations;
- unseen scenes;
- compositional constraint novelty;
- trajectory-length extrapolation.

Do not treat the existing validation/test distinction as OOD without demonstrating which
distributional property changed. Report absolute ID and OOD performance beside the gap.

## P2 — Improve diagnostics and reporting

Add:

- performance by observation and context length;
- common versus rare actions;
- successful versus failed source trajectories;
- action category and state-change magnitude;
- leakage-flagged versus clean trajectory subsets;
- calibration curves for state variables;
- compute-normalized performance;
- a qualitative failure taxonomy;
- per-environment tables before aggregate scores.

The aggregate must use a predefined normalization formula and include a leave-one-environment-
out sensitivity analysis.

## Recommended implementation sequence

1. Add fixture/training parity tests.
2. Build the multi-checkpoint, multi-seed comparison command.
3. Add iterative diffusion decoding.
4. Implement the ALFWorld replay/state adapter.
5. Generate oracle-action closed-loop rollouts.
6. Add ID/OOD manifests.
7. Implement counterfactual and corruption generators.
8. Add probes.
9. Extend to WebShop and ScienceWorld.
10. Freeze the statistical analysis specification before running the full matrix.

## Highest-value next milestone

The environment replay layer is the highest-value enhancement. It converts the current
evaluation framework from text reconstruction analysis into a genuine test of learned
transition dynamics.
