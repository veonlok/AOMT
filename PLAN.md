# AOMT Hugging Face Audit and Multi-Benchmark Validation Plan

## Summary

Use the Hugging Face AOMT dataset as the primary source of truth, with the local `data/*.jsonl` files treated as a cached snapshot for reproducibility checks only. The audit should explain whether D-Flex's advantage over D-Random is due to dataset structure, split leakage, or genuine representation gains, while keeping the downstream evaluation framing explicit: models are evaluated on ScienceWorld, ALFWorld, and WebShop.

The core logic is:

1. Audit the HF dataset and reconstruct its real structure and split properties.
2. Build leakage-robust alternative splits for the trajectory dataset itself.
3. Compare D-Random vs D-Flex on dataset slices and then interpret those findings against downstream results on ScienceWorld, ALFWorld, and WebShop.
4. Recommend a validation protocol that better predicts cross-environment generalization.

## Implementation Changes

### 1. Make Hugging Face the canonical dataset input

Build the harness around the HF dataset repo first, then optionally verify that the local JSONL snapshot matches or differs.

Data ingestion rules:
- Primary input: the HF dataset card/files for `AK2802/AOMT`.
- Secondary input: local `data/train.jsonl`, `data/validation.jsonl`, `data/test.jsonl` as a pinned snapshot.
- Record dataset revision, file hashes, row counts, and any schema differences between HF and local copies.
- If HF contains multiple environments or revisions, preserve that structure instead of collapsing prematurely to the local ScienceWorld-only snapshot.

Outputs:
- `source_of_truth_report.md` describing HF contents, available splits/files, schema, and any mismatch vs local data.
- `snapshot_diff.json` summarizing row-count, field-level, and split-level differences.

### 2. Generalize the audit schema to support ScienceWorld, ALFWorld, and WebShop

The row-feature schema should not assume ScienceWorld-only semantics, even if the currently cached local files are ScienceWorld.

Required normalized fields:
- `env_name`: `scienceworld`, `alfworld`, `webshop`, or other HF-provided env ids
- `trajectory_id`
- `split`
- `goal_text`
- `goal_family`
- `goal_instance_key`
- `n_steps`, `n_blocks`, `n_actions`, `n_observations`
- `trajectory_length_bin`
- `action_skeleton_hash`
- `full_trajectory_hash`
- `duplicate_cluster_id`
- `has_leakage_flag`
- `source_revision`

Environment-specific normalization:
- ScienceWorld: preserve `Goal/Think/Action/Observation` structure.
- ALFWorld: map its action/observation format into the same canonical event schema.
- WebShop: normalize browsing/search/click/cart/purchase actions into the same trajectory abstraction.
- If a field is environment-specific and not portable, store it in an `env_metadata` column rather than forcing it into the shared schema.

### 3. Build a portable trajectory taxonomy

Define taxonomy in two layers.

Shared layer across all environments:
- `goal_family`: retrieval, comparison, classification, manipulation, multi-step planning, navigation, commerce, etc.
- `trajectory_length_bin`: short / medium / long.
- `interaction_pattern`: low-branching, repetitive, search-heavy, stateful, confirmation-heavy.
- `completion_status`: success / failure / unknown, only when inferable.
- `ambiguity_pattern`: includes disambiguation turns, retries, repeated observations, or repair loops.

Environment-specific layer:
- ScienceWorld-specific families like conductivity, temperature, melting point, plant growth, object finding, lifespan comparison.
- ALFWorld-specific families like put/pick/clean/heat/cool/examine/place.
- WebShop-specific families like attribute filtering, comparison shopping, budget-constrained selection, checkout-oriented search.

This gives a single slice framework that can be used both for dataset auditing and downstream error analysis.

### 4. Audit leakage in the HF split, not just the local ScienceWorld cache

Run leakage analysis at four levels against the HF-defined train/validation split.

Checks:
- Exact duplicate trajectories via `full_trajectory_hash`.
- Near-duplicate trajectories via `goal + action skeleton` and fuzzy observation similarity.
- Template leakage via shared `goal_instance_key` and normalized goal templates.
- Environment overlap via shared domains/tasks within ScienceWorld, ALFWorld, and WebShop.
- Distribution mismatch via environment mix, task-family mix, trajectory length, ambiguity patterns, and completion rates.

Required reports:
- HF split overlap matrix.
- Per-environment leakage summary.
- Per-family memorization-risk summary.
- A validation-row risk label:
  - `exact_seen`
  - `template_seen`
  - `family_seen`
  - `env_unseen`
  - `strong_ood`

The current local ScienceWorld snapshot already suggests high template overlap; the updated plan should verify whether that is also true in HF proper or only in the cached subset.

### 5. Tie dataset leakage findings to model behavior

Use the audit harness to compare D-Random vs D-Flex on slices that matter for memorization vs generalization.

Primary slice comparisons:
- seen-template vs unseen-template
- seen-family vs unseen-family
- short vs long trajectories
- low-ambiguity vs high-ambiguity trajectories
- per-environment: ScienceWorld, ALFWorld, WebShop
- within ScienceWorld: task families like conductivity, temperature, growth, retrieval
- duplicate-risk tiers from the leakage audit

Interpretation rule:
- If D-Random's advantage is concentrated on seen-template or duplicate-prone slices, that supports the memorization hypothesis.
- If D-Flex gains grow on unseen-template, cross-family, or cross-environment slices, that supports the representation-learning hypothesis.

Because you evaluate on ScienceWorld, ALFWorld, and WebShop, the report should explicitly connect dataset-side slice results to downstream benchmark behavior:
- Does the split that best predicts ScienceWorld validation also best predict ALFWorld/WebShop transfer?
- Does removing template leakage reduce D-Random's apparent strength and improve correlation with downstream OOD metrics?

### 6. Produce two recommended validation benchmarks

Keep the two-benchmark structure, but make it multi-environment aware.

#### A. Balanced IID benchmark

Purpose:
- stable training-time model selection.

Construction:
- group exact/near duplicates so they cannot cross splits.
- stratify by environment, goal family, length bin, and completion status where available.
- preserve approximate environment proportions from the HF dataset.

Use:
- quick sanity checks and variance control.

#### B. Grouped OOD benchmark

Purpose:
- primary benchmark for representation quality and generalization.

Construction:
- group by duplicate cluster plus normalized goal template.
- assign groups wholesale across train/validation/test.
- enforce no shared exact templates between train and validation/test.
- preserve environment coverage where possible, but prioritize leakage prevention over perfect balance.
- if HF contains multiple environments, prefer holding out template groups within each environment rather than fully removing an environment unless you explicitly want cross-environment OOD.

Use:
- headline validation metric for explaining transfer to ScienceWorld, ALFWorld, and WebShop.

### 7. Deliver a reusable harness and final report

Deliverables should be reusable for future dataset revisions and future environments.

Artifacts:
- `trajectory_features.parquet` or `.csv`
- `duplicate_clusters.json`
- `split_current_hf_audit.json`
- `split_iid_balanced.json`
- `split_grouped_ood.json`
- `dataset_audit_report.md` or notebook export
- `model_slice_report.md`
- plots for overlap, distribution shift, and per-slice D-Random vs D-Flex gaps

The final writeup should include a section called `Relevance to Downstream Benchmarks` that explicitly maps the dataset audit findings to ScienceWorld, ALFWorld, and WebShop evaluation outcomes.

## Important Interfaces / Outputs

Stable interfaces:
- HF dataset loader that can resolve revision/file names explicitly.
- Canonical trajectory dataframe with shared fields across environments.
- Split-builder API that accepts grouping keys and stratification keys.
- Per-trajectory evaluator output keyed by `trajectory_id`, `env_name`, and slice labels.

Expected outputs from model comparison:
- per-trajectory loss or score
- slice-level aggregate metrics
- seen/unseen generalization tables
- environment-level transfer summary

## Test Plan

Validation checks:
- HF loader reproduces the exact published row counts for the chosen revision.
- Local snapshot reconciliation reports exact differences instead of silently merging.
- Feature extraction is deterministic.
- Duplicate clustering catches exact duplicates and stable near-duplicate groups on fixtures.
- IID split blocks duplicate leakage across splits.
- Grouped OOD split blocks shared templates across splits.
- Environment proportions are reported for every split.
- Model-slice aggregation joins cleanly across ScienceWorld, ALFWorld, and WebShop evaluations.

Analysis scenarios:
- HF current split leakage audit.
- local-snapshot-vs-HF reconciliation.
- D-Random vs D-Flex on seen/unseen slices.
- correlation between validation results and downstream performance on ScienceWorld, ALFWorld, and WebShop.
- recommendation of which validation protocol best predicts downstream OOD behavior.

## Assumptions

- HF is the canonical dataset source; local JSONL files are a cached subset/snapshot.
- ScienceWorld, ALFWorld, and WebShop are the downstream evaluation environments that matter for the conclusion, even if the local cached training data currently emphasizes ScienceWorld.
- If HF currently exposes only part of that environment coverage, the harness should still be built to support all three and document what is present vs absent.
- The recommended final protocol remains a two-benchmark suite:
  - Balanced IID for sanity/model selection
  - Grouped OOD as the primary generalization benchmark
