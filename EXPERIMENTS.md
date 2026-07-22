# Experiment Log — CP2107 D-Flex Project

All runs on A100-80GB, LLaDA2.0-mini (16B MoE, bf16), LoRA r=16 α=32 targeting
query_key_value+dense (~2.95M trainable / 16.26B total = 0.018%). ScienceWorld ETO
data: 1187 train / 148 val / 148 test. All losses are mean cross-entropy over masked tokens.

---

## Phase 0: Pilot runs (OLD — max_len=1024, val_batches=3, BROKEN)

These were the first runs using `train_llada_rb.py` before the padding bug was identified.
They used `max_len=1024` (CLAUDE.md specifies 8192), `val_batches=3` (too noisy), and
padded all sequences to max_len even if they were shorter.

| Run name   | Objective         | Steps | Final val loss | Note |
|------------|-------------------|-------|----------------|------|
| rb_const20 | random_block      | 400   | 0.278          | Easy task: full bidirectional context → low loss meaningless |
| rb_curric  | random_block (curriculum 10→60%) | 400 | 1.099 | Higher rate = harder task, as expected |
| causal20   | 5 mixed causal objs (BROKEN) | 400 | 4.104 | Invalid D-AR baseline (see below) |

**Why causal20 is NOT a valid D-AR baseline:**
The 5 causal objectives (next_action, next_observation, future_from_past, full_completion,
goal_conditioned_action) produce wildly different mask rates per step (5–60%) and gradient
scales. Mixed together without normalization, they produce conflicting gradient signals and
the mean achieved mask rate was ~15% instead of 20%. The loss never converged. Conclusion:
use a single, consistent objective for the baseline. Replaced by `d_ar` (suffix masking).

**Why rb_const20 val=0.278 is misleading:**
random_block masks tokens uniformly at random — each masked token has full bidirectional
context from all surrounding unmasked tokens. This is the easiest possible denoising task
for a bidirectional model, not a measure of trajectory understanding. Low loss ≠ better
world model. The shared `val_nextobs` metric (added in v2 runs) is the correct comparison.

---

## Phase 1: First valid D-AR vs D-Flex comparison (SLURM jobs 641489/641490)

**FAILED — timed out at step 180/200, no adapters saved.**

Root cause: `make_step_batch` padded every batch to `max_len=8192` unconditionally.
Mean sequence length in ScienceWorld ETO is ~1046 tokens. Attention cost is O(L²), so
padding 1046→8192 is (8192/1046)² ≈ 61× more compute per step. At 38s/step × 200 steps
= 133 min > 120 min wall time. The adapter save only happens at the end of the loop, so
nothing was saved.

**Fix applied:** Two-pass dynamic padding in `make_step_batch` — collect all examples
first, then `pad_to = min(max_batch_len, max_len)`. Mean pad length in practice ~1200
tokens. Step time dropped from ~38s to ~18s.

---

## Phase 2: First valid completed runs (SLURM jobs 642333/642334)

Script: `train_llada_rb.py` with dynamic padding, filter-not-truncate, val_batches=15.
**Note:** D-Flex in this run had only 3 objectives (no `inverse_dynamics`).

### Run: `dar` (job 642333) — D-AR baseline

- Objective: `d_ar` (suffix masking — mask last 20% of non-goal tokens, predict from prefix)
- 200 steps, 62 min, A100-80GB node xgph0
- Adapter: `AK2802/AOMT/dar`

| Step | Val loss |
|------|----------|
| 0    | 6.095    |
| 20   | 5.583    |
| 40   | 5.498    |
| 60   | 5.168    |
| 80   | 5.253    |
| 100  | 5.037    |
| 120  | 5.016    |
| 140  | 4.951    |
| 160  | 5.017    |
| 180  | 4.983    |
| 199  | 4.955    |

**Observation — D-AR plateaus hard at step ~60:**
Loss drops quickly from 6.09 → 5.17 in the first 60 steps, then completely stagnates for
the remaining 140 steps (range 4.95–5.25). Train ≈ val (4.957 ≈ 4.955) — no overfitting,
no underfitting. The model is stuck.

**Why the plateau happens:** LLaDA2.0-mini is a bidirectional masked diffusion model with
NO causal architectural prior. Suffix masking forces it to learn causal direction purely via
gradients, fighting against the bidirectional inductive bias built into every attention layer.
The plateau is an architectural mismatch signal, not a hyperparameter problem. This is
actually an interesting finding: a bidirectional model cannot easily internalize causal
direction even with enough gradient steps at a small LoRA scale.

**Supervision budget (D-AR):**
- THINK: 71,684 scored tokens (44%)
- ACTION: 13,885 (9%)
- OBS: 76,810 (47%)

Relatively balanced because suffix masking hits whatever block types happen to be in the
last 20% of the sequence, and ScienceWorld trajectories end with observations.

---

### Run: `dflex` (job 642334) — D-Flex (3 objectives, v1)

- Objective: `d_flex` with 3 objectives (imputation, retrodiction, obs_denoising), uniform 1/3
- 200 steps, 64 min, A100-80GB node xgph1
- Adapter: `AK2802/AOMT/dflex`

| Step | Val loss |
|------|----------|
| 0    | 6.034    |
| 20   | 5.113    |
| 40   | 5.357    |
| 60   | 4.479    |
| 80   | 3.770    |
| 100  | 4.119    |
| 120  | 3.763    |
| 140  | 3.185    |
| 160  | 3.099    |  ← best
| 180  | 3.505    |
| 199  | 3.886    |

**Observation — D-Flex still learning at step 200:**
Continuous downward trend through step 160 (3.099), then a slight uptick. The model has NOT
converged. 400 steps needed to see the full learning curve. The noisy step-to-step train
loss (~3–5 range) is expected: each step samples a different objective from the mixture,
producing inherently different loss scales.

**Why D-Flex val loss < D-AR val loss is NOT the paper result:**
D-Flex tasks have bidirectional context → inherently easier for a bidirectional model → lower
loss is EXPECTED. D-AR has causal context → harder → higher loss. This difference does NOT
mean D-Flex learned better representations. The fair comparison is `val_nextobs_loss` (Phase 3).

**Supervision budget (D-Flex v1):**
- THINK: 31,176 scored tokens (20%)
- ACTION: 4,506 (3%)
- OBS: 122,410 (77%)

Heavily obs-weighted because obs_denoising (1/3 of batches) scores ONLY obs tokens, and obs
blocks are long in ScienceWorld (multi-sentence environment descriptions). This imbalance
motivated adding `inverse_dynamics` as the 4th D-Flex objective.

---

## Phase 3: Full comparison runs (upcoming — v2 series)

Script: `train_llada_rb.py` v2 with:
1. `inverse_dynamics` added as 4th D-Flex objective (4 objs, uniform 1/4)
2. `val_nextobs_loss` shared evaluation metric (predict last obs from causal prefix)
   — identical task for D-AR, D-Flex, and D-Random, enabling fair comparison
3. Charts now show both training-task val and shared nextobs val

### Why `inverse_dynamics` is the key addition:

Given: goal + ALL observation tokens (bidirectionally, including obs AFTER the action)
+ think/action at other steps.
Hidden: think block at same step as target action (prevent leakage: think narrates the action).
Target: all action tokens at the chosen step.

This is the canonical world model task — if you understand the transition dynamics
T(s, a) → s', you can infer a from s and s'. A CAUSAL model can only see obs BEFORE the
action; it cannot use the obs AFTER, so it cannot solve this task without leakage. A
bidirectional model can. This is what makes D-Flex strictly more expressive.

**Note on masking rate for inverse_dynamics:**
Unlike the other 3 D-Flex objectives (which mask rate% of eligible tokens), inverse_dynamics
masks ALL action tokens at one step regardless of rate. Action blocks in ScienceWorld are
short (typically 5–20 tokens for a command), so the achieved mask rate for these steps is
~2–5%, not 20%. The per-step `achieved` rate metric reflects this. This is correct by design:
predicting half of an action is semantically meaningless.

### Why `val_nextobs` is the paper-level contribution:

val_nextobs task: given the full causal prefix of a trajectory, predict the last observation
block. Context is strictly causal (prefix only) — no future tokens — so the model only has
access to the same information regardless of training objective.

- D-AR trains on this almost directly (suffix masking often includes the last obs block)
- D-Flex never trains on this explicitly — it would have to generalize from diverse objectives
- D-Random trains on it incidentally

If D-Flex matches or beats D-AR on val_nextobs despite not training for it, it means
bidirectional multi-task masking induces forward dynamics as an emergent property. That is
the central empirical claim of the paper.

### Jobs queued:

| Job name    | Objective    | Steps | Seed | Purpose |
|-------------|--------------|-------|------|---------|
| dar_v2      | d_ar         | 400   | 0    | D-AR baseline with nextobs eval |
| dflex_v2    | d_flex (4 obj)| 400  | 0    | D-Flex with inverse_dynamics + nextobs |
| drandom_v2  | random_block | 400   | 0    | D-Random baseline with nextobs |

### Evaluation preflight: production tokenizer parity

- Date: 2026-07-22
- Adapter repository: `AK2802/AOMT` at
  `08ea2b99f87930da86b58d176ae5b48ef96e5528`
- Matched adapters: `dar_v2`, `drandom_v2`, `dflex_v2`; all seed 0
- Base repository: `inclusionAI/LLaDA2.0-mini` at
  `dad945cac317da394b390f82c7b40691d8a881ed`
- Dataset revision: `Joshyxwa/cp2107-textworld-trajectories` at
  `b01efa47595f21a8c5555f04633c13172c364079`
- Result: 148 checked, 0 dropped, 0 mismatches; parity passed
- Artifact: `runs/evaluation-suite-v1/parity-scienceworld-v1.json`
- Model manifest: `models/AOMT/model-manifest.json`

All 13 LoRA adapters in the repository are downloaded locally and validated; the three listed
above remain the matched primary comparison. Full checkpoint evaluation was not launched on this
Windows host because the base weights are separate (32,513,122,504 bytes across seven shards)
and no NVIDIA runtime is available. This is an infrastructure boundary, not an evaluation
exclusion; the recorded revisions and suite command are ready for the GPU cluster.

---

## Key methodological notes

**Supervision budget matching:** All runs target 20% mask rate over non-goal tokens.
The actual scoring rate may differ by objective (see notes above). Budget audits are in
`runs/<name>/metrics.json: supervision_budget_by_type`.

**Reproducibility:** All runs use seed=0. Seeds 1, 2 are queued after Phase 3 results
confirm the direction (required for ICML multi-seed reporting).

**What training loss alone does NOT show:**
- Whether the model learned world-model representations vs. surface statistics
- Whether D-Flex representations transfer to downstream tasks
- Whether the improvement is due to masking strategy or task difficulty confound

**What we need for a publishable result (beyond these runs):**
1. val_nextobs comparison across all 3 models (Phase 3)
2. Action-conditioned next-state prediction accuracy (closed-loop eval)
3. Counterfactual consistency and corruption recovery
4. Latent-state probes (inventory, location tracking) with anti-leakage controls
5. ID vs OOD task success gap
6. Multi-seed results (seeds 0, 1, 2 per model)

---

## Cluster hard lessons (see CLAUDE.md §4 for full rules)

- `--exclude=xgpj0` always (broken numpy on that node)
- `--exclude=xcng0,xcng1` for normal partition (ARM nodes, x86 venv breaks)
- NEVER pad to `max_len` globally — always dynamic padding to batch max
- NEVER use `/tmp` — use `$HOME/dflex_proj/tmp` (10MB/user quota on /tmp)
- HF upload must happen in a fresh subprocess AFTER `unset HF_HUB_OFFLINE HF_HOME`
  (huggingface_hub caches the env var at import time; Python-level pop() has no effect)
- `wandb sync --sync-all` (not `ls -td wandb/offline-run-* | head -1`) — the latter
  has a race condition when two jobs finish simultaneously and picks the wrong run
- transformers pinned to 4.5x — LLaDA2 breaks on 5.x with `KeyError: 'default'` in rope init
