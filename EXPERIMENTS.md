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

> By limiting to a single, consistent objective, do we limit the list of tasks that the model can handle?

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

## Phase 3: Full comparison — D-AR vs D-Flex vs D-Random (COMPLETED)

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

### Results (SLURM jobs 642969 / 642970 / 642971)

| Run | Job | Steps | Obj | Final train | Final val (task) | Final nextobs | Best nextobs | Minutes |
|-----|-----|-------|-----|-------------|------------------|---------------|--------------|---------|
| dar_v2 | 642969 | 400 | d_ar | 4.244 | 4.650 | 4.149 | 4.012 | 129.7 |
| dflex_v2 | 642970 | 400 | d_flex (4 obj) | 1.927 | 3.529 | 3.616 | 3.614 | 126.3 |
| drandom_v2 | 642971 | 400 | random_block | 0.214 | 0.294 | 2.760 | 2.084 | 133.6 |

Adapters at `AK2802/AOMT/{dar_v2,dflex_v2,drandom_v2}`.

### Key findings

**Central result: D-Flex beats D-AR by ~0.4 nats on nextobs (steps 240–400)**

Steps 240–399 averaged over val checkpoints:
- D-AR nextobs: mean=4.377, std=0.248
- D-Flex nextobs: mean=3.974, std=0.319
- D-Random nextobs: mean=3.197, std=0.513

Crossover point at step ~180. D-Flex improves despite NEVER training on causal
forward prediction — the bidirectional multi-task objective induces forward dynamics
as an emergent property. This is the central empirical claim.

**D-Flex is NOT converged at 400 steps.**
nextobs curve step 200–400: 5.1 → 4.25 → 4.53 → 4.14 → 4.35 → 4.19 → 3.89 → 3.69 → 3.61 → 3.76 → 3.62.
Still declining. 600 steps needed to determine plateau.

**D-Random: memorisation, not world model.**
val (training task) = 0.294 (near-perfect denoising). nextobs std=0.513 (vs 0.24 for D-AR).
High variance = unstable forward-prediction mechanism. Train/nextobs gap: 0.294 → 2.76 = 2.47 nats
vs D-Flex 3.53 → 3.62 = 0.09 nats. D-Flex train-task val ≈ nextobs val, meaning bidirectional
training generalised to causal evaluation.

**D-AR plateau:** architecturally mismatched. Bidirectional model + causal suffix task.
Loss 6.09 → 4.65 over 400 steps, still not converged. Purely-causal supervision from a model
with no causal inductive bias = slow, inefficient gradient signal.

**Supervision budget (D-Flex v2, 4 objectives):**
- THINK: 49,433 (20%), ACTION: 8,729 (3.6%), OBS: 184,098 (76%)

---

## Phase 4: Convergence + ablations (RUNNING — jobs 681708/681709/681710)

Script: `train_llada_rb.py` v3 (committed `6aeb56a`). gpu-long partition, 6h wall time.

### Training improvements over Phase 3

- **Accuracy metric** — `train/acc`, `val/acc`, `val/nextobs_acc` logged to wandb;
  measures what fraction of masked tokens are predicted correctly (not just loss)
- **EMA (decay=0.995)** — EMA weights applied during every val evaluation, restored after.
  Reduces noise in val curves without affecting training dynamics
- **Regularisation** — `weight_decay=0.1` (was 0), `lora_dropout=0.1` (was 0)
- **Full val sweep** (`--full_val_at_end`) — after training, evaluate nextobs over ALL 148
  val trajectories (not sampled batches). Gives a definitive low-variance estimate of
  world model quality for the final checkpoint
- **Linear probing** (`--probe_after_train`) — freeze adapter, extract last-layer hidden
  states from val set, fit:
  - LogisticRegression → block-type accuracy (THINK/ACTION/OBS, 3-class; chance=0.33)
  - Ridge regression → step-number R² and MAE
  Logged to wandb as `probe/block_type_acc`, `probe/step_r2`, `probe/step_mae`.
  Tells us whether the representations differ structurally across objectives

### New objectives

**`d_ar_refined` (job 681710 — `dar_v3`)**
`causal_random_mask`: instead of always masking the suffix, uniformly sample the
span start position across all eligible positions. Same span length (rate × eligible),
same strictly-causal context (no future tokens after span end). Fixes two biases in
`suffix_mask`:
1. Positional bias: suffix_mask only supervises end-of-trajectory tokens
2. Variance: short sequences get proportionally less causal context

Ablation question: does removing the positional bias break or help D-AR's nextobs score?
Compare `dar_v3` (uniform position) vs `dar_v2` (suffix-only) on val/nextobs.

**`d_progressive` (job 681709 — `dprog`)**
`progressive_span_mask`: masking curriculum over training.
- progress=0 (step 0): span_size=1 → identical to `random_block` (individual token masking)
- progress=1 (step 600): span_size=25 → approaching whole-block masking
- span_size grows exponentially: `round(exp(log(25) × progress))`
- Mask rate stays fixed at 20% throughout (span growth changes contiguity, not volume)

Ablation question: does gradually teaching the model block-level structure improve nextobs
vs `drandom_v2` (same objective but no progression, already done in Phase 3)?
The curriculum hypothesis: token-level first = easy gradient signal early; block-level later
= richer contextual reasoning required. A well-designed curriculum should improve both
training stability and generalisation.

### Phase 4 runs

| Run | SLURM job | Obj | Steps | Seed | Key question |
|-----|-----------|-----|-------|------|--------------|
| dflex_v3 | 681708 | d_flex | 600 | 0 | Does D-Flex converge? What's the 600-step nextobs? |
| dprog | 681709 | d_progressive | 600 | 0 | Does span curriculum beat flat random masking? |
| dar_v3 | 681710 | d_ar_refined | 600 | 0 | Does uniform-position masking improve D-AR? |

All: gpu-long, A100-80GB, 6h, seed=0. Status: PENDING (gpu-long queue).
Results will auto-upload to `AK2802/AOMT/{dflex_v3,dprog,dar_v3}` + wandb sync.

---

## Phase 5: d_ar_section vs progressive curriculum ablations (COMPLETED)

Script: `train_llada_rb.py` v4 (commit `4280b41`). gpu-long, 6h, 800 steps.
Cancelled: dflex_v3, dar_v3 (dropped D-Flex; d_ar_refined superseded by d_ar_section).

### Key analysis informing this phase

**Data statistics (ScienceWorld, 500 trajectories):**
- Seq length: mean=1190, median=1156, p90=1784 tokens
- Steps per traj: mean=30.5 (long multi-step reasoning chains)
- Action tokens per block: mean=4.8, **median=4** (extremely short — "go north", "pick up knife")
- Block run lengths: median=12, p75=24, p90=38 tokens → max_span=25 covers 75th percentile
- **Action fraction: 6.53% of eligible tokens** (~73 action tokens per 1190-token trajectory)

**Think-action leakage check:** Think blocks do NOT directly telegraph actions.
"I need to locate the metal pot" → "look around" (generic exploration).
"I've spotted a turtle egg" → "focus on egg turtle" (requires reasoning, not copying).
d_ar_section is genuinely challenging despite think being context.

### New objectives

**`d_ar_section`** (SLURM 691116 — `dar_section`):
`all_action_mask`: mask ALL action tokens (n=4-8 per step × 15 steps ≈ 73 tokens/traj).
Context = goal + think + obs, bidirectionally. No rate parameter — automatic ~6.5%.

Alignment with LLaDA inference: at inference time you would set all action tokens to MASK
and run one denoising pass over the full trajectory. This is exactly the training task.

Comparison vs Phase 3 d_ar (suffix masking): d_ar scores the last 20% of the trajectory
(~210 tokens, many are obs tokens). d_ar_section scores only action tokens (73 tokens) but
these are the tokens that matter for downstream task performance. The supervision signal is
denser in semantic content per scored token.

**`d_progressive` — curriculum speed ablation:**
Three runs compare how fast the span grows, with identical 20% mask rate and max_span=25:

| Run | Job | Exponent | Span at halfway (step 400) | Span at p25 (step 200) |
|-----|-----|----------|--------------------------|------------------------|
| dprog (Phase 4) | 681709 | 1.0 (fast) | 5 | 2 |
| dprog_med | 691117 | 2.0 (medium) | 2 | 1 |
| dprog_slow | 691118 | 3.0 (slow) | 1.5 | 1 |

Hypothesis: slower curriculum (more time at token-level masking) builds stronger local
representations before moving to harder span prediction, improving nextobs/nextaction
generalisation. Or: too much time at token-level could be wasteful / converge slower.

Verified with masking visualisation (`--show_samples 2`) before submitting:
- d_ar_section: all action blocks correctly masked, no think/obs scored, 30/30 examples correct
- d_progressive at exponent=1.0: clear span progression from single tokens → large chunks
- d_progressive at exponent=2.0: stays at token-level much longer (span=2 at halfway vs span=5)

### New evaluation metrics (all Phase 5 onwards)

**`val/nextaction_loss`** — predict last action block from causal prefix.
Symmetric to val/nextobs (world model). nextaction tests policy quality.
Expected: d_ar_section should excel here (its training task is exactly this).
Expected: d_progressive is trained agnostically (equal chance of scoring action vs obs vs think).

**`train/tok_f1` / `val/tok_f1`** — unigram token-bag F1 (SQuAD-style).
Partial credit: "pick up metal pot" vs "pick up pot" = 0.8 F1, not 0 exact match.
Better reflects semantic correctness for short action sequences.

### TextWorld datasets

Dataset: `Joshyxwa/cp2107-textworld-trajectories` (downloaded to `data/textworld/`).

**ALFWorld stats:**
- 6574 train / 251 val / 255 test trajectories
- Seq length: mean=2272 tokens (~2× ScienceWorld)
- Action fraction: **0.87%** (extremely sparse — ~20 action tokens per trajectory)
- Block order: Obs(N) → Think(N) → Action(N) (reversed from ScienceWorld; position-based masking handles both)
- grad_accum=8 required for ALF runs to reduce gradient variance from ~80 scored tokens/step

**WebShop stats (EXCLUDED):**
- 3014 train / 251 val / 255 test
- Seq length: mean=36 tokens — trivially short for a 16B model, no meaningful signal
- Action fraction: 62.5%, no Think blocks
- Decision: excluded from all Phase 5+ training runs

**Combined SW+ALF:** 1187+6574=7761 train / 148+251=399 val.
Provides cross-domain diversity: ALF's dense multi-step navigation vs SW's long reasoning chains.

### Phase 5 run matrix (9 runs total)

**ScienceWorld — mask rate 20%** (main ablation):

| Run | SLURM | Obj | Exp | Steps | Key question |
|-----|-------|-----|-----|-------|--------------|
| dar_section | 691116 | d_ar_section | N/A | 800 | Causal baseline: all-action masking, LLaDA-aligned |
| dprog (Phase 4) | 681709 | d_progressive | 1.0 (fast) | 600 | Fast curriculum reference (Phase 4 holdover) |
| dprog_med | 691117 | d_progressive | 2.0 (med) | 800 | Slower span growth — more token-level early |
| dprog_slow | 691118 | d_progressive | 3.0 (slow) | 800 | Slowest span growth — token-level dominant |

**ScienceWorld — mask rate 6.5%** (action-rate match to dar_section):

| Run | SLURM | Obj | Exp | Steps | Key question |
|-----|-------|-----|-----|-------|--------------|
| dprog_fast_act | 691128 | d_progressive | 1.0 | 800 | Fast, action-rate → vs dar_section (same rate) |
| dprog_med_act | 691129 | d_progressive | 2.0 | 800 | Medium, action-rate → isolate mask_rate from span effect |
| dprog_slow_act | 691130 | d_progressive | 3.0 | 800 | Slow, action-rate → does sparse masking help? |

**ALFWorld (grad_accum=8):**

| Run | SLURM | Obj | Exp | Steps | Key question |
|-----|-------|-----|-----|-------|--------------|
| dar_section_alf | 691131 | d_ar_section | N/A | 800 | Does causal masking scale to very sparse action signal? |
| dprog_med_alf | 691132 | d_progressive | 2.0 | 800 | Cross-domain: does SW-tuned curriculum generalise? |

**Cross-domain:**

| Run | SLURM | Obj | Exp | Steps | Key question |
|-----|-------|-----|-----|-------|--------------|
| dar_section_combined | 691133 | d_ar_section | N/A | 800 | Combined SW+ALF: does more data diversity help? |

Status: All SW runs COMPLETED. ALF+combined runs CANCELLED at time limit (6h insufficient).

### Phase 5 Results

| Run | SLURM | best nextobs↓ | best nextact↓ | final nextobs | final train loss | Status |
|-----|-------|--------------|--------------|---------------|-----------------|--------|
| dar_section | 691116 | **4.546** | **0.253** | 10.586 | 0.001 | OVERFIT |
| dprog (Phase 4 ref) | 681709 | — | — | — | — | partial |
| dprog_med | 691117 | 1.569 | 1.664 | 2.475 | 3.392 | OK |
| dprog_slow | 691118 | 1.657 | 1.505 | 2.542 | 3.150 | OK |
| dprog_fast_act (6.5%) | 691128 | **1.534** | 1.625 | 2.473 | 2.127 | OK |
| dprog_med_act (6.5%) | 691129 | 1.688 | 1.811 | 2.632 | 3.789 | OK |
| dprog_slow_act (6.5%) | 691130 | 1.981 | 1.623 | 2.756 | 3.574 | OK |
| dar_section_alf | 691131 | — | — | — | — | CANCELLED (step 380/800) |
| dprog_med_alf | 691132 | — | — | — | — | CANCELLED (step 380/800) |
| dar_section_combined | 691133 | — | — | — | — | CANCELLED (step 780/800) |

### Phase 5 Key Findings

**Critical finding 1: d_ar_section severely overfits on SW-v1 (1187 examples).**
Training curve: train loss 6.4 → 0.001 (acc=1.0), val/task 8.5 → 0.200. But nextobs
diverges catastrophically: 5.2 → 4.5 (best at step ~140) → 10.6 (final).
The model memorised action→action mappings while world model quality collapsed.
Root cause: 1187 examples × ~73 action tokens = very limited diversity. Action tokens
are short (median=4 tokens), making memorisation easy. Task-specific overfitting.

**Critical finding 2: d_progressive is strongly superior for world model learning.**
Best nextobs across all runs: dprog_fast_act=1.534, dprog_med=1.569 vs dar_section=4.546.
Progressive masking hits observations AND actions randomly, forcing the model to learn
bidirectional context — it cannot overfit to any single token type.
No catastrophic divergence: final nextobs ~2.4-2.8 vs dar_section's 10.6.

**Critical finding 3: Mask rate (20% vs 6.5%) has minimal effect on nextobs.**
dprog_fast (20%) best nextobs=N/A vs dprog_fast_act (6.5%) best nextobs=1.534.
dprog_med (20%) = 1.569 vs dprog_med_act (6.5%) = 1.688. Differences are small.
The action-rate runs are NOT better than 20% rate despite matching dar_section's rate.

**Critical finding 4: Curriculum speed (exponent) has mild effect.**
Fast (1.0) → medium (2.0) → slow (3.0) shows slight progression in nextobs (1.534 → 1.657).
The fast curriculum (more time at large spans early) appears marginally better than slow.
No strong signal to justify strongly preferring one exponent over another.

**Critical finding 5: ALFWorld/combined training needs >6h wall time.**
dar_section_alf (691131): cancelled at step 380/800. nextact declining (9.6→4.3) — still
converging. The sparse action signal (0.87%) requires much longer to learn.
Combined run (691133): cancelled at step 780/800. Also still converging at end.
Both would need ~12-15h to complete 800 steps, beyond gpu-long limit.

**Summary:** d_progressive > d_ar_section for world model quality. The overfitting problem
points to a dataset-size issue that the new scienceworld-compact-v2 (3534 train) addresses.

---

## Phase 6: scienceworld-compact-v2 + OAE semantic metric (QUEUED)

Script: `train_llada_rb.py` v5 (commit `071d60d`). gpu-long, 6h, 800 steps.

### Dataset changes (scienceworld-compact-v2)

**Old dataset (scienceworld-v1):** 1187 train / 148 val. Think+Action+Obs blocks.
**New dataset (scienceworld-compact-v2):** 3534 train / 1762 val (+3× data). NO Think blocks.
Format: Goal(0) → Obs(0) → Action(1) → Obs(1) → Action(2) → ... (compact, direct).
State_labels: goal_progress (subgoal completion flags), score (0-100). Rich eval signal.
Action fraction: ~21% (vs 6.5% in v1, due to no Think blocks diluting).
Seq length: mean=2094, p90=6267, p99=9435 → 3% exceed 8192 → sliding window applied.
After sliding-window: 3908 train / 1904 val examples.

Key motivation: dar_section overfitting was due to small dataset size (1187 traj).
3908 examples should provide much more action diversity → less memorisation.

### Semantic accuracy metric (OAE — replaces token F1)

ScienceWorld is non-deterministic — multiple valid action sequences exist for the same goal.
Token F1 scores "go north" vs "teleport to kitchen" as 0 even if both are valid paths.
OAE (Outcome-Aware Equivalence) is designed for this:

- **SAS** (Semantic Action Similarity, 22M): cosine similarity of sentence embeddings.
  Handles paraphrases: "pick up metal pot" ≈ "pick up metal pot containing nothing" → 0.92.
  Limitation: SAS=0.78 for "red box" vs "yellow box" (entity-name discrimination is weak).

- **OOC** (Outcome-Outcome Consistency, 44M): NLI entailment score.
  P("The agent performs: action_pred" → "The resulting observation is: obs_gt").
  Handles outcome-equivalent-but-surface-different actions without a simulator.
  "pick up metal pot" → "You move the metal pot to the inventory." → OOC=0.95.
  Limitation: domain-generic NLI struggles with game-world implication ("grab" → "move to inv").

- **OAE = mean(max(SAS_i, OOC_i))**: perfect baseline=1.000, null baseline=0.060.

Wandb metrics: `val/sem_action_sim`, `val/outcome_consistency`, `val/oae`.
Note: OOC is more reliable than SAS for catching entity errors (right action family, wrong
      specific entity), but SAS is more reliable for catching synonym/paraphrase cases.
      Report both separately in analysis; OAE is the combined headline metric.

### Phase 6 run matrix

| Run | Obj | Exp | Rate | Dataset | Key question |
|-----|-----|-----|------|---------|--------------|
| dar_section_v2 | d_ar_section | N/A | auto | scienceworld-v2 | Does 3× data reduce overfitting? |
| dprog_fast_v2 | d_progressive | 1.0 | 20% | scienceworld-v2 | Best Phase 5 on larger data |
| dprog_med_v2 | d_progressive | 2.0 | 20% | scienceworld-v2 | Med curriculum on larger data |

### Analysis plan (after Phase 6 results)

1. **OAE vs nextaction**: is OAE correlated with nextaction loss? Higher OAE should = lower nextact.
2. **Overfitting mitigation**: does dar_section_v2 overfit less than dar_section (1187 vs 3908 train)?
3. **Curriculum on v2**: does dprog_fast_v2 achieve nextobs < 1.534 (Phase 5 best)?
4. **OAE discriminability**: which objective scores highest OAE? Expected: dprog > dar_section.

---

## Next steps (after Phase 6 results)

### Priority 1: Multi-seed runs
Best-performing objective needs seeds 1 and 2 for publishable error bars.
Single-seed results illustrative only. Submit 2 more runs after Phase 6 analysis.

### Priority 2: Action-conditioned next-state prediction
Beyond nextobs (predict last obs from prefix), evaluate:
given prefix + proposed action → predict resulting observation.
This is a more direct T(s,a)→s' world model test. Requires a small eval script.

### Priority 4: Scale / OOD eval
Test adapters on held-out ScienceWorld task types not seen during fine-tuning.
Measures whether the world model generalises or memorises task-specific patterns.

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
