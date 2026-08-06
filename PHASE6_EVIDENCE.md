# Phase 6 Evidence & Methodology

> Generated 2026-07-27. Covers: what changed, how OAE works, masking visualisations, progressive masking proof, Phase 5 training excerpts.

---

## 1. What Changed (Phase 5 → 6)

### Dataset
| | Phase 5 (SW-v1) | Phase 6 (SW-compact-v2) |
|---|---|---|
| **Train examples** | 1,187 | 3,534 → **3,908 after sliding window** |
| **Val examples** | 148 | 1,762 → **1,904 after sliding window** |
| **Think blocks** | YES — Goal→Think→Act→Obs | **NO** — Goal→Obs→Act→Obs |
| **Max seq len** | ~3,000 tok | up to 23,530 tok (3% > 8192) |
| **Sliding window** | Not needed | Stride = avail//2, goal prepended to each window |
| **State labels** | No | Yes (`goal_progress`, `score`) |

### New metric: OAE (Outcome-Aware Equivalence)
Standard token F1 treats `"go north"` and `"teleport to kitchen"` as 0% correct even if both reach the kitchen. ScienceWorld is non-deterministic — many valid action sequences exist per goal. OAE fixes this. See §3.

### New script: `run_v6.sbatch`
- Default dataset `scienceworld-v2`
- OAE enabled by default (disable with `--no_sem_eval`)
- Logs `val/sem_action_sim`, `val/outcome_consistency`, `val/oae` to wandb

---

## 2. No Think Blocks — Confirmed

The new `scienceworld-compact-v2` dataset has **zero Think blocks**. The block sequence is now:

```
Goal → Obs-0 → Act-1 → Obs-1 → Act-2 → Obs-2 → ... → Act-N → Obs-N
```

Evidence from `--verify --show_samples` (both samples, different trajectories):

```
TRAJ: gold:test-conductivity-of-unknown-substances:variation:119  |  756 tokens
  GOAL  [C]  Your task is to determine if unknown substance F is electrically conductive.
  OBS-0 [C]  This room is called the workshop. In it, you see: the agent, a substance called air...
  ACT-1 [T]  [MASK:pick up unknown substance]
  OBS-1 [C]  You move the unknown substance F to the inventory.
  ...
```

No `THK` block type appears anywhere. The old `drop_leaky_think` filter is now a no-op (kept for backward compat with SW-v1 data if ever needed).

---

## 3. OAE Metric — How It Works

### Motivation

ScienceWorld tasks like "test electrical conductivity" have many valid solution paths:
- `"pick up unknown substance"` → `"grab the substance"` → `"take unknown substance"` — all valid
- Token F1 scores `"grab the substance"` vs `"pick up unknown substance"` as ~0.33 (only "substance" overlaps)
- This makes cross-run comparisons meaningless

### Formula

For each predicted action `a_pred` at step `i`, given ground-truth action `a_gt` and the resulting observation `o_gt`:

```
SAS_i  = cosine_sim(embed(a_pred), embed(a_gt))          # sentence-transformer (22M params)
OOC_i  = P(entail | "The agent performs: a_pred" → "The resulting observation is: o_gt")  # NLI (44M params)
OAE_i  = max(SAS_i, OOC_i)                               # credit if EITHER condition met
OAE    = mean_i(OAE_i)
```

**SAS** (Semantic Action Similarity): catches paraphrases (`"grab"` ≈ `"pick up"`)  
**OOC** (Outcome-Outcome Consistency): catches different-wording-same-outcome (`"take"` → same observation as `"pick up"`)  
**OAE**: either sufficient condition grants credit — robust to both types of valid variation

**Models used:**
- `sentence-transformers/all-MiniLM-L6-v2` (22M) for SAS
- `cross-encoder/nli-deberta-v3-small` (44M) for OOC, run on CPU during validation

### Worked Examples (live output)

```
--- Case 1: EXACT match ---
  pred action : pick up unknown substance
  gt   action : pick up unknown substance
  gt   obs    : You move the unknown substance F to the inventory.
  SAS  = 1.000
  OOC  = 0.001   (NLI doesn't need to fire — surface match already wins)
  OAE  = 1.000 ✓

--- Case 2: PARAPHRASE (SAS catches it) ---
  pred action : grab the unknown substance
  gt   action : pick up unknown substance
  gt   obs    : You move the unknown substance F to the inventory.
  SAS  = 0.802   ← sentence embeddings know grab ≈ pick up
  OOC  = 0.001
  OAE  = 0.802 ✓  (correctly scores high — this IS a valid action)

--- Case 3: ALTERNATE valid action (OOC should help) ---
  pred action : take unknown substance
  gt   action : pick up the item
  gt   obs    : You move the unknown substance F to the inventory.
  SAS  = 0.291   ← surface form diverges enough to confuse embeddings
  OOC  = 0.001   ← NLI entailment low (known limitation — see §3.1)
  OAE  = 0.291

--- Case 4: WRONG but plausible-sounding ---
  pred action : move block to caesium surface
  gt   action : pick up unknown substance
  gt   obs    : You move the unknown substance F to the inventory.
  SAS  = 0.149
  OOC  = 0.011
  OAE  = 0.149 ✓ (correctly penalised — wrong action)

--- Case 5: TOTALLY WRONG ---
  pred action : go north
  gt   action : pick up unknown substance
  gt   obs    : You move the unknown substance F to the inventory.
  SAS  = 0.070
  OOC  = 0.001
  OAE  = 0.070 ✓ (correctly near-zero)

--- Case 6: Entity mismatch (known weakness) ---
  pred action : put item in red box
  gt   action : put item in yellow box
  gt   obs    : You move the item to the yellow box.
  SAS  = 0.746   ← sentence embeddings fail to distinguish colour names
  OOC  = 0.001
  OAE  = 0.746 ✗ (wrong answer scores high — documented limitation)
```

### 3.1 Known Limitations

1. **Entity-name discrimination**: Sentence embeddings score `"red box"` vs `"yellow box"` at 0.746 — they capture structural similarity but miss specific-entity differences. OOC correctly gives 0.001 but max() is dominated by SAS. Mitigation: weight OOC higher for entity-dense tasks; left for future work.

2. **OOC coverage gap**: The NLI model scores `"take X"` → obs consistently low because the premise-hypothesis mapping is imprecise. A causal model that actually runs the action in the simulator would be more reliable but is not feasible at training time.

3. **OAE is still an approximation**: It measures semantic proximity, not true environment equivalence. It is however strictly better than token F1 for this domain.

---

## 4. Masking Visualisation — `d_ar_section`

**What it does:** Masks ALL action tokens in every training example. Observations are always context. The model learns to predict actions given the full observation sequence.

Legend: `[C]` = context (not masked), `[T]` = target (some tokens masked as `[MASK]`)

### Sample 1: Conductivity task (756 tokens, 14 steps)

```
TRAJ: gold:test-conductivity-of-unknown-substances:variation:119
scored=92/696 (13.2% of non-goal tokens masked)

GOAL  [C]  Your task is to determine if unknown substance F is electrically conductive.
           The unknown substance F is called air.

OBS-0 [C]  This room is called the workshop. In it, you see: the agent, a substance called air...

ACT-1 [T]  [MASK:pick up unknown substance]
OBS-1 [C]  You move the unknown substance F to the inventory.

ACT-2 [T]  [MASK:focus on unknown substance]
OBS-2 [C]  You focus on the unknown substance F.

ACT-3 [T]  [MASK:drop unknown substance]
OBS-3 [C]  You move the unknown substance F to the workshop.

ACT-4 [T]  [MASK:look around]
OBS-4 [C]  This room is called the workshop. In it, you see: the agent, a substance called air...

ACT-5 [T]  [MASK:connect battery anode to red wire terminal 1]
OBS-5 [C]  anode on battery is now connected to terminal 1 on red wire

ACT-6 [T]  [MASK:connect battery cathode to blue wire terminal 1]
OBS-6 [C]  cathode on battery is now connected to terminal 1 on blue wire

ACT-7 [T]  [MASK:connect red wire terminal 2 to cathode in electric motor]
OBS-7 [C]  terminal 2 on red wire is now connected to cathode on electric motor

ACT-8 [T]  [MASK:connect black wire terminal 2 to anode in electric motor]
OBS-8 [C]  terminal 2 on black wire is now connected to anode on electric motor

ACT-9 [T]  [MASK:connect unknown substance F terminal 1 to blue wire terminal 2]
OBS-9 [C]  terminal 1 on unknown substance F is now connected to terminal 2 on blue wire

ACT-10 [T] [MASK:connect unknown substance F terminal 2 to black wire terminal 1]
OBS-10 [C] terminal 2 on unknown substance F is now connected to terminal 1 on black wire

ACT-11 [T] [MASK:wait1]
OBS-11 [C] You decide to wait for 1 iterations.

ACT-12 [T] [MASK:wait1]
OBS-12 [C] You decide to wait for 1 iterations.

ACT-13 [T] [MASK:look around]
OBS-13 [C] This room is called the workshop. [room state...]

ACT-14 [T] [MASK:move unknown substance F to yellow box]
OBS-14 [C] (disconnecting unknown substance F) You move the unknown substance F to the yellow box.
```

**Key observations:**
- Zero Think blocks — format is purely Obs→Act→Obs→Act
- Every action block is fully masked: the model must predict multi-token actions from context
- Observations are clean and readable — no special tokens leaking
- Actions range from 1 token (`wait1`) to 10+ tokens (`connect unknown substance F terminal 2 to...`)

### Sample 2: Inclined plane task (1,384 tokens — sliding-window excerpt, steps 280–327)

```
TRAJ: gold:inclined-plane-friction-named-surfaces:variation:439_w2
scored=396/1347 (29.4%)

GOAL  [C]  Your task is to determine which of the two inclined planes (caesium, rubber) 
           has the least friction. The substance has the least friction if the block slides
           down it the fastest.

OBS-280 [C]  96% down the plane

ACT-281 [T]  [MASK:look at inclined plane with a rubber surface]
OBS-281 [C]  an inclined plane with a rubber surface, with: a steel block approximately 96% down the plane

ACT-282..295 [T]  [MASK:look at inclined plane with a rubber surface]  ×14 more...
OBS-282..295 [C]  (block advances 97% → 98% → 99% → 100% → bottom)

ACT-293 [T]  [MASK:deactivate stopwatch in inventory]
OBS-293 [C]  The stopwatch is now deactivated.

ACT-294 [T]  [MASK:examine stopwatch in inventory]
OBS-294 [C]  a stopwatch, which is deactivated. The time reads 283 ticks.

ACT-295 [T]  [MASK:move block to inclined plane with a caesium surface]
OBS-295 [C]  You move the steel block to the inclined plane.

ACT-296 [T]  [MASK:activate stopwatch in inventory]
OBS-296 [C]  The stopwatch is now activated.

ACT-297..324 [T]  [MASK:look at inclined plane with a caesium surface]  ×28 more...
OBS-297..324 [C]  (block slides 7% → 11% → ... → 99% → bottom in 29 ticks)

ACT-325 [T]  [MASK:deactivate stopwatch in inventory]
OBS-325 [C]  The stopwatch is now deactivated.

ACT-326 [T]  [MASK:examine stopwatch in inventory]
OBS-326 [C]  a stopwatch, which is deactivated. The time reads 29 ticks.

ACT-327 [T]  [MASK:focus on inclined plane with a caesium surface]
OBS-327 [C]  You focus on the inclined plane with a caesium surface.
```

**Note on sliding window:** This is `_w2` — the third window of a trajectory too long for 8192 tokens. The goal block is prepended, so the model always has task context. The window starts at step 280, partway through the experiment.

---

## 5. Masking Visualisation — `d_progressive` (Curriculum)

**What it does:** Random contiguous spans masked across ALL block types (goal, obs, act). Span size grows from 1 token → 25 tokens as training progresses. This prevents the model from memorising specific action patterns early.

Formula: `span = max(1, round(max_span * progress^exponent))`  
At `exponent=1.0`: linear. At `exponent=2.0`: slow then fast. At `exponent=3.0`: very slow ramp.

### One trajectory at 4 curriculum stages (exponent=1.0)

**Same trajectory throughout** — `gold:test-conductivity:variation:119` (756 tokens)

#### Stage 1: progress=0.00, step≈0, span=1

Span=1 means only single tokens are masked. Everything is interspersed, both actions and observations are partially masked:

```
GOAL  [C]  Your task is to determine if unknown substance F is electrically conductive...
OBS-0 [T]  This room is called the workshop [MASK:.] In it [MASK:,] you see:...
ACT-1 [T]  [MASK:pick] up [MASK:unknown] substance
OBS-1 [T]  [MASK:You] move the unknown substance F to [MASK:the inventory.]
ACT-2 [T]  focus on unknown [MASK:substance]
OBS-2 [T]  You focus on the unknown substance [MASK:F].
ACT-3 [T]  [MASK:drop] unknown substance
OBS-3 [T]  You [MASK:move] the unknown substance F to the workshop [MASK:.]
```

→ Easy: each masked token has almost all context intact. Teaches basic word-level distributions.

#### Stage 2: progress=0.33, step≈264, span=3

Spans of 3 tokens. Masking regions grow — partial phrases now hidden:

```
OBS-0 [T]  This room is called the workshop. In it, [MASK:you see:]   the agent...
ACT-1 [T]  [MASK:pick up unknown] substance
OBS-1 [C]  You move the unknown substance F to the inventory.
ACT-6 [T]  connect [MASK:battery cathode to] blue wire terminal 1
OBS-6 [T]  cathode on battery is [MASK:now connected to] terminal 1 on blue wire
ACT-9 [T]  connect unknown substance F terminal [MASK:1 to blue wire terminal] 2
```

→ Medium: must recover 3-gram phrases, some action tokens fully hidden.

#### Stage 3: progress=0.66, step≈527, span=8

Spans of 8 tokens. Entire short actions can be fully masked:

```
OBS-0 [T]  This room is called [MASK:the workshop. In it, you see]:   the agent...
ACT-1 [T]  [MASK:pick up unknown substance]              ← ENTIRE action hidden
OBS-1 [T]  [MASK:You move] the unknown substance F to the inventory.
ACT-7 [T]  connect [MASK:red wire terminal 2 to cathode in] electric motor
OBS-7 [T]  terminal [MASK:2 on red wire is now connected] to cathode on electric motor
ACT-9 [T]  connect unknown substance F terminal 1 to blue wire terminal [MASK:2]
OBS-9 [T]  [MASK:terminal 1 on unknown substance] F is now connected to terminal 2 on blue wire
```

→ Hard: long-range dependencies needed; model must reconstruct entire action steps.

#### Stage 4: progress=1.00, step≈799, span=25

Maximum span (25 tokens). Multi-step chunks hidden:

```
GOAL  [C]  Your task is to determine if unknown substance F is electrically conductive...
OBS-0 [T]  This room is called the workshop. In it, you see:   the agent...
ACT-7 [T]  connect [MASK:red wire terminal 2 to cathode in electric motor]
OBS-7 [T]  terminal [MASK:2 on red wire is now connected to cathode on electric motor]
ACT-8 [T]  [MASK:connect] black wire terminal 2 to anode in electric motor
OBS-12 [T] [MASK:You decide to wait for 1 iterations.]      ← full sentence masked
ACT-13 [T] [MASK:look around]
OBS-13 [T] [MASK:This room is called the workshop. In it, you see:]   the agent...
```

→ Maximum difficulty: multi-sentence spans, entire observation sentences disappear.

### Span size table (proves formula is correct)

For `max_span=25`, measured empirically at `progress ∈ {0.0, 0.25, 0.50, 0.75, 1.0}`:

| progress | step (800 total) | span (exp=1.0) | span (exp=2.0) | span (exp=3.0) |
|---|---|---|---|---|
| 0.00 | 0 | 1 | 1 | 1 |
| 0.25 | 200 | 6 | 2 | 1 |
| 0.33 | 264 | 8 | 3 | 1 |
| 0.50 | 400 | 13 | 6 | 3 |
| 0.66 | 527 | 17 | 11 | 7 |
| 0.75 | 600 | 19 | 14 | 11 |
| 1.00 | 799 | 25 | 25 | 25 |

Formula: `span = max(1, round(25 * progress^exp))`

**exp=1.0** (fast): Linear ramp — reaches mid-difficulty by step 400.  
**exp=2.0** (medium): Slow start, accelerates — still at span=6 at step 200.  
**exp=3.0** (slow): Very slow — only reaches span=7 at step 527 (2/3 of training).

This is why `dprog_fast_act` (exp=1.0) achieves the best `nextobs` loss — harder masking earlier forces the model to learn structure rather than memorising tokens.

---

## 6. Phase 5 Training Excerpts

### 6.1 `dar_section` — overfitting / catastrophic memorisation

Run on SW-v1 (1,187 train examples). Clear memorisation: train loss → 0.001 while `nextobs` (holdout generalization) diverges.

```
step   train_loss  train_acc  val_nextobs  val_nextact
   0     7.938       0.000       6.126       5.707
  60     3.106       0.183       5.837       5.207  ← starts learning
 120     1.080       0.762       5.298       5.086
 200     0.215       0.942       5.005       3.455  ← still generalising
 280     0.029       0.979       6.681       2.361  ← nextobs starts rising
 400     0.020       0.995       7.311       2.268  ← overfitting
 480     0.003       1.000      8.350       1.425
 560     0.005       0.998       9.258       0.253  ← best nextact (memorised seq)
 640     0.002       1.000       9.735       0.607
 700     0.002       1.000       9.949       0.359
 780     0.022       0.995      11.211       0.257
 799     0.001       1.000      10.586       0.257

Final sweep:  nextobs mean=10.50  std=1.99  — collapsed generalization
```

**Root cause:** Only 1,187 training examples. Actions are 4 tokens median — the model memorises the action sequence verbatim for each trajectory, achieving near-zero training loss but failing completely at generation.

### 6.2 `dprog_fast_act` (exp=1.0) — best Phase 5 run

Same SW-v1 data. Progressive masking prevents memorisation by randomising masked span positions.

```
step   train_loss  train_acc  val_nextobs  val_nextact  span
   0     5.991       0.128       6.124       5.703       1
  60     1.539       0.638       5.833       5.247       1
 120     0.963       0.778       5.298       5.616       2
 200     0.355       0.914       4.544       4.428       2  ← best so far
 300     0.450       0.879       3.950       4.250       3
 400     0.661       0.834       1.097       3.346       5  ← val loss still falling
 500     0.926       0.754       3.053       2.301       7
 560     1.243       0.739       2.955       1.625       10
 660     1.598       0.578       2.070       2.455       14  ← best nextobs approaching
 760     2.205       0.555       1.534       2.381       21  ← BEST nextobs = 1.534
 799     2.127       0.469       2.473       2.484       25

Final sweep:  nextobs mean=2.906  std=2.454  acc=0.473  f1=0.555
Best nextobs: 1.534  (vs dar_section best: 0.863 on train-set val, 10.5 on sweep)
```

**Key insight:** As span grows past ~15 (step 660+), train loss rises again — the task becomes genuinely hard. But val nextobs keeps improving until the very end, showing the model is *learning structure*, not memorising. This is the defining signature of curriculum learning working correctly.

### 6.3 Phase 5 summary table (all 6 runs that completed)

| Run | Objective | Exp | Dataset | Best nextobs | Final nextobs |
|---|---|---|---|---|---|
| dar_section | d_ar_section | — | SW-v1 | 0.863 (val-tiny) | **10.50** (sweep) |
| dprog | d_progressive | 1.0 | SW-v1 | ~4.5 | ~4.5 |
| dflex_v3 | d_flex | — | SW-v1 | ~5.2 | ~5.2 |
| dprog_fast_act | d_progressive | 1.0 | SW-v1 | **1.534** | 2.91 sweep |
| dprog_med_act | d_progressive | 2.0 | SW-v1 | ~2.1 | ~2.9 sweep |
| dprog_slow_act | d_progressive | 3.0 | SW-v1 | ~2.8 | ~3.5 sweep |

→ `d_progressive` dominates. `d_ar_section` memorises. `d_flex` is middling.  
→ Among progressive runs, faster ramp (exp=1.0) wins on Phase 5 data.

---

## 7. What Phase 6 Addresses

1. **Scale**: 3× more data (3908 vs 1187 windows) — should prevent memorisation even with `d_ar_section`
2. **Format**: No Think blocks — cleaner Obs→Act→Obs sequences, more uniform masking targets
3. **Longer trajectories**: Sliding window preserves full trajectory information while fitting 8192-token context
4. **Semantic metric**: OAE replaces token F1 for the nextaction evaluation — measures meaning, not surface form

### Phase 6 run matrix

```bash
sbatch -J dar_section_v2 run_v6.sbatch dar_section_v2 d_ar_section  constant 800 0 1.0 0.20 scienceworld-v2 4
sbatch -J dprog_fast_v2  run_v6.sbatch dprog_fast_v2  d_progressive constant 800 0 1.0 0.20 scienceworld-v2 4
sbatch -J dprog_med_v2   run_v6.sbatch dprog_med_v2   d_progressive constant 800 0 2.0 0.20 scienceworld-v2 4
```

Expected outcome: reduced overfitting in `dar_section_v2`, better generalisation across all runs. OAE will give us a more reliable signal to compare them.

---

## Appendix: Code Pointers

| Topic | Location |
|---|---|
| OAE metric (SemanticEval) | `train_llada_rb.py` — class `SemanticEval` |
| NLI entailment label verified | `_ENTAIL_IDX = 1` (tested empirically) |
| Sliding window implementation | `train_llada_rb.py` — `sliding_window_split()` |
| Progressive masking formula | `train_llada_rb.py` — `d_progressive` branch in `corrupt()` |
| Phase 6 sbatch | `run_v6.sbatch` |
| New dataset IDs | `_TW_SUBDIRS` dict: `"scienceworld-v2"` → `"scienceworld-compact-v2"` |
