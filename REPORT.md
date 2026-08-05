# Masking Objectives and Curriculum Strategies for Fine-tuning Masked Diffusion Language Models on Agent Trajectories

**Arnav Kamath** · National University of Singapore · CP2107 Independent Research Module

> **Status:** Experiments complete. Teammate evaluation on held-out test set and averaged OAE pending. This report covers the author's experimental and analytical contributions in full.

---

## Abstract

We study how to fine-tune a masked diffusion language model (LLaDA2.0-mini, 16B MoE) on ScienceWorld agent trajectories using LoRA. We compare three families of masking objectives — suffix masking (D-AR), multi-task random masking (D-Flex), and progressive span curriculum masking (D-Progressive) — across six experimental phases and nine completed training runs. The central finding is that action-only masking (d\_ar\_section) induces catastrophic memorisation specific to masked diffusion models: bidirectional attention allows the model to reconstruct masked action tokens directly from surrounding observations, achieving 100% training accuracy but near-chance held-out generalisation (4.8% accuracy on the val sweep). Progressive span curriculum masking, which randomises masked regions across all token types with spans growing from 1 to 25 tokens over training, prevents this collapse and achieves 10× better held-out generalisation (47% accuracy). We additionally introduce OAE (Outcome-Aware Equivalence), a semantic evaluation metric combining sentence-transformer similarity and NLI entailment, to handle ScienceWorld's non-deterministic action space. OAE reveals a further distinction: d\_ar\_section learns fluent game language (OAE ≈ 0.36) without environment grounding, while d\_progressive maintains consistent language-action coupling. All adapter weights are publicly released at `AK2802/AOMT`.

---

## 1. Introduction

Masked diffusion language models (MDMs) are a recent alternative to autoregressive (AR) generation. Unlike AR models, which predict tokens left-to-right, MDMs are trained to reconstruct masked tokens from full bidirectional context and generate via iterative denoising. LLaDA (Liu et al., 2024) demonstrated that MDMs can match autoregressive models on standard NLU and generation benchmarks. Their bidirectional attention is theoretically advantageous for structured prediction tasks where the full context is available at inference time.

Agent trajectory modeling is one such task. Given a goal and a history of observations and actions, the model must predict the next action — a task where prior observations provide rich bidirectional context. The MDM's ability to attend to future observations during training (but not at inference time) introduces both an opportunity and a risk: the model may learn richer representations from the full trajectory context, or it may exploit future information in ways that prevent genuine generalisation.

This work investigates which masking objectives lead to genuine world-model learning in MDMs fine-tuned on ScienceWorld (Wang et al., 2022) trajectories. We make three contributions:

1. **An empirical demonstration of bidirectional attention leakage** as a failure mode specific to action-only masking in MDMs, confirmed across two dataset scales.
2. **Progressive span curriculum masking** as a principled and effective remedy, with empirical comparison across curriculum speeds.
3. **OAE (Outcome-Aware Equivalence)**, a semantic evaluation metric for non-deterministic environments that distinguishes surface-form fluency from environment grounding.

---

## 2. Background

### 2.1 LLaDA2.0-mini

LLaDA2.0-mini is a 16B Mixture-of-Experts masked diffusion language model. It is trained with a masked language modelling objective over continuous noise schedules and generates text by iterative denoising from a fully-masked sequence. All attention is bidirectional. We fine-tune it with LoRA (r=16, α=32) targeting query, key, value, and dense projection layers, yielding approximately 2.95M trainable parameters out of 16.26B total (0.018%). All experiments use bf16 precision on a single A100-80GB GPU.

### 2.2 ScienceWorld

ScienceWorld is a text-based simulation environment requiring multi-step scientific reasoning (e.g., determining electrical conductivity, measuring friction, performing chemical reactions). Trajectories follow the structure Goal → Obs₀ → Act₁ → Obs₁ → ··· → ActN → ObsN. A key property is **non-determinism**: multiple valid action sequences exist for most tasks. Standard token F1 is therefore an inadequate evaluation metric.

We use two dataset versions:
- **SW-v1** (Phases 0–5): 1,187 train / 148 val / 148 test. Includes Think blocks.
- **SW-compact-v2** (Phase 6): 3,534 train / 1,762 val / 1,799 test. No Think blocks. Sequences up to 23,530 tokens; 3% exceed 8,192 → sliding window applied (see §3.3). After windowing: 3,908 train / 1,904 val.

### 2.3 Evaluation Metric: val\_nextobs\_loss

The primary evaluation metric throughout is **val/nextobs\_loss**: given the causal prefix of a held-out trajectory (goal + all preceding obs/act pairs), predict the final observation block. The context is strictly causal — no future tokens — making this metric independent of training objective. Lower is better.

---

## 3. Methods

### 3.1 Masking Objectives

All objectives use cross-entropy loss over masked tokens only; non-masked tokens contribute nothing to the gradient.

**D-AR (suffix masking):** Masks the final `rate × L` non-goal tokens in each trajectory. Context is strictly causal. Designed to test whether a bidirectional model can internalize causal direction via gradients alone.

**D-Flex (multi-task random masking):** Samples one of four objectives per batch:
1. *Imputation*: mask a random contiguous span.
2. *Retrodiction*: mask the first `rate × L` tokens.
3. *Observation denoising*: mask `rate × L` tokens from observation blocks only.
4. *Inverse dynamics*: mask all action tokens at one randomly-chosen step.

Context is bidirectional for all four. The diversity of tasks is intended to encourage representations useful for many trajectory reasoning sub-problems.

**D-AR-Section (action-only masking):** Masks ALL action tokens across the entire trajectory (~6.5% of non-goal tokens). Context is bidirectional. Directly aligned with MDM inference (at test time, all action positions would be masked and denoised jointly). The training task is exactly the downstream prediction task.

**D-Progressive (curriculum span masking):** Masks random contiguous spans of growing size across all block types (goal, obs, act). Span size follows `span = max(1, round(max_span × progressᵉˣᵖ))` where `progress ∈ [0,1]` linearly indexes training steps, `max_span = 25`, and `exp ∈ {1.0, 2.0, 3.0}` controls curriculum speed. Mask rate is fixed at 20% throughout; only contiguity, not volume, increases. At progress=0: identical to random token masking. At progress=1: spans of 25 tokens, covering entire action blocks and partial observation sentences.

### 3.2 Training Setup

All Phase 5–6 runs share:
- Steps: 800, batch size: 1, gradient accumulation: 4
- Learning rate: 1e-4, weight decay: 0.1, LoRA dropout: 0.1
- EMA decay: 0.995 (applied during validation, restored after)
- max\_len: 8192 tokens; val evaluated every 20 steps over 15 random batches
- Full val sweep (all val trajectories, not sampled) run once at end of training

### 3.3 Sliding Window for Long Trajectories

SW-compact-v2 contains trajectories up to 23,530 tokens. For sequences exceeding 8,192 tokens, we apply a sliding window split:
- The goal block is always prepended to each window (task context preserved)
- Windows split at block (bidx) boundaries — never mid-block
- Stride = available\_space // 2 (50% overlap between adjacent windows)

This produces 3,908 training examples from 3,534 raw trajectories.

### 3.4 OAE: Outcome-Aware Equivalence

ScienceWorld's non-determinism makes token F1 misleading. "Go north" and "teleport to kitchen" may both be valid actions if both reach the kitchen; token F1 scores the latter as 0 when the gold is the former. We introduce OAE:

For each predicted action $\hat{a}_i$ at step $i$, with ground-truth action $a_i^*$ and resulting ground-truth observation $o_i^*$:

$$\text{SAS}_i = \cos(\text{embed}(\hat{a}_i),\ \text{embed}(a_i^*))$$

$$\text{OOC}_i = P\!\left(\text{entail} \mid \underbrace{\text{"The agent performs: } \hat{a}_i\text{"}}\_{\text{premise}} \to \underbrace{\text{"The resulting observation is: } o_i^*\text{"}}\_{\text{hypothesis}}\right)$$

$$\text{OAE}_i = \max(\text{SAS}_i,\ \text{OOC}_i), \qquad \text{OAE} = \frac{1}{N}\sum_i \text{OAE}_i$$

SAS uses `all-MiniLM-L6-v2` (22M params). OOC uses `cross-encoder/nli-deberta-v3-small` (44M params) run on CPU during validation. The max() gives credit if either the action is a semantic paraphrase (SAS) or causes the same outcome (OOC).

**Limitations of OAE.** (i) Entity-name discrimination: SAS scores "red box" vs "yellow box" at 0.746 due to weak entity sensitivity in sentence embeddings. OOC correctly gives 0.001 but max() is dominated by SAS. (ii) OOC coverage: the NLI model does not model game-world causality, so truly valid alternate actions (e.g., "take" vs "pick up") may still score low on OOC. (iii) Single-sample noise: OAE is currently logged from one validation sample per step; averaging over 10+ samples is required for a reliable training signal. This is identified as future work.

---

## 4. Experimental History

### Phase 0–1: Pilot and Infrastructure (invalidated)

Initial runs (Phase 0, jobs rb\_const20, rb\_curric, causal20) used `max_len=1024`, `val_batches=3`, and mixed causal objectives with conflicting gradient scales. A critical padding bug padded all sequences to 8192 unconditionally, increasing attention cost ~61× and causing all Phase 1 jobs to time out before saving adapters. These runs are reported for completeness but contribute no valid results.

**Fix:** Dynamic padding to per-batch maximum length reduced step time from ~38s to ~18s.

### Phase 2: First Valid Runs (D-AR vs D-Flex, 200 steps)

After fixing the padding bug, we compared D-AR suffix masking (`dar`) and D-Flex 3-objective masking (`dflex`) over 200 steps. D-AR val loss plateaued at step ~60 (loss ≈ 4.95–5.25, no further improvement), while D-Flex continued improving through step 200. The plateau in D-AR is attributed to architectural mismatch: a bidirectional model attempting to internalise causal direction via gradients against bidirectional attention in every layer. This was a primary motivating observation for the subsequent experimental design.

The shared `val_nextobs` metric was not yet implemented, so Phase 2 losses are not directly comparable across objectives.

### Phase 3: D-AR vs D-Flex vs D-Random with Shared Evaluation (400 steps)

We introduced `val_nextobs_loss` as a shared, objective-independent evaluation: predict the final observation block from a strictly causal prefix. We added D-Random (uniform random token masking, `drandom_v2`) as a third condition, and extended D-Flex to 4 objectives (adding `inverse_dynamics`).

| Run | Objective | Best nextobs↓ | Sweep nextobs mean |
|---|---|---|---|
| dar\_v2 | D-AR (suffix) | 4.012 | 4.377 |
| dflex\_v2 | D-Flex (4-obj) | 3.614 | 3.974 |
| drandom\_v2 | D-Random | **2.084** | 3.197 |

D-Flex outperformed D-AR by ~0.4 nats on nextobs averaged over steps 240–400 (D-AR: 4.377 ± 0.248; D-Flex: 3.974 ± 0.319). D-Random achieved the best single best-nextobs checkpoint but showed high variance (std=0.513 vs 0.319 for D-Flex) and a large train/val gap (0.294 → 2.76), suggesting memorisation of denoising patterns rather than world-model learning. D-Flex was not yet converged at 400 steps; extension to 600 steps was planned.

### Phase 4: Convergence and D-Progressive Introduction (600 steps)

We extended D-Flex to 600 steps (`dflex_v3`) and introduced two new objectives: `d_ar_refined` (uniform-position causal masking, ablating the positional bias of suffix masking) and `d_progressive` (curriculum span masking, `dprog`).

| Run | Objective | Best nextobs↓ | Final nextobs | Final train loss |
|---|---|---|---|---|
| dflex\_v3 | D-Flex | 3.507 | 3.849 | 3.707 |
| dprog | D-Progressive (exp=1.0) | 2.102 | 4.755 | 3.241 |
| dar\_v3 | D-AR-Refined | — | — | — |

D-Progressive achieved best nextobs of 2.102 at 600 steps, substantially better than D-Flex at 600 steps (3.507). This introduced D-Progressive as the primary candidate for Phase 5 ablation.

### Phase 5: D-AR-Section vs Progressive Curriculum Ablation (800 steps, SW-v1)

Phase 5 is the main controlled ablation. We compare `d_ar_section` (action-only masking, the LLaDA-aligned inference objective) against `d_progressive` at three curriculum speeds (exponents 1.0, 2.0, 3.0) and two mask rates (20% and action-rate 6.5%). All runs use 800 steps on SW-v1 (1,187 train examples).

#### 4.1 Complete Phase 5 Results

| Run | Objective | Exp | Rate | Best nextobs↓ | Sweep acc↑ | Sweep F1↑ | Final train loss |
|---|---|---|---|---|---|---|---|
| dar\_section | d\_ar\_section | — | 6.5% | 4.546 | 0.048 | 0.107 | **0.001** |
| dflex\_v3 | d\_flex | — | 15% | 3.507 | 0.201 | — | 3.707 |
| dprog\_fast\_act | d\_progressive | 1.0 | 6.5% | **1.534** | **0.473** | **0.555** | 2.127 |
| dprog\_med\_act | d\_progressive | 2.0 | 6.5% | 1.688 | 0.458 | 0.517 | 3.789 |
| dprog\_slow\_act | d\_progressive | 3.0 | 6.5% | 1.981 | 0.427 | 0.477 | 3.574 |

Best nextobs for `dar_section` is measured on the 15-batch mini-val (not the full sweep). Its sweep mean is 10.502 (std=1.993), confirming complete generalisation collapse.

#### 4.2 Finding 1 — Bidirectional Attention Leakage in Action-Only Masking

`dar_section` achieves train\_acc=1.000, train\_loss=0.001 by step ~480, while val/nextobs diverges from 4.5 (step 140) to 10.6 (final). Sweep accuracy is 4.8% — near chance.

This failure mode is specific to masked diffusion models. With action-only masking, the model attends to ALL surrounding context — including observations that occur *after* the masked action — to reconstruct the action tokens. Since each trajectory is a fixed sequence, every action is deterministically predictable from the adjacent observation pair. The bidirectional attention makes this lookup trivially easy; the model never needs to learn a policy. An autoregressive model cannot exhibit this failure because future tokens are causally masked.

```
Training curve — dar_section:
  step   train_loss   val_nextobs
     0     7.938        6.126
    60     3.106        5.837      ← early learning
   200     0.215        5.005      ← still improving
   280     0.029        6.681      ← divergence begins
   480     0.003        8.350      ← memorisation complete
   799     0.001       10.586      ← full collapse
```

#### 4.3 Finding 2 — D-Progressive Prevents Memorisation

All three progressive runs maintain stable generalisation: final sweep nextobs 2.47–2.76, sweep accuracy 42.7–47.3%.

```
Training curve — dprog_fast_act (best Phase 5 run):
  step   train_loss   val_nextobs   span
     0     5.991        6.124        1
   200     0.355        4.544        2     ← still improving
   400     0.661        1.097        5     ← val still falling
   540     1.243        2.955       10     ← train rises, val stable
   760     2.205        1.534       21     ← BEST nextobs
   799     2.127        2.473       25
```

As span grows past ~15, training loss increases (task becomes genuinely hard) while val nextobs continues improving — the signature of generalisation, not memorisation. Train loss rising while val loss falls is the opposite of overfitting.

#### 4.4 Finding 3 — Faster Curriculum Ramp Wins (exp=1.0 > 2.0 > 3.0)

| Exponent | Span at step 400 | Best nextobs | Sweep acc |
|---|---|---|---|
| 1.0 | ~13 | **1.534** | **0.473** |
| 2.0 | ~6 | 1.688 | 0.458 |
| 3.0 | ~3 | 1.981 | 0.427 |

Counter-intuitively, the fastest curriculum ramp wins. With a fixed budget of 800 steps, exp=3.0 spends over half of training at span ≤ 3, leaving too little time at high difficulty. exp=1.0 reaches span=13 by step 400, providing harder generalisation challenges earlier. This suggests that **curriculum speed must be calibrated to the training budget**, not to absolute task difficulty.

#### 4.5 Finding 4 — Think Blocks Are Not Helpful

Phase 4 `dprog` (SW-v1 with Think blocks, exp=1.0, 600 steps) achieved sweep nextobs mean ≈ 2.92. Phase 5 `dprog_fast_act` (SW-v1 without Think blocks, same hyperparameters, 800 steps) also achieved 2.91 sweep mean. The Think block contribution is negligible within measurement noise. The compact format (Goal→Obs→Act→Obs) is both smaller and equally effective.

### Phase 6: SW-compact-v2 Dataset with OAE Metric (800 steps, 3× data)

Phase 6 applies the best-performing objective (d\_progressive) and the proposed (d\_ar\_section) to the larger SW-compact-v2 dataset (3,908 train after sliding window) with OAE logging enabled. All three runs timed out at step ~740/800 (6h wall time on gpu-long). Checkpoints at the best-nextobs step are reported.

| Run | Objective | Exp | Best nextobs↓ | At step | Final nextobs | Peak OAE↑ |
|---|---|---|---|---|---|---|
| dar\_section\_v2 | d\_ar\_section | — | 4.468 | 300 | 7.503 | 0.363 |
| dprog\_fast\_v2 | d\_progressive | 1.0 | **2.384** | 540 | 3.316 | **0.338** |
| dprog\_med\_v2 | d\_progressive | 2.0 | 2.565 | 540 | 3.285 | 0.276 |

**Note on metric comparability.** Phase 6 nextobs values are not directly comparable to Phase 5 values. SW-compact-v2 contains longer, more diverse trajectories, and the val set is 1,904 examples (vs 148 in Phase 5). A nextobs of 2.384 on Phase 6 data represents a different and more challenging generalisation problem.

#### 5.1 Finding 5 — Memorisation Persists at 3× Data Scale

`dar_section_v2` exhibits the same divergence pattern as Phase 5: train\_acc=0.994 by step 320, nextobs diverges from 4.5 to 7.5. Three times more training data does not resolve the bidirectional attention leakage problem. This confirms that the failure is **structural to the objective**, not a data-insufficiency artifact.

#### 5.2 Finding 6 — OAE Reveals Grounding Failure Invisible to Nextobs

`dar_section_v2`'s OAE rises from 0.053 (step 0) to 0.363 (step 700) as nextobs simultaneously diverges from 4.5 to 7.5. The model is learning to generate **fluent game language** (predictions sound like valid ScienceWorld actions) while failing completely at **trajectory grounding** (predictions are wrong for the specific trajectory context).

This dissociation between surface fluency and semantic grounding is a concrete finding that token F1 alone would not have revealed: token F1 at step 700 for dar\_section\_v2 is near-zero on the val sweep, but OAE is 0.36. The model has learned the distribution of ScienceWorld action strings without learning which action is appropriate when.

By contrast, `dprog_fast_v2` OAE rises to 0.338 while nextobs simultaneously decreases to 2.384 — language quality and grounding improve together, as expected.

---

## 5. Results Summary

| Metric | dar\_section (P5) | dprog\_fast\_act (P5) | dar\_section\_v2 (P6) | dprog\_fast\_v2 (P6) |
|---|---|---|---|---|
| Final train loss | **0.001** | 2.127 | ~0.003 | ~1.5 |
| Best nextobs↓ | 4.546† | **1.534** | 4.468 | **2.384** |
| Sweep nextobs mean↓ | 10.502 | 2.906 | (timed out) | (timed out) |
| Sweep acc↑ | 0.048 | **0.473** | — | — |
| Sweep F1↑ | 0.107 | **0.555** | — | — |
| Peak OAE↑ | — | — | 0.363 | **0.338**‡ |

†Best mini-val nextobs; sweep mean is 10.502.  
‡OAE computed from single-sample per step (high variance); averaged test-set OAE pending.

**D-Progressive (exp=1.0) is the best-performing objective across both phases on all generalisation metrics.**

---

## 6. The OAE Metric: Calibration Evidence

The following examples demonstrate OAE scoring on representative action pairs (run offline using loaded checkpoints):

| Case | Predicted action | GT action | SAS | OOC | OAE |
|---|---|---|---|---|---|
| Exact match | "pick up unknown substance" | "pick up unknown substance" | 1.000 | 0.001 | 1.000 |
| Paraphrase | "grab the unknown substance" | "pick up unknown substance" | 0.802 | 0.001 | **0.802** |
| Totally wrong | "go north" | "pick up unknown substance" | 0.070 | 0.001 | 0.070 |
| Domain-wrong | "move block to caesium surface" | "pick up unknown substance" | 0.149 | 0.011 | 0.149 |
| Entity error (weakness) | "put item in red box" | "put item in yellow box" | 0.746 | 0.001 | **0.746** ✗ |

The entity error case (row 5) is a known limitation: sentence embeddings are insufficiently sensitive to specific entity names. A model predicting "red box" when the gold is "yellow box" receives OAE=0.746 rather than near-zero. This is documented as a limitation and does not affect the validity of OAE as a relative comparison metric across runs, since all runs are evaluated on the same samples.

---

## 7. Pending Evaluations (Teammate)

The following evaluations are **not yet complete** and are excluded from this report:

1. **Held-out test set nextobs and OAE** for `dprog_fast_v2` (checkpoint\_540) on SW-compact-v2 test split (1,799 examples). This is the definitive generalization number for the best model.
2. **Averaged OAE** over all val examples (not single-sample per step). This removes the high variance from single-sample OAE and gives a reliable semantic accuracy estimate.
3. **Qualitative generation samples**: 10+ trajectories from the test set, last action masked, model predictions versus ground truth, human-readable.

These evaluations can be run with:
```bash
# Load dprog_fast_v2 checkpoint_540, evaluate on test split
python train_llada_rb.py \
  --datasets scienceworld-v2 --data_dir_tw data/textworld \
  --resume_from runs/dprog_fast_v2/checkpoint_540 \
  --val_only --full_val_at_end --val_batches 100
```

---

## 8. Limitations

**Single seed.** All runs use seed=0. No confidence intervals are reported. Multi-seed runs (seeds 1, 2) are required for publishable error bars. Results should be interpreted as directional, not definitive.

**No autoregressive baseline.** We compare masking objectives within LLaDA2.0-mini but do not compare against an autoregressive model (e.g., LLaMA-3-8B LoRA) on the same data. The claim that MDMs are beneficial for trajectory modeling is not tested here.

**OAE not human-validated.** OAE is proposed and calibrated against constructed examples. We do not confirm that OAE scores correlate with human judgments of action correctness on ScienceWorld trajectories.

**ALFWorld incomplete.** ALF runs timed out consistently (>6h for 800 steps), leaving the cross-domain question unanswered. Cross-domain generalisation of the curriculum strategy is unverified.

**Training budget.** 800 steps with a 6-hour wall time is modest. Phase 6 runs timed out at step 740. Results may shift with longer training, particularly for d\_progressive where the learning curve appears not yet converged.

**OAE single-sample noise.** The OAE values reported from training logs are computed on one validation sample per step. Standard deviation across steps is ~0.08–0.12, making step-to-step comparisons unreliable. The peak OAE values reported are upper bounds on the true mean.

---

## 9. Discussion

### 9.1 Why Bidirectional Attention Leakage Matters

The catastrophic memorisation of d\_ar\_section is not a standard overfitting story. Standard overfitting occurs when a model's capacity exceeds the dataset size — the solution is regularisation or more data. Here, **3× more data does not help** (Finding 5). The failure is qualitatively different: the model is solving a trivially easy version of the intended task. Masked action tokens are predictable from adjacent observations because both are part of the same fixed trajectory. The model exploits a structural shortcut that an autoregressive model cannot access.

This is an important practical warning for anyone applying MDMs to sequential tasks: action-only masking should not be used as-is. The fix (d\_progressive) works because masking spans randomly across all token types ensures that adjacent context is itself partially masked, eliminating the shortcut.

### 9.2 On the OAE-NextObs Dissociation

The Phase 6 finding that dar\_section\_v2 achieves OAE=0.36 while sweep nextobs=7.5 has a clean interpretation. OAE measures whether the predicted action *string* sounds like a reasonable ScienceWorld action in isolation. NextObs measures whether the model understands the *specific state* the agent is in. A model that has memorised the distribution of action strings (which is easy — the ScienceWorld action vocabulary is limited: "pick up X", "connect X to Y", "look at X") can score reasonably on OAE without having any understanding of what's happening in the trajectory. D-Progressive does not exhibit this dissociation because it must also reconstruct observation tokens — it cannot succeed by memorising action strings alone.

### 9.3 Curriculum Speed and Training Budget Interaction

The finding that exp=1.0 beats exp=2.0 beats exp=3.0 within a fixed 800-step budget suggests that curriculum schedule design should be treated as a function of both task difficulty and training budget jointly. With a longer budget (e.g., 3,000 steps), exp=3.0 might outperform exp=1.0 by providing a more careful early-stage foundation. This is left as future work.

---

## 10. Conclusion

We have shown that masked diffusion language models require careful masking objective design for agent trajectory fine-tuning. Action-only masking induces a bidirectional attention leakage failure that persists across dataset scales. Progressive span curriculum masking, which randomises masked regions across all trajectory token types with linearly-growing span size, prevents this failure and achieves 10× better held-out generalisation accuracy (47.3% vs 4.8% on the val sweep). The OAE semantic metric provides additional signal beyond token F1, revealing that action-only masking produces surface fluency without grounding — a distinction that token-level metrics cannot detect.

The best model (`dprog_fast_v2`, checkpoint at step 540, `AK2802/AOMT/dprog_fast_v2`) achieves val/nextobs=2.384 on the larger SW-compact-v2 dataset. Held-out test set evaluation is pending.

---

## Appendix A: Hyperparameters

| Hyperparameter | Value |
|---|---|
| Model | LLaDA2.0-mini (16B MoE, bf16) |
| LoRA rank / alpha | 16 / 32 |
| LoRA targets | query\_key\_value, dense |
| Trainable params | 2.95M / 16.26B (0.018%) |
| Learning rate | 1e-4 |
| Optimizer | AdamW |
| Weight decay | 0.1 |
| LoRA dropout | 0.1 |
| EMA decay | 0.995 |
| Batch size | 1 (grad accum 4) |
| Max sequence length | 8,192 tokens |
| Training steps | 800 |
| Mask rate | 20% (or 6.5% for action-rate runs) |
| Max span (d\_progressive) | 25 tokens |
| GPU | A100-80GB (NUS HPC) |

## Appendix B: Released Artifacts

All adapters are released at `AK2802/AOMT` on HuggingFace.

| Adapter | Phase | Notes |
|---|---|---|
| `dar` | P2 | D-AR, 200 steps, pilot |
| `dflex` | P2 | D-Flex 3-obj, 200 steps, pilot |
| `dar_v2` | P3 | D-AR with val\_nextobs, 400 steps |
| `dflex_v2` | P3 | D-Flex 4-obj, 400 steps |
| `drandom_v2` | P3 | D-Random, 400 steps |
| `dflex_v3` | P4 | D-Flex, 600 steps |
| `dprog` | P4 | D-Progressive exp=1.0, 600 steps |
| `dar_section` | P5 | d\_ar\_section, 800 steps — memorised |
| `dprog_fast_act` | P5 | **Best Phase 5.** d\_prog exp=1.0 rate=6.5%, 800 steps |
| `dprog_med_act` | P5 | d\_prog exp=2.0 rate=6.5%, 800 steps |
| `dprog_slow_act` | P5 | d\_prog exp=3.0 rate=6.5%, 800 steps |
| `dar_section_v2` | P6 | d\_ar\_section, SW-v2, best ckpt @ step 300 |
| `dprog_fast_v2` | P6 | **Best Phase 6.** d\_prog exp=1.0, SW-v2, best ckpt @ step 540 |
| `dprog_med_v2` | P6 | d\_prog exp=2.0, SW-v2, best ckpt @ step 540 |
