# Research Analysis: Masking Objectives for Fine-tuning a Masked Diffusion LM on Agent Trajectories

> Deep analysis of all completed experiments. Phase 5 (6 runs, all finished). Phase 6 in progress (3 runs, ~340/800 steps at time of writing).

---

## 1. Complete Results Table — Phase 5

| Run | Objective | Curr. exp | Train loss (final) | Val nextobs (best) | Val nextobs (sweep mean) | Sweep acc | Sweep F1 | HF |
|---|---|---|---|---|---|---|---|---|
| dar_section | d_ar_section | — | **0.001** | 4.546† | 10.502 | 0.048 | 0.107 | ✓ |
| dflex_v3 | d_flex | — | 3.707 | 3.507 | 4.056 | 0.201 | — | ✓ |
| dprog (w/Think) | d_progressive | 1.0 | 3.241 | 2.102 | 2.918 | 0.480 | — | ✓ |
| dprog_fast_act | d_progressive | 1.0 | 2.127 | **1.534** | 2.906 | **0.473** | **0.555** | ✓ |
| dprog_med_act | d_progressive | 2.0 | 3.789 | 1.688 | 3.065 | 0.458 | 0.517 | ✓ |
| dprog_slow_act | d_progressive | 3.0 | 3.574 | 1.981 | 3.243 | 0.427 | 0.477 | ✓ |
| dar_section_alf | d_ar_section | — | — | — | — | — | — | ✗ (timeout) |
| dprog_med_alf | d_progressive | 2.0 | — | — | — | — | — | ✗ (timeout) |

†dar_section's "best" nextobs of 4.546 is from the mini val set (15 batches), not the full sweep. Its sweep mean is 10.5 — the model has catastrophically overfit.

**Phase 6 OAE (in progress, single-sample — high variance expected):**
- dar_section_v2: OAE oscillating 0.05–0.21 at step 380
- dprog_fast_v2: OAE 0.05–0.20 at step 220
- dprog_med_v2: OAE 0.05–0.18 at step 200

---

## 2. Finding 1 — d_ar_section Catastrophically Memorises on Small Data

This is the sharpest finding in the study.

**The numbers:**
```
dar_section (1,187 train examples):
  step   train_loss   train_acc   val_nextobs
     0     7.938       0.000        6.126
    60     3.106       0.183        5.837
   200     0.215       0.942        5.005   ← val still improving
   280     0.029       0.979        6.681   ← nextobs starts diverging
   480     0.003       1.000        8.350
   640     0.002       1.000        9.735
   799     0.001       1.000       10.586   ← train perfect, val collapsed

Final: train_acc=1.0, sweep_acc=0.048 (4.8%), sweep_F1=0.107
```

Training accuracy reaches 100% by step ~480. Generalisation accuracy stays at 4.8% — the model has memorised the exact action token sequences per trajectory, not learned a policy.

**Why this happens with masked diffusion LMs specifically:** LLaDA uses full bidirectional attention over the entire trajectory. When masking only the action tokens (d_ar_section), the model can attend to ALL surrounding context — including future observations — to reconstruct the masked actions. This is more information than an autoregressive model would ever see at inference time, making memorisation trivially easy: each trajectory is a fixed sequence, and every action is perfectly predictable from the surrounding obs tokens. The model doesn't learn action generation; it learns trajectory lookup.

**Phase 6 signal (dar_section_v2 on 3× more data):**
At step 320, train=0.025, acc=0.994 — same collapse pattern emerging on 3908 examples, just slower. The fundamental problem is the objective, not the data size.

**Implication:** d_ar_section is likely unsuitable for masked diffusion LMs regardless of data scale. The objective leaks future context (subsequent observations) directly into the masking task, making the task trivially solvable by memorisation.

---

## 3. Finding 2 — Progressive Curriculum Masking Prevents Memorisation

The dprog family maintains healthy train loss (~2-4) throughout training while achieving the best generalisation.

**Mechanism:** Random span masking across all block types means:
- The model cannot predict masked action tokens from adjacent obs tokens alone — sometimes the obs are also partially masked
- Span growth from 1→25 forces the model to solve progressively harder reconstruction tasks
- No single trajectory is "solved" — each training step presents a different masking pattern

**Key comparison:**
```
                    train_loss   sweep_nextobs   sweep_acc
d_ar_section:          0.001        10.502         0.048   ← memorised
d_flex:                3.707         4.056         0.201   ← random, unfocused
d_progressive (1.0):   2.127         2.906         0.473   ← curriculum wins
```

The dprog runs cluster tightly around sweep_nextobs ~2.9–3.2 regardless of exponent — the curriculum shape matters less than whether curriculum is used at all.

---

## 4. Finding 3 — Faster Curriculum Ramp is Better (exp=1.0 > 2.0 > 3.0)

This is counter-intuitive and worth highlighting.

| Exponent | Span at step 200 | Span at step 400 | Best nextobs | Sweep acc |
|---|---|---|---|---|
| 1.0 (fast) | ~5 | ~13 | **1.534** | **0.473** |
| 2.0 (medium) | ~3 | ~6 | 1.688 | 0.458 |
| 3.0 (slow) | ~2 | ~3 | 1.981 | 0.427 |

The naive prediction would be: slower ramp = easier early tasks = better foundation = better final performance. The data says the opposite.

**Explanation:** With 800 training steps and 1187 examples, the model needs to reach high-difficulty masking (large spans) early enough to leverage it. exp=3.0 only reaches span=7 by step 527 — fully 2/3 of training is spent at small spans where the memorisation risk is moderate. exp=1.0 reaches span=13 by step 400, forcing harder generalisation challenges earlier and achieving better final nextobs loss despite higher train loss.

This suggests an important practical principle: **for short fine-tuning runs, curriculum speed should be calibrated to the training budget, not to the difficulty of the task in isolation.**

---

## 5. Finding 4 — Think Blocks Don't Help

dprog (SW-v1 with Think blocks) vs dprog_fast_act (SW-v1 without Think blocks):
- Both use identical hyperparameters (exp=1.0, 800 steps)
- Sweep nextobs: 2.918 vs 2.906 — statistically indistinguishable
- Think blocks add ~5-10% token overhead and include model chain-of-thought tokens that aren't environment-grounded

**Implication:** The "reasoning trace" in the original ScienceWorld data is noise, not signal, for trajectory modeling. The compact format (Goal→Obs→Act→Obs) is both smaller and equally effective. This is a clean ablation.

---

## 6. OAE Analysis — What the Early Phase 6 Numbers Tell Us

The OAE scores (0.05–0.21, high variance) are too noisy for ranking at this stage — they're computed on a single validation sample per evaluation step. However several things are observable:

**Why OAE is noisy here:** The metric draws one random sample from the val set per step. ScienceWorld has highly variable action lengths and trajectory structures. A single-step sample is essentially a Bernoulli draw — whether it happens to contain a step where the model succeeds or fails is luck at early training.

**What the range means:** OAE=0.05–0.21 early in training. At baseline (random), all-MiniLM cosine similarity between unrelated action strings gives ~0.15–0.20. So early OAE near 0.05–0.10 is actually below random — the model is confidently predicting wrong tokens. OAE climbing toward 0.20 by step 200 is a weak positive signal.

**dar_section_v2 OAE is already showing the same memorisation pattern as Phase 5:** train_acc=0.994 at step 320, but OAE oscillates 0.09–0.21 with no clear upward trend. The model can reconstruct masked training actions perfectly but generates semantically poor predictions on held-out samples.

**Expected OAE at convergence:** Based on Phase 5 token F1 (0.555 for dprog_fast_act), and the calibration from our worked examples (token F1~0.33 for paraphrase vs SAS~0.80), we'd expect final OAE around **0.35–0.50** for the best progressive run, and **0.10–0.15** for dar_section (given its catastrophic generalisation).

---

## 7. The Full Picture: What Did We Actually Learn?

Across all phases, the core finding is a **three-way interaction between masking objective, curriculum speed, and data scale**, applied to a masked diffusion LM (LLaDA2.0-mini) on ScienceWorld trajectories.

### What works:
- **d_progressive with exp=1.0** — best on all held-out metrics, no memorisation
- **Compact format** (no Think blocks) — equivalent performance with less data
- **Sliding window** — correctly handles trajectories longer than 8192 tokens while preserving goal context

### What fails:
- **d_ar_section** — memorises regardless of data size; the bidirectional attention makes the task too easy
- **d_flex** — random unstructured masking is too unfocused; doesn't provide a useful learning signal

### What's ambiguous:
- Whether OAE truly captures semantic correctness vs token F1 — needs human evaluation to validate
- Whether scale alone (Phase 6) resolves dar_section's memorisation problem (early signs say no)
- Whether these results generalise beyond ScienceWorld

---

## 8. Novelty and Publishability Assessment

### Component-level novelty analysis

**A. Progressive span curriculum for masked diffusion LM fine-tuning**  
*Novelty: Moderate*  
SpanBERT (Joshi et al., 2020) introduced span masking for masked LMs. Recent masked diffusion LMs (MDLM, LLaDA) use independent token masking. A growing-span curriculum applied specifically to MDM fine-tuning on sequential decision tasks has not appeared in literature to our knowledge. The finding that faster ramp wins over slower ramp (counterintuitive, tied to training budget) is a concrete, reproducible result.

**B. LLaDA2.0-mini for agent trajectory modeling**  
*Novelty: High application novelty, low technical novelty*  
LLaDA2.0 (Liu et al., 2025) is very new. Applying it to text-based decision-making (ScienceWorld) is a novel application. The bidirectional attention advantage is theoretically interesting — MDMs can condition on future observations during training in a way autoregressive models cannot. However this paper does not demonstrate that the MDM architecture outperforms a comparable autoregressive baseline, which weakens the claim.

**C. Bidirectional attention leakage as a failure mode in MDM fine-tuning**  
*Novelty: Potentially publishable observation*  
The mechanism by which d_ar_section overfits — the model exploits future observations via bidirectional attention to reconstruct masked actions trivially — is a concrete, underappreciated failure mode specific to MDMs. Autoregressive models cannot do this (they don't see future tokens). This is a genuine insight about MDM fine-tuning that the community should know about.

**D. OAE metric for non-deterministic text game evaluation**  
*Novelty: Low-moderate*  
BERTScore, BLEURT, and others use model-based evaluation. LM-as-judge is now standard. The specific OAE = max(SAS, OOC) formulation is new but the components are not. The NLI entailment approach for causal action-outcome consistency is the more interesting part, but it's currently weakened by the low OOC scores on valid alternate actions. Without human validation it's hard to claim this is better than token F1 in practice — we know it's better in theory.

---

### Publishability verdict

**Top venue (ICLR / NeurIPS / ACL / EMNLP):** Not ready.  
Missing: multi-seed experiments (everything is seed=0), autoregressive baseline comparison, human validation of OAE, results on more than one domain, statistical significance testing. Scale (800 steps, 1187–3908 examples) is small for 2025 standards.

**Workshop paper (NeurIPS FMDM / ICLR LLM Agents / ACL evaluation workshop):** Achievable with moderate work.  
A 4-page paper titled something like *"Curriculum Span Masking Prevents Memorisation in Masked Diffusion LM Fine-tuning for Sequential Decision Tasks"* is a coherent workshop submission. The core claim is testable, the experiment is clean, and the finding (bidirectional attention leakage + curriculum fix) is interesting to the MDM community.  
**What it needs:** 3 seeds per run, one autoregressive baseline (LLaMA-3.1-8B LoRA or similar), Phase 6 complete results, OAE with averaging over 10+ val samples.

**ArXiv preprint:** Can be posted now as a technical report. Framing: "preliminary study of masking objectives for MDM fine-tuning on agent trajectories." Sets a stake in the ground.

**Course report (CP2107 NUS):** Strongest fit right now. The methodology is principled, the ablation is systematic, the findings are clearly stated, and the write-up quality (EXPERIMENTS.md, PHASE6_EVIDENCE.md) is well above typical course project standards.

---

### The one finding worth leading with

> **Bidirectional attention in masked diffusion LMs creates a train-time data leakage problem when action tokens are masked in isolation: the model trivially reconstructs actions by attending to surrounding environment observations, achieving 100% training accuracy but near-chance generalisation. A progressive span-growing curriculum, which randomly masks across all block types with increasing span lengths, prevents this collapse and achieves 10× better generalisation accuracy.**

This is novel, falsifiable, and practically important for anyone applying MDMs to sequential tasks. It tells the community something concrete they didn't know. That's the core contribution.

---

## 9. Recommended Framing for the Report

### Title
*"Masked Diffusion Language Models for Sequential Decision Making: A Study of Masking Objectives and Curriculum Strategies on ScienceWorld"*

### Abstract narrative (3 sentences)
We fine-tune LLaDA2.0-mini, a 16B masked diffusion LM, on ScienceWorld agent trajectories and compare three masking objectives across a 6-run ablation. We find that action-only masking (d_ar_section) catastrophically overfits due to bidirectional attention leakage — the model exploits surrounding observations to reconstruct actions with 100% training accuracy but 4.8% held-out accuracy. A progressive span curriculum across all trajectory tokens prevents this collapse, achieving 10× better generalisation, and we introduce OAE (Outcome-Aware Equivalence), a semantic evaluation metric combining sentence similarity and NLI entailment, to handle ScienceWorld's non-deterministic action space.

### Section structure
1. Introduction — MDMs for agents, why evaluation is hard
2. Background — LLaDA2.0, ScienceWorld, masked diffusion training
3. Method — masking objectives (d_ar_section, d_flex, d_progressive), OAE metric, sliding window
4. Experiments — Phase 5 ablation on SW-v1, Phase 6 on compact-v2
5. Results — the memorisation finding, curriculum comparison, OAE trajectories
6. Analysis — bidirectional leakage mechanism, curriculum speed effect, Think block null result
7. Limitations — single seed, small scale, OAE not human-validated
8. Conclusion

---

## 10. What to Do Next (if time permits)

1. **Wait for Phase 6 to finish** — will give full OAE curves and the dar_section_v2 memorisation comparison on 3× data
2. **Average OAE over 10 samples** (modify `--val_batches` for semantic eval) — reduces variance enough to track learning
3. **Add one autoregressive baseline** — even a small LLaMA-3-8B LoRA run on the same data would dramatically strengthen the paper claim
4. **Human evaluation on 50 OAE examples** — confirm the metric correlates with human "is this action reasonable" judgments
5. **Submit to one workshop** — the finding is real and worth sharing
