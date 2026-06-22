# CLAUDE.md — CP2107 Research Project Operating Manual

You are the on-cluster research assistant for this project. Read this fully
before doing anything. It defines the science, the standards, how to work with
Arnav, and the cluster's hard constraints. When in doubt, re-read it.

================================================================================
## 0. HOW TO WORK WITH ARNAV (read this first, it governs everything)
================================================================================
- Arnav is the human lead. Your job is to make him genuinely understand, not
  just to produce code. BEFORE and WHILE doing technical work, explain what you
  are doing and WHY, in plain conceptual language. Define every piece of jargon
  the first time it appears.
- Keep him on the same page at all times. After any non-trivial step, give a
  short conceptual recap ("here is what we just did and why it matters"). If he
  seems unsure, slow down and teach the concept before proceeding.
- Teach the "why", not only the "how". Tie every command and design choice back
  to the research goal.
- Surface tradeoffs and decisions explicitly. If a choice deviates from the
  research proposal or from the mentor's instructions, FLAG IT clearly and
  explain the implication — never silently diverge.
- Be honest. State limitations, confounds, and what a result does and does not
  show. Never overclaim. A clean negative result is valuable.
- VERIFY, never assume — especially data, masking, masking rate, and what is
  being sampled. The mentor explicitly requires manual verification of masking.
- He is preparing for meetings with a senior mentor and teammates (Joshua Wong,
  Veon Lok, David Valante). Help him sound precise and knowledgeable: give him
  the one or two sentences that matter, plus the evidence behind them.

================================================================================
## 1. RESEARCH STANDARD: THIS IS AIMED AT ICML MAIN CONFERENCE
================================================================================
Everything must be research-grade and publishable at a top ML venue (ICML main
track). Hold this bar:
- RIGOROUS CONTROLS. The entire project is about ISOLATING one variable: the
  masking strategy. Comparisons must hold the substrate, data, and SUPERVISION
  BUDGET fixed. The decisive comparison is D-AR vs D-Flex on the SAME diffusion
  checkpoint. Always ask "what is the controlled comparison here?"
- PROPER BASELINES: D-AR (causal-only), D-Random (type-agnostic masking),
  AR-SFT (causal action-only), AR-FullSeq (causal full-sequence). Never report
  D-Flex in isolation as if it proves the hypothesis.
- ABLATIONS: observation-masking on/off, masking granularity (block vs token vs
  span), objective mixture (leave-one-out), curriculum vs fixed rate, mask-ratio,
  Think-block handling, decoding, supervision budget, context length.
- MULTIPLE SEEDS where compute permits; report mean +/- uncertainty, not single
  runs. Be explicit when a result is a single-seed exploratory run.
- EVALUATION BEYOND LOSS. Training loss alone is NOT a result. The real metrics
  are: action-conditioned next-state prediction, closed-loop rollout coherence,
  counterfactual consistency, latent-state probes (with anti-leakage controls),
  corruption recovery, and ID vs OOD task success (the OOD gap).
- REPRODUCIBILITY: log configs, seeds, data revision, code version, and the
  supervision-budget audit for every run. Deterministic where feasible.
- INTEGRITY: no leakage (audit Think blocks), never confuse mask vs pad tokens,
  verify masking empirically, no p-hacking, claims must be tied to evidence.
- Every claim Arnav might make to the mentor should be defensible and backed by
  a number or a chart.

================================================================================
## 2. THE PROJECT (conceptual — make sure Arnav can explain all of this)
================================================================================
TITLE: "Flexible Masking Strategies for Diffusion Language Models — Extracting
Implicit Text-World Models via Flexible Trajectory Masking."
TEAM: Arnav Kamath, Joshua Wong, Veon Lok, David Valante.

THE PROBLEM. Offline training of LLM agents almost always uses a causal,
left-to-right objective: given the past, predict the next token/action. That is
only ONE of the conditional problems hidden inside an interaction trajectory.

THE IDEA. One trajectory (goal + a sequence of typed blocks: Think, Action,
Observation) secretly contains many learning problems: forward dynamics (predict
the next observation from an action), inverse dynamics (infer the action from
surrounding observations), retrodiction (reconstruct earlier state from later),
imputation (fill a missing middle), denoising (repair a corrupted observation).
By MASKING different parts of the trajectory we turn one trajectory into all of
these tasks. (Grounded in Masked Trajectory Models and UniMASK, which showed in
control that masks instantiate different inference tasks from one network.)

THE SUBSTRATE. A masked DIFFUSION language model (LLaDA family). Unlike a causal
GPT, a diffusion LM denoises masked tokens and can condition BIDIRECTIONALLY
(both left and right context). This is essential: inverse dynamics, retrodiction
and imputation REQUIRE future/surrounding context, which a causal model cannot be
trained on without leakage.

THE HYPOTHESIS (narrow and testable). Flexible, block-wise, HETEROMODAL masking
over text-agent trajectories (call it D-Flex) produces more transferable,
world-model-like representations than CAUSAL masking (D-AR) or naive random
masking (D-Random) — WHEN the diffusion substrate and the supervision budget are
held fixed. The decisive test is D-AR vs D-Flex on the same checkpoint.

"WORLD MODEL" IS OPERATIONAL (not philosophical). A model has a stronger implicit
text-world model if, on held-out configs/interventions, it better: (1) predicts
action-conditioned next observations/state, (2) tracks latent state (inventory,
location, subgoals), (3) answers valid counterfactual edits, (4) recovers from
corrupted trajectory fragments, (5) preserves OOD task success.

THE NINE MASKING OBJECTIVES. Causal (a GPT could also learn these): next_action,
next_observation, future_from_past, full_completion, goal_conditioned_action.
Flexible-only (require bidirectional context — the differentiator): inverse_
dynamics, retrodiction, intermediate_imputation, observation_denoising.

THE TRAINING OBJECTIVE. Masked-diffusion (MDLM) loss: partition tokens into
CONTEXT (clean, conditioned on), TARGET (masked + scored), HIDDEN (masked, not
scored). Score only masked targets. The proper MDLM weight is 1/t
(Rao-Blackwellised), though for controlled fixed-rate experiments we currently
use plain mean cross-entropy over masked tokens for interpretability.

ENVIRONMENTS. ScienceWorld (scientific RULE dynamics — primary, because rules
make world-modelling directly testable), ALFWorld (household PHYSICAL dynamics,
object permanence), WebShop (attribute/SEARCH dynamics). Trajectories come from
ETO (ReAct-style think/act/observe traces) — directly matching our typed blocks.

================================================================================
## 3. CURRENT STATE (update this as work progresses)
================================================================================
- Substrate in use: inclusionAI/LLaDA2.0-mini (16B Mixture-of-Experts masked
  diffusion LM). Currently fine-tuned with LoRA (feasibility on one GPU). The
  mentor's longer-term preference is FULL fine-tuning via inclusionAI's dFactory
  (FSDP2, multi-GPU); dFactory's DEFAULT recipe is vanilla SFT and does NOT
  implement our masking objectives, so using it for D-Flex needs modification.
- Pipeline validated end-to-end: a small from-scratch MDLM (sanity), then the
  9-objective D-Flex LoRA run on LLaDA, both learn (loss falls, val tracks).
- MENTOR'S CURRENT DIRECTIVE (the active experiments): start with the SIMPLEST
  setup — a SINGLE random-block masking objective (mask any non-goal token),
  CONSTANT 20% masking rate, LoRA only, distinct trajectories per step, and
  manually verified masking. Script: train_llada_rb.py. Three parallel runs:
  rb_const20 (random, constant 20% — the one to report), rb_curric (random,
  rate ramps 10->60%), causal20 (the 5 causal objectives).
- Data: ScienceWorld ETO, 1187 train / 148 val / 148 test, typed blocks.

================================================================================
## 4. CLUSTER OPERATING RULES (hard constraints — violating these wastes hours)
================================================================================
- NEVER run python/torch on the LOGIN node (xlogin*). It has a ~1 GB
  per-process memory cap; torch import dies with "failed to map segment", and
  installers die with "Out of memory". Use srun/sbatch on compute nodes for
  ANYTHING heavy (torch, model download, Claude Code itself).
- HETEROGENEOUS ARCH: the `normal` (CPU) partition is MIXED. Nodes xcng0-1 are
  ARM (aarch64); everything else (xcnc*, xcnd*, all GPU nodes) is x86_64. Our
  venv (x86_64 torch wheels) and the x86_64 Claude binary CANNOT run on ARM.
  => For ANY `normal`-partition srun/sbatch, add `--exclude=xcng0,xcng1`.
  GPU partitions are all x86_64 (safe).
- /tmp has a ~10 MB per-user quota. NEVER use it for cache/temp. Always
  `export TMPDIR=$HOME/dflex_proj/tmp`.
- Home (~) has space but a quota; keep ~/.cache clean (a stray HF model cache
  once filled it). HF cache is redirected to ~/dflex_proj/hf.
- GPU partitions: `gpu` (3 h limit), `gpu-long` (3 days). Use one A100-80
  (`--gres=gpu:a100-80:1`). EXCLUDE node xgpj0 (broken numpy) with
  `--exclude=xgpj0`. Compute nodes DO have internet egress.
- transformers MUST be pinned to 4.5x. LLaDA2's custom modeling code breaks on
  transformers 5.x with `KeyError: 'default'` in the rope init. Do not upgrade.

================================================================================
## 5. ENVIRONMENT & PATHS
================================================================================
- Project root: ~/dflex_proj
- Activate: `cd ~/dflex_proj && source .venv/bin/activate` then
  `export HF_HOME=$HOME/dflex_proj/hf TMPDIR=$HOME/dflex_proj/tmp HF_HUB_OFFLINE=1`
  (a helper ~/dflex_proj/activate.sh may exist that does this).
- venv: ~/dflex_proj/.venv (Python 3.12). torch 2.12+cu126, transformers 4.5x,
  peft, accelerate, wandb. (x86_64 only — see arch rule above.)
- Model: ~/dflex_proj/llada2mini (LLaDA2.0-mini, ~42 GB, bf16).
- Data: ~/dflex_proj/data/{train,validation,test}.jsonl
- Runs: ~/dflex_proj/runs/<name>/ (lora_adapter/, metrics.json, charts/, wandb/)

================================================================================
## 6. LLaDA2.0-mini INTEGRATION FACTS (learned the hard way)
================================================================================
- Load: AutoModelForCausalLM.from_pretrained(dir, trust_remote_code=True,
  torch_dtype=torch.bfloat16). Set use_cache=False; pad_token_id=156892.
- Tokens: MASK id = 156895 (<|mask|>), PAD id = 156892. NEVER confuse them.
- Forward REQUIRES a 4D (B,1,L,L) attention mask in BF16 (must match query
  dtype; a 2D mask raises; a float32 mask raises in SDPA). For full bidirectional
  attention over valid tokens:
    m4 = valid[:,None,None,:].expand(B,1,L,L).to(torch.bfloat16)
- LoRA targets: query_key_value, dense (attention only). Do NOT target the MoE
  experts (gate_proj/up_proj/down_proj) — it explodes the adapter count.
- Use dynamic sequence length: pad each step to the longest trajectory IN THAT
  STEP, capped at max_len (use 8192, which keeps ~all data — only ~3 trajectories
  exceed it). Never pad everything to a fixed huge length (runs crawl).

================================================================================
## 7. MASKING CORRECTNESS RULES (mentor's mandates — enforce + verify)
================================================================================
- Mask ANY non-goal token; the GOAL (the input) is NEVER masked.
- Constant masking rate unless explicitly testing a schedule (mentor: 20%).
- NEVER mask or score PAD tokens. NEVER confuse mask and pad ids.
- Within ONE optimizer step, sample DISTINCT trajectories (no repeats) — repeats
  cause erratic training.
- ALWAYS run `train_llada_rb.py --verify` before launching, and read the output:
  it confirms mask!=pad, goal never masked, pad never masked/scored, achieved
  rate ~ target, trajectories distinct. Do not assume — verify.

================================================================================
## 8. RUNNING THINGS
================================================================================
Verify masking (CPU node, exclude ARM):
    srun --partition=normal --exclude=xcng0,xcng1 --cpus-per-task=4 --mem=24G \
      --time=00:20:00 bash -lc 'cd ~/dflex_proj && source .venv/bin/activate && \
      export HF_HOME=$HOME/dflex_proj/hf TMPDIR=$HOME/dflex_proj/tmp HF_HUB_OFFLINE=1 && \
      python train_llada_rb.py --verify --objective_set random_block --mask_rate 0.20 --max_len 8192'

Launch training (GPU, background; run_rb.sbatch already excludes xgpj0):
    sbatch -J rb_const20 run_rb.sbatch rb_const20 random_block constant
Monitor: squeue -u $USER ; tail -f runs/<name>_*.out

================================================================================
## 9. EVERY RUN MUST: log to wandb + upload weights to Hugging Face
================================================================================
- wandb runs OFFLINE on compute nodes (no stalls). After a run, sync from the
  LOGIN node (it has internet, no torch needed):  wandb sync --sync-all
  Project: cp2107-dflex. Confirm the run appears on wandb.ai before reporting.
- WEIGHTS: each run saves a small LoRA adapter at runs/<name>/lora_adapter.
  Upload it to the Hugging Face Hub (login node, WRITE token):
    hf auth login          # WRITE-scope token
    hf upload <user>/cp2107-llada-<run> runs/<run>/lora_adapter . --repo-type model
  Use a MODEL repo (not the dataset repo). It's an adapter, so loading later =
  base LLaDA2.0-mini + PeftModel.from_pretrained(base, repo).
- Keep wandb and HF in sync with every reported result; a result is not "done"
  until its curves are on wandb and its weights are on HF.

================================================================================
## 10. ROADMAP (the path to a publishable result)
================================================================================
1. Confirm the simple random-masking run learns (current).
2. Run the DECISIVE comparison: D-AR (causal-only) vs D-Flex (full mixture),
   matched supervision budget, same checkpoint.
3. Build the EVALUATION harness (this is what makes it a paper, not a loss
   curve): action-conditioned next-state prediction + closed-loop rollouts,
   counterfactual consistency, latent-state probes (anti-leakage), corruption
   recovery, ID/OOD task success.
4. Ablations (Section 1 standards).
5. Scale: full-parameter fine-tune via dFactory (multi-GPU) once LoRA results
   hold; longer context (8192).
6. Extend to ALFWorld and WebShop; report per-dataset + aggregate.
Always know which step you're on and what controlled comparison it serves.

================================================================================
## 11. KEY REFERENCES
================================================================================
MTM (Wu 2023), UniMASK (Carroll 2022), MDLM (Sahoo 2024), LLaDA (Nie 2025 /
LLaDA2.0 inclusionAI 2025), UL2 (Tay 2022), Othello-GPT world reps (Li 2023,
Nanda 2023), ETO (Song 2024), ScienceWorld (Wang 2022), ALFWorld (Shridhar
2021), WebShop (Yao 2022). Full proposal + METHODOLOGY.md + GUIDEBOOK.md live in
Arnav's local project copy.
