#!/usr/bin/env python3
"""Controlled LoRA fine-tune of LLaDA2.0-mini on trajectory data.

Objective sets:
  d_ar_section  — mask ALL action blocks; context = goal+think+obs (bidirectional).
                  Matches LLaDA inference: set every action to MASK, denoise the full traj.
                  Rate is automatic (~6.5% for ScienceWorld).
  d_progressive — span-masking curriculum: span grows 1→max_span over training.
                  --prog_exponent controls growth speed (higher = slower growth).
  random_block  — random token masking, bidirectional context (Phase 3 baseline)
  d_ar          — suffix masking, causal context (Phase 3 baseline)
  d_flex        — 4 bidirectional objectives (Phase 3)
  d_ar_refined  — uniform-position span, causal context

Shared validation metrics (computed for all objectives):
  val/nextobs_loss     — predict last obs block from causal prefix
  val/nextaction_loss  — predict last action block from causal prefix
Both are directly relevant to inference: nextaction = policy evaluation,
nextobs = world model / transition dynamics evaluation.

Full val sweep (--full_val_at_end): nextobs over all 148 val trajectories.

Multi-dataset support: --datasets scienceworld,textworld
  textworld = ALFWorld + WebShop from Joshyxwa/cp2107-textworld-trajectories.
  Download first with: python fetch_textworld_data.py

Verify mode: --verify checks all masking invariants.
             --show_samples N also prints N annotated trajectory examples for human review.
"""
from __future__ import annotations
import argparse, json, os, random, sys, time, collections
from dataclasses import dataclass
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

PADB, GOAL, THINK, ACTION, OBS = 0, 1, 2, 3, 4
TYPE_FROM_STR = {"Goal": GOAL, "Think": THINK, "Action": ACTION, "Observation": OBS}
TYPE_NAME     = {GOAL: "goal", THINK: "think", ACTION: "action", OBS: "obs"}

D_FLEX_OBJS   = ("intermediate_imputation", "retrodiction", "obs_denoising", "inverse_dynamics")
CAUSAL_OBJS   = ("next_action", "next_observation", "future_from_past",
                 "full_completion", "goal_conditioned_action")

MASK_ID = None
PAD_ID  = 0


# ── data loading ──────────────────────────────────────────────────────────────

@dataclass
class Example:
    ids: np.ndarray; btype: np.ndarray; step: np.ndarray; bidx: np.ndarray; traj_id: str

def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def row_to_arrays(row, tok, max_len, drop_leaky_think=True):
    leaky = set()
    if drop_leaky_think:
        for fl in row.get("leakage_flags", []) or []:
            if isinstance(fl, dict) and "step" in fl:
                leaky.add(fl["step"])
    ids, btype, step, bidx = [], [], [], []
    bi = 0
    for blk in row["blocks"]:
        bt = TYPE_FROM_STR.get(blk["type"])
        if bt is None:
            continue
        if bt == THINK and drop_leaky_think and blk.get("step") in leaky:
            continue
        toks = tok.encode(blk["text"], add_special_tokens=False)
        if not toks:
            continue
        for t in toks:
            ids.append(int(t)); btype.append(bt)
            step.append(int(blk.get("step", -1))); bidx.append(bi)
        bi += 1
    if not ids or len(ids) > max_len:
        return None
    return Example(np.array(ids, np.int64), np.array(btype, np.int64),
                   np.array(step, np.int64), np.array(bidx, np.int64),
                   str(row.get("trajectory_id", "?")))

def load_split(path, tok, max_len):
    examples, dropped = [], 0
    for r in read_jsonl(path):
        e = row_to_arrays(r, tok, max_len)
        if e is None: dropped += 1
        else: examples.append(e)
    if dropped:
        print(f"[data] {os.path.basename(path)}: dropped {dropped} seqs > {max_len} tokens")
    return examples

def load_datasets(args, tok):
    """Load one or more datasets (comma-separated in --datasets).

    Dataset identifiers:
      scienceworld — data/             (1187 train / 148 val)
      alfworld     — data/textworld/alfworld  (6574 train / 251 val, seq_len mean=2272)
      webshop      — data/textworld/webshop   (3014 train / 377 val, seq_len mean=36 — WARNING)

    NOTE: WebShop sequences average only 36 tokens (trivial for 16B model).
          ALFWorld has action_frac=0.87% (vs ScienceWorld 6.5%) — gradient is very sparse
          for d_ar_section on ALFWorld; consider --grad_accum 8.
    """
    _TW_SUBDIRS = {
        "alfworld": "alfworld",
        "webshop":  "webshop",
    }
    train_all, val_all = [], []
    for ds in args.datasets.split(","):
        ds = ds.strip()
        if ds == "scienceworld":
            tr = load_split(os.path.join(args.data_dir, "train.jsonl"),      tok, args.max_len)
            vl = load_split(os.path.join(args.data_dir, "validation.jsonl"), tok, args.max_len)
        elif ds in _TW_SUBDIRS:
            sub = os.path.join(args.data_dir_tw, _TW_SUBDIRS[ds])
            if not os.path.isfile(os.path.join(sub, "train.jsonl")):
                raise FileNotFoundError(
                    f"{ds} data not found at {sub}.\n"
                    f"Run: python fetch_textworld_data.py")
            tr = load_split(os.path.join(sub, "train.jsonl"),      tok, args.max_len)
            vl = load_split(os.path.join(sub, "validation.jsonl"), tok, args.max_len)
        else:
            raise ValueError(f"Unknown dataset: {ds!r}. Choose from: scienceworld, alfworld, webshop")
        print(f"[data] {ds}: train={len(tr)} val={len(vl)}")
        train_all.extend(tr); val_all.extend(vl)
    return train_all, val_all


# ── masking primitives ────────────────────────────────────────────────────────

def random_block_mask(ex, nrng, rate):
    """D-Random: mask rate% of non-goal tokens uniformly at random (bidirectional context)."""
    L = len(ex.ids)
    elig = ex.btype != GOAL
    tgt = elig & (nrng.random(L) < rate)
    if not tgt.any():
        idx = np.where(elig)[0]
        if idx.size:
            tgt[int(nrng.choice(idx))] = True
    return ~tgt, tgt

def all_action_mask(ex):
    """D-AR-Section: mask ALL action blocks simultaneously.

    Context (bidirectional): goal + ALL think + ALL obs tokens, including future obs.
    Target: every action token in the trajectory (no rate parameter needed).
    Hidden: none.

    Training/inference alignment: at LLaDA inference time, you set all action
    tokens to MASK and run a single denoising pass over the full trajectory.
    Think blocks here provide planning context ('I should go to kitchen') but do
    NOT directly state the action — the model must infer 'teleport to kitchen' or
    'look around' from the abstract plan + environment state.

    Mean scored fraction: ~6.5% of eligible tokens (ScienceWorld)."""
    L = len(ex.ids)
    tgt = (ex.btype == ACTION)
    if not tgt.any():
        return np.ones(L, bool), np.zeros(L, bool)
    return ~tgt, tgt

def suffix_mask(ex, rate):
    """D-AR: mask last (rate × eligible) non-goal tokens; causal prefix as context."""
    L = len(ex.ids)
    elig_idx = np.where(ex.btype != GOAL)[0]
    if len(elig_idx) == 0:
        return np.ones(L, bool), np.zeros(L, bool)
    n_mask = max(1, min(int(round(rate * len(elig_idx))), len(elig_idx)))
    tgt = np.zeros(L, bool)
    tgt[elig_idx[-n_mask:]] = True
    return ~tgt, tgt

def causal_random_mask(ex, rng, rate):
    """D-AR-Refined: uniform-position span masking with causal context."""
    L = len(ex.ids)
    elig_idx = np.where(ex.btype != GOAL)[0]
    if len(elig_idx) == 0:
        return np.ones(L, bool), np.zeros(L, bool)
    n_mask = max(1, min(int(round(rate * len(elig_idx))), len(elig_idx)))
    start_i = rng.randint(0, max(0, len(elig_idx) - n_mask) + 1)
    tgt = np.zeros(L, bool)
    tgt[elig_idx[start_i:start_i + n_mask]] = True
    first_tgt = int(np.where(tgt)[0][0])
    ctx = np.zeros(L, bool)
    ctx[:first_tgt] = True
    ctx &= ~tgt
    return ctx, tgt

def progressive_span_mask(ex, nrng, rate, progress, exponent=1.0, max_span=25):
    """D-Progressive: token-level → span-level masking curriculum.

    Span grows from 1 (individual tokens) to max_span over training:
      span_size = round(exp(log(max_span) * (progress ^ exponent)))

    exponent=1.0: moderate growth (span=5 at halfway) — the 'fast' curriculum
    exponent=2.0: slower growth (span≈2 at halfway) — more time at token-level
    exponent=3.0: very slow growth (span≈1.5 at halfway) — nearly all time on easy

    max_span=25 covers the 75th percentile of ScienceWorld block lengths (p75=24 tokens).
    At max_span, most individual blocks are fully masked, approximating block-level masking.

    Rate stays fixed (default 20%) throughout — only contiguity changes, not volume."""
    L = len(ex.ids)
    elig_idx = np.where(ex.btype != GOAL)[0]
    if len(elig_idx) == 0:
        return np.ones(L, bool), np.zeros(L, bool)
    n_mask_target = max(1, min(int(round(rate * len(elig_idx))), len(elig_idx)))

    eff_progress = progress ** exponent
    span_size = max(1, round(np.exp(np.log(max_span) * eff_progress)))

    tgt = np.zeros(L, bool)
    if span_size == 1:
        draw = nrng.random(L) < rate
        tgt = (ex.btype != GOAL) & draw
        if not tgt.any():
            tgt[elig_idx[int(nrng.integers(0, len(elig_idx)))]] = True
        true_idx = np.where(tgt)[0]
        if len(true_idx) > n_mask_target:
            tgt[true_idx[n_mask_target:]] = False
    else:
        pool = elig_idx.copy(); nrng.shuffle(pool)
        for start_pos in pool:
            for offset in range(span_size):
                pos = int(start_pos) + offset
                if pos < L and ex.btype[pos] != GOAL:
                    tgt[pos] = True
            if tgt.sum() >= n_mask_target:
                break
        true_idx = np.where(tgt)[0]
        if len(true_idx) > n_mask_target:
            tgt[true_idx[n_mask_target:]] = False
        if not tgt.any():
            tgt[elig_idx[int(nrng.integers(0, len(elig_idx)))]] = True

    return ~tgt, tgt

def _steps_present(ex):
    return sorted({int(s) for s in ex.step.tolist() if s >= 1})

def _first_idx(ex, step_val, bt):
    idx = np.where((ex.step == step_val) & (ex.btype == bt))[0]
    return int(idx[0]) if idx.size else None

def imputation_mask(ex, nrng, rate):
    L = len(ex.ids)
    elig_idx = np.where(ex.btype != GOAL)[0]
    if len(elig_idx) == 0:
        return np.ones(L, bool), np.zeros(L, bool)
    n_mask = max(1, min(int(round(rate * len(elig_idx))), len(elig_idx)))
    lo = max(0, int(0.10 * len(elig_idx)))
    hi = min(int(0.70 * len(elig_idx)), max(lo, len(elig_idx) - n_mask))
    if lo > hi: lo = 0; hi = max(0, len(elig_idx) - n_mask)
    start_i = int(nrng.integers(lo, max(lo + 1, hi + 1)))
    tgt = np.zeros(L, bool)
    tgt[elig_idx[start_i:start_i + n_mask]] = True
    return ~tgt, tgt

def retrodiction_mask(ex, rate):
    L = len(ex.ids)
    elig_idx = np.where(ex.btype != GOAL)[0]
    if len(elig_idx) == 0:
        return np.ones(L, bool), np.zeros(L, bool)
    n_mask = max(1, min(int(round(rate * len(elig_idx))), len(elig_idx)))
    tgt = np.zeros(L, bool)
    tgt[elig_idx[:n_mask]] = True
    return ~tgt, tgt

def obs_denoising_mask(ex, nrng, rate):
    L = len(ex.ids)
    elig_idx = np.where(ex.btype != GOAL)[0]
    obs_idx  = np.where(ex.btype == OBS)[0]
    if len(elig_idx) == 0 or len(obs_idx) == 0:
        return np.ones(L, bool), np.zeros(L, bool)
    n_mask = max(1, min(int(round(rate * len(elig_idx))), len(obs_idx)))
    chosen = nrng.choice(obs_idx, size=n_mask, replace=False)
    tgt = np.zeros(L, bool)
    tgt[chosen] = True
    return ~tgt, tgt

def inverse_dynamics_mask(ex, rng, rate):
    """D-Flex: given ALL obs bidirectionally, predict action at chosen step.
    Think at same step is HIDDEN (not scored, not context) to prevent leakage."""
    L = len(ex.ids)
    steps = _steps_present(ex)
    action_steps = [s for s in steps if _first_idx(ex, s, ACTION) is not None]
    if not action_steps:
        return imputation_mask(ex, np.random.default_rng(rng.randint(0, 1 << 30)), rate)
    t = rng.choice(action_steps)
    tgt = (ex.step == t) & (ex.btype == ACTION)
    if not tgt.any():
        return imputation_mask(ex, np.random.default_rng(rng.randint(0, 1 << 30)), rate)
    ctx = np.zeros(L, bool)
    ctx |= (ex.btype == GOAL)
    ctx |= (ex.btype == OBS)
    ctx |= ((ex.btype == THINK)  & (ex.step != t))
    ctx |= ((ex.btype == ACTION) & (ex.step != t))
    ctx &= ~tgt
    return ctx, tgt

def causal_mask(obj, ex, rng, rate):
    L = len(ex.ids); steps = _steps_present(ex); goal = ex.btype == GOAL
    def fb(): return random_block_mask(ex, np.random.default_rng(rng.randint(0, 1 << 30)), rate)
    if not steps: return fb()
    if obj == "next_action":
        cand = [s for s in steps if _first_idx(ex, s, ACTION) is not None]
        if not cand: return fb()
        t = rng.choice(cand); st = _first_idx(ex, t, ACTION)
        tgt = (ex.step == t) & (ex.btype == ACTION)
        ctx = (np.arange(L) < st) & ~tgt
    elif obj == "next_observation":
        cand = [s for s in steps if _first_idx(ex, s, OBS) is not None]
        if not cand: return fb()
        t = rng.choice(cand); st = _first_idx(ex, t, OBS)
        tgt = (ex.step == t) & (ex.btype == OBS)
        ctx = (np.arange(L) < st) & ~tgt
    elif obj == "future_from_past":
        t = rng.choice(steps[:-1]) if len(steps) > 1 else steps[0]
        ctx = (ex.step <= t) | goal; tgt = ~ctx
    elif obj == "full_completion":
        ctx = goal | (ex.bidx == 0)
        interior = sorted(set(int(b) for b in ex.bidx) - {0})
        keep = max(0, int(round((1 - rate) * len(interior))))
        for b in (rng.sample(interior, min(keep, len(interior))) if interior else []):
            ctx = ctx | (ex.bidx == b)
        tgt = ~ctx
    elif obj == "goal_conditioned_action":
        ab = sorted(set(int(ex.bidx[i]) for i in range(L) if ex.btype[i] == ACTION))
        if not ab: return fb()
        k = max(1, int(round(rate * len(ab)))); chosen = rng.sample(ab, min(k, len(ab)))
        tgt = np.isin(ex.bidx, chosen); ctx = ~tgt
        for b in chosen:
            a_step = int(ex.step[np.where(ex.bidx == b)[0][0]])
            ctx = ctx & ~((ex.step > a_step) & (ex.btype == OBS))
    else:
        return fb()
    ctx = ctx & ~tgt
    if not tgt.any(): return fb()
    return ctx, tgt


# ── shared evaluation tasks (objective-agnostic val metrics) ──────────────────

def next_obs_mask(ex):
    """Val metric: predict last obs block from causal prefix.
    Tests world-model quality (forward dynamics T(s,a)→s').
    Identical task for all objectives — lower is better."""
    L = len(ex.ids)
    obs_steps = sorted(s for s in set(ex.step.tolist())
                       if s >= 1 and np.any((ex.step == s) & (ex.btype == OBS)))
    if not obs_steps:
        return None
    last_step = obs_steps[-1]
    tgt = (ex.step == last_step) & (ex.btype == OBS)
    if not tgt.any():
        return None
    first_tgt = int(np.where(tgt)[0][0])
    if first_tgt == 0:
        return None
    ctx = np.zeros(L, bool)
    ctx[:first_tgt] = True
    ctx &= ~tgt
    return ctx, tgt

def next_action_mask(ex):
    """Val metric: predict last action block from causal prefix.
    Tests policy quality — given trajectory prefix, what action to take next.
    Directly relevant to inference-time use. Identical for all objectives."""
    L = len(ex.ids)
    action_steps = sorted(s for s in set(ex.step.tolist())
                          if s >= 1 and np.any((ex.step == s) & (ex.btype == ACTION)))
    if not action_steps:
        return None
    last_step = action_steps[-1]
    tgt = (ex.step == last_step) & (ex.btype == ACTION)
    if not tgt.any():
        return None
    first_tgt = int(np.where(tgt)[0][0])
    if first_tgt == 0:
        return None
    ctx = np.zeros(L, bool)
    ctx[:first_tgt] = True
    ctx &= ~tgt
    return ctx, tgt


# ── batch construction ────────────────────────────────────────────────────────

def corrupt(ex, ctx, tgt, pad_to):
    L = min(len(ex.ids), pad_to)
    ctx = ctx[:L]; tgt = tgt[:L]
    ids   = np.full(pad_to, PAD_ID,  np.int64)
    btype = np.full(pad_to, PADB,    np.int64)
    valid = np.zeros(pad_to, bool)
    ids[:L] = ex.ids[:L]; btype[:L] = ex.btype[:L]; valid[:L] = True
    inp  = ids.copy()
    tmsk = np.zeros(pad_to, bool); tmsk[:L] = tgt
    hmsk = np.zeros(pad_to, bool); hmsk[:L] = (~ctx) & (~tgt)
    inp[tmsk | hmsk] = MASK_ID
    return inp, btype, valid, tmsk, ids

def sample_one(obj_set, ex, rng, nrng, rate, progress=0.0, prog_exp=1.0, prog_max_span=25):
    if obj_set == "d_ar_section":
        return "all_action", all_action_mask(ex)
    if obj_set == "random_block":
        return "random_block", random_block_mask(ex, nrng, rate)
    if obj_set == "d_ar":
        return "suffix", suffix_mask(ex, rate)
    if obj_set == "d_ar_refined":
        return "causal_random", causal_random_mask(ex, rng, rate)
    if obj_set == "d_progressive":
        return "prog_span", progressive_span_mask(ex, nrng, rate, progress, prog_exp, prog_max_span)
    if obj_set == "d_flex":
        obj = rng.choice(D_FLEX_OBJS)
        if obj == "intermediate_imputation": return obj, imputation_mask(ex, nrng, rate)
        if obj == "retrodiction":            return obj, retrodiction_mask(ex, rate)
        if obj == "obs_denoising":           return obj, obs_denoising_mask(ex, nrng, rate)
        return obj, inverse_dynamics_mask(ex, rng, rate)
    obj = rng.choice(CAUSAL_OBJS)
    return obj, causal_mask(obj, ex, rng, rate)

def make_step_batch(train, k, obj_set, rng, nrng, max_len, rate, progress=0.0,
                    prog_exp=1.0, prog_max_span=25):
    idxs = rng.sample(range(len(train)), min(k, len(train)))
    prepared = []
    for i in idxs:
        ex = train[i]
        obj, (ctx, tgt) = sample_one(obj_set, ex, rng, nrng, rate, progress,
                                      prog_exp, prog_max_span)
        prepared.append((ex, obj, ctx, tgt))
    pad_to = min(max(len(ex.ids) for ex, _, _, _ in prepared), max_len)
    I, B, V, S, Lb, objs, tids = [], [], [], [], [], [], []
    elig_tot = scored_tot = 0
    for ex, obj, ctx, tgt in prepared:
        inp, bt, val, sc, lab = corrupt(ex, ctx, tgt, pad_to)
        I.append(inp); B.append(bt); V.append(val); S.append(sc); Lb.append(lab)
        objs.append(obj); tids.append(ex.traj_id)
        elig_tot  += int(((ex.btype != GOAL)).sum())
        scored_tot += int(sc.sum())
    achieved = scored_tot / max(1, elig_tot)
    return (np.stack(I), np.stack(B), np.stack(V), np.stack(S), np.stack(Lb),
            objs, tids, achieved)

def _make_eval_batch(val_data, k, rng, max_len, mask_fn):
    """Generic eval batch builder; mask_fn returns (ctx, tgt) or None."""
    idxs = rng.sample(range(len(val_data)), min(k * 3, len(val_data)))
    prepared = []
    for i in idxs:
        result = mask_fn(val_data[i])
        if result is not None:
            prepared.append((val_data[i], result[0], result[1]))
        if len(prepared) >= k:
            break
    if not prepared:
        return None
    pad_to = min(max(len(ex.ids) for ex, _, _ in prepared), max_len)
    I, B, V, S, Lb = [], [], [], [], []
    for ex, ctx, tgt in prepared:
        inp, bt, vld, sc, lab = corrupt(ex, ctx, tgt, pad_to)
        I.append(inp); B.append(bt); V.append(vld); S.append(sc); Lb.append(lab)
    return np.stack(I), np.stack(B), np.stack(V), np.stack(S), np.stack(Lb)

def make_val_nextobs_batch(val_data, k, rng, max_len):
    return _make_eval_batch(val_data, k, rng, max_len, next_obs_mask)

def make_val_nextaction_batch(val_data, k, rng, max_len):
    return _make_eval_batch(val_data, k, rng, max_len, next_action_mask)

def full_val_nextobs_sweep(run_micro_fn, val_data, max_len):
    """Evaluate nextobs on EVERY val trajectory for a definitive low-variance estimate."""
    losses, accs, f1s = [], [], []
    for ex in val_data:
        result = next_obs_mask(ex)
        if result is None:
            continue
        ctx, tgt = result
        pad_to = min(len(ex.ids), max_len)
        inp_a = np.full(pad_to, PAD_ID, np.int64)
        bt_a  = np.full(pad_to, PADB,   np.int64)
        vld_a = np.zeros(pad_to, bool)
        L = min(len(ex.ids), pad_to)
        inp_a[:L] = ex.ids[:L]; bt_a[:L] = ex.btype[:L]; vld_a[:L] = True
        tgt_a   = np.zeros(pad_to, bool); tgt_a[:L]   = tgt[:L]
        hmask_a = np.zeros(pad_to, bool); hmask_a[:L] = (~ctx[:L]) & (~tgt[:L])
        lab_a   = inp_a.copy()
        inp_a[tgt_a | hmask_a] = MASK_ID
        arrs = (inp_a[None], bt_a[None], vld_a[None], tgt_a[None], lab_a[None])
        loss, acc, f1, _, _ = run_micro_fn(arrs)
        losses.append(float(loss.item())); accs.append(acc); f1s.append(f1)
    if not losses:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    return (float(np.mean(losses)), float(np.std(losses)),
            float(np.mean(accs)), float(np.mean(f1s)), len(losses))


# ── EMA ───────────────────────────────────────────────────────────────────────

class EMA:
    """Exponential moving average of LoRA weights for stable val evaluation."""
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {n: p.data.clone().float()
                       for n, p in model.named_parameters() if p.requires_grad}

    def update(self, model):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and n in self.shadow:
                    self.shadow[n].mul_(self.decay).add_(p.data.float(), alpha=1 - self.decay)

    def apply_to(self, model):
        self._backup = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and n in self.shadow:
                    p.data.copy_(self.shadow[n].to(p.dtype))

    def restore_from(self, model):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and n in self._backup:
                    p.data.copy_(self._backup[n])


# ── metrics ───────────────────────────────────────────────────────────────────

def _unigram_f1(pred_ids, true_ids):
    """Unigram token-bag F1 — partial credit for semantically close predictions.

    Better than exact match: 'pick up metal pot' vs 'pick up pot' scores ~0.8 F1
    rather than 0. Computed at subword-token level without any additional model.
    Note: true_ids should NOT include special tokens (MASK_ID, PAD_ID)."""
    from collections import Counter
    if len(pred_ids) == 0 or len(true_ids) == 0:
        return 1.0 if (len(pred_ids) == 0 and len(true_ids) == 0) else 0.0
    # Convert to Python lists for Counter
    pl = pred_ids.tolist() if hasattr(pred_ids, 'tolist') else list(pred_ids)
    tl = true_ids.tolist() if hasattr(true_ids, 'tolist') else list(true_ids)
    pc = Counter(pl); tc = Counter(tl)
    overlap = sum(min(pc[k], tc[k]) for k in pc if k in tc)
    p = overlap / max(1, sum(pc.values()))
    r = overlap / max(1, sum(tc.values()))
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# ── masking visualisation ─────────────────────────────────────────────────────

def show_masked_samples(examples, args, tok):
    """Print annotated trajectory examples showing masking applied.

    Called via --show_samples N in verify mode. No model required.
    For d_progressive, shows masking at 4 stages (progress = 0, 0.33, 0.66, 1.0)
    so the curriculum progression is visually apparent.

    Legend:
      [MASK:text] = target tokens (shown with original text for reference)
      (HID:text)  = hidden tokens (masked but not scored)
      plain text  = context tokens (unchanged input)
    """
    rng  = random.Random(args.seed)
    nrng = np.random.default_rng(args.seed)
    is_tty = sys.stdout.isatty()

    # Terminal colors
    R   = "\033[0m"  if is_tty else ""
    BLU = "\033[94m" if is_tty else ""
    YEL = "\033[93m" if is_tty else ""
    RED = "\033[91m" if is_tty else ""
    GRN = "\033[92m" if is_tty else ""
    DIM = "\033[2m"  if is_tty else ""
    INV = "\033[7m"  if is_tty else ""   # reverse video for masked tokens

    BT_COL  = {GOAL: BLU, THINK: YEL, ACTION: RED, OBS: GRN}
    BT_NAME = {GOAL: "GOAL", THINK: "THINK", ACTION: "ACT", OBS: "OBS"}

    n = args.show_samples
    for _ in range(n):
        ex = rng.choice(examples)
        print(f"\n{'═'*72}")
        print(f"TRAJ: {ex.traj_id}  |  {len(ex.ids)} tokens  |  obj={args.objective_set}")
        print('═'*72)

        if args.objective_set == "d_progressive":
            for prog in [0.0, 0.33, 0.66, 1.0]:
                eff = prog ** args.prog_exponent
                span_sz = max(1, round(np.exp(np.log(args.prog_max_span) * eff)))
                ctx, tgt = progressive_span_mask(ex, nrng, args.mask_rate,
                                                  prog, args.prog_exponent, args.prog_max_span)
                n_s  = int(tgt.sum())
                n_el = int((ex.btype != GOAL).sum())
                step_approx = round(prog * max(1, args.steps - 1))
                print(f"\n  ── progress={prog:.2f}  step≈{step_approx}  "
                      f"span={span_sz}  scored={n_s}/{n_el} "
                      f"({100*n_s/max(1,n_el):.1f}%) ──")
                _print_traj_masked(ex, ctx, tgt, tok, BT_COL, BT_NAME, R, INV, DIM, GRN, is_tty)
        else:
            _, (ctx, tgt) = sample_one(args.objective_set, ex, rng, nrng,
                                        args.mask_rate, 0.0,
                                        args.prog_exponent, args.prog_max_span)
            n_s  = int(tgt.sum())
            n_el = int((ex.btype != GOAL).sum())
            print(f"  scored={n_s}/{n_el} ({100*n_s/max(1,n_el):.1f}%)")
            _print_traj_masked(ex, ctx, tgt, tok, BT_COL, BT_NAME, R, INV, DIM, GRN, is_tty)

def _print_traj_masked(ex, ctx, tgt, tok, BT_COL, BT_NAME, R, INV, DIM, GRN, is_tty):
    """Print a trajectory with tokens annotated as context / masked / hidden."""
    L = len(ex.ids)
    # Group into contiguous same-type same-step blocks
    blocks, cur_bt, cur_step, cur_tok, cur_ctx, cur_tgt = [], None, None, [], [], []
    for pos in range(L):
        bt, step = int(ex.btype[pos]), int(ex.step[pos])
        if bt != cur_bt or step != cur_step:
            if cur_tok:
                blocks.append((cur_bt, cur_step, cur_tok, cur_ctx, cur_tgt))
            cur_bt, cur_step = bt, step
            cur_tok, cur_ctx, cur_tgt = [], [], []
        cur_tok.append(int(ex.ids[pos]))
        cur_ctx.append(bool(ctx[pos]))
        cur_tgt.append(bool(tgt[pos]))
    if cur_tok:
        blocks.append((cur_bt, cur_step, cur_tok, cur_ctx, cur_tgt))

    for bt, step, tokens, b_ctx, b_tgt in blocks:
        col  = BT_COL.get(bt, "") if is_tty else ""
        lbl  = BT_NAME.get(bt, "???")
        step_s = f"-{step}" if step >= 0 and bt != GOAL else ""
        role = "[T]" if any(b_tgt) else ("[H]" if (not any(b_ctx) and not any(b_tgt)) else "[C]")
        prefix = f"{col}{lbl}{step_s:4s}{R}"

        parts = []
        i = 0
        while i < len(tokens):
            if b_tgt[i]:
                # Collect contiguous masked (target) run
                run = []
                while i < len(tokens) and b_tgt[i]:
                    run.append(tokens[i]); i += 1
                text = tok.decode(run, skip_special_tokens=True).strip()
                if is_tty:
                    parts.append(f"{INV}[{text}]{R}")
                else:
                    parts.append(f"[MASK:{text}]")
            elif b_ctx[i]:
                word = tok.decode([tokens[i]], skip_special_tokens=True)
                parts.append(word); i += 1
            else:
                # Hidden token
                word = tok.decode([tokens[i]], skip_special_tokens=True)
                if is_tty:
                    parts.append(f"{DIM}({word}){R}")
                else:
                    parts.append(f"(HID:{word})")
                i += 1

        full_text = " ".join(parts)
        if len(full_text) > 110:
            full_text = full_text[:107] + "…"
        print(f"  {prefix:16s} {role}  {full_text}")


# ── charts ────────────────────────────────────────────────────────────────────

def make_charts(hist, obj_loss, budget, out_dir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    def smooth(y, k=15):
        if len(y) < k: return np.array(y)
        c = np.cumsum(np.insert(y, 0, 0)); return (c[k:] - c[:-k]) / k

    # Loss + nextobs + nextaction
    plt.figure(figsize=(9, 5))
    plt.plot(hist["step"], hist["loss"], alpha=0.2, color="tab:blue")
    sm = smooth(np.array(hist["loss"]))
    plt.plot(hist["step"][len(hist["step"]) - len(sm):], sm,
             color="tab:blue", label="train/loss (smoothed)")
    if hist["val_step"]:
        plt.plot(hist["val_step"], hist["val_loss"], "o-", color="tab:red",    label="val/loss (task)")
        no = hist.get("val_nextobs", [])
        na = hist.get("val_nextaction", [])
        if no and any(not np.isnan(v) for v in no):
            plt.plot(hist["val_step"], no, "s--", color="tab:green",  label="val/nextobs")
        if na and any(not np.isnan(v) for v in na):
            plt.plot(hist["val_step"], na, "^:", color="tab:orange", label="val/nextaction")
    plt.xlabel("step"); plt.ylabel("cross-entropy"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out_dir}/loss_curve.png", dpi=130); plt.close()

    # Accuracy + F1
    if hist.get("val_acc") or hist.get("acc"):
        plt.figure(figsize=(9, 4))
        if hist.get("acc"):
            sm_a = smooth(np.array(hist["acc"]))
            plt.plot(hist["step"][len(hist["step"]) - len(sm_a):], sm_a,
                     color="tab:blue", alpha=0.7, label="train/acc (smoothed)")
        if hist.get("tok_f1"):
            sm_f = smooth(np.array(hist["tok_f1"]))
            plt.plot(hist["step"][len(hist["step"]) - len(sm_f):], sm_f,
                     color="tab:purple", alpha=0.7, label="train/tok_f1 (smoothed)")
        if hist.get("val_step"):
            if hist.get("val_acc"):
                plt.plot(hist["val_step"], hist["val_acc"], "o-", color="tab:red",  label="val/acc")
            if hist.get("val_nextobs_acc") and any(not np.isnan(v) for v in hist["val_nextobs_acc"]):
                plt.plot(hist["val_step"], hist["val_nextobs_acc"], "s--",
                         color="tab:green",  label="val/nextobs_acc")
            if hist.get("val_nextaction_acc") and any(not np.isnan(v) for v in hist.get("val_nextaction_acc",[])):
                plt.plot(hist["val_step"], hist["val_nextaction_acc"], "^:",
                         color="tab:orange", label="val/nextaction_acc")
        plt.ylim(0, 1); plt.xlabel("step"); plt.ylabel("accuracy / F1")
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(f"{out_dir}/accuracy_curve.png", dpi=130); plt.close()

    # Mask rate
    plt.figure(figsize=(7, 4))
    plt.plot(hist["step"], hist["mask_rate"], color="tab:orange")
    plt.ylim(0, 1); plt.xlabel("step"); plt.ylabel("achieved mask rate")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{out_dir}/mask_rate.png", dpi=130); plt.close()

    # Per-objective loss
    if any(obj_loss[k][1] for k in obj_loss):
        names = [k for k in obj_loss if obj_loss[k][1] > 0]
        vals  = [obj_loss[k][0] / obj_loss[k][1] for k in names]
        plt.figure(figsize=(7, 4))
        plt.bar(range(len(names)), vals, color="tab:purple")
        plt.xticks(range(len(names)), names, rotation=40, ha="right")
        plt.ylabel("avg loss"); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
        plt.savefig(f"{out_dir}/loss_by_objective.png", dpi=130); plt.close()

    # Supervision budget
    bt_names = [TYPE_NAME[k] for k in (GOAL, THINK, ACTION, OBS)]
    bv = [budget.get(k, 0) for k in (GOAL, THINK, ACTION, OBS)]
    plt.figure(figsize=(6, 4))
    plt.bar(bt_names, bv, color="tab:gray")
    plt.ylabel("scored tokens"); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(f"{out_dir}/supervision_budget.png", dpi=130); plt.close()


# ── verify mode ───────────────────────────────────────────────────────────────

def verify(args, tok, train, val):
    rng  = random.Random(args.seed)
    nrng = np.random.default_rng(args.seed)
    print("=" * 72)
    print(f"MASK_ID={MASK_ID}  PAD_ID={PAD_ID}  distinct? {MASK_ID != PAD_ID}")
    assert MASK_ID != PAD_ID, "MASK and PAD must differ!"

    k = args.grad_accum * args.batch_size
    I, B, V, S, Lb, objs, tids, achieved = make_step_batch(
        train, k, args.objective_set, rng, nrng, args.max_len, args.mask_rate,
        prog_exp=args.prog_exponent, prog_max_span=args.prog_max_span)
    print(f"sampled {len(tids)} trajs: {tids}")
    print(f"objective(s): {sorted(set(objs))}")
    print(f"achieved mask rate: {achieved:.4f} (target {args.mask_rate})")
    print("-" * 72)

    for bi in range(len(tids)):
        inp, bt, val_, sc, lab = I[bi], B[bi], V[bi], S[bi], Lb[bi]
        goal_pos = bt == GOAL
        nonscored_valid = val_ & ~sc
        unexpected = nonscored_valid & (inp != lab) & (inp != MASK_ID)
        hidden_cnt = int((nonscored_valid & (inp == MASK_ID)).sum())
        ctx_cnt    = int((nonscored_valid & (inp == lab)).sum())
        print(f"[ex{bi}] valid={int(val_.sum())} scored={int(sc.sum())} "
              f"goal_scored={bool((sc & goal_pos).any())} "
              f"pad_masked={bool((inp==MASK_ID)[~val_].any())} "
              f"scored_are_mask={bool((inp[sc]==MASK_ID).all())} "
              f"ctx={ctx_cnt} hidden={hidden_cnt} unexpected={int(unexpected.sum())}")
        assert not (sc & goal_pos).any(),     "Goal tokens were scored!"
        assert not (inp == MASK_ID)[~val_].any(), "Padding was masked!"
        assert (inp[sc] == MASK_ID).all(),    "Scored tokens must be MASK!"
        assert unexpected.sum() == 0,         "Non-scored valid tokens must be ctx or hidden!"

    print("=" * 72)
    print(f"VERIFYING {args.objective_set} objective:")

    if args.objective_set == "d_ar_section":
        n_ok = 0
        for ex in train[:30]:
            ctx_v, tgt_v = all_action_mask(ex)
            # All action tokens should be targets
            all_act_tgt = ((ex.btype == ACTION) == tgt_v).all()
            # No non-action token should be target
            no_nonact_tgt = not tgt_v[ex.btype != ACTION].any()
            # Check context = all non-action
            n_act = int((ex.btype == ACTION).sum())
            n_tgt = int(tgt_v.sum())
            rate_achieved = n_tgt / max(1, int((ex.btype != GOAL).sum()))
            if n_ok < 5:
                print(f"  {ex.traj_id}: all_act_tgt={all_act_tgt}  "
                      f"no_nonact_tgt={no_nonact_tgt}  "
                      f"action_tokens={n_act}  rate={rate_achieved:.3f}")
            assert all_act_tgt and no_nonact_tgt, "d_ar_section masking incorrect!"
            n_ok += 1
        print(f"  d_ar_section: {n_ok}/30 examples correct")

    if args.objective_set == "d_progressive":
        print("  Span sizes at key progress values:")
        for prog in [0.0, 0.25, 0.5, 0.75, 1.0]:
            eff = prog ** args.prog_exponent
            span = max(1, round(np.exp(np.log(args.prog_max_span) * eff)))
            step_approx = round(prog * max(1, args.steps - 1))
            print(f"    progress={prog:.2f}  step≈{step_approx}  span={span}")
        # Verify rate is consistent across progress values
        ex = train[0]
        rates = []
        for prog in [0.0, 0.5, 1.0]:
            ctx_v, tgt_v = progressive_span_mask(ex, nrng, args.mask_rate,
                                                   prog, args.prog_exponent, args.prog_max_span)
            r = tgt_v.sum() / max(1, (ex.btype != GOAL).sum())
            rates.append(float(r))
        print(f"  Rate at progress [0, 0.5, 1.0]: {[f'{r:.3f}' for r in rates]} "
              f"(target {args.mask_rate:.3f})")

    if args.objective_set == "d_flex":
        print("  inverse_dynamics checks:")
        id_rng = random.Random(42)
        n_ok = 0
        for ex in train[:20]:
            ctx_v, tgt_v = inverse_dynamics_mask(ex, id_rng, args.mask_rate)
            tgt_steps = {int(ex.step[i]) for i in np.where(tgt_v)[0]}
            for t_step in tgt_steps:
                think_at_t = np.where((ex.step == t_step) & (ex.btype == THINK))[0]
                if think_at_t.size:
                    hidden = not ctx_v[think_at_t].any() and not tgt_v[think_at_t].any()
                    future_obs = np.where((ex.step > t_step) & (ex.btype == OBS))[0]
                    if future_obs.size and n_ok < 3:
                        print(f"    step {t_step}: think_hidden={hidden}  "
                              f"future_obs_in_ctx={ctx_v[future_obs].all()}")
                    n_ok += 1
                    if n_ok >= 3: break
            if n_ok >= 3: break

    print("=" * 72)
    print("VERIFYING next_obs_mask:")
    n_valid = sum(1 for ex in train[:100] if next_obs_mask(ex) is not None)
    print(f"  succeeds on {n_valid}/100 train examples")
    for ex in train[:3]:
        result = next_obs_mask(ex)
        if result:
            ctx_v, tgt_v = result
            ft = int(np.where(tgt_v)[0][0]) if tgt_v.any() else -1
            lc = int(np.where(ctx_v)[0][-1]) if ctx_v.any() else -1
            print(f"  {ex.traj_id}: tgt={int(tgt_v.sum())} obs-tokens  "
                  f"causal={lc < ft}  goal_in_tgt={bool(tgt_v[ex.btype==GOAL].any())}")

    print("VERIFYING next_action_mask:")
    n_valid = sum(1 for ex in train[:100] if next_action_mask(ex) is not None)
    print(f"  succeeds on {n_valid}/100 train examples")
    for ex in train[:3]:
        result = next_action_mask(ex)
        if result:
            ctx_v, tgt_v = result
            ft = int(np.where(tgt_v)[0][0]) if tgt_v.any() else -1
            lc = int(np.where(ctx_v)[0][-1]) if ctx_v.any() else -1
            print(f"  {ex.traj_id}: tgt={int(tgt_v.sum())} action-tokens  "
                  f"causal={lc < ft}  bt_check={all(ex.btype[i]==ACTION for i in np.where(tgt_v)[0])}")

    print("=" * 72)
    print("ALL CHECKS PASSED")

    if args.show_samples > 0:
        print(f"\n{'═'*72}")
        print(f"MASKING SAMPLES (n={args.show_samples})")
        show_masked_samples(train, args, tok)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global MASK_ID, PAD_ID
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir",    default=os.path.expanduser("~/dflex_proj/llada2mini"))
    ap.add_argument("--data_dir",     default="data",
                    help="ScienceWorld data directory")
    ap.add_argument("--data_dir_tw",  default="data/textworld",
                    help="TextWorld (ALFWorld+WebShop) data. Download with fetch_textworld_data.py")
    ap.add_argument("--datasets",     default="scienceworld",
                    help="Comma-separated: scienceworld, textworld, or both")
    ap.add_argument("--out_dir",      default="runs/llada_rb")
    ap.add_argument("--objective_set", default="d_ar_section",
                    choices=["d_ar_section", "d_progressive", "random_block",
                             "d_ar", "d_ar_refined", "d_flex", "causal"])
    ap.add_argument("--mask_schedule", default="constant",
                    choices=["constant", "curriculum"])
    ap.add_argument("--mask_rate",       type=float, default=0.20)
    ap.add_argument("--mask_rate_start", type=float, default=0.10)
    ap.add_argument("--mask_rate_end",   type=float, default=0.60)
    # Progressive masking
    ap.add_argument("--prog_exponent",  type=float, default=1.0,
                    help="Growth speed for d_progressive: 1=fast, 2=medium, 3=slow")
    ap.add_argument("--prog_max_span",  type=int,   default=25,
                    help="Max span size for d_progressive (25 covers p75 of ScienceWorld blocks)")
    # Training
    ap.add_argument("--steps",       type=int,   default=800)
    ap.add_argument("--batch_size",  type=int,   default=1)
    ap.add_argument("--grad_accum",  type=int,   default=4)
    ap.add_argument("--max_len",     type=int,   default=8192)
    ap.add_argument("--lr",          type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--lora_r",       type=int,   default=16)
    ap.add_argument("--lora_alpha",   type=int,   default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    ap.add_argument("--lora_targets", default="query_key_value,dense")
    ap.add_argument("--ema_decay",    type=float, default=0.995)
    # Validation
    ap.add_argument("--val_every",   type=int,  default=20)
    ap.add_argument("--val_batches", type=int,  default=15)
    ap.add_argument("--full_val_at_end", action="store_true",
                    help="Sweep nextobs over ALL val trajectories after training")
    # Legacy / deprecated
    ap.add_argument("--probe_after_train", action="store_true",
                    help="(Deprecated — linear probing removed) No-op, kept for sbatch compat")
    # Misc
    ap.add_argument("--seed",          type=int,  default=0)
    ap.add_argument("--run_name",      default="rb")
    ap.add_argument("--wandb_mode",    default="offline",
                    choices=["online", "offline", "disabled"])
    ap.add_argument("--wandb_project", default="cp2107-dflex")
    # Verify / visualise
    ap.add_argument("--verify",        action="store_true")
    ap.add_argument("--show_samples",  type=int, default=0,
                    help="In verify mode: print N examples showing masking applied")
    args = ap.parse_args()

    if args.probe_after_train:
        print("[warn] --probe_after_train is deprecated (linear probing removed). Skipping.")

    random.seed(args.seed); np.random.seed(args.seed)
    if HAS_TORCH: torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    PAD_ID  = tok.pad_token_id if tok.pad_token_id is not None else 0
    MASK_ID = tok.mask_token_id
    if MASK_ID is None:
        raise SystemExit("tokenizer has no mask_token_id")

    train, val = load_datasets(args, tok)
    print(f"[data] total: train={len(train)} val={len(val)} | MASK={MASK_ID} PAD={PAD_ID}")

    if args.verify:
        verify(args, tok, train, val); return

    device = "cuda"
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    print("[load] LLaDA2.0-mini bf16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    model.config.use_cache = False
    model.config.pad_token_id = PAD_ID
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                      lora_dropout=args.lora_dropout, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=args.lora_targets.split(","))
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)

    ema = EMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    wb = None
    if args.wandb_mode != "disabled":
        try:
            import wandb
            wb = wandb.init(project=args.wandb_project, name=args.run_name,
                            mode=args.wandb_mode, config=vars(args))
        except Exception as e:
            print(f"[wandb] off ({e})")

    rng  = random.Random(args.seed)
    nrng = np.random.default_rng(args.seed)

    hist = {
        "step": [], "loss": [], "acc": [], "tok_f1": [], "mask_rate": [],
        "val_step": [], "val_loss": [], "val_acc": [], "val_tok_f1": [],
        "val_nextobs": [], "val_nextobs_acc": [],
        "val_nextaction": [], "val_nextaction_acc": [],
    }
    obj_loss = collections.defaultdict(lambda: [0.0, 0])
    budget   = collections.Counter()
    t0 = time.time()

    def rate_at(p):
        if args.objective_set == "d_ar_section":
            return 0.0  # rate not used for d_ar_section
        return args.mask_rate if args.mask_schedule == "constant" else \
            args.mask_rate_start + (args.mask_rate_end - args.mask_rate_start) * p

    def run_micro(arrs):
        inp, bt, val_, sc, lab = arrs
        inp_t = torch.from_numpy(inp).to(device)
        vb_t  = torch.from_numpy(val_).to(device)
        sc_t  = torch.from_numpy(sc).to(device)
        lab_t = torch.from_numpy(lab).to(device)
        B_, L_ = inp_t.shape
        m4 = vb_t[:, None, None, :].expand(B_, 1, L_, L_).to(torch.bfloat16)
        out    = model(input_ids=inp_t, attention_mask=m4)
        logits = out.logits.float()
        nll    = F.cross_entropy(logits.transpose(1, 2), lab_t, reduction="none")
        sel    = sc_t.float()
        loss   = (nll * sel).sum() / sel.sum().clamp(min=1.0)
        with torch.no_grad():
            preds = logits.detach().argmax(dim=-1)
            acc   = ((preds == lab_t) & sc_t).sum().float() / sc_t.sum().clamp(min=1).float()
            # Unigram F1: token-bag overlap at scored positions
            sc_flat   = sc_t.flatten()
            p_ids     = preds.flatten()[sc_flat].cpu()
            t_ids     = lab_t.flatten()[sc_flat].cpu()
            tok_f1    = _unigram_f1(p_ids, t_ids)
        return loss, float(acc.item()), float(tok_f1), sc_t, torch.from_numpy(bt).to(device)

    model.train()
    for step in range(args.steps):
        progress = step / max(1, args.steps - 1)
        rate     = rate_at(progress)

        I, B, V, S, Lb, objs, tids, achieved = make_step_batch(
            train, args.grad_accum * args.batch_size,
            args.objective_set, rng, nrng, args.max_len, rate,
            progress, args.prog_exponent, args.prog_max_span)

        opt.zero_grad()
        accloss = acccacc = acctf1 = 0.0
        for j in range(args.grad_accum):
            sl = slice(j * args.batch_size, (j + 1) * args.batch_size)
            loss, acc, tf1, sc_t, bt_t = run_micro((I[sl], B[sl], V[sl], S[sl], Lb[sl]))
            (loss / args.grad_accum).backward()
            accloss += float(loss.item()) / args.grad_accum
            acccacc += acc / args.grad_accum
            acctf1  += tf1 / args.grad_accum
            sbt = bt_t[sc_t]
            for kk in (GOAL, THINK, ACTION, OBS):
                budget[kk] += int((sbt == kk).sum().item())

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step()
        if ema: ema.update(model)

        for o in objs:
            obj_loss[o][0] += accloss / len(objs)
            obj_loss[o][1] += 1.0 / len(objs)

        hist["step"].append(step); hist["loss"].append(accloss)
        hist["acc"].append(acccacc); hist["tok_f1"].append(acctf1)
        hist["mask_rate"].append(achieved)
        if wb:
            wb.log({"train/loss": accloss, "train/acc": acccacc,
                    "train/tok_f1": acctf1, "train/mask_rate": achieved,
                    "lr": sched.get_last_lr()[0]}, step=step)

        if step % args.val_every == 0 or step == args.steps - 1:
            if ema: ema.apply_to(model)
            model.eval()
            with torch.no_grad():
                # 1. Training-task val
                vls, vaccs, vtf1s = [], [], []
                for _ in range(args.val_batches):
                    Iv, Bv, Vv, Sv, Lv, _, _, _ = make_step_batch(
                        val, args.batch_size, args.objective_set,
                        rng, nrng, args.max_len, rate,
                        progress, args.prog_exponent, args.prog_max_span)
                    l, a, f, _, _ = run_micro((Iv, Bv, Vv, Sv, Lv))
                    vls.append(float(l.item())); vaccs.append(a); vtf1s.append(f)
                vm = np.mean(vls); vm_acc = np.mean(vaccs); vm_f1 = np.mean(vtf1s)

                # 2. Shared nextobs eval
                no_ls, no_accs = [], []
                for _ in range(args.val_batches):
                    arrs = make_val_nextobs_batch(val, args.batch_size, rng, args.max_len)
                    if arrs is not None:
                        l, a, f, _, _ = run_micro(arrs)
                        no_ls.append(float(l.item())); no_accs.append(a)
                vm_no     = float(np.mean(no_ls))     if no_ls     else float("nan")
                vm_no_acc = float(np.mean(no_accs)) if no_accs else float("nan")

                # 3. Shared nextaction eval
                na_ls, na_accs = [], []
                for _ in range(args.val_batches):
                    arrs = make_val_nextaction_batch(val, args.batch_size, rng, args.max_len)
                    if arrs is not None:
                        l, a, f, _, _ = run_micro(arrs)
                        na_ls.append(float(l.item())); na_accs.append(a)
                vm_na     = float(np.mean(na_ls))     if na_ls     else float("nan")
                vm_na_acc = float(np.mean(na_accs)) if na_accs else float("nan")

            if ema: ema.restore_from(model)

            hist["val_step"].append(step)
            hist["val_loss"].append(float(vm))
            hist["val_acc"].append(float(vm_acc))
            hist["val_tok_f1"].append(float(vm_f1))
            hist["val_nextobs"].append(vm_no);       hist["val_nextobs_acc"].append(vm_no_acc)
            hist["val_nextaction"].append(vm_na);    hist["val_nextaction_acc"].append(vm_na_acc)

            if wb:
                wb.log({"val/loss": float(vm), "val/acc": float(vm_acc),
                        "val/tok_f1": float(vm_f1),
                        "val/nextobs_loss": vm_no, "val/nextobs_acc": vm_no_acc,
                        "val/nextaction_loss": vm_na, "val/nextaction_acc": vm_na_acc},
                       step=step)

            eff_prog = progress ** args.prog_exponent
            span_now = max(1, round(np.exp(np.log(args.prog_max_span) * eff_prog))) \
                       if args.objective_set == "d_progressive" else 0
            span_str = f" span={span_now}" if span_now else ""
            print(f"[{step:4d}/{args.steps}] "
                  f"train={accloss:.3f} acc={acccacc:.3f} f1={acctf1:.3f} "
                  f"val={float(vm):.3f} va={float(vm_acc):.3f} "
                  f"nextobs={vm_no:.3f} nextact={vm_na:.3f} "
                  f"rate={achieved:.3f}{span_str} "
                  f"({time.time()-t0:.0f}s)")

            model.save_pretrained(os.path.join(args.out_dir, f"checkpoint_{step}"))
            model.train()

    model.save_pretrained(os.path.join(args.out_dir, "lora_adapter"))

    # ── post-training ─────────────────────────────────────────────────────────
    sweep_metrics = {}
    if args.full_val_at_end:
        print("[sweep] nextobs over all val trajectories ...")
        if ema: ema.apply_to(model)
        model.eval()
        with torch.no_grad():
            mn, sd, mn_acc, mn_f1, n = full_val_nextobs_sweep(run_micro, val, args.max_len)
        if ema: ema.restore_from(model)
        sweep_metrics = {"sweep/nextobs_mean": mn, "sweep/nextobs_std": sd,
                         "sweep/nextobs_acc": mn_acc, "sweep/nextobs_f1": mn_f1,
                         "sweep/n": n}
        print(f"[sweep] mean={mn:.4f} std={sd:.4f} acc={mn_acc:.4f} f1={mn_f1:.4f} n={n}")
        if wb:
            wb.log(sweep_metrics); wb.summary.update(sweep_metrics)

    valid_nextobs = [v for v in hist["val_nextobs"] if not np.isnan(v)]
    valid_nextact = [v for v in hist["val_nextaction"] if not np.isnan(v)]
    metrics = {
        "final_train_loss":         hist["loss"][-1],
        "final_train_acc":          hist["acc"][-1],
        "final_train_tok_f1":       hist["tok_f1"][-1],
        "final_val_loss":           hist["val_loss"][-1]       if hist["val_loss"]    else None,
        "final_val_nextobs_loss":   hist["val_nextobs"][-1]    if hist["val_nextobs"] else None,
        "final_val_nextaction_loss":hist["val_nextaction"][-1] if hist["val_nextaction"] else None,
        "best_val_nextobs_loss":    float(min(valid_nextobs))  if valid_nextobs else None,
        "best_val_nextaction_loss": float(min(valid_nextact))  if valid_nextact else None,
        "mean_mask_rate":           float(np.mean(hist["mask_rate"])),
        "objective_set":            args.objective_set,
        "prog_exponent":            args.prog_exponent,
        "prog_max_span":            args.prog_max_span,
        "weight_decay":             args.weight_decay,
        "lora_dropout":             args.lora_dropout,
        "ema_decay":                args.ema_decay,
        "datasets":                 args.datasets,
        "supervision_budget":       {TYPE_NAME[k]: budget[k]
                                     for k in (GOAL, THINK, ACTION, OBS)},
        "minutes": round((time.time() - t0) / 60, 2),
        **sweep_metrics,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    make_charts(hist, dict(obj_loss), budget, os.path.join(args.out_dir, "charts"))
    if wb: wb.finish()
    print(f"[done] {json.dumps(metrics)}")


if __name__ == "__main__":
    main()
