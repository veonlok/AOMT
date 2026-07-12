#!/usr/bin/env python3
"""Controlled LoRA fine-tune of LLaDA2.0-mini.

Objective sets:
  random_block  — D-Random: random token masking, full bidirectional context
  d_ar          — D-AR (original): suffix masking, causal context
  d_ar_refined  — D-AR (refined): uniformly-sampled span position, causal context
  d_flex        — 4 bidirectional objectives (imputation/retrodiction/obs_denoising/inverse_dynamics)
  d_progressive — curriculum: starts token-level random masking, grows to block-level over training

Shared evaluation: val/nextobs_loss — predict last obs block from causal prefix, same for all models.
Full sweep at end: nextobs over every val trajectory (not sampled batches) for a definitive estimate.
Linear probing: block-type classification + step-number regression on frozen representations.
Includes --verify mode for mandatory correctness checking before launch."""
from __future__ import annotations
import argparse, json, os, random, time, collections
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
TYPE_NAME = {GOAL: "goal", THINK: "think", ACTION: "action", OBS: "obs"}
CAUSAL_OBJS = ("next_action", "next_observation", "future_from_past",
               "full_completion", "goal_conditioned_action")
# D-Flex: 4 objectives requiring bidirectional attention (impossible for causal models without leakage)
D_FLEX_OBJS = ("intermediate_imputation", "retrodiction", "obs_denoising", "inverse_dynamics")
MASK_ID = None
PAD_ID  = 0

@dataclass
class Example:
    ids: np.ndarray; btype: np.ndarray; step: np.ndarray; bidx: np.ndarray; traj_id: str

# ── data loading ──────────────────────────────────────────────────────────────

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
    if not ids:
        return None
    if len(ids) > max_len:
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

# ── masking primitives ────────────────────────────────────────────────────────

def random_block_mask(ex, nrng, rate):
    """D-Random: mask rate% of non-goal tokens uniformly at random."""
    L = len(ex.ids)
    elig = ex.btype != GOAL
    tgt = elig & (nrng.random(L) < rate)
    if not tgt.any():
        idx = np.where(elig)[0]
        if idx.size:
            tgt[int(nrng.choice(idx))] = True
    return ~tgt, tgt

def _steps_present(ex):
    return sorted({int(s) for s in ex.step.tolist() if s >= 1})

def _first_idx(ex, step_val, bt):
    idx = np.where((ex.step == step_val) & (ex.btype == bt))[0]
    return int(idx[0]) if idx.size else None

def causal_mask(obj, ex, rng, rate):
    L = len(ex.ids); steps = _steps_present(ex); goal = ex.btype == GOAL
    def fb():
        return random_block_mask(ex, np.random.default_rng(rng.randint(0, 1 << 30)), rate)
    if not steps: return fb()
    if obj == "next_action":
        cand = [s for s in steps if _first_idx(ex, s, ACTION) is not None]
        if not cand: return fb()
        t = rng.choice(cand); st = _first_idx(ex, t, ACTION)
        tgt = (ex.step == t) & (ex.btype == ACTION); ctx = (np.arange(L) < st) & ~tgt
    elif obj == "next_observation":
        cand = [s for s in steps if _first_idx(ex, s, OBS) is not None]
        if not cand: return fb()
        t = rng.choice(cand); st = _first_idx(ex, t, OBS)
        tgt = (ex.step == t) & (ex.btype == OBS); ctx = (np.arange(L) < st) & ~tgt
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

def suffix_mask(ex, rate):
    """D-AR (original): mask the last (rate × eligible) non-goal tokens; context = causal prefix."""
    L = len(ex.ids)
    elig_idx = np.where(ex.btype != GOAL)[0]
    if len(elig_idx) == 0:
        return np.ones(L, bool), np.zeros(L, bool)
    n_mask = max(1, min(int(round(rate * len(elig_idx))), len(elig_idx)))
    tgt = np.zeros(L, bool)
    tgt[elig_idx[-n_mask:]] = True
    return ~tgt, tgt

def causal_random_mask(ex, rng, rate):
    """D-AR (refined): mask a contiguous span of (rate × eligible) tokens at a UNIFORMLY SAMPLED
    position, not always the suffix. Context is strictly causal (prefix before the span start);
    tokens after the span are hidden (masked, not scored) so no future information leaks.

    Fixes two biases in suffix_mask:
    1. Positional bias: suffix_mask only ever supervises the end of trajectories.
    2. Context-length variance: short sequences get proportionally less causal context.
    With causal_random_mask every token position has equal probability of being a target."""
    L = len(ex.ids)
    elig_idx = np.where(ex.btype != GOAL)[0]
    if len(elig_idx) == 0:
        return np.ones(L, bool), np.zeros(L, bool)
    n_mask = max(1, min(int(round(rate * len(elig_idx))), len(elig_idx)))
    max_start = max(0, len(elig_idx) - n_mask)
    start_i = rng.randint(0, max_start + 1)
    tgt = np.zeros(L, bool)
    tgt[elig_idx[start_i:start_i + n_mask]] = True
    first_tgt = int(np.where(tgt)[0][0])
    ctx = np.zeros(L, bool)
    ctx[:first_tgt] = True   # strictly causal prefix only
    ctx &= ~tgt
    return ctx, tgt

def imputation_mask(ex, nrng, rate):
    """D-Flex: mask contiguous middle span; both sides are context.
    Requires bidirectional attention."""
    L = len(ex.ids)
    elig_idx = np.where(ex.btype != GOAL)[0]
    if len(elig_idx) == 0:
        return np.ones(L, bool), np.zeros(L, bool)
    n_mask = max(1, min(int(round(rate * len(elig_idx))), len(elig_idx)))
    lo = max(0, int(0.10 * len(elig_idx)))
    hi = min(int(0.70 * len(elig_idx)), max(lo, len(elig_idx) - n_mask))
    if lo > hi:
        lo = 0; hi = max(0, len(elig_idx) - n_mask)
    start_i = int(nrng.integers(lo, max(lo + 1, hi + 1)))
    tgt = np.zeros(L, bool)
    tgt[elig_idx[start_i:start_i + n_mask]] = True
    return ~tgt, tgt

def retrodiction_mask(ex, rate):
    """D-Flex: mask first (rate × eligible) tokens; context = later trajectory + goal."""
    L = len(ex.ids)
    elig_idx = np.where(ex.btype != GOAL)[0]
    if len(elig_idx) == 0:
        return np.ones(L, bool), np.zeros(L, bool)
    n_mask = max(1, min(int(round(rate * len(elig_idx))), len(elig_idx)))
    tgt = np.zeros(L, bool)
    tgt[elig_idx[:n_mask]] = True
    return ~tgt, tgt

def obs_denoising_mask(ex, nrng, rate):
    """D-Flex: randomly mask obs tokens (up to rate% of all eligible) from surrounding context."""
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
    """D-Flex: given obs context BOTH before AND after, predict the action that caused the transition.
    CONTEXT: goal + ALL obs tokens (bidirectional, includes future obs) + other steps' think/action.
    HIDDEN:  think at same step as target action (prevents leakage — think narrates the action).
    TARGET:  all action tokens at the chosen step.
    Requires bidirectional attention."""
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
    ctx |= ((ex.btype == THINK) & (ex.step != t))
    ctx |= ((ex.btype == ACTION) & (ex.step != t))
    ctx &= ~tgt
    return ctx, tgt

def progressive_span_mask(ex, nrng, rate, progress):
    """D-Progressive: start at token-level (span=1), grow exponentially to block-level (span→∞).
    At progress=0: identical to random_block_mask (individual token masking).
    At progress=1: masks contiguous spans of ~max_span tokens (approaching whole-block masking).
    Maintains exactly rate% eligible tokens masked at all stages.

    This curriculum trains the model on easy individual tokens first, then gradually
    shifts to harder span-level prediction that requires richer contextual reasoning."""
    L = len(ex.ids)
    elig_idx = np.where(ex.btype != GOAL)[0]
    if len(elig_idx) == 0:
        return np.ones(L, bool), np.zeros(L, bool)
    n_mask_target = max(1, min(int(round(rate * len(elig_idx))), len(elig_idx)))

    # Span size grows exponentially: 1 at progress=0 → max_span at progress=1
    max_span = 25
    span_size = max(1, round(np.exp(np.log(max_span) * progress)))

    tgt = np.zeros(L, bool)
    if span_size == 1:
        # Token-level: same as random_block_mask
        draw = nrng.random(L) < rate
        tgt = (ex.btype != GOAL) & draw
        if not tgt.any():
            tgt[elig_idx[int(nrng.integers(0, len(elig_idx)))]] = True
        # Trim if overshot
        true_idx = np.where(tgt)[0]
        if len(true_idx) > n_mask_target:
            tgt[true_idx[n_mask_target:]] = False
    else:
        n_spans = max(1, round(n_mask_target / span_size))
        pool = elig_idx.copy(); nrng.shuffle(pool)
        for start in pool:
            for offset in range(span_size):
                pos = int(start) + offset
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

def next_obs_mask(ex):
    """Shared evaluation metric — identical for D-AR, D-Flex, and D-Random.
    Task: predict the LAST observation block from its strict causal prefix.
    Lower val_nextobs_loss = better implicit world model."""
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

# ── batch construction ────────────────────────────────────────────────────────

def corrupt(ex, ctx, tgt, max_len):
    L = min(len(ex.ids), max_len); ctx = ctx[:L]; tgt = tgt[:L]
    ids   = np.full(max_len, PAD_ID, np.int64)
    btype = np.full(max_len, PADB,   np.int64)
    valid = np.zeros(max_len, bool)
    ids[:L] = ex.ids[:L]; btype[:L] = ex.btype[:L]; valid[:L] = True
    inp = ids.copy()
    tmask = np.zeros(max_len, bool); tmask[:L] = tgt
    hmask = np.zeros(max_len, bool); hmask[:L] = (~ctx) & (~tgt)
    inp[tmask | hmask] = MASK_ID
    return inp, btype, valid, tmask, ids

def sample_one(obj_set, ex, rng, nrng, rate, progress=0.0):
    if obj_set == "random_block":
        return "random_block", random_block_mask(ex, nrng, rate)
    if obj_set == "d_ar":
        return "suffix", suffix_mask(ex, rate)
    if obj_set == "d_ar_refined":
        return "causal_random", causal_random_mask(ex, rng, rate)
    if obj_set == "d_progressive":
        return "progressive_span", progressive_span_mask(ex, nrng, rate, progress)
    if obj_set == "d_flex":
        obj = rng.choice(D_FLEX_OBJS)
        if obj == "intermediate_imputation":
            return obj, imputation_mask(ex, nrng, rate)
        if obj == "retrodiction":
            return obj, retrodiction_mask(ex, rate)
        if obj == "obs_denoising":
            return obj, obs_denoising_mask(ex, nrng, rate)
        return obj, inverse_dynamics_mask(ex, rng, rate)
    obj = rng.choice(CAUSAL_OBJS)
    return obj, causal_mask(obj, ex, rng, rate)

def make_step_batch(train, k, obj_set, rng, nrng, max_len, rate, progress=0.0):
    idxs = rng.sample(range(len(train)), min(k, len(train)))
    prepared = []
    for i in idxs:
        ex = train[i]
        obj, (ctx, tgt) = sample_one(obj_set, ex, rng, nrng, rate, progress)
        prepared.append((ex, obj, ctx, tgt))
    # Dynamic padding: pad to longest in this batch, not global max_len
    pad_to = min(max(len(ex.ids) for ex, _, _, _ in prepared), max_len)
    I, B, V, S, Lb, objs, tids = [], [], [], [], [], [], []
    elig_tot = scored_tot = 0
    for ex, obj, ctx, tgt in prepared:
        inp, bt, val, sc, lab = corrupt(ex, ctx, tgt, pad_to)
        I.append(inp); B.append(bt); V.append(val); S.append(sc); Lb.append(lab)
        objs.append(obj); tids.append(ex.traj_id)
        elig_tot  += int(((bt == THINK) | (bt == ACTION) | (bt == OBS)).sum())
        scored_tot += int(sc.sum())
    achieved = scored_tot / max(1, elig_tot)
    return (np.stack(I), np.stack(B), np.stack(V), np.stack(S), np.stack(Lb),
            objs, tids, achieved)

def make_val_nextobs_batch(val_data, k, rng, max_len):
    """Shared forward-dynamics eval batch. Oversamples 3× to handle None returns."""
    idxs = rng.sample(range(len(val_data)), min(k * 3, len(val_data)))
    prepared = []
    for i in idxs:
        result = next_obs_mask(val_data[i])
        if result is not None:
            prepared.append((val_data[i], result[0], result[1]))
        if len(prepared) >= k:
            break
    if not prepared:
        return None
    pad_to = min(max(len(ex.ids) for ex, _, _ in prepared), max_len)
    I, B, V_, S, Lb = [], [], [], [], []
    for ex, ctx, tgt in prepared:
        inp, bt, vld, sc, lab = corrupt(ex, ctx, tgt, pad_to)
        I.append(inp); B.append(bt); V_.append(vld); S.append(sc); Lb.append(lab)
    return np.stack(I), np.stack(B), np.stack(V_), np.stack(S), np.stack(Lb)

def full_val_nextobs_sweep(run_micro_fn, val_data, max_len):
    """Evaluate nextobs on EVERY val trajectory — definitive low-variance estimate.
    Returns (mean_loss, std_loss, mean_acc, n)."""
    losses, accs = [], []
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
        loss, acc, _, _ = run_micro_fn(arrs)
        losses.append(float(loss.item())); accs.append(acc)
    if not losses:
        return float("nan"), float("nan"), float("nan"), 0
    return float(np.mean(losses)), float(np.std(losses)), float(np.mean(accs)), len(losses)

# ── EMA ───────────────────────────────────────────────────────────────────────

class EMA:
    """Exponential moving average of LoRA weights for stable evaluation."""
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
        """Copy EMA weights into model (call restore_from to undo)."""
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

# ── linear probing ────────────────────────────────────────────────────────────

def run_linear_probe(model, val_data, device, max_len, wb):
    """Fit linear probes on frozen representations extracted from the last transformer layer.
    Probe 1: block-type classification (THINK / ACTION / OBS) — measures structural encoding.
    Probe 2: step-number regression (predict trajectory step from token hidden state) — measures
             temporal position encoding.
    Results logged to wandb under probe/."""
    try:
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score
    except ImportError:
        print("[probe] sklearn not available — skipping linear probing"); return {}

    print("[probe] extracting hidden states from val set ...")
    model.eval()
    all_h, all_bt, all_step_lbl = [], [], []

    with torch.no_grad():
        for ex in val_data:
            pad_to = min(len(ex.ids), max_len)
            ids_a = np.full(pad_to, PAD_ID, np.int64)
            vld_a = np.zeros(pad_to, bool)
            L = min(len(ex.ids), pad_to)
            ids_a[:L] = ex.ids[:L]; vld_a[:L] = True

            inp_t = torch.from_numpy(ids_a).unsqueeze(0).to(device)
            vb_t  = torch.from_numpy(vld_a).unsqueeze(0).to(device)
            m4 = vb_t[:, None, None, :].expand(1, 1, pad_to, pad_to).to(torch.bfloat16)
            try:
                out = model(input_ids=inp_t, attention_mask=m4, output_hidden_states=True)
                h = out.hidden_states[-1][0, :L].float().cpu().numpy()  # (L, D)
            except Exception as e:
                print(f"[probe] output_hidden_states failed: {e}"); return {}

            all_h.append(h)
            all_bt.append(ex.btype[:L])
            all_step_lbl.append(ex.step[:L])

    H      = np.concatenate(all_h, axis=0)
    bt_lbl = np.concatenate(all_bt, axis=0)
    s_lbl  = np.concatenate(all_step_lbl, axis=0)

    scaler = StandardScaler()
    X = scaler.fit_transform(H)
    mid = len(X) // 2
    results = {}

    # Probe 1: block type (THINK / ACTION / OBS — exclude GOAL)
    mask1 = np.isin(bt_lbl, [THINK, ACTION, OBS])
    if mask1.sum() > 200:
        Xc = X[mask1]; yc = bt_lbl[mask1]
        clf = LogisticRegression(max_iter=300, C=1.0, n_jobs=4)
        clf.fit(Xc[:len(Xc)//2], yc[:len(yc)//2])
        acc1 = accuracy_score(yc[len(yc)//2:], clf.predict(Xc[len(Xc)//2:]))
        results["probe/block_type_acc"] = float(acc1)
        print(f"[probe] block_type_acc={acc1:.4f}  "
              f"(chance={1/3:.3f}, strong=0.90+)")

    # Probe 2: step number regression
    mask2 = s_lbl >= 1
    if mask2.sum() > 200:
        Xr = X[mask2]; yr = s_lbl[mask2].astype(float)
        reg = Ridge(alpha=1.0)
        reg.fit(Xr[:len(Xr)//2], yr[:len(yr)//2])
        yhat = reg.predict(Xr[len(Xr)//2:])
        yr_test = yr[len(yr)//2:]
        ss_res = np.sum((yr_test - yhat) ** 2)
        ss_tot = np.sum((yr_test - yr_test.mean()) ** 2)
        r2  = float(1 - ss_res / max(ss_tot, 1e-12))
        mae = float(np.mean(np.abs(yr_test - yhat)))
        results["probe/step_r2"]  = r2
        results["probe/step_mae"] = mae
        print(f"[probe] step_r2={r2:.4f}  step_mae={mae:.2f} steps")

    if wb and results:
        wb.log(results)
        wb.summary.update(results)
    return results

# ── charts ────────────────────────────────────────────────────────────────────

def make_charts(hist, obj_loss, budget, out_dir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    def smooth(y, k=15):
        if len(y) < k: return np.array(y)
        c = np.cumsum(np.insert(y, 0, 0)); return (c[k:] - c[:-k]) / k

    # Loss + nextobs curve
    plt.figure(figsize=(8, 5))
    plt.plot(hist["step"], hist["loss"], alpha=0.25, color="tab:blue", label="train/loss (raw)")
    sm = smooth(np.array(hist["loss"]))
    plt.plot(hist["step"][len(hist["step"]) - len(sm):], sm, color="tab:blue", label="train/loss (smoothed)")
    if hist["val_step"]:
        plt.plot(hist["val_step"], hist["val_loss"], "o-", color="tab:red", label="val/loss (training task)")
        no = hist.get("val_nextobs", [])
        if no and any(not np.isnan(v) for v in no):
            plt.plot(hist["val_step"], no, "s--", color="tab:green", label="val/nextobs (shared)")
    plt.xlabel("step"); plt.ylabel("cross-entropy"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out_dir}/loss_curve.png", dpi=130); plt.close()

    # Accuracy curve
    if hist.get("val_acc") or hist.get("acc"):
        plt.figure(figsize=(8, 4))
        if hist.get("acc"):
            sm_acc = smooth(np.array(hist["acc"]))
            plt.plot(hist["step"][len(hist["step"]) - len(sm_acc):], sm_acc,
                     color="tab:blue", alpha=0.7, label="train/acc (smoothed)")
        if hist.get("val_step") and hist.get("val_acc"):
            plt.plot(hist["val_step"], hist["val_acc"], "o-", color="tab:red", label="val/acc (task)")
        if hist.get("val_nextobs_acc") and any(not np.isnan(v) for v in hist.get("val_nextobs_acc", [])):
            plt.plot(hist["val_step"], hist["val_nextobs_acc"], "s--", color="tab:green",
                     label="val/nextobs_acc (shared)")
        plt.ylim(0, 1); plt.xlabel("step"); plt.ylabel("accuracy (masked tokens)")
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(f"{out_dir}/accuracy_curve.png", dpi=130); plt.close()

    # Mask rate
    plt.figure(figsize=(7, 4))
    plt.plot(hist["step"], hist["mask_rate"], color="tab:orange")
    plt.ylim(0, 1); plt.xlabel("step"); plt.ylabel("achieved mask rate")
    plt.title("Masking rate"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out_dir}/mask_rate.png", dpi=130); plt.close()

    # Per-objective loss
    if any(obj_loss[k][1] for k in obj_loss):
        names = [k for k in obj_loss if obj_loss[k][1] > 0]
        vals  = [obj_loss[k][0] / obj_loss[k][1] for k in names]
        plt.figure(figsize=(7, 4)); plt.bar(range(len(names)), vals, color="tab:purple")
        plt.xticks(range(len(names)), names, rotation=40, ha="right")
        plt.ylabel("avg loss"); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
        plt.savefig(f"{out_dir}/loss_by_objective.png", dpi=130); plt.close()

    # Supervision budget
    bt_names = [TYPE_NAME[k] for k in (GOAL, THINK, ACTION, OBS)]
    bv = [budget.get(k, 0) for k in (GOAL, THINK, ACTION, OBS)]
    plt.figure(figsize=(6, 4)); plt.bar(bt_names, bv, color="tab:gray")
    plt.ylabel("scored tokens"); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(f"{out_dir}/supervision_budget.png", dpi=130); plt.close()

# ── verify mode ───────────────────────────────────────────────────────────────

def verify(args, tok, train, val):
    rng = random.Random(args.seed); nrng = np.random.default_rng(args.seed)
    print("=" * 70)
    print(f"MASK_ID={MASK_ID}  PAD_ID={PAD_ID}  -> distinct? {MASK_ID != PAD_ID}")
    assert MASK_ID != PAD_ID, "MASK and PAD must differ!"

    k = args.grad_accum * args.batch_size
    I, B, V, S, Lb, objs, tids, achieved = make_step_batch(
        train, k, args.objective_set, rng, nrng, args.max_len, args.mask_rate)
    print(f"sampled {len(tids)} trajectories: {tids}")
    print(f"all distinct? {len(set(tids)) == len(tids)}")
    print(f"objective(s): {sorted(set(objs))}")
    print(f"achieved mask rate: {achieved:.3f} (target {args.mask_rate})")

    inp, bt, val_, sc, lab = I[0], B[0], V[0], S[0], Lb[0]
    goal_pos = bt == GOAL
    print("-" * 70)
    print(f"[ex0] valid={int(val_.sum())} goal_tokens={int(goal_pos.sum())} scored={int(sc.sum())}")
    print(f"  goal ever scored?          {bool((sc & goal_pos).any())}  (must be False)")
    print(f"  any pad masked?            {bool((inp == MASK_ID)[~val_].any())}  (must be False)")
    print(f"  any pad scored?            {bool(sc[~val_].any())}  (must be False)")
    print(f"  scored positions are MASK? {bool((inp[sc] == MASK_ID).all())}  (must be True)")
    nonscored_valid = val_ & ~sc
    unexpected = nonscored_valid & (inp != lab) & (inp != MASK_ID)
    hidden_count = int((nonscored_valid & (inp == MASK_ID)).sum())
    ctx_count    = int((nonscored_valid & (inp == lab)).sum())
    print(f"  non-scored valid = ctx({ctx_count}) + hidden({hidden_count}), "
          f"unexpected={int(unexpected.sum())}  (unexpected must be 0)")
    first = np.where(val_)[0][:60]
    shown = ["[MASK]" if sc[p] else tok.decode([int(inp[p])]) for p in first]
    print("-" * 70); print(" ".join(shown))

    if args.objective_set == "d_flex":
        print("=" * 70); print("VERIFYING inverse_dynamics_mask:")
        id_rng = random.Random(42); n_ok = 0
        for ex in train[:20]:
            ctx_v, tgt_v = inverse_dynamics_mask(ex, id_rng, args.mask_rate)
            tgt_steps = set(int(ex.step[i]) for i in np.where(tgt_v)[0])
            for t_step in tgt_steps:
                think_at_t = np.where((ex.step == t_step) & (ex.btype == THINK))[0]
                if think_at_t.size:
                    hidden_ok = not ctx_v[think_at_t].any() and not tgt_v[think_at_t].any()
                    print(f"  step {t_step}: think hidden? {hidden_ok}  (must be True)")
                if tgt_steps:
                    t = min(tgt_steps)
                    future_obs = np.where((ex.step > t) & (ex.btype == OBS))[0]
                    if future_obs.size:
                        print(f"  future obs in ctx (bidirectional)? {ctx_v[future_obs].all()}  (must be True)")
                        n_ok += 1
                        if n_ok >= 3: break
            if n_ok >= 3: break

    if args.objective_set in ("d_ar_refined",):
        print("=" * 70); print("VERIFYING causal_random_mask:")
        cr_rng = random.Random(42)
        for ex in train[:5]:
            ctx_v, tgt_v = causal_random_mask(ex, cr_rng, args.mask_rate)
            if ctx_v.any() and tgt_v.any():
                last_ctx  = int(np.where(ctx_v)[0][-1])
                first_tgt = int(np.where(tgt_v)[0][0])
                print(f"  {ex.traj_id}: last_ctx={last_ctx} first_tgt={first_tgt} "
                      f"causal={last_ctx < first_tgt} (must be True)  "
                      f"rate={tgt_v.sum()/max(1,(ex.btype!=GOAL).sum()):.3f}")

    print("=" * 70); print("VERIFYING next_obs_mask:")
    n_valid = sum(1 for ex in train[:100] if next_obs_mask(ex) is not None)
    print(f"  succeeds on {n_valid}/100 train examples")
    for ex in train[:4]:
        result = next_obs_mask(ex)
        if result is None: continue
        ctx_v, tgt_v = result
        first_tgt = int(np.where(tgt_v)[0][0]) if tgt_v.any() else -1
        last_ctx  = int(np.where(ctx_v)[0][-1]) if ctx_v.any() else -1
        print(f"  {ex.traj_id}: tgt={int(tgt_v.sum())} obs-tokens  "
              f"causal={last_ctx < first_tgt}  goal_leak={(tgt_v & (ex.btype==GOAL)).any()}")
    print("=" * 70); print("ALL CHECKS DONE")

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global MASK_ID, PAD_ID
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir",    default=os.path.expanduser("~/dflex_proj/llada2mini"))
    ap.add_argument("--data_dir",     default="data")
    ap.add_argument("--out_dir",      default="runs/llada_rb")
    ap.add_argument("--objective_set", default="random_block",
                    choices=["random_block", "causal", "d_ar", "d_ar_refined",
                             "d_flex", "d_progressive"])
    ap.add_argument("--mask_schedule", default="constant", choices=["constant", "curriculum"])
    ap.add_argument("--mask_rate",       type=float, default=0.20)
    ap.add_argument("--mask_rate_start", type=float, default=0.10)
    ap.add_argument("--mask_rate_end",   type=float, default=0.60)
    ap.add_argument("--steps",      type=int,   default=400)
    ap.add_argument("--batch_size", type=int,   default=1)
    ap.add_argument("--grad_accum", type=int,   default=4)
    ap.add_argument("--max_len",    type=int,   default=8192)
    ap.add_argument("--lr",         type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--lora_r",      type=int,   default=16)
    ap.add_argument("--lora_alpha",  type=int,   default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    ap.add_argument("--lora_targets", default="query_key_value,dense")
    ap.add_argument("--ema_decay",   type=float, default=0.995,
                    help="EMA decay for val evaluation (0 = disable EMA)")
    ap.add_argument("--val_every",   type=int, default=20)
    ap.add_argument("--val_batches", type=int, default=15)
    ap.add_argument("--full_val_at_end", action="store_true",
                    help="Run nextobs over all val trajectories after training (definitive estimate)")
    ap.add_argument("--probe_after_train", action="store_true",
                    help="Run linear probing on frozen representations after training")
    ap.add_argument("--seed",         type=int,  default=0)
    ap.add_argument("--run_name",     default="rb")
    ap.add_argument("--wandb_mode",   default="offline",
                    choices=["online", "offline", "disabled"])
    ap.add_argument("--wandb_project", default="cp2107-dflex")
    ap.add_argument("--hf_repo",      default="AK2802/AOMT")
    ap.add_argument("--verify",       action="store_true")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    if HAS_TORCH: torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else 0
    MASK_ID = tok.mask_token_id
    if MASK_ID is None: raise SystemExit("tokenizer has no mask_token_id")

    train = load_split(os.path.join(args.data_dir, "train.jsonl"), tok, args.max_len)
    val   = load_split(os.path.join(args.data_dir, "validation.jsonl"), tok, args.max_len)
    print(f"[data] train={len(train)} val={len(val)} | MASK={MASK_ID} PAD={PAD_ID}")

    if args.verify:
        verify(args, tok, train, val); return

    device = "cuda"
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    print("[load] LLaDA2.0-mini bf16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    model.config.use_cache = False; model.config.pad_token_id = PAD_ID
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()

    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                      lora_dropout=args.lora_dropout,
                      bias="none", task_type="CAUSAL_LM",
                      target_modules=args.lora_targets.split(","))
    model = get_peft_model(model, lcfg); model.print_trainable_parameters()

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
    hist = {"step": [], "loss": [], "acc": [], "mask_rate": [],
            "val_step": [], "val_loss": [], "val_acc": [],
            "val_nextobs": [], "val_nextobs_acc": []}
    obj_loss = collections.defaultdict(lambda: [0.0, 0])
    budget   = collections.Counter()
    t0 = time.time()

    def rate_at(p):
        return args.mask_rate if args.mask_schedule == "constant" else \
            args.mask_rate_start + (args.mask_rate_end - args.mask_rate_start) * p

    def run_micro(arrs):
        inp, bt, val_, sc, lab = arrs
        inp_t  = torch.from_numpy(inp).to(device)
        vb_t   = torch.from_numpy(val_).to(device)
        sc_t   = torch.from_numpy(sc).to(device)
        lab_t  = torch.from_numpy(lab).to(device)
        B_, L_ = inp_t.shape
        m4 = vb_t[:, None, None, :].expand(B_, 1, L_, L_).to(torch.bfloat16)
        out    = model(input_ids=inp_t, attention_mask=m4)
        logits = out.logits.float()
        nll    = F.cross_entropy(logits.transpose(1, 2), lab_t, reduction="none")
        sel    = sc_t.float()
        loss   = (nll * sel).sum() / sel.sum().clamp(min=1.0)
        with torch.no_grad():
            preds = logits.detach().argmax(dim=-1)
            acc   = ((preds == lab_t) & sc_t).sum().float() / sc_t.sum().float().clamp(min=1)
        return loss, float(acc.item()), sc_t, torch.from_numpy(bt).to(device)

    model.train()
    for step in range(args.steps):
        progress = step / max(1, args.steps - 1)
        rate = rate_at(progress)
        I, B, V, S, Lb, objs, tids, achieved = make_step_batch(
            train, args.grad_accum * args.batch_size,
            args.objective_set, rng, nrng, args.max_len, rate, progress)

        opt.zero_grad(); accloss = 0.0; accacc = 0.0
        for j in range(args.grad_accum):
            sl = slice(j * args.batch_size, (j + 1) * args.batch_size)
            loss, acc, sc_t, bt_t = run_micro((I[sl], B[sl], V[sl], S[sl], Lb[sl]))
            (loss / args.grad_accum).backward()
            accloss += float(loss.item()) / args.grad_accum
            accacc  += acc / args.grad_accum
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
        hist["acc"].append(accacc); hist["mask_rate"].append(achieved)
        if wb:
            wb.log({"train/loss": accloss, "train/acc": accacc,
                    "train/mask_rate": achieved, "lr": sched.get_last_lr()[0]}, step=step)

        if step % args.val_every == 0 or step == args.steps - 1:
            # Apply EMA weights for val evaluation
            if ema: ema.apply_to(model)
            model.eval()
            with torch.no_grad():
                # 1. Training-task val
                vls, vaccs = [], []
                for _ in range(args.val_batches):
                    Iv, Bv, Vv, Sv, Lv, _, _, _ = make_step_batch(
                        val, args.batch_size, args.objective_set,
                        rng, nrng, args.max_len, rate, progress)
                    l, a, _, _ = run_micro((Iv, Bv, Vv, Sv, Lv))
                    vls.append(float(l.item())); vaccs.append(a)
                vm = float(np.mean(vls)); vm_acc = float(np.mean(vaccs))

                # 2. Shared nextobs eval
                nextobs_ls, nextobs_accs = [], []
                for _ in range(args.val_batches):
                    arrs = make_val_nextobs_batch(val, args.batch_size, rng, args.max_len)
                    if arrs is not None:
                        l, a, _, _ = run_micro(arrs)
                        nextobs_ls.append(float(l.item())); nextobs_accs.append(a)
                vm_nextobs     = float(np.mean(nextobs_ls))     if nextobs_ls     else float("nan")
                vm_nextobs_acc = float(np.mean(nextobs_accs)) if nextobs_accs else float("nan")

            if ema: ema.restore_from(model)
            hist["val_step"].append(step)
            hist["val_loss"].append(vm);     hist["val_acc"].append(vm_acc)
            hist["val_nextobs"].append(vm_nextobs)
            hist["val_nextobs_acc"].append(vm_nextobs_acc)
            if wb:
                wb.log({"val/loss": vm, "val/acc": vm_acc,
                        "val/nextobs_loss": vm_nextobs,
                        "val/nextobs_acc": vm_nextobs_acc}, step=step)
            print(f"[step {step:3d}/{args.steps}] "
                  f"train={accloss:.3f} acc={accacc:.3f} "
                  f"val={vm:.3f} val_acc={vm_acc:.3f} "
                  f"nextobs={vm_nextobs:.3f} nextobs_acc={vm_nextobs_acc:.3f} "
                  f"rate={achieved:.2f} ({time.time()-t0:.0f}s)")

            ckpt_dir = os.path.join(args.out_dir, f"checkpoint_{step}")
            model.save_pretrained(ckpt_dir)
            model.train()

    model.save_pretrained(os.path.join(args.out_dir, "lora_adapter"))

    # ── post-training evaluations ────────────────────────────────────────────
    sweep_metrics = {}
    if args.full_val_at_end:
        print("[sweep] running nextobs over all val trajectories ...")
        if ema: ema.apply_to(model)
        model.eval()
        with torch.no_grad():
            mn, sd, mn_acc, n = full_val_nextobs_sweep(run_micro, val, args.max_len)
        if ema: ema.restore_from(model)
        model.train()
        sweep_metrics = {"sweep/nextobs_mean": mn, "sweep/nextobs_std": sd,
                         "sweep/nextobs_acc": mn_acc, "sweep/n": n}
        print(f"[sweep] nextobs mean={mn:.4f} std={sd:.4f} acc={mn_acc:.4f} n={n}")
        if wb:
            wb.log(sweep_metrics)
            wb.summary.update(sweep_metrics)

    probe_metrics = {}
    if args.probe_after_train:
        if ema: ema.apply_to(model)
        probe_metrics = run_linear_probe(model, val, device, args.max_len, wb)
        if ema: ema.restore_from(model)
        model.train()

    valid_nextobs = [v for v in hist["val_nextobs"] if not np.isnan(v)]
    metrics = {
        "final_train_loss":      hist["loss"][-1],
        "final_train_acc":       hist["acc"][-1],
        "final_val_loss":        hist["val_loss"][-1] if hist["val_loss"] else None,
        "final_val_acc":         hist["val_acc"][-1]  if hist["val_acc"]  else None,
        "final_val_nextobs_loss": hist["val_nextobs"][-1] if hist["val_nextobs"] else None,
        "best_val_nextobs_loss":  float(min(valid_nextobs)) if valid_nextobs else None,
        "mean_mask_rate":        float(np.mean(hist["mask_rate"])),
        "objective_set":         args.objective_set,
        "schedule":              args.mask_schedule,
        "weight_decay":          args.weight_decay,
        "lora_dropout":          args.lora_dropout,
        "ema_decay":             args.ema_decay,
        "supervision_budget_by_type": {TYPE_NAME[k]: budget[k]
                                        for k in (GOAL, THINK, ACTION, OBS)},
        "minutes": round((time.time() - t0) / 60, 2),
        **sweep_metrics, **probe_metrics,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    make_charts(hist, dict(obj_loss), budget, os.path.join(args.out_dir, "charts"))
    if wb: wb.finish()
    print(f"[done] {json.dumps(metrics)}")

if __name__ == "__main__":
    main()
