#!/usr/bin/env python3
"""Controlled LoRA fine-tune of LLaDA2.0-mini.
Objective sets: random_block, causal (legacy), d_ar (suffix masking), d_flex (3 bidirectional objectives).
All objectives use a fixed rate% of non-goal tokens as the target for fair comparison.
Includes --verify mode for mandatory correctness checking before launch."""
from __future__ import annotations
import argparse, json, os, random, time, collections, re
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

PADB, GOAL, THINK, ACTION, OBS = 0, 1, 2, 3, 4
TYPE_FROM_STR = {"Goal": GOAL, "Think": THINK, "Action": ACTION, "Observation": OBS}
TYPE_NAME = {GOAL: "goal", THINK: "think", ACTION: "action", OBS: "obs"}
CAUSAL_OBJS = ("next_action", "next_observation", "future_from_past",
               "full_completion", "goal_conditioned_action")
D_FLEX_OBJS = ("intermediate_imputation", "retrodiction", "obs_denoising")
MASK_ID = None
PAD_ID = 0

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
    if not ids:
        return None
    sl = slice(0, max_len)
    return Example(np.array(ids[sl], np.int64), np.array(btype[sl], np.int64),
                   np.array(step[sl], np.int64), np.array(bidx[sl], np.int64),
                   str(row.get("trajectory_id", "?")))

def load_split(path, tok, max_len):
    return [e for e in (row_to_arrays(r, tok, max_len) for r in read_jsonl(path)) if e]

def random_block_mask(ex, nrng, rate):
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
def _first_idx(ex, step, bt):
    idx = np.where((ex.step == step) & (ex.btype == bt))[0]
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
    """D-AR: mask the last (rate × eligible) non-goal tokens; context = prefix + goal.
    Single stable objective — all masked tokens are a contiguous suffix in token order."""
    L = len(ex.ids)
    elig_idx = np.where(ex.btype != GOAL)[0]
    if len(elig_idx) == 0:
        return np.ones(L, bool), np.zeros(L, bool)
    n_mask = max(1, min(int(round(rate * len(elig_idx))), len(elig_idx)))
    tgt = np.zeros(L, bool)
    tgt[elig_idx[-n_mask:]] = True
    return ~tgt, tgt

def imputation_mask(ex, nrng, rate):
    """D-Flex: mask a contiguous middle span of ~rate% of non-goal tokens.
    Start sampled from [10%, 70%] of eligible positions so both sides always have context.
    Requires bidirectional attention — impossible for any causal model without leakage."""
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
    """D-Flex: mask the first (rate × eligible) non-goal tokens; context = later tokens + goal.
    Given what happened LATER in a trajectory, reconstruct what happened EARLY.
    Requires bidirectional attention — impossible for any causal model without leakage."""
    L = len(ex.ids)
    elig_idx = np.where(ex.btype != GOAL)[0]
    if len(elig_idx) == 0:
        return np.ones(L, bool), np.zeros(L, bool)
    n_mask = max(1, min(int(round(rate * len(elig_idx))), len(elig_idx)))
    tgt = np.zeros(L, bool)
    tgt[elig_idx[:n_mask]] = True
    return ~tgt, tgt

def obs_denoising_mask(ex, nrng, rate):
    """D-Flex: randomly mask obs tokens until rate% of ALL non-goal tokens are masked.
    Predicts corrupted environment observations from surrounding context.
    Rate is computed over all eligible tokens (not just obs) to match D-AR budget."""
    L = len(ex.ids)
    elig_idx = np.where(ex.btype != GOAL)[0]
    obs_idx = np.where(ex.btype == OBS)[0]
    if len(elig_idx) == 0 or len(obs_idx) == 0:
        return np.ones(L, bool), np.zeros(L, bool)
    n_mask = max(1, min(int(round(rate * len(elig_idx))), len(obs_idx)))
    chosen = nrng.choice(obs_idx, size=n_mask, replace=False)
    tgt = np.zeros(L, bool)
    tgt[chosen] = True
    return ~tgt, tgt

def corrupt(ex, ctx, tgt, max_len):
    L = min(len(ex.ids), max_len); ctx = ctx[:L]; tgt = tgt[:L]
    ids = np.full(max_len, PAD_ID, np.int64)
    btype = np.full(max_len, PADB, np.int64)
    valid = np.zeros(max_len, bool)
    ids[:L] = ex.ids[:L]; btype[:L] = ex.btype[:L]; valid[:L] = True
    inp = ids.copy()
    tmask = np.zeros(max_len, bool); tmask[:L] = tgt
    hmask = np.zeros(max_len, bool); hmask[:L] = (~ctx) & (~tgt)
    inp[tmask | hmask] = MASK_ID
    return inp, btype, valid, tmask, ids

def sample_one(obj_set, ex, rng, nrng, rate):
    if obj_set == "random_block":
        return "random_block", random_block_mask(ex, nrng, rate)
    if obj_set == "d_ar":
        return "suffix", suffix_mask(ex, rate)
    if obj_set == "d_flex":
        obj = rng.choice(D_FLEX_OBJS)
        if obj == "intermediate_imputation":
            return obj, imputation_mask(ex, nrng, rate)
        if obj == "retrodiction":
            return obj, retrodiction_mask(ex, rate)
        return obj, obs_denoising_mask(ex, nrng, rate)
    obj = rng.choice(CAUSAL_OBJS)
    return obj, causal_mask(obj, ex, rng, rate)

def make_step_batch(train, k, obj_set, rng, nrng, max_len, rate):
    idxs = rng.sample(range(len(train)), min(k, len(train)))
    I, B, V, S, Lb, objs, tids = [], [], [], [], [], [], []
    elig_tot = scored_tot = 0
    for i in idxs:
        ex = train[i]
        obj, (ctx, tgt) = sample_one(obj_set, ex, rng, nrng, rate)
        inp, bt, val, sc, lab = corrupt(ex, ctx, tgt, max_len)
        I.append(inp); B.append(bt); V.append(val); S.append(sc); Lb.append(lab)
        objs.append(obj); tids.append(ex.traj_id)
        elig_tot += int(((bt == THINK) | (bt == ACTION) | (bt == OBS)).sum())
        scored_tot += int(sc.sum())
    achieved = scored_tot / max(1, elig_tot)
    return (np.stack(I), np.stack(B), np.stack(V), np.stack(S), np.stack(Lb),
            objs, tids, achieved)

def make_charts(hist, obj_loss, budget, out_dir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    def smooth(y, k=15):
        if len(y) < k: return np.array(y)
        c = np.cumsum(np.insert(y, 0, 0)); return (c[k:] - c[:-k]) / k
    plt.figure(figsize=(7, 4.5))
    plt.plot(hist["step"], hist["loss"], alpha=0.25, color="tab:blue", label="train (raw)")
    sm = smooth(np.array(hist["loss"]))
    plt.plot(hist["step"][len(hist["step"]) - len(sm):], sm, color="tab:blue", label="train (smoothed)")
    if hist["val_step"]:
        plt.plot(hist["val_step"], hist["val_loss"], "o-", color="tab:red", label="validation")
    plt.xlabel("step"); plt.ylabel("cross-entropy (masked tokens)")
    plt.title("LLaDA2.0-mini + LoRA random-block masking"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out_dir}/loss_curve.png", dpi=130); plt.close()
    plt.figure(figsize=(7, 4)); plt.plot(hist["step"], hist["mask_rate"], color="tab:orange")
    plt.ylim(0, 1); plt.xlabel("step"); plt.ylabel("achieved mask rate")
    plt.title("Masking rate (should sit at target)"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out_dir}/mask_rate.png", dpi=130); plt.close()
    if any(obj_loss[k][1] for k in obj_loss):
        names = [k for k in obj_loss if obj_loss[k][1] > 0]
        vals = [obj_loss[k][0] / obj_loss[k][1] for k in names]
        plt.figure(figsize=(7, 4)); plt.bar(range(len(names)), vals, color="tab:purple")
        plt.xticks(range(len(names)), names, rotation=40, ha="right"); plt.ylabel("avg loss")
        plt.title("Loss by objective"); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
        plt.savefig(f"{out_dir}/loss_by_objective.png", dpi=130); plt.close()
    bt = [TYPE_NAME[k] for k in (GOAL, THINK, ACTION, OBS)]
    bv = [budget.get(k, 0) for k in (GOAL, THINK, ACTION, OBS)]
    plt.figure(figsize=(6, 4)); plt.bar(bt, bv, color="tab:gray")
    plt.ylabel("scored tokens"); plt.title("Supervision budget by block type")
    plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(f"{out_dir}/supervision_budget.png", dpi=130); plt.close()

def verify(args, tok, train):
    rng = random.Random(args.seed); nrng = np.random.default_rng(args.seed)
    print("=" * 70)
    print(f"MASK_ID={MASK_ID}  PAD_ID={PAD_ID}  -> distinct? {MASK_ID != PAD_ID}")
    assert MASK_ID != PAD_ID, "MASK and PAD must differ!"
    k = args.grad_accum * args.batch_size
    I, B, V, S, Lb, objs, tids, achieved = make_step_batch(
        train, k, args.objective_set, rng, nrng, args.max_len, args.mask_rate)
    print(f"sampled {len(tids)} trajectories this step: {tids}")
    print(f"all distinct? {len(set(tids)) == len(tids)}")
    print(f"objective(s): {sorted(set(objs))}")
    print(f"achieved mask rate over non-goal tokens: {achieved:.3f} (target {args.mask_rate})")
    inp, bt, val, sc, lab = I[0], B[0], V[0], S[0], Lb[0]
    goal_pos = bt == GOAL
    print("-" * 70)
    print(f"[ex0] valid={int(val.sum())} goal_tokens={int(goal_pos.sum())} masked/scored={int(sc.sum())}")
    print(f"  goal ever masked?          {bool((sc & goal_pos).any())}  (must be False)")
    print(f"  any pad masked?            {bool((inp == MASK_ID)[~val].any())}  (must be False)")
    print(f"  any pad scored?            {bool(sc[~val].any())}  (must be False)")
    print(f"  scored positions are MASK? {bool((inp[sc] == MASK_ID).all())}  (must be True)")
    print(f"  context tokens unchanged?  {bool((inp[val & ~sc] == lab[val & ~sc]).all())}  (must be True)")
    first = np.where(val)[0][:60]
    shown = ["[MASK]" if sc[p] else tok.decode([int(inp[p])]) for p in first]
    print("-" * 70); print("first ~60 tokens (masked shown as [MASK]):"); print(" ".join(shown))
    print("=" * 70)
    print("goal never masked + pad never masked/scored + rate~target + trajs distinct => CORRECT")

def main():
    global MASK_ID, PAD_ID
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=os.path.expanduser("~/dflex_proj/llada2mini"))
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="runs/llada_rb")
    ap.add_argument("--objective_set", default="random_block",
                    choices=["random_block", "causal", "d_ar", "d_flex"])
    ap.add_argument("--mask_schedule", default="constant", choices=["constant", "curriculum"])
    ap.add_argument("--mask_rate", type=float, default=0.20)
    ap.add_argument("--mask_rate_start", type=float, default=0.10)
    ap.add_argument("--mask_rate_end", type=float, default=0.60)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_targets", default="query_key_value,dense")
    ap.add_argument("--val_every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_name", default="rb")
    ap.add_argument("--wandb_mode", default="offline", choices=["online", "offline", "disabled"])
    ap.add_argument("--wandb_project", default="cp2107-dflex")
    ap.add_argument("--verify", action="store_true")
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
    val = load_split(os.path.join(args.data_dir, "validation.jsonl"), tok, args.max_len)
    print(f"[data] train={len(train)} val={len(val)} | MASK={MASK_ID} PAD={PAD_ID}")
    if args.verify:
        verify(args, tok, train); return

    device = "cuda"
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    print("[load] LLaDA2.0-mini bf16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    model.config.use_cache = False; model.config.pad_token_id = PAD_ID
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                      bias="none", task_type="CAUSAL_LM", target_modules=args.lora_targets.split(","))
    model = get_peft_model(model, lcfg); model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)

    wb = None
    if args.wandb_mode != "disabled":
        try:
            import wandb
            wb = wandb.init(project=args.wandb_project, name=args.run_name, mode=args.wandb_mode, config=vars(args))
        except Exception as e:
            print(f"[wandb] off ({e})")

    rng = random.Random(args.seed); nrng = np.random.default_rng(args.seed)
    hist = {"step": [], "loss": [], "mask_rate": [], "val_step": [], "val_loss": []}
    obj_loss = collections.defaultdict(lambda: [0.0, 0]); budget = collections.Counter(); t0 = time.time()

    def rate_at(p):
        return args.mask_rate if args.mask_schedule == "constant" else \
            args.mask_rate_start + (args.mask_rate_end - args.mask_rate_start) * p

    def run_micro(arrs):
        inp, bt, val_, sc, lab = arrs
        inp = torch.from_numpy(inp).to(device); vb = torch.from_numpy(val_).to(device)
        sc_t = torch.from_numpy(sc).to(device); lab_t = torch.from_numpy(lab).to(device)
        B_, L_ = inp.shape
        m4 = vb[:, None, None, :].expand(B_, 1, L_, L_).to(torch.bfloat16)
        out = model(input_ids=inp, attention_mask=m4)
        logits = out.logits.float()
        nll = F.cross_entropy(logits.transpose(1, 2), lab_t, reduction="none")
        sel = sc_t.float()
        loss = (nll * sel).sum() / sel.sum().clamp(min=1.0)
        return loss, sc_t, torch.from_numpy(bt).to(device)

    model.train()
    for step in range(args.steps):
        progress = step / max(1, args.steps - 1); rate = rate_at(progress)
        I, B, V, S, Lb, objs, tids, achieved = make_step_batch(
            train, args.grad_accum * args.batch_size, args.objective_set, rng, nrng, args.max_len, rate)
        opt.zero_grad(); accloss = 0.0
        for j in range(args.grad_accum):
            sl = slice(j * args.batch_size, (j + 1) * args.batch_size)
            loss, sc_t, bt_t = run_micro((I[sl], B[sl], V[sl], S[sl], Lb[sl]))
            (loss / args.grad_accum).backward(); accloss += float(loss.item()) / args.grad_accum
            sbt = bt_t[sc_t]
            for k in (GOAL, THINK, ACTION, OBS): budget[k] += int((sbt == k).sum().item())
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step()
        for o in objs: obj_loss[o][0] += accloss / len(objs); obj_loss[o][1] += 1.0 / len(objs)
        hist["step"].append(step); hist["loss"].append(accloss); hist["mask_rate"].append(achieved)
        if wb: wb.log({"train/loss": accloss, "train/mask_rate": achieved, "lr": sched.get_last_lr()[0]}, step=step)
        if step % args.val_every == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                vls = []
                for _ in range(3):
                    Iv, Bv, Vv, Sv, Lv, _, _, _ = make_step_batch(
                        val, args.batch_size, args.objective_set, rng, nrng, args.max_len, rate)
                    l, _, _ = run_micro((Iv, Bv, Vv, Sv, Lv)); vls.append(float(l.item()))
            vm = float(np.mean(vls)); hist["val_step"].append(step); hist["val_loss"].append(vm)
            if wb: wb.log({"val/loss": vm}, step=step)
            print(f"[step {step:3d}/{args.steps}] train={accloss:.3f} val={vm:.3f} rate={achieved:.2f} ({time.time()-t0:.0f}s)")
            model.train()

    model.save_pretrained(os.path.join(args.out_dir, "lora_adapter"))
    metrics = {"final_train_loss": hist["loss"][-1],
               "final_val_loss": hist["val_loss"][-1] if hist["val_loss"] else None,
               "mean_mask_rate": float(np.mean(hist["mask_rate"])),
               "objective_set": args.objective_set, "schedule": args.mask_schedule,
               "supervision_budget_by_type": {TYPE_NAME[k]: budget[k] for k in (GOAL, THINK, ACTION, OBS)},
               "minutes": round((time.time() - t0) / 60, 2)}
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    make_charts(hist, dict(obj_loss), budget, os.path.join(args.out_dir, "charts"))
    if wb: wb.finish()
    print(f"[done] {json.dumps(metrics)}")

if __name__ == "__main__":
    main()
