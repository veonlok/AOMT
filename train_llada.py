#!/usr/bin/env python3
"""
train_llada.py -- LoRA fine-tune LLaDA2.0-mini (inclusionAI, 16B MoE diffusion LM)
with the D-Flex flexible masking objectives on ScienceWorld trajectories.

Reuses the exact masking machinery validated in train_dflex.py (nine objectives,
curriculum, MDLM context/target/hidden corruption, 1/t-weighted scoring). Only
the tokenizer (LLaDA's), mask token (LLaDA's), model (LLaDA2.0-mini bf16 + LoRA +
gradient checkpointing), and forward (model(input_ids, attention_mask).logits)
differ. Saves only the small LoRA adapter, not the 32GB base.
"""
from __future__ import annotations
import argparse, json, math, os, random, time, collections, re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PADB, GOAL, THINK, ACTION, OBS = 0, 1, 2, 3, 4
TYPE_FROM_STR = {"Goal": GOAL, "Think": THINK, "Action": ACTION, "Observation": OBS}
TYPE_NAME = {GOAL: "goal", THINK: "think", ACTION: "action", OBS: "obs"}
CAUSAL_OBJS = ("next_action", "next_observation", "future_from_past",
               "full_completion", "goal_conditioned_action")
FLEX_OBJS = ("inverse_dynamics", "retrodiction",
             "intermediate_imputation", "observation_denoising")
ALL_OBJS = CAUSAL_OBJS + FLEX_OBJS
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

def _steps_present(ex):
    return sorted({int(s) for s in ex.step.tolist() if s >= 1})

def _first_idx(ex, step, bt):
    idx = np.where((ex.step == step) & (ex.btype == bt))[0]
    return int(idx[0]) if idx.size else None

def sample_mask(obj, ex, rng, target_ratio):
    L = len(ex.ids); steps = _steps_present(ex); goal = ex.btype == GOAL
    ctx = np.zeros(L, bool); tgt = np.zeros(L, bool)
    def fallback():
        n = max(1, int(target_ratio * L)); s = rng.randint(0, max(0, L - n))
        t = np.zeros(L, bool); t[s:s + n] = True
        return ~t, t
    if not steps:
        return fallback()
    if obj == "next_action":
        cand = [s for s in steps if _first_idx(ex, s, ACTION) is not None]
        if not cand: return fallback()
        t = rng.choice(cand); start = _first_idx(ex, t, ACTION)
        tgt = (ex.step == t) & (ex.btype == ACTION); ctx = (np.arange(L) < start) & ~tgt
    elif obj == "next_observation":
        cand = [s for s in steps if _first_idx(ex, s, OBS) is not None]
        if not cand: return fallback()
        t = rng.choice(cand); start = _first_idx(ex, t, OBS)
        tgt = (ex.step == t) & (ex.btype == OBS); ctx = (np.arange(L) < start) & ~tgt
    elif obj == "future_from_past":
        t = rng.choice(steps[:-1]) if len(steps) > 1 else steps[0]
        ctx = (ex.step <= t) | goal; tgt = ~ctx
    elif obj == "full_completion":
        ctx = goal | (ex.bidx == 0)
        interior = sorted(set(int(b) for b in ex.bidx) - {0})
        keep = max(0, int(round((1 - target_ratio) * len(interior))))
        for b in (rng.sample(interior, min(keep, len(interior))) if interior else []):
            ctx = ctx | (ex.bidx == b)
        tgt = ~ctx
    elif obj == "inverse_dynamics":
        cand = [s for s in steps if _first_idx(ex, s, ACTION) is not None]
        if not cand: return fallback()
        t = rng.choice(cand); tgt = (ex.step == t) & (ex.btype == ACTION)
        ctx = ((ex.step == t) | (ex.step == t + 1)) & (ex.btype == OBS)
        if not ctx.any(): ctx = (ex.step == t) & (ex.btype != ACTION)
    elif obj == "retrodiction":
        t = rng.choice(steps[1:]) if len(steps) > 1 else steps[0]
        ctx = ex.step >= t; tgt = (ex.step >= 1) & (ex.step < t)
        if not tgt.any(): return fallback()
    elif obj == "intermediate_imputation":
        if len(steps) < 3: return fallback()
        t1, t2 = sorted(rng.sample(steps, 2))
        if t2 - t1 < 1: t2 = min(steps[-1], t1 + 1)
        ctx = (ex.step <= t1) | (ex.step >= t2) | goal; tgt = ~ctx
        if not tgt.any(): return fallback()
    elif obj == "observation_denoising":
        cand = [s for s in steps if _first_idx(ex, s, OBS) is not None]
        if not cand: return fallback()
        t = rng.choice(cand); tgt = (ex.step == t) & (ex.btype == OBS)
        ob = int(ex.bidx[np.where(tgt)[0][0]])
        ctx = (goal | (ex.bidx == ob - 1) | (ex.bidx == ob + 1)) & ~tgt
    elif obj == "goal_conditioned_action":
        ab = sorted(set(int(ex.bidx[i]) for i in range(L) if ex.btype[i] == ACTION))
        if not ab: return fallback()
        k = max(1, int(round(target_ratio * len(ab))))
        chosen = rng.sample(ab, min(k, len(ab)))
        tgt = np.isin(ex.bidx, chosen); ctx = ~tgt
        for b in chosen:
            a_step = int(ex.step[np.where(ex.bidx == b)[0][0]])
            ctx = ctx & ~((ex.step > a_step) & (ex.btype == OBS))
    else:
        return fallback()
    ctx = ctx & ~tgt
    if not tgt.any(): return fallback()
    return ctx, tgt

def target_ratio_at(p, start=0.10, end=0.60):
    return float(start + (end - start) * min(max(p, 0.0), 1.0))

BASE_WEIGHTS = {"next_action": .15, "next_observation": .20, "future_from_past": .10,
                "full_completion": .10, "inverse_dynamics": .15, "retrodiction": .10,
                "intermediate_imputation": .10, "observation_denoising": .05,
                "goal_conditioned_action": .05}
WARMUP = {"inverse_dynamics": (.1, .4), "retrodiction": (.2, .5),
          "intermediate_imputation": (.2, .5), "observation_denoising": (.3, .6)}

def sample_objective(p, rng):
    w = {}
    for n, base in BASE_WEIGHTS.items():
        if n in WARMUP:
            lo, hi = WARMUP[n]; g = 0.0 if p <= lo else (1.0 if p >= hi else (p - lo) / (hi - lo))
        else:
            g = 1.0
        w[n] = base * g
    names = list(w); tot = sum(w.values()) or 1.0; r = rng.random() * tot; c = 0.0
    for n in names:
        c += w[n]
        if r <= c: return n
    return names[-1]

def corrupt(ex, ctx, tgt, t, nrng, max_len):
    L = min(len(ex.ids), max_len); ctx = ctx[:L]; tgt = tgt[:L]
    ids = np.full(max_len, PAD_ID, np.int64); btype = np.full(max_len, PADB, np.int64)
    valid = np.zeros(max_len, bool)
    ids[:L] = ex.ids[:L]; btype[:L] = ex.btype[:L]; valid[:L] = True
    inp = ids.copy()
    tgt_masked = np.zeros(max_len, bool); tgt_masked[:L] = tgt & (nrng.random(L) < t)
    if not tgt_masked.any():
        ti = np.where(tgt)[0]; tgt_masked[int(nrng.choice(ti)) if ti.size else 0] = True
    hidden = np.zeros(max_len, bool); hidden[:L] = valid[:L] & ~ctx & ~tgt
    inp[tgt_masked | hidden] = MASK_ID
    return inp, btype, valid, tgt_masked, ids

def make_batch(exs, obj, progress, rng, nrng, max_len):
    tr = target_ratio_at(progress); I, B, V, S, Lb, T = [], [], [], [], [], []
    for ex in exs:
        ctx, tgt = sample_mask(obj, ex, rng, tr)
        t = float(nrng.uniform(1e-3, 1 - 1e-3))
        inp, bt, val, sc, lab = corrupt(ex, ctx, tgt, t, nrng, max_len)
        I.append(inp); B.append(bt); V.append(val); S.append(sc); Lb.append(lab); T.append(t)
    return (np.stack(I), np.stack(B), np.stack(V), np.stack(S),
            np.stack(Lb), np.array(T, np.float32), tr)

def find_lora_targets(model, explicit=None):
    if explicit:
        return explicit.split(",")
    leafs = set()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            leafs.add(name.split(".")[-1])
    attn = [n for n in leafs if any(k in n.lower() for k in
            ["q_proj", "k_proj", "v_proj", "o_proj", "qkv", "wq", "wk", "wv", "wo", "query", "key", "value"])]
    if attn:
        return sorted(attn)
    excl = lambda n: any(k in n.lower() for k in
                         ["embed", "lm_head", "router", "gate", "expert", "shared"])
    return sorted([n for n in leafs if not excl(n)])

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
    plt.xlabel("step"); plt.ylabel("masked-diffusion loss")
    plt.title("LLaDA2.0-mini + D-Flex (LoRA) training"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out_dir}/loss_curve.png", dpi=130); plt.close()
    names = [k for k in ALL_OBJS if obj_loss.get(k, [0, 0])[1] > 0]
    vals = [obj_loss[k][0] / obj_loss[k][1] for k in names]
    cols = ["tab:green" if k in CAUSAL_OBJS else "tab:purple" for k in names]
    plt.figure(figsize=(8, 4.5)); plt.bar(range(len(names)), vals, color=cols)
    plt.xticks(range(len(names)), names, rotation=40, ha="right"); plt.ylabel("avg loss")
    plt.title("Loss by objective (green=causal, purple=flexible-only)")
    plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(f"{out_dir}/loss_by_objective.png", dpi=130); plt.close()
    plt.figure(figsize=(7, 4)); plt.plot(hist["step"], hist["target_ratio"], color="tab:orange")
    plt.xlabel("step"); plt.ylabel("target ratio r-bar"); plt.title("Incremental masking curriculum")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(f"{out_dir}/curriculum.png", dpi=130); plt.close()
    bt = [TYPE_NAME[k] for k in (GOAL, THINK, ACTION, OBS)]
    bv = [budget.get(k, 0) for k in (GOAL, THINK, ACTION, OBS)]
    plt.figure(figsize=(6, 4)); plt.bar(bt, bv, color="tab:gray")
    plt.ylabel("scored tokens"); plt.title("Supervision budget by block type")
    plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(f"{out_dir}/supervision_budget.png", dpi=130); plt.close()
    return [f"{out_dir}/loss_curve.png", f"{out_dir}/loss_by_objective.png",
            f"{out_dir}/curriculum.png", f"{out_dir}/supervision_budget.png"]

def main():
    global MASK_ID, PAD_ID
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=os.path.expanduser("~/dflex_proj/llada2mini"))
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="runs/llada_dflex")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_targets", default=None)
    ap.add_argument("--mask_token_id", type=int, default=-1)
    ap.add_argument("--val_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_name", default="LLaDA-DFlex-scienceworld")
    ap.add_argument("--wandb_mode", default="offline", choices=["online", "offline", "disabled"])
    ap.add_argument("--wandb_project", default="cp2107-dflex")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.val_every, args.grad_accum, args.max_len = 12, 6, 2, 256

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = random.Random(args.seed); nrng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    print("[load] tokenizer + model (bf16)...")
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else 0
    MASK_ID = args.mask_token_id if args.mask_token_id >= 0 else tok.mask_token_id
    if MASK_ID is None:
        raise SystemExit("No mask_token_id; pass --mask_token_id <id>.")
    print(f"[load] PAD_ID={PAD_ID} MASK_ID={MASK_ID} vocab={len(tok)}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    model.config.use_cache = False
    model.config.pad_token_id = PAD_ID
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    targets = find_lora_targets(model, args.lora_targets)
    print(f"[lora] target_modules={targets}")
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                      bias="none", task_type="CAUSAL_LM", target_modules=targets)
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    train = load_split(os.path.join(args.data_dir, "train.jsonl"), tok, args.max_len)
    val = load_split(os.path.join(args.data_dir, "validation.jsonl"), tok, args.max_len)
    print(f"[data] train={len(train)} val={len(val)}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)

    wb = None
    if args.wandb_mode != "disabled":
        try:
            import wandb
            wb = wandb.init(project=args.wandb_project, name=args.run_name,
                            mode=args.wandb_mode, config=vars(args))
        except Exception as e:
            print(f"[wandb] off ({e})")

    hist = {"step": [], "loss": [], "target_ratio": [], "val_step": [], "val_loss": []}
    obj_loss = {k: [0.0, 0] for k in ALL_OBJS}; budget = collections.Counter(); t0 = time.time()

    def loss_on(exs, obj, progress):
        inp, bt, val_, sc, lab, ts, tr = make_batch(exs, obj, progress, rng, nrng, args.max_len)
        inp = torch.from_numpy(inp).to(device); attn = torch.from_numpy(val_).to(device)
        sc_t = torch.from_numpy(sc).to(device); lab_t = torch.from_numpy(lab).to(device)
        ts_t = torch.from_numpy(ts).to(device)
        vb = attn.bool()
        B_, L_ = inp.shape
        m4 = vb[:, None, None, :].expand(B_, 1, L_, L_).to(torch.bfloat16)
        out = model(input_ids=inp, attention_mask=m4)
        logits = out.logits.float()
        nll = F.cross_entropy(logits.transpose(1, 2), lab_t, reduction="none")
        w = (1.0 / ts_t.clamp(min=1e-2)).clamp(max=50.0)[:, None]
        sel = sc_t.float()
        loss = (nll * sel * w).sum() / sel.sum().clamp(min=1.0)
        return loss, sc_t, torch.from_numpy(bt).to(device)

    model.train()
    for step in range(args.steps):
        progress = step / max(1, args.steps - 1)
        obj = sample_objective(progress, rng)
        opt.zero_grad(); acc = 0.0
        for _ in range(args.grad_accum):
            exs = [train[rng.randrange(len(train))] for _ in range(args.batch_size)]
            loss, sc_t, bt_t = loss_on(exs, obj, progress)
            (loss / args.grad_accum).backward()
            acc += float(loss.item()) / args.grad_accum
            sbt = bt_t[sc_t]
            for k in (GOAL, THINK, ACTION, OBS):
                budget[k] += int((sbt == k).sum().item())
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step()
        obj_loss[obj][0] += acc; obj_loss[obj][1] += 1
        hist["step"].append(step); hist["loss"].append(acc)
        hist["target_ratio"].append(target_ratio_at(progress))
        if wb:
            wb.log({"train/loss": acc, "train/objective_idx": ALL_OBJS.index(obj),
                    "train/target_ratio": target_ratio_at(progress),
                    "lr": sched.get_last_lr()[0]}, step=step)
        if step % args.val_every == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                vl = []
                for _ in range(3):
                    vexs = [val[rng.randrange(len(val))] for _ in range(args.batch_size)]
                    l, _, _ = loss_on(vexs, sample_objective(progress, rng), progress)
                    vl.append(float(l.item()))
            vm = float(np.mean(vl)); hist["val_step"].append(step); hist["val_loss"].append(vm)
            if wb: wb.log({"val/loss": vm}, step=step)
            print(f"[step {step:3d}/{args.steps}] train={acc:.3f} val={vm:.3f} obj={obj} "
                  f"r={target_ratio_at(progress):.2f} ({time.time()-t0:.0f}s)")
            model.train()

    adapter_dir = os.path.join(args.out_dir, "lora_adapter")
    model.save_pretrained(adapter_dir)
    metrics = {"final_train_loss": hist["loss"][-1],
               "final_val_loss": hist["val_loss"][-1] if hist["val_loss"] else None,
               "objective_avg_loss": {k: obj_loss[k][0] / obj_loss[k][1]
                                      for k in ALL_OBJS if obj_loss[k][1] > 0},
               "supervision_budget_by_type": {TYPE_NAME[k]: budget[k] for k in (GOAL, THINK, ACTION, OBS)},
               "lora_targets": targets, "minutes": round((time.time() - t0) / 60, 2)}
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    charts = make_charts(hist, obj_loss, budget, os.path.join(args.out_dir, "charts"))
    if wb:
        import wandb
        wb.log({os.path.basename(c)[:-4]: wandb.Image(c) for c in charts}); wb.finish()
    print(f"[done] {json.dumps(metrics)}")
    print(f"[charts] {charts}\n[adapter] {adapter_dir}")


if __name__ == "__main__":
    main()
