#!/usr/bin/env python3
"""Granular per-trajectory metrics, cross-environment leakage keys, and clustering
for ScienceWorld / ALFWorld / WebShop.

Everything here works on the canonical record dicts produced by
``trajectory_analysis.load_env`` (fields: env, trajectory_id, split, goal, actions,
observations, thinks, state_labels). Returns plain dicts / lists so the notebook can
build DataFrames and plots.

Token counts are whitespace-word approximations (labelled ``*_words``) — true model
tokens would need the LLaDA tokenizer; char counts (``*_chars``) are exact.
"""

from __future__ import annotations

import hashlib
from collections import Counter

from scripts.transform import taxonomy as tx
from scripts.transform import env_taxonomy as et


# ------------------------------------------------------ action categorisation
def action_category(action: str, env: str) -> str:
    """Coarse role of an action: 'navigation', 'manipulation', or 'other'."""
    a = action.strip().lower()
    if not a:
        return "other"
    if env == "webshop":
        if a.startswith("search["):        return "navigation"   # query the catalogue
        if "buy now" in a:                 return "manipulation"  # commit purchase
        if a.startswith("click["):         return "navigation"   # browse results/pages/options
        return "other"
    if env == "alfworld":
        v = a.split(" ")[0]
        if v in {"gotolocation", "openobject", "closeobject"}:            return "navigation"
        if v in {"pickupobject", "putobject", "toggleobject", "sliceobject"}: return "manipulation"
        return "other"   # noop
    # scienceworld (and any text-world fallback)
    for p in ("teleport to", "go to", "look around", "look at", "examine", "read ", "open ", "close "):
        if a.startswith(p): return "navigation"
    for p in ("pick up", "move ", "drop", "put down", "focus", "connect", "use ",
              "pour", "mix", "activate", "deactivate"):
        if a.startswith(p): return "manipulation"
    return "other"   # wait / numeric disambiguation / misc


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def completion_status(rec) -> str:
    if rec["env"] == "scienceworld":
        return tx.guess_completion_status({"goal": rec["goal"]}, rec["actions"], rec["observations"])
    return "assumed_success_demo"   # ALFWorld plan-only / WebShop offline logs are demonstrations


# ------------------------------------------------------ per-trajectory metrics
def trajectory_metrics(rec) -> dict:
    env, goal = rec["env"], rec["goal"]
    actions, obs, thinks = rec["actions"], rec["observations"], rec["thinks"]
    na, no, nt = len(actions), len(obs), len(thinks)
    n_blocks = 1 + na + no + nt   # goal block + interleaved events

    a_chars = [len(a) for a in actions]
    o_chars = [len(str(o)) for o in obs]
    t_chars = [len(t) for t in thinks]
    a_words = [len(a.split()) for a in actions]
    o_words = [len(str(o).split()) for o in obs]
    t_words = [len(t.split()) for t in thinks]

    cats = Counter(action_category(a, env) for a in actions)
    lower = [a.lower() for a in actions]
    consec_rep = sum(1 for i in range(1, na) if lower[i] == lower[i - 1])

    return {
        "env": env,
        "trajectory_id": rec["trajectory_id"],
        "split": rec["split"],
        "goal_family": et.goal_family(env, goal, rec.get("state_labels")),
        # goal
        "goal_chars": len(goal),
        "goal_words": len(goal.split()),
        # block composition
        "n_blocks": n_blocks,
        "n_actions": na,
        "n_observations": no,
        "n_thinks": nt,
        "think_ratio": round(nt / n_blocks, 3),
        "action_ratio": round(na / n_blocks, 3),
        "obs_ratio": round(no / n_blocks, 3),
        # per-block-type sizes
        "avg_action_chars": round(_mean(a_chars), 1),
        "avg_action_words": round(_mean(a_words), 1),
        "avg_obs_chars": round(_mean(o_chars), 1),
        "avg_obs_words": round(_mean(o_words), 1),
        "avg_think_chars": round(_mean(t_chars), 1),
        "avg_think_words": round(_mean(t_words), 1),
        # whole-trajectory size
        "traj_chars": len(goal) + sum(a_chars) + sum(o_chars) + sum(t_chars),
        "traj_words": len(goal.split()) + sum(a_words) + sum(o_words) + sum(t_words),
        # repetition
        "repeated_actions_consecutive": consec_rep,
        "repeated_actions_total": na - len(set(lower)),
        # action-type mix
        "n_navigation": cats.get("navigation", 0),
        "n_manipulation": cats.get("manipulation", 0),
        "n_other_actions": cats.get("other", 0),
        "nav_ratio": round(cats.get("navigation", 0) / na, 3) if na else 0.0,
        "manip_ratio": round(cats.get("manipulation", 0) / na, 3) if na else 0.0,
        # interaction structure (for within-env difficulty slices)
        "interaction_pattern": tx.classify_interaction_pattern(actions, obs),
        "ambiguity_pattern": tx.classify_ambiguity_pattern(actions, obs),
        "source_format": (rec.get("state_labels") or {}).get("source_format"),
        # completion
        "completion_status": completion_status(rec),
    }


# ------------------------------------------------------- hashing / leakage keys
def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def full_hash(rec) -> str:
    body = rec["goal"] + "\n" + "\n".join(
        f"A|{a}" for a in rec["actions"]) + "\n" + "\n".join(f"O|{o}" for o in rec["observations"])
    return _sha(body)


def skeleton_hash(rec) -> str:
    return _sha(" || ".join(rec["actions"]))


def instance_key(rec) -> str:
    """Env-specific 'same task instance' key used for leakage/OOD scoring."""
    env, sl = rec["env"], rec.get("state_labels") or {}
    if env == "alfworld":
        p = sl.get("pddl_params") or {}
        return "alf|" + "|".join([
            str(sl.get("task_type", "")), str(p.get("object_target", "")),
            str(p.get("parent_target", "")), str(p.get("mrecep_target", "")),
            str(p.get("toggle_target", "")),
        ])
    if env == "webshop":
        return "ws|" + tx.normalize_whitespace(rec["goal"]).lower()
    return "sw|" + tx.normalize_goal_template(rec["goal"])


def _family(rec):
    return et.goal_family(rec["env"], rec["goal"], rec.get("state_labels"))


# ------------------------------------------------------- cross-env leakage table
def split_risk_rows(loaded) -> list:
    """One row per (env, eval-split) with counts in each leakage tier:
    exact_seen > template_seen (same instance key) > family_seen > strong_ood."""
    rows = []
    for env, recs in loaded.items():
        if not recs:
            continue
        train = [r for r in recs if r["split"] == "train"]
        train_full = {full_hash(r) for r in train}
        train_key = {instance_key(r) for r in train}
        train_fam = {_family(r) for r in train}
        for split in ("validation", "test"):
            items = [r for r in recs if r["split"] == split]
            if not items:
                continue
            tier = Counter()
            for r in items:
                if full_hash(r) in train_full:      tier["exact_seen"] += 1
                elif instance_key(r) in train_key:  tier["template_seen"] += 1
                elif _family(r) in train_fam:        tier["family_seen"] += 1
                else:                                tier["strong_ood"] += 1
            n = len(items)
            seen = tier["exact_seen"] + tier["template_seen"]
            rows.append({
                "env": env, "split": split, "n": n,
                "exact_seen": tier["exact_seen"],
                "template_seen": tier["template_seen"],
                "family_seen": tier["family_seen"],
                "strong_ood": tier["strong_ood"],
                "seen_pct": round(100 * seen / n, 1),
                "ood_pct": round(100 * tier["strong_ood"] / n, 1),
            })
    return rows


# --------------------------------------------------------- clustering / patterns
def cluster_summary(records) -> dict:
    full = Counter(full_hash(r) for r in records)
    skel = Counter(skeleton_hash(r) for r in records)
    key = Counter(instance_key(r) for r in records)
    return {
        "n_trajectories": len(records),
        "unique_full": len(full),
        "exact_dup_trajectories": sum(c for c in full.values() if c > 1),
        "exact_dup_clusters": sum(1 for c in full.values() if c > 1),
        "unique_action_skeletons": len(skel),
        "skeleton_dup_trajectories": sum(c for c in skel.values() if c > 1),
        "largest_skeleton_cluster": max(skel.values()) if skel else 0,
        "unique_instance_keys": len(key),
        "largest_instance_cluster": max(key.values()) if key else 0,
    }


def top_action_ngrams(records, env, k=3, top=10):
    """Most common action-verb k-grams — surfaces repeated behavioural patterns."""
    c = Counter()
    for r in records:
        vs = [et.action_verb(a, env) for a in r["actions"]]
        for i in range(len(vs) - k + 1):
            c["  ->  ".join(vs[i:i + k])] += 1
    return c.most_common(top)


def top_skeletons(records, top=10):
    """Most common full action-verb skeletons (whole-trajectory repeated patterns)."""
    c = Counter()
    env = records[0]["env"] if records else ""
    for r in records:
        c["  ->  ".join(et.action_verb(a, env) for a in r["actions"])] += 1
    return c.most_common(top)


# --------------------------------------------------------- split-overlap matrix
def overlap_matrix(records) -> list:
    """Shared-key counts between splits within one environment (the audit report's
    train/val/test overlap, generalized to any env via the per-env instance key)."""
    by = {s: [r for r in records if r["split"] == s] for s in ("train", "validation", "test")}
    kset = lambda items, fn: {fn(r) for r in items}
    rows = []
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        A, B = by[a], by[b]
        rows.append({
            "pair": f"{a}->{b}",
            "n_right": len(B),
            "shared_full_traj": len(kset(A, full_hash) & kset(B, full_hash)),
            "shared_skeleton": len(kset(A, skeleton_hash) & kset(B, skeleton_hash)),
            "shared_instance": len(kset(A, instance_key) & kset(B, instance_key)),
            "shared_family": len(kset(A, _family) & kset(B, _family)),
        })
    return rows


# --------------------------------------------- alternative benchmark construction
def build_grouped_split(records, key_fn, seed=0):
    """Assign each trajectory to train/val/test so that **no key_fn group crosses a
    split boundary**, greedily filling toward the original split proportions.
    Returns {trajectory_id: split}."""
    from collections import defaultdict
    orig = Counter(r["split"] for r in records)
    total = sum(orig.values()) or 1
    targets = {s: orig.get(s, 0) / total * len(records) for s in ("train", "validation", "test")}
    filled = {s: 0 for s in targets}
    groups = defaultdict(list)
    for r in records:
        groups[key_fn(r)].append(r)
    assignment = {}
    for _key, items in sorted(groups.items(), key=lambda kv: (-len(kv[1]), str(kv[0]))):
        s = max(targets, key=lambda s: targets[s] - filled[s])   # split most below its target
        for r in items:
            assignment[r["trajectory_id"]] = s
        filled[s] += len(items)
    return assignment


def _risk_counts(train, val):
    tf = {full_hash(r) for r in train}
    tk = {instance_key(r) for r in train}
    tfam = {_family(r) for r in train}
    tier = Counter()
    for r in val:
        if full_hash(r) in tf:        tier["exact_seen"] += 1
        elif instance_key(r) in tk:   tier["template_seen"] += 1
        elif _family(r) in tfam:      tier["family_seen"] += 1
        else:                         tier["strong_ood"] += 1
    return tier


def protocol_comparison(records) -> list:
    """Validation risk-tier distribution under three split protocols:
    current (as-shipped) vs IID-balanced (group near-dup skeletons) vs grouped-OOD
    (group by task instance so no template crosses)."""
    protocols = {
        "current": {r["trajectory_id"]: r["split"] for r in records},
        "iid_balanced": build_grouped_split(records, skeleton_hash),
        "grouped_ood": build_grouped_split(records, instance_key),
    }
    rows = []
    for name, assign in protocols.items():
        train = [r for r in records if assign.get(r["trajectory_id"]) == "train"]
        val = [r for r in records if assign.get(r["trajectory_id"]) == "validation"]
        tier = _risk_counts(train, val)
        n = len(val) or 1
        seen = tier["exact_seen"] + tier["template_seen"]
        rows.append({
            "protocol": name, "val_n": len(val),
            "exact_seen": tier["exact_seen"], "template_seen": tier["template_seen"],
            "family_seen": tier["family_seen"], "strong_ood": tier["strong_ood"],
            "seen_pct": round(100 * seen / n, 1),
        })
    return rows


def family_leakage(records) -> list:
    """Per-goal-family template-seen rate of the (as-shipped) validation split."""
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "validation"]
    tk = {instance_key(r) for r in train}
    rows = []
    for fam in sorted({_family(r) for r in val}):
        items = [r for r in val if _family(r) == fam]
        seen = sum(1 for r in items if instance_key(r) in tk)
        rows.append({
            "goal_family": fam, "val": len(items), "template_seen": seen,
            "seen_pct": round(100 * seen / len(items), 1) if items else 0.0,
        })
    return sorted(rows, key=lambda r: -r["val"])


# ------------------------------------------------ length distributions by granularity
def word_tokens(text) -> int:
    """Whitespace-word token count (fallback when no tokenizer is supplied)."""
    return len(str(text).split())


def unit_lengths(records, tok=word_tokens):
    """Yield one row per text unit for length-distribution study, tagged with the
    granularity it was measured at. ``tok`` is any ``text -> int`` tokenizer.

    Granularities: goal, action, observation, thought (per-unit), and trajectory
    (whole). 'block'-level = pool of {goal, action, observation, thought}."""
    for r in records:
        env = r["env"]
        traj_tokens = traj_chars = 0
        units = [("goal", r["goal"])]
        units += [("action", a) for a in r["actions"]]
        units += [("observation", str(o)) for o in r["observations"]]
        units += [("thought", t) for t in r["thinks"]]
        for gran, txt in units:
            n = tok(txt)
            traj_tokens += n
            traj_chars += len(str(txt))
            yield {"env": env, "granularity": gran, "tokens": n, "chars": len(str(txt))}
        yield {"env": env, "granularity": "trajectory", "tokens": traj_tokens, "chars": traj_chars}
