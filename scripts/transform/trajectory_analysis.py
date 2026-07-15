#!/usr/bin/env python3
"""Loading + enrichment (pure data transforms) for
``notebooks/trajectory_taxonomy.ipynb``.

This module holds **no plotting or display side effects** — everything here maps
raw records to canonical records, feature dicts, DataFrames, and summary rows.
The presentation layer (matplotlib panels, ``render``) lives in ``trajectory_viz``.
Keeping the two apart means the transforms are importable and testable without a
display, and the notebook can reuse the same features for its own inline figures.

Layers on the taxonomy modules:

- ``taxonomy``      — ScienceWorld / generic feature classifiers
- ``env_taxonomy``  — cross-environment goal-family / objective / action-verb dispatch

Loading:     load_jsonl, canonical_record, load_env
Enrichment:  enrich  (raw record -> taxonomy-labelled feature dict)
Aggregation: build_features, action_verb_counts, objective_stats, summary_row
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.transform import taxonomy as tx
from scripts.transform import env_taxonomy as et

SPLITS = ["train", "validation", "test"]


# ------------------------------------------------------------------- loading
def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _streams_from_blocks(blocks):
    """Flatten an interleaved ``blocks`` list into (actions, observations, thinks, goal)."""
    actions, observations, thinks, goal = [], [], [], ""
    for b in blocks or []:
        t = b.get("type")
        txt = b.get("text", "")
        if t == "Action":        actions.append(tx.normalize_whitespace(txt))
        elif t == "Observation": observations.append(str(txt))
        elif t == "Think":       thinks.append(tx.normalize_whitespace(txt))
        elif t == "Goal" and not goal: goal = tx.normalize_whitespace(txt)
    return actions, observations, thinks, goal


def canonical_record(row, env_hint=None):
    actions, obs, thinks, goal_from_block = _streams_from_blocks(row.get("blocks") or [])
    goal = tx.normalize_whitespace(row.get("goal", "") or goal_from_block)
    return {
        "env": row.get("env") or env_hint or "unknown",
        "trajectory_id": str(row.get("trajectory_id", "")),
        "split": row.get("split"),
        "goal": goal,
        "actions": actions, "observations": obs, "thinks": thinks,
        "n_actions": len(actions), "n_observations": len(obs),
        "state_labels": row.get("state_labels") or {},
        "_schema": "blocks",   # all three envs verified to use the blocks schema
    }


def load_env(env_name, data_dir, splits=SPLITS):
    """Load an environment from ``<data_dir>/<env>/<split>.jsonl`` (canonical per-env
    layout). Falls back to the flat ``<data_dir>/<split>.jsonl`` for scienceworld."""
    data_dir = Path(data_dir)
    records, found_any = [], False
    for split in splits:
        candidates = [data_dir / env_name / f"{split}.jsonl"]
        if env_name == "scienceworld":
            candidates.append(data_dir / f"{split}.jsonl")
        for path in candidates:
            if path.exists():
                for r in load_jsonl(path):
                    r.setdefault("split", split)
                    records.append(canonical_record(r, env_hint=env_name))
                found_any = True
                break
    return records, found_any


# ---------------------------------------------------------------- enrichment
def enrich(rec):
    """Attach taxonomy labels + coarse features to a canonical record."""
    env, goal = rec["env"], rec["goal"]
    sl = rec.get("state_labels") or {}
    fam = et.goal_family(env, goal, sl)     # ALFWorld uses exact state_labels.task_type
    obj = et.objective_type(env, fam)
    out = dict(rec)
    out.update({
        "goal_family": fam,
        "objective_type": obj,
        "source_format": sl.get("source_format"),
        "goal_template_key": tx.normalize_goal_template(goal),
        "length_bin": tx.classify_length_bin(rec["n_actions"]),
        "interaction_pattern": tx.classify_interaction_pattern(rec["actions"], rec["observations"]),
        "ambiguity_pattern": tx.classify_ambiguity_pattern(rec["actions"], rec["observations"]),
        "avg_obs_chars": float(np.mean([len(str(o)) for o in rec["observations"]])) if rec["observations"] else 0.0,
        "repeated_action_ratio": tx.repeated_action_ratio([a.lower() for a in rec["actions"]]),
    })
    return out


# --------------------------------------------------------------- aggregation
def build_features(records):
    """Enrich every record into a taxonomy-labelled feature DataFrame.

    Pure transform (no plotting): this is the data half of the old ``profile_env``,
    so the viz layer and the notebook's inline figures share one feature table."""
    return pd.DataFrame(enrich(r) for r in records)


def action_verb_counts(records, env):
    """Counter of coarse action verbs (``env_taxonomy.action_verb``) across records."""
    return Counter(et.action_verb(a, env) for r in records for a in r["actions"])


def objective_stats(df):
    """Per-objective-type action-length summary (count / median / mean / max)."""
    return df.groupby("objective_type")["n_actions"].agg(["count", "median", "mean", "max"]).round(1)


def summary_row(env, df, records):
    """One cross-environment summary row (pure dict — the notebook tabulates/displays)."""
    if df is None or not len(df):
        return {"env": env, "status": "missing (needs HF auth)", "n_traj": 0,
                "goal_families": None, "objective_types": None,
                "median_actions": None, "dominant_interaction": None, "top_verb": None}
    verbs = action_verb_counts(records, env)
    return {"env": env, "status": "empirical", "n_traj": len(df),
            "goal_families": df["goal_family"].nunique(),
            "objective_types": ", ".join(sorted(df["objective_type"].unique())),
            "median_actions": int(df["n_actions"].median()),
            "dominant_interaction": df["interaction_pattern"].mode()[0],
            "top_verb": verbs.most_common(1)[0][0] if verbs else None}
