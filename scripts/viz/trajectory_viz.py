#!/usr/bin/env python3
"""Visualisation layer for ``notebooks/trajectory_taxonomy.ipynb``.

Everything with a display side effect (matplotlib panels, console ``render``) lives
here, split out of ``trajectory_analysis`` so the data transforms stay importable and
testable without a display. This module depends on the transforms — never the reverse:

    trajectory_viz  ->  trajectory_analysis (action_verb_counts, ...)
                    ->  plot_theme          (shared surface / ink / per-env hue)

``plot_env_panels`` is a pure drawing helper (takes a prepared feature DataFrame and
renders it); ``profile_env`` is the thin orchestrator on top of it — it composes
``trajectory_analysis.build_features`` + ``plot_env_panels`` + ``objective_stats`` and
holds no chart logic itself, so panels can be restyled without touching it.

Profile: profile_env
Panels:  plot_env_panels
Text:    render
"""

from __future__ import annotations

from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from scripts.transform import trajectory_analysis as ta
from scripts.viz import plot_theme as pt


def profile_env(env, records):
    """Build one environment's taxonomy profile: features -> panels -> stats table.

    Returns the enriched feature DataFrame (``None`` if there are no records). This is
    orchestration only — the figure is drawn by ``plot_env_panels``, so this function
    stays readable as "what a profile is made of"."""
    if not records:
        print(f"[{env}] no local data — run Section 1 with a valid HF token first.")
        return None
    df = ta.build_features(records)
    print(f"[{env}]  {len(df)} trajectories | schemas: {dict(Counter(df['_schema']))}")
    plot_env_panels(df, records, env)
    display(ta.objective_stats(df))
    return df


def plot_env_panels(df, records, env, palette=None):
    """Draw the standard 4-panel taxonomy figure for one environment.

    All chart logic lives here; the caller supplies the enriched feature DataFrame
    (``trajectory_analysis.build_features``) and the raw ``records`` (for the action
    verbs). Every panel is a single series about one environment, so all four are
    drawn in that environment's hue (``plot_theme.ENV_COLORS``) — the section is
    colour-coded by env, and no panel needs a legend."""
    color = palette or pt.env_color(env)

    _, ax = plt.subplots(2, 2, figsize=(14, 9))
    df["goal_family"].value_counts().plot.barh(ax=ax[0, 0], color=color)
    ax[0, 0].set_title(f"{env}: goal family"); ax[0, 0].invert_yaxis()
    pt.style_axes(ax[0, 0], xgrid=True)
    df["objective_type"].value_counts().plot.barh(ax=ax[0, 1], color=color)
    ax[0, 1].set_title("objective type"); ax[0, 1].invert_yaxis()
    pt.style_axes(ax[0, 1], xgrid=True)

    df["n_actions"].plot.hist(bins=30, ax=ax[1, 0], color=color)
    ax[1, 0].axvline(df["n_actions"].median(), color=pt.INK, ls="--", lw=1.2,
                     label=f'median={df["n_actions"].median():.0f}')
    ax[1, 0].set_title("actions / trajectory"); ax[1, 0].set_xlabel("n_actions"); ax[1, 0].legend()
    pt.style_axes(ax[1, 0])

    verbs = ta.action_verb_counts(records, env)
    pd.Series(dict(verbs.most_common(15))).sort_values().plot.barh(ax=ax[1, 1], color=color)
    ax[1, 1].set_title(f"top action verbs ({len(verbs)} unique)")
    pt.style_axes(ax[1, 1], xgrid=True)
    plt.tight_layout(); plt.show()


def render(rec, max_events=20):
    """Print a single trajectory (goal + interleaved action/observation events)."""
    print("=" * 90)
    print(f"[{rec['env']}] id={rec['trajectory_id']}  split={rec['split']}  schema={rec['_schema']}")
    print("GOAL:", rec["goal"][:280]); print("-" * 90)
    n, shown = max(len(rec["actions"]), len(rec["observations"])), 0
    for i in range(n):
        if i < len(rec["actions"]):
            print(f"  ACTION: {rec['actions'][i][:150]}"); shown += 1
        if i < len(rec["observations"]):
            print(f"  OBS   : {str(rec['observations'][i]).replace(chr(10),' ')[:150]}"); shown += 1
        if shown >= max_events:
            print("  ...(truncated)"); break
