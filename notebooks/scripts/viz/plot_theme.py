#!/usr/bin/env python3
"""Shared matplotlib theme for the AOMT notebooks.

One import (``import plot_theme as pt; pt.apply_theme()``) gives every figure the
same surface, ink, grid and fonts, and — more importantly — the same *colour
meanings*: an environment keeps its hue in every chart it appears in, so a reader
who learns "ScienceWorld is blue" is never contradicted later.

Palettes were checked with the dataviz validator (lightness band, chroma floor,
CVD separation, contrast) rather than picked by eye:

- ``ENV_COLORS``   blue / orange / violet  — all checks pass (worst adjacent CVD dE 96.7)
- ``BLOCK_COLORS`` green / aqua / red      — aqua is under 3:1 on the light surface, so
- ``MIX_COLORS``   yellow / magenta / gray — likewise: both stacked charts that use these
  carry in-segment value labels (``label_stacked``), which is the required relief.
"""

from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- palettes
# Environment identity. Fixed mapping — never assign by rank or plot order.
ENV_COLORS = {
    "scienceworld": "#2a78d6",   # blue
    "alfworld":     "#eb6834",   # orange
    "webshop":      "#4a3aa7",   # violet
}

# Block composition (think / action / observation).
BLOCK_COLORS = {
    "think":       "#008300",
    "action":      "#1baf7a",
    "observation": "#e34948",
}

# Action-type mix. "other" is a residual bucket, so it stays deliberately gray.
MIX_COLORS = {
    "navigation":   "#eda100",
    "manipulation": "#e87ba4",
    "other":        "#898781",
}

# Chart chrome.
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_MUTED = "#898781"
GRID      = "#e1e0d9"
BASELINE  = "#c3c2b7"

# Sequential (single hue, light -> dark) for any magnitude encoding.
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]


def env_color(env: str) -> str:
    """Hue for an environment; falls back to muted ink for anything unmapped."""
    return ENV_COLORS.get(env, INK_MUTED)


def apply_theme() -> None:
    """Install the theme globally. Call once, in the notebook's setup cell."""
    mpl.rcParams.update({
        "figure.figsize":      (9, 4.5),
        "figure.facecolor":    SURFACE,
        "figure.dpi":          110,
        "savefig.facecolor":   SURFACE,
        "axes.facecolor":      SURFACE,
        "axes.edgecolor":      BASELINE,
        "axes.linewidth":      0.8,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.grid":           True,
        "axes.axisbelow":      True,
        "axes.titlesize":      11,
        "axes.titleweight":    "semibold",
        "axes.titlelocation":  "left",
        "axes.titlepad":       8,
        "axes.labelsize":      9.5,
        "axes.labelcolor":     INK,
        "axes.prop_cycle":     mpl.cycler(color=list(ENV_COLORS.values())),
        "grid.color":          GRID,
        "grid.linewidth":      0.8,
        "grid.linestyle":      "-",          # solid hairline; dashes read as "threshold"
        "grid.alpha":          1.0,
        "text.color":          INK,
        "xtick.color":         INK_MUTED,
        "ytick.color":         INK_MUTED,
        "xtick.labelsize":     9,
        "ytick.labelsize":     9,
        "xtick.labelcolor":    INK,
        "ytick.labelcolor":    INK,
        "lines.linewidth":     2.0,
        "lines.markersize":    8,
        "legend.frameon":      False,
        "legend.fontsize":     9,
        "font.size":           10,
        "font.family":         "sans-serif",
        "font.sans-serif":     ["Segoe UI", "DejaVu Sans", "sans-serif"],
    })


def style_axes(ax, xgrid: bool = False) -> None:
    """Grid on the value axis only — a grid across the category axis is noise."""
    ax.grid(axis="x" if xgrid else "y", color=GRID, linewidth=0.8)
    ax.grid(axis="y" if xgrid else "x", visible=False)


def label_stacked(ax, min_share: float = 0.06, fmt: str = "{:.0%}") -> None:
    """Write each stacked segment's share inside it, skipping segments too small to
    hold the text. This is the contrast relief for BLOCK_COLORS / MIX_COLORS, and it
    also means no value is reachable only by hovering a colour."""
    for container in ax.containers:
        horizontal = getattr(container, "orientation", "vertical") == "horizontal"
        sizes = [b.get_width() if horizontal else b.get_height() for b in container]
        labels = [fmt.format(s) if s >= min_share else "" for s in sizes]
        ax.bar_label(labels=labels, container=container, label_type="center",
                     color=SURFACE, fontsize=8, fontweight="semibold")


def separate_segments(ax, gap: float = 2.0) -> None:
    """A 2px surface gap between stacked fills, instead of a border around each."""
    for container in ax.containers:
        for bar in container:
            bar.set_edgecolor(SURFACE)
            bar.set_linewidth(gap)


def color_env_bars(ax, envs) -> None:
    """Recolour a bar chart whose categories are environments, by entity."""
    for bar, env in zip(ax.patches, envs):
        bar.set_facecolor(env_color(env))


def env_legend(ax, envs=None, **kw):
    """A legend keyed to environments, present whenever >= 2 envs are drawn."""
    envs = list(envs or ENV_COLORS)
    handles = [plt.Line2D([], [], color=env_color(e), lw=3, label=e) for e in envs]
    return ax.legend(handles=handles, **kw)


def ecdf(ax, vals, label=None, color=None):
    """Plot the empirical CDF of ``vals`` (a pandas Series or any array-like) on ``ax``.

    Non-positive values are dropped so the curve stays meaningful on a log x-axis;
    an empty input draws nothing (so callers can loop over sparse slices safely)."""
    v = np.sort(np.asarray(vals, dtype=float))
    v = v[v > 0]
    if len(v):
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), label=label, color=color)
