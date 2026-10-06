"""Figure generation.  One palette, one typographic scale, one convention throughout.

Protocol A and Protocol B quantities are never placed on the same axis, and every axis
label names the protocol it belongs to.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import protocol_b as pb
from wsdata import DOMAINS, POOL_LABEL, POOLS

# MixtureVitae, the permissive pool, is warm; the non-permissive pools are cool.
MV_LIGHT, MV_DARK = "#f0b27a", "#c0552b"
REF_LIGHT, REF_DARK = "#c7d3e0", "#3d5a80"
GREY = "#5a5a5a"
POOL_COLORS = {
    "MV": (MV_LIGHT, MV_DARK),
    "Nemotron": (REF_LIGHT, REF_DARK),
    "FineWeb-Edu": (REF_LIGHT, REF_DARK),
}
LINE_COLORS = {"MV": MV_DARK, "Nemotron": "#2a9d8f", "FineWeb-Edu": "#5c6bc0"}
POOL_TICK = {"MV": "MixtureVitae\n(permissive)", "Nemotron": "Nemotron-\nCC-HQ",
             "FineWeb-Edu": "FineWeb-\nEdu"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#e4e4e4",
    "grid.linewidth": 0.6,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})

STAGE_LABEL = {"base": "Base", "sft": "SFT", "dpo": "DPO"}


def _save(fig: plt.Figure, out: Path, name: str) -> Path:
    path = out / name
    fig.savefig(path)
    plt.close(fig)
    return path


def _paired_panel(ax, suite, context_length, stage, value_col="mean", err_col="sd",
                  ylabel=None, title=None, ylim_pad=1.35):
    """Grouped `without / with front-loading` bars for the three source pools."""
    # Paired bars are drawn with a small central gap so that the two value labels
    # clear each other even when the pair differs by only a few tenths.
    width, offset = 0.32, 0.19
    xs = np.arange(len(POOLS))
    tops = []
    for i, pool in enumerate(POOLS):
        light, dark = POOL_COLORS[pool]
        row_no = suite[(suite.model_group == pool) & (~suite.front_loading)
                       & (suite.context_length == context_length) & (suite.stage == stage)]
        row_yes = suite[(suite.model_group == pool) & (suite.front_loading)
                        & (suite.context_length == context_length) & (suite.stage == stage)]
        v0, v1 = float(row_no[value_col].iloc[0]), float(row_yes[value_col].iloc[0])
        e0 = float(row_no[err_col].iloc[0]) if err_col else 0.0
        e1 = float(row_yes[err_col].iloc[0]) if err_col else 0.0
        ax.bar(xs[i] - offset, v0, width, color=light, edgecolor="white",
               yerr=e0, error_kw=dict(ecolor=GREY, lw=0.8, capsize=1.8))
        ax.bar(xs[i] + offset, v1, width, color=dark, edgecolor="white",
               yerr=e1, error_kw=dict(ecolor=GREY, lw=0.8, capsize=1.8))
        ax.text(xs[i] - offset, v0 + e0, f"{v0:.1f}", ha="center", va="bottom",
                fontsize=6.5, color=GREY)
        ax.text(xs[i] + offset, v1 + e1, f"{v1:.1f}", ha="center", va="bottom",
                fontsize=6.5, fontweight="bold", color=dark)
        tops.append(max(v0 + e0, v1 + e1))
        ax.text(xs[i], max(v0, v1) * 0.0 + max(tops) * 1.16, f"$\\Delta$ {v1 - v0:+.1f}",
                ha="center", va="bottom", fontsize=7.5, style="italic",
                color=MV_DARK if pool == "MV" else GREY)
    ax.set_xticks(xs)
    ax.set_xticklabels([POOL_TICK[p] for p in POOLS], fontsize=7)
    ax.set_ylim(0, max(tops) * ylim_pad)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, style="italic", color=GREY)
    return max(tops)


def factorial_main(suite: pd.DataFrame, language: pd.DataFrame, out: Path) -> Path:
    """Figure 2: the headline factorial result."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25))
    _paired_panel(axes[0], suite, 4096, "sft",
                  ylabel="Reasoning suite avg.", title="Reasoning, 4K, after Tulu3 SFT")
    _paired_panel(axes[1], suite, 16384, "sft", title="Reasoning, 16K, after Tulu3 SFT")

    lang = language.copy()
    lang["stage"] = "base"
    lang = lang.rename(columns={"language": "mean"})
    lang["sd"] = 0.0
    _paired_panel(axes[2], lang, 16384, "base",
                  ylabel="Language understanding avg.",
                  title="Language understanding, 16K base")

    # Reference line at the permissive pool's front-loaded level, so the reader can read
    # the permissiveness contrast at matched front-loading straight off the axis.
    for ax, ctx, stage, frame in ((axes[0], 4096, "sft", suite), (axes[1], 16384, "sft", suite),
                                  (axes[2], 16384, "base", lang)):
        mv = frame[(frame.model_group == "MV") & (frame.front_loading)
                   & (frame.context_length == ctx) & (frame.stage == stage)]["mean"].iloc[0]
        ax.axhline(float(mv), color=MV_DARK, lw=0.8, ls=(0, (4, 3)), alpha=0.75, zorder=0)

    handles = [Patch(facecolor=REF_LIGHT, label="no reasoning front-loading"),
               Patch(facecolor=REF_DARK, label="reasoning front-loaded"),
               Patch(facecolor=MV_DARK, label="permissive substrate (highlighted)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout()
    return _save(fig, out, "factorial_main.pdf")


def frontloading_trajectory(contrasts: pd.DataFrame, out: Path) -> Path:
    """Figure 3: the front-loading effect across post-training stages."""
    fl = contrasts[contrasts.family == "front-loading"]
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.2), sharey=True)
    stages = ["base", "sft", "dpo"]
    for ax, ctx, title in ((axes[0], 4096, "4K context"), (axes[1], 16384, "16K context")):
        for pool in POOLS:
            sub = fl[(fl.label == pool) & (fl.context_length == ctx)].set_index("stage")
            y = [sub.loc[s, "estimate"] for s in stages]
            lo = [sub.loc[s, "lo"] for s in stages]
            hi = [sub.loc[s, "hi"] for s in stages]
            xs = np.arange(len(stages))
            ax.plot(xs, y, "-o", ms=3.5, lw=1.6, color=LINE_COLORS[pool],
                    label=POOL_LABEL[pool] + (" (permissive)" if pool == "MV" else ""))
            ax.fill_between(xs, lo, hi, color=LINE_COLORS[pool], alpha=0.14, lw=0)
        ax.set_xticks(np.arange(len(stages)))
        ax.set_xticklabels([STAGE_LABEL[s] for s in stages])
        ax.set_title(title, style="italic", color=GREY)
        ax.set_ylim(0, None)
    axes[0].set_ylabel("Front-loading effect (pp)")
    axes[0].legend(frameon=False, loc="lower left", fontsize=7)
    fig.tight_layout()
    return _save(fig, out, "frontloading_trajectory.pdf")


ZOO_MATCHED = [("MV", 16384, MV_DARK), ("Comma0.1", 16384, "#7f8c8d"),
               ("Nemotron", 16384, "#2a9d8f"), ("FineWeb-Edu", 16384, "#5c6bc0")]
ZOO_EXTERNAL = [("MV", 16384, MV_DARK), ("SmolLM2", 16384, "#6ab04c"),
                ("Qwen2.5", 4096, "#e1b12c"), ("Qwen3", 4096, "#8e6a3f")]
ZOO_NAME = {"MV": "MixtureVitae 16K", "Comma0.1": "Comma0.1 16K", "Nemotron": "Nemotron-CC-HQ 16K",
            "FineWeb-Edu": "FineWeb-Edu 16K", "SmolLM2": "SmolLM2 16K",
            "Qwen2.5": "Qwen2.5", "Qwen3": "Qwen3"}


def phase1_zoo(zoo_suite: pd.DataFrame, out: Path) -> Path:
    """Figure 4: Phase 1 trajectories across the model zoo."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.1), sharey=True)
    stages = ["base", "sft", "dpo"]
    for ax, spec, title in (
        (axes[0], ZOO_MATCHED, "Matched compute, 300B tokens"),
        (axes[1], ZOO_EXTERNAL, "Open-weights context, 11\u201336T tokens"),
    ):
        for group, ctx, color in spec:
            sub = zoo_suite[(zoo_suite.model_group == group)
                            & (zoo_suite.context_length == ctx)].set_index("stage")
            xs = np.arange(len(stages))
            y = [sub.loc[s, "mean"] for s in stages]
            e = [sub.loc[s, "sd"] for s in stages]
            ax.errorbar(xs, y, yerr=e, fmt="-o", ms=3.5, lw=1.6, capsize=2,
                        color=color, label=ZOO_NAME[group],
                        zorder=3 if group == "MV" else 2)
        ax.set_xticks(np.arange(len(stages)))
        ax.set_xticklabels([STAGE_LABEL[s] for s in stages])
        ax.set_title(title, style="italic", color=GREY)
        # "best" puts each legend in that panel's emptiest region rather than a
        # hard-coded corner, and the translucent patch keeps it legible if a future
        # line ever runs underneath it.
        ax.legend(fontsize=7, loc="best", frameon=True, framealpha=0.9,
                  edgecolor="none", facecolor="white", borderpad=0.3,
                  labelspacing=0.35, handlelength=1.6)
    axes[0].set_ylabel("Reasoning suite avg.")
    axes[0].margins(y=0.14)
    fig.tight_layout()
    return _save(fig, out, "phase1_zoo.pdf")


def phase2_receptivity(out: Path) -> Path:
    """Figure 5: paired change under reasoning post-training, Protocol B, single seed."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.45),
                             gridspec_kw={"width_ratios": [0.95, 1.3]})

    ax = axes[0]
    # Sorted by effect so the "one model gains, the rest lose" shape reads at a glance,
    # and labelled with names only: six of the seven start from Tulu3 SFT, so repeating
    # the starting point on every tick cost two lines each to say nothing. The one
    # exception is stated in the caption.
    ordered = sorted(pb.PHASE2_PAIRED, key=lambda r: r[3] - r[2])
    names = [n for n, _, _, _ in ordered]
    deltas = [b - a for _, _, a, b in ordered]
    colors = [MV_DARK if n == "MixtureVitae" else REF_DARK for n in names]
    ys = np.arange(len(names))
    ax.barh(ys, deltas, color=colors, height=0.66)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=7.5)
    for y, d in zip(ys, deltas):
        ax.text(d + (0.5 if d >= 0 else -0.5), y, f"{d:+.1f}", va="center",
                ha="left" if d >= 0 else "right", fontsize=7)
    ax.set_xlim(min(deltas) * 1.25, max(deltas) * 3.2)
    ax.set_xlabel(r"$\Delta$ reasoning suite avg. under OT3-30K")
    ax.set_title("Each model from its own starting checkpoint", style="italic", color=GREY)
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    bases = ["MixtureVitae", "SmolLM2", "Comma0.1", "FineWeb-Edu", "Nemotron-CC-HQ"]
    width = 0.19
    xs = np.arange(len(bases))
    corpus_colors = [MV_DARK, "#5c6bc0", "#2a9d8f", "#b0a08c"]
    for j, corpus in enumerate(pb.HELDOUT_CORPORA):
        est = [pb.HELDOUT_MATH500[b][j][0] for b in bases]
        lo = [pb.HELDOUT_MATH500[b][j][0] - pb.HELDOUT_MATH500[b][j][1] for b in bases]
        hi = [pb.HELDOUT_MATH500[b][j][2] - pb.HELDOUT_MATH500[b][j][0] for b in bases]
        ax.bar(xs + (j - 1.5) * width, est, width, color=corpus_colors[j], label=corpus,
               yerr=[lo, hi], error_kw=dict(ecolor=GREY, lw=0.7, capsize=1.4))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(["MixtureVitae", "SmolLM2", "Comma0.1",
                        "FineWeb-\nEdu", "Nemotron-\nCC-HQ"], fontsize=7)
    ax.set_ylabel(r"$\Delta$ MATH500 (pp)")
    ax.set_title("Four reasoning corpora, from the Tulu3 SFT checkpoint",
                 style="italic", color=GREY)
    ax.set_ylim(min(-6, ax.get_ylim()[0]), 52)
    ax.legend(frameon=False, fontsize=6.8, ncol=2, loc="upper right",
              bbox_to_anchor=(1.02, 1.03), handlelength=1.2, columnspacing=1.0)
    fig.tight_layout()
    return _save(fig, out, "phase2_receptivity.pdf")


def frontloading_domains(scores: pd.DataFrame, out: Path) -> Path:
    """Appendix figure: the front-loading effect by capability domain."""
    dom = (scores.groupby(["model_group", "front_loading", "context_length", "stage",
                           "seed", "domain"], observed=True)["score"].mean()
           .groupby(["model_group", "front_loading", "context_length", "stage", "domain"])
           .mean().reset_index(name="score"))
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4), sharey=True)
    width = 0.26
    xs = np.arange(len(DOMAINS))
    for ax, ctx, title in ((axes[0], 4096, "4K context"), (axes[1], 16384, "16K context")):
        for i, pool in enumerate(POOLS):
            eff = []
            for d in DOMAINS:
                sel = dom[(dom.model_group == pool) & (dom.context_length == ctx)
                          & (dom.stage == "sft") & (dom.domain == d)]
                eff.append(float(sel[sel.front_loading]["score"].iloc[0])
                           - float(sel[~sel.front_loading]["score"].iloc[0]))
            ax.bar(xs + (i - 1) * width, eff, width, color=LINE_COLORS[pool])
            for x, v in zip(xs + (i - 1) * width, eff):
                ax.text(x, v + (0.5 if v >= 0 else -0.5), f"{v:+.0f}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=6.2, color=GREY)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels([d.capitalize() for d in DOMAINS], fontsize=7)
        ax.set_title(title, style="italic", color=GREY)
        ax.set_ylim(-6, 38)
    axes[0].set_ylabel("Front-loading effect (pp)")
    handles = [Patch(facecolor=LINE_COLORS[p], label=POOL_LABEL[p]) for p in POOLS]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.09))
    fig.tight_layout()
    return _save(fig, out, "frontloading_domains.pdf")
