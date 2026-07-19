"""
Matplotlib charts for the Streamlit UI (A1-A3).

Three live, per-proposal charts built only from the agent's result:
    A1  fig_trigger_words, why this decision (signed word contributions)
    A2  fig_policy_scores, why these policies (RAG match score per policy)
    A3  fig_probability, the Red-Flag probability against the threshold

Colors come from the project data-viz palette (validated status + diverging
values), so they stay colorblind-safe and readable in light and dark themes:
critical/High  #d03b3b; warning/Medium  #fab219; toward Red Flag #d03b3b; toward compliant #2a78d6 (diverging blue<->red)
Identity is never colour-alone: every mark carries an axis label and a legend.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Palette (from references/palette.md - validated, do not eyeball).
RED = "#d03b3b"        # critical / High / toward Red Flag
AMBER = "#fab219"      # warning / Medium
BLUE = "#2a78d6"       # diverging cool pole / toward compliant
DIM = "#b7b7b3"        # not matched / track


def light_ink():
    return {"text": "#222222", "muted": "#6b6b6b", "grid": "#dcdcda"}


def dark_ink():
    return {"text": "#e6e6e6", "muted": "#9aa0a6", "grid": "#3a3a38"}


def _style(ax, ink):
    # Recessive axes: transparent background, thin muted spines, no clutter.
    ax.set_facecolor("none")
    ax.figure.patch.set_alpha(0.0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(ink["grid"])
    ax.tick_params(colors=ink["muted"], labelsize=9)
    ax.xaxis.label.set_color(ink["muted"])
    ax.yaxis.label.set_color(ink["muted"])
    ax.title.set_color(ink["text"])


def fig_trigger_words(trigger_words, n=10, ink=None):
    # A1 - horizontal diverging bars: red pushes to Red Flag, blue to compliant.
    ink = ink or light_ink()
    items = list(trigger_words)[:n]
    items = list(reversed(items))            # strongest on top after barh
    words = [w for w, _ in items]
    scores = [s for _, s in items]
    colors = [RED if s > 0 else BLUE for s in scores]

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.barh(words, scores, color=colors, height=0.68)
    ax.axvline(0, color=ink["grid"], linewidth=1)
    ax.set_title("Why this decision — trigger words", fontsize=11, loc="left")
    ax.set_xlabel("contribution (tfidf x coefficient)")
    for label in ax.get_yticklabels():
        label.set_color(ink["text"])
    ax.legend(handles=[Patch(color=RED, label="toward Red Flag"),
                       Patch(color=BLUE, label="toward compliant")],
              loc="lower right", fontsize=8, frameon=False,
              labelcolor=ink["text"])
    _style(ax, ink)
    fig.tight_layout()
    return fig


def fig_policy_scores(retrieved, matched_ids, severities, ink=None):
    # A2 - one bar per policy coloured by severity (matched = solid, else dim); severities maps policy_id -> "High"/"Medium".
    ink = ink or light_ink()
    hits = list(retrieved)
    ids = [h["policy_id"] for h in hits]
    scores = [h["score"] for h in hits]
    matched = set(matched_ids)

    colors = []
    for pid in ids:
        base = RED if severities.get(pid) == "High" else AMBER
        colors.append(base if pid in matched else DIM)

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.bar(ids, scores, color=colors, width=0.66)
    # Cutoff line at the lowest matched score - everything above it was kept.
    handles = [Patch(color=RED, label="High"),
               Patch(color=AMBER, label="Medium"),
               Patch(color=DIM, label="not matched")]
    matched_scores = [h["score"] for h in hits if h["policy_id"] in matched]
    if matched_scores:
        cut = min(matched_scores)
        ax.axhline(cut, color=ink["muted"], linewidth=1, linestyle="--")
        handles.append(Line2D([0], [0], color=ink["muted"], linestyle="--",
                              label="match cutoff"))
    ax.set_title("Why these policies — RAG match score", fontsize=11, loc="left")
    ax.set_ylabel("match score")
    for label in ax.get_xticklabels():
        label.set_color(ink["text"])
        label.set_rotation(45)
        label.set_ha("right")
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=False,
              labelcolor=ink["text"], ncol=2)
    _style(ax, ink)
    fig.tight_layout()
    return fig


def fig_probability(proba, threshold=0.40, ink=None):
    # A3 - a single horizontal gauge: fill to proba, threshold marked
    ink = ink or light_ink()
    fill = RED if proba >= threshold else "#0ca30c"      # good green if below

    fig, ax = plt.subplots(figsize=(5.6, 1.25))
    ax.barh([0], [1.0], color=DIM, height=0.5)           # full 0..1 track
    ax.barh([0], [proba], color=fill, height=0.5)
    ax.axvline(threshold, color=ink["text"], linewidth=2)
    ax.text(threshold, 0.55, " threshold " + str(threshold), va="bottom",
            ha="left", fontsize=8, color=ink["text"])
    ax.text(proba, -0.55, "p = " + str(round(proba, 3)), va="top",
            ha="center", fontsize=10, color=ink["text"])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.9, 0.9)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("Red-Flag probability", fontsize=11, loc="left")
    _style(ax, ink)
    fig.tight_layout()
    return fig
