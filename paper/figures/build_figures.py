"""Build Fig. 1 (architecture) and Fig. 2 (two-turn ranking) for the paper.

Design language: one accent color (teal #20808D), neutrals only, no decoration.
Output: 300 DPI PNG, locked aspect ratio (T&F template requirement).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent

# Palette
TEAL = "#20808D"
TEAL_DARK = "#1B474D"
TEAL_LIGHT = "#BCE2E7"
TERRA = "#A84B2F"
INK = "#28251D"
MUTED = "#7A7974"
FAINT = "#D4D1CA"
PAPER = "#FBFBF9"

plt.rcParams.update({
    "font.family": "DejaVu Sans",   # close to Arial; embedded in PDF
    "font.size": 9,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})


# -----------------------------------------------------------------------------
# Fig. 1 — Architecture
# -----------------------------------------------------------------------------
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)

    def box(x, y, w, h, text, *, bold=False, fill=PAPER, edge=INK,
            text_color=INK, fontsize=9, rounded=True, lw=1.2):
        style = "round,pad=0.04,rounding_size=0.15" if rounded else "square,pad=0.04"
        patch = FancyBboxPatch((x, y), w, h, boxstyle=style,
                                linewidth=lw, edgecolor=edge, facecolor=fill)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight=("bold" if bold else "normal"),
                color=text_color)

    def arrow(x1, y1, x2, y2, *, color=INK, lw=1.0, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                      arrowstyle="-|>", mutation_scale=12,
                                      linewidth=lw, color=color, linestyle=ls,
                                      shrinkA=2, shrinkB=2))

    # Inputs
    box(0.1, 3.6, 1.9, 0.7, "User query  q", fill=PAPER, edge=MUTED, fontsize=9)
    box(0.1, 2.7, 1.9, 0.7, "Session history\nH = (h\u2081, \u2026, h\u2099)", fill=PAPER, edge=MUTED, fontsize=8.5)
    box(0.1, 1.8, 1.9, 0.7, "Farmer profile p\n(crop, district, season)", fill=PAPER, edge=MUTED, fontsize=8.5)

    # Three signal blocks
    box(3.0, 3.45, 3.3, 1.0,
        "Gemma 4 dense retriever\ngoogle/embeddinggemma-300m\n(768-d cosine s\u2094(q,v))",
        bold=True, fill=TEAL_LIGHT, edge=TEAL_DARK, fontsize=8.8)
    box(3.0, 2.25, 3.3, 0.95,
        "Profile mask\nm(v, p) \u2208 {0, 1}\ncrop/season/language filter",
        bold=True, fill=PAPER, edge=INK, fontsize=8.8)
    box(3.0, 1.0, 3.3, 1.05,
        "Co-activation cache  \u03c6_c(v | H)\nedge update: w \u2190 (1\u2013\u03c1)w + \u03b4\u00b71[success]\ndecay: matrix \u00d7 (1\u2013\u03c1) per session",
        bold=True, fill=PAPER, edge=TERRA, text_color=TERRA, fontsize=8.6)

    # Combine -> score
    box(7.4, 2.0, 2.4, 1.4,
        "score(v | q, H)\n=  w_d \u00b7 s_d(q,v)\n+ w_m \u00b7 m(v,p)\n+ w_c \u00b7 \u03c6_c(v|H)",
        bold=True, fill=PAPER, edge=INK, fontsize=9)

    # Connections from inputs to signals
    arrow(2.0, 3.95, 3.0, 3.95, color=TEAL_DARK, lw=1.4)   # q -> Gemma
    arrow(2.0, 2.1, 3.0, 2.7, color=MUTED)                  # profile -> mask
    arrow(2.0, 3.05, 3.0, 1.5, color=TERRA, lw=1.2)        # H -> cache

    # Signal blocks -> score combiner
    arrow(6.3, 3.95, 7.4, 3.0, color=TEAL_DARK, lw=1.4)
    arrow(6.3, 2.7, 7.4, 2.7, color=INK)
    arrow(6.3, 1.55, 7.4, 2.4, color=TERRA, lw=1.2)

    # Output
    box(7.4, 0.3, 2.4, 0.9, "Top-k API ranking\n(returned to LLM agent)",
        bold=True, fill=TEAL, edge=TEAL_DARK, text_color="white", fontsize=9)
    arrow(8.6, 1.95, 8.6, 1.2, color=INK, lw=1.4)

    # Annotation: online update loop (cache learns from success)
    arrow(7.4, 0.75, 5.4, 1.0, color=TERRA, lw=1.0, ls="--")
    ax.text(6.4, 0.55, "successful (h\u2099 \u2192 v) \u21d2 update edge",
            ha="center", va="center", fontsize=7.5, color=TERRA, style="italic")

    out = HERE / "fig1_architecture.png"
    fig.savefig(out, bbox_inches="tight", dpi=300, facecolor=PAPER)
    plt.close(fig)
    print("wrote", out)


# -----------------------------------------------------------------------------
# Fig. 2 — Two-turn ranking
# -----------------------------------------------------------------------------
def fig2_two_turn():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), dpi=300, sharey=True)
    fig.patch.set_facecolor(PAPER)

    apis_t1 = ["KSDA Pest Alert", "Raitha Mitra", "UAS-B Plantix",
               "KSNDMC Drought", "K-Kisan Helpline", "PMFBY", "Bhoomi RTC"]
    dense_t1 = [0.71, 0.66, 0.63, 0.55, 0.52, 0.47, 0.32]
    coact_t1 = [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]

    apis_t2 = ["PMFBY", "Raitha Mitra", "KSDA Pest Alert",
               "K-Kisan Helpline", "Soil Health Card", "Agmarknet KA", "Bhoomi RTC"]
    dense_t2 = [0.47, 0.66, 0.51, 0.40, 0.36, 0.34, 0.30]
    coact_t2 = [1.39, 0.92, 0.48, 0.21, 0.05, 0.00, 0.00]

    def plot(ax, apis, dense, coact, title, highlight=None):
        y = range(len(apis))
        ax.barh(y, dense, color=TEAL_LIGHT, edgecolor=TEAL_DARK,
                linewidth=0.8, label="Gemma 4 dense score")
        ax.barh(y, coact, left=dense, color=TERRA, edgecolor=TEAL_DARK,
                linewidth=0.8, label="Co-activation bonus", alpha=0.85)
        ax.set_yticks(list(y))
        ax.set_yticklabels(apis, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(0, 2.4)
        ax.set_xlabel("score", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
        # value labels at right edge of each bar
        for i, (d, c) in enumerate(zip(dense, coact)):
            total = d + c
            label = f"{total:.2f}" + (f"  (+{c:.2f})" if c > 0 else "")
            ax.text(total + 0.04, i, label, va="center", fontsize=7.5, color=INK)
        if highlight is not None:
            for j, lab in enumerate(ax.get_yticklabels()):
                if j == highlight:
                    lab.set_fontweight("bold")
                    lab.set_color(TERRA)
            # also annotate the climb
            ax.annotate("PMFBY: rank 6 → rank 1",
                        xy=(dense[0] + coact[0], 0), xytext=(1.55, 1.2),
                        fontsize=7.8, color=TERRA, style="italic",
                        arrowprops=dict(arrowstyle="->", color=TERRA, lw=0.8))

    plot(axes[0], apis_t1, dense_t1, coact_t1,
         "Turn 1 \u2014 cold start\n\u201cMy ragi crop has yellowing leaves\u201d")
    plot(axes[1], apis_t2, dense_t2, coact_t2,
         "Turn 2 \u2014 after Plantix \u2713\n\u201cWhen can I claim crop insurance?\u201d",
         highlight=0)

    # Single shared legend below
    handles = [
        mpatches.Patch(color=TEAL_LIGHT, label="Gemma 4 dense  s_d(q,v)"),
        mpatches.Patch(color=TERRA, alpha=0.85, label="Co-activation bonus  w_c\u00b7\u03c6_c(v|H)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = HERE / "fig2_two_turn.png"
    fig.savefig(out, bbox_inches="tight", dpi=300, facecolor=PAPER)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig1_architecture()
    fig2_two_turn()
