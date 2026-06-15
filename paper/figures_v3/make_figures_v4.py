#!/usr/bin/env python3
"""
Generate clean, non-overlapping figures for SessionRerank+ paper.
- Fig 1: Architecture (block diagram, all text inside boxes)
- Fig 2: Algorithm flow (numbered steps, vertical timeline)
Constraints: no text overlap, all fonts 9+ pt, single accent color, white bg,
fits CRC column width (8.4cm) at 200 dpi.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT_DIR = Path('/home/user/workspace/sessionrerank/paper/figures_v3')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Design tokens
ACCENT = '#01696F'          # Hydra teal
ACCENT_SOFT = '#D6E8E9'
BORDER = '#28251D'
TEXT = '#28251D'
MUTED = '#7A7974'
SURFACE = '#F9F8F5'
CACHE_FILL = '#FFF5E6'      # warm tint for cache
CACHE_BORDER = '#964219'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.edgecolor': BORDER,
    'axes.linewidth': 0.6,
    'savefig.dpi': 220,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# =================== FIG 1 — ARCHITECTURE ===================
def make_fig1():
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 64)
    ax.axis('off')

    def box(x, y, w, h, label, sub=None, fill=SURFACE, border=BORDER,
            text_color=TEXT, fontsize=10, bold=True):
        patch = FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.5,rounding_size=1.2",
            linewidth=1.2, edgecolor=border, facecolor=fill)
        ax.add_patch(patch)
        cx, cy = x + w/2, y + h/2
        if sub:
            ax.text(cx, cy + 2.5, label, ha='center', va='center',
                    fontsize=fontsize, color=text_color,
                    fontweight='bold' if bold else 'normal')
            ax.text(cx, cy - 2.8, sub, ha='center', va='center',
                    fontsize=8.5, color=MUTED, style='italic')
        else:
            ax.text(cx, cy, label, ha='center', va='center',
                    fontsize=fontsize, color=text_color,
                    fontweight='bold' if bold else 'normal')

    def arrow(x1, y1, x2, y2, color=ACCENT, lw=1.4, style='->'):
        ar = FancyArrowPatch((x1, y1), (x2, y2),
            arrowstyle=style, mutation_scale=14,
            color=color, linewidth=lw)
        ax.add_patch(ar)

    def label_on(x, y, text, color=MUTED, fontsize=8, italic=True):
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, color=color,
                style='italic' if italic else 'normal')

    # ---- TOP ROW: Query input ----
    box(2, 53, 22, 9, 'User Query', sub='q_{t+1}',
        fill='#FFFFFF', fontsize=10)

    # ---- Base retriever (center-left) ----
    box(30, 51, 26, 11, 'Base Dense Retriever',
        sub='Sentence-BERT  (frozen)',
        fill=ACCENT_SOFT, fontsize=10)

    # ---- Candidate set ----
    box(62, 53, 22, 9, 'Top-K Candidates',
        sub='K = 100', fill='#FFFFFF', fontsize=9.5)

    arrow(24, 57.5, 30, 57.5)
    arrow(56, 57, 62, 57)

    # ---- Session history (left, vertical) ----
    box(2, 28, 22, 18, 'Session History',
        sub='H_t = (a_{i1}…a_{it})', fill='#FFFFFF', fontsize=10)

    # ---- Co-activation Cache (right) ----
    box(74, 28, 22, 18, 'Co-activation\nCache',
        sub='C_co  (sparse)', fill=CACHE_FILL,
        border=CACHE_BORDER, fontsize=10)

    # ---- Re-ranker (center, prominent) — NO sub label, internal rows only ----
    patch = FancyBboxPatch((28, 20), 42, 24,
        boxstyle="round,pad=0.5,rounding_size=1.2",
        linewidth=1.4, edgecolor=ACCENT, facecolor=ACCENT)
    ax.add_patch(patch)

    # Title at top of box
    ax.text(49, 40, 'SessionRerank+',
            ha='center', va='center', fontsize=13,
            color='white', fontweight='bold')

    # Divider line under title
    ax.plot([34, 64], [37, 37], color='white',
            linewidth=0.8, alpha=0.5)

    # Three internal rows, well spaced
    ax.text(49, 33.5, 's_cat   s_sch   s_ept   s_rec',
            ha='center', va='center', fontsize=10,
            color='white', fontweight='bold')
    ax.text(49, 29.5, 'fused with  w_base · s_base',
            ha='center', va='center', fontsize=9,
            color='white', style='italic')
    ax.text(49, 25, 'Eq. (1)    O(|H|) lookup',
            ha='center', va='center', fontsize=9,
            color='white')

    # Arrows into reranker
    arrow(24, 37, 28, 33)                 # history -> reranker
    arrow(74, 37, 70, 33)                 # cache -> reranker
    arrow(49, 51, 49, 44)                 # base retriever -> reranker
    label_on(54, 47.5, 's_base', color=MUTED)

    # ---- Output ----
    box(34, 6, 30, 10, 'Re-ranked APIs',
        sub='sorted by  s(q,a | H_t)',
        fill='#FFFFFF', fontsize=10)

    arrow(49, 20, 49, 16)

    # ---- Update loop (dashed) ----
    arrow(64, 11, 90, 11, color=CACHE_BORDER, style='->', lw=1.0)
    arrow(90, 11, 90, 28, color=CACHE_BORDER, style='->', lw=1.0)
    label_on(82, 13, 'increment-decay update  (Eq. 2)',
             color=CACHE_BORDER, fontsize=8)

    # Title
    ax.text(50, 60.5, '',
            ha='center', va='top', fontsize=11,
            color=TEXT, fontweight='bold')

    plt.savefig(OUT_DIR / 'fig1_architecture.png',
                facecolor='white', dpi=220)
    plt.close()
    print(f"Wrote {OUT_DIR/'fig1_architecture.png'}")


# =================== FIG 2 — ALGORITHM FLOW ===================
def make_fig2():
    fig, ax = plt.subplots(figsize=(7.0, 7.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis('off')

    steps = [
        ('1', 'Receive query q_{t+1} and session history H_t',
         'Anonymous, in-session only. No user ID needed.'),
        ('2', 'Base retriever scores all APIs:  s_base(q, a)',
         'Sentence-BERT cosine similarity.'),
        ('3', 'Read Co-activation Cache for every (a, a_ik) pair',
         'O(|H|) lookups; no neural inference.'),
        ('4', 'Compute three affinity signals',
         's_cat  (categorical)   s_sch  (schema Jaccard)   s_ept  (endpoint overlap)'),
        ('5', 'Add recency-decayed prior  s_rec  (window k = 3)',
         'Linearly decays influence of older turns.'),
        ('6', 'Fuse with InfoNCE-learned weights  →  s(q, a | H_t)',
         'Eq. (1). If H_t is empty, score = w_base · s_base exactly.'),
        ('7', 'Return top-N re-ranked APIs to the agent',
         'Sorted by fused score.'),
        ('8', 'On a successful call, update Co-activation Cache',
         'C_co[i,j] ← (1 − ρ) · C_co[i,j] + τ_0 · δ_{ij}   (Eq. 2)'),
    ]

    n = len(steps)
    top = 95; bot = 5
    step_h = (top - bot) / n
    x_circle = 12
    x_text = 22

    for i, (num, title, sub) in enumerate(steps):
        y = top - (i + 0.5) * step_h

        # numbered circle
        circ = mpatches.Circle((x_circle, y), 3.6,
            facecolor=ACCENT, edgecolor=ACCENT, zorder=3)
        ax.add_patch(circ)
        ax.text(x_circle, y, num, ha='center', va='center',
                fontsize=11, color='white', fontweight='bold', zorder=4)

        # box (right of circle)
        box = FancyBboxPatch((x_text, y - step_h*0.35),
            74, step_h*0.7,
            boxstyle="round,pad=0.5,rounding_size=1.0",
            linewidth=1.0, edgecolor=BORDER, facecolor=SURFACE, zorder=1)
        ax.add_patch(box)

        ax.text(x_text + 3, y + step_h*0.10, title,
                ha='left', va='center', fontsize=10,
                color=TEXT, fontweight='bold')
        ax.text(x_text + 3, y - step_h*0.16, sub,
                ha='left', va='center', fontsize=8.5,
                color=MUTED, style='italic')

        # connector to next step
        if i < n - 1:
            y_next = top - (i + 1.5) * step_h
            ar = FancyArrowPatch((x_circle, y - 3.6),
                                  (x_circle, y_next + 3.6),
                arrowstyle='->', mutation_scale=10,
                color=ACCENT, linewidth=1.2, zorder=2)
            ax.add_patch(ar)

    # legend / footer
    ax.text(50, 1.5,
            'All retrieval-time steps are O(|H|).  No GPU at serve time.',
            ha='center', va='center', fontsize=8.5,
            color=ACCENT, style='italic')

    plt.savefig(OUT_DIR / 'fig2_algorithm.png',
                facecolor='white', dpi=220)
    plt.close()
    print(f"Wrote {OUT_DIR/'fig2_algorithm.png'}")


if __name__ == '__main__':
    make_fig1()
    make_fig2()
