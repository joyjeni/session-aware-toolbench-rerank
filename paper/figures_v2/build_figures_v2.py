"""Journal-grade figures for the SessionRerank+ T&F CRC paper.

Palette (professional/journal-suitable):
  Primary navy  #0B2545   (boxes, primary text)
  Slate         #13315C   (secondary boxes)
  Steel         #8DA9C4   (tertiary, supporting)
  Accent teal   #1B6E8A   (highlight, key flows)
  Accent gold   #C49A2C   (single warm callout, sparingly)
  Page          #F8F6F1
  Faint grid    #E5E1D6
  Ink (text)    #1A1A1A

Style:
  - Single sans-serif (DejaVu Sans) for body
  - 300 DPI, exact figure widths (7.2 in for 2-col-wide; 3.5 in for 1-col)
  - Minimal chartjunk, thin axis lines, light grid
  - No gradients, no shadows, no 3D
  - No element overlap — strict bounding-box layout
"""

from pathlib import Path
import json, math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
NAVY   = "#0B2545"
SLATE  = "#13315C"
STEEL  = "#8DA9C4"
TEAL   = "#1B6E8A"
GOLD   = "#C49A2C"
PAPER  = "#F8F6F1"
GRID   = "#E5E1D6"
INK    = "#1A1A1A"
MUTED  = "#5A5A5A"
WHITE  = "#FFFFFF"
TEAL_FILL = "#E6EEF1"
GOLD_FILL = "#FAF1DC"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK,
    "axes.titlesize": 10.5,
    "axes.titleweight": "semibold",
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": WHITE,
})


def _box(ax, xy, w, h, text, fc=WHITE, ec=NAVY, tc=INK, fs=9.5,
         lw=1.0, rad=0.04, weight="normal", align="center"):
    x, y = xy
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0.012,rounding_size={rad}",
                         linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(box)
    if align == "center":
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                color=tc, fontsize=fs, fontweight=weight)
    elif align == "left":
        ax.text(x + 0.1, y + h/2, text, ha="left", va="center",
                color=tc, fontsize=fs, fontweight=weight)
    return box


def _arrow(ax, p1, p2, color=NAVY, lw=1.2, style="-|>", connect="arc3,rad=0",
           text=None, text_xy=None, fs=8.5, ls="-"):
    arr = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=10,
                          color=color, lw=lw, connectionstyle=connect,
                          linestyle=ls)
    ax.add_patch(arr)
    if text and text_xy:
        ax.text(text_xy[0], text_xy[1], text, ha="center", va="center",
                color=color, fontsize=fs, style="italic")
    return arr


# ===========================================================================
# Fig. 1 — System Architecture (5-column grid, no overlap)
# ===========================================================================
def fig1_architecture():
    """Layout grid (14 x 9 units):
       Col A (inputs):   x = 0.3 .. 2.7   (w=2.4)
       Col B (stage 1):  x = 3.3 .. 6.1   (w=2.8)
       Col C (stage 2):  x = 6.7 .. 9.5   (w=2.8)
       Col D (fusion):   x = 10.1 .. 13.7 (w=3.6)
       Cache band:       y = 1.4 .. 3.4   (full width across B-C)
       Top boxes:        y = 5.0 .. 6.5   (h=1.5)
    """
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 9); ax.axis("off")

    # Title strip (own band, y=8.0–8.8)
    ax.text(7, 8.5, "SessionRerank+ : Session-Aware Re-Ranking for API Tool Retrieval",
            ha="center", va="center", fontsize=12, fontweight="bold", color=NAVY)
    ax.text(7, 8.0, "Two-stage cascade with a co-activation cache over interaction history",
            ha="center", va="center", fontsize=9, color=MUTED, style="italic")

    # ---- Inputs column (A) ----
    _box(ax, (0.3, 6.4), 2.4, 0.9,
         "User query  q_t\n(turn t)",
         fc=PAPER, ec=NAVY, weight="bold", fs=9.5)
    _box(ax, (0.3, 5.0), 2.4, 0.9,
         "Session history\n H_{<t}",
         fc=PAPER, ec=NAVY, fs=9.5)

    # ---- Stage 1 (B) ----
    _box(ax, (3.3, 5.0), 2.8, 2.3,
         "Stage 1\nCandidate Generator\n\n• BM25 lexical\n• Dense (MiniLM)\n• Rank-fusion → Top-100",
         fc=WHITE, ec=SLATE, lw=1.2, fs=8.8)

    # ---- Stage 2 (C) ----
    _box(ax, (6.7, 5.0), 2.8, 2.3,
         "Stage 2\nGemma 4 Re-Ranker\n\n• Prompted scoring\n• LoRA fine-tuned\n  on ToolBench",
         fc=WHITE, ec=SLATE, lw=1.2, fs=8.8)

    # ---- Score fusion (D) ----
    _box(ax, (10.1, 5.0), 3.6, 2.3,
         "Score Fusion\n\n s* = α · s_gemma\n      + β · s_cache\n      + γ · s_dense",
         fc=WHITE, ec=NAVY, lw=1.2, fs=9)

    # ---- Co-activation cache band (full middle row) ----
    _box(ax, (3.3, 2.4), 6.2, 1.6,
         "Co-Activation Cache  C\n"
         "C(a,b) = w · 1[a,b co-fire] + (1−w) · γ · C(a,b)\n"
         "edge weight w ∈ [0,1],  decay γ",
         fc=TEAL_FILL, ec=TEAL, lw=1.4, fs=8.8, weight="bold")

    # ---- Output (D-bottom) ----
    _box(ax, (10.1, 2.4), 3.6, 1.6,
         "Top-K  API tools\nfor turn t\n→ executed by agent",
         fc=GOLD_FILL, ec=GOLD, lw=1.4, fs=9, weight="bold")

    # ---- Arrows (no overlap with boxes) ----
    # query -> stage1
    _arrow(ax, (2.7, 6.85), (3.3, 6.4))
    # history -> stage1
    _arrow(ax, (2.7, 5.45), (3.3, 5.7), ls=":")
    # history -> cache
    _arrow(ax, (1.5, 5.0), (1.5, 3.2), color=TEAL, lw=1.2)
    _arrow(ax, (1.5, 3.2), (3.3, 3.2), color=TEAL, lw=1.2)
    # stage1 -> stage2
    _arrow(ax, (6.1, 6.15), (6.7, 6.15),
           text="Top-100", text_xy=(6.4, 6.45), fs=8)
    # stage2 -> fusion
    _arrow(ax, (9.5, 6.15), (10.1, 6.15))
    # cache -> fusion
    _arrow(ax, (9.5, 3.2), (10.1, 5.5), color=TEAL, lw=1.4,
           text="cache score", text_xy=(9.85, 4.3), fs=8)
    # fusion -> output
    _arrow(ax, (11.9, 5.0), (11.9, 4.0), color=GOLD, lw=1.4)
    # feedback loop (output -> history) — drawn as a gentle arc that stays
    # in the empty band below the cache (y≈1.5–1.9), not crossing it
    _arrow(ax, (10.1, 2.6), (2.7, 5.0),
           color=MUTED, lw=0.9, connect="arc3,rad=-0.4", ls="--")
    ax.text(6.4, 1.55, "turn-level feedback (executed result → H)",
            ha="center", fontsize=8, color=MUTED, style="italic")

    # ---- Legend (own band y=0.3–0.9) ----
    ax.text(0.3, 0.55, "Legend:", fontsize=8.5, color=INK, fontweight="bold")
    lx = 1.5
    for col, lbl in [(NAVY, "primary flow"),
                     (TEAL, "co-activation cache"),
                     (GOLD, "output"),
                     (MUTED, "feedback (dashed)")]:
        ax.add_patch(Rectangle((lx, 0.45), 0.25, 0.22, color=col))
        ax.text(lx + 0.4, 0.56, lbl, fontsize=8.3, color=INK, va="center")
        lx += 3.0

    out = HERE / "fig1_architecture.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


# ===========================================================================
# Fig. 2 — Two-Turn Co-Activation Example (3-column layout, no overlap)
# ===========================================================================
def fig2_two_turn():
    """Layout (14 x 8):
       Col A (Turn 1):   x = 0.3 .. 4.3
       Col B (Cache):    x = 4.9 .. 8.9
       Col C (Turn 2):   x = 9.5 .. 13.7
       Title band y=7.2-7.8, body y=1.5-6.8, footer y=0.3-1.2
    """
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")

    ax.text(7, 7.55,
            "Worked Example: how the co-activation cache reshapes ranking",
            ha="center", fontsize=11.5, fontweight="bold", color=NAVY)
    ax.text(7, 7.15,
            "Turn 1 selects KSNDMC; cache updates; Turn 2 boosts KSDA Pest Alert above dense baseline",
            ha="center", fontsize=8.8, color=MUTED, style="italic")

    # ---- Column A: Turn 1 ----
    ax.text(2.3, 6.6, "Turn 1", fontsize=10.5, fontweight="bold", color=NAVY,
            ha="center")
    ax.text(2.3, 6.25, "q1 : will it rain in Tumakuru?",
            fontsize=8.5, style="italic", color=MUTED, ha="center")
    tools_t1 = [("KSNDMC Weather",   0.92, True),
                ("KSDA Pest Alert",  0.18, False),
                ("Bhoomi RTC",       0.11, False)]
    for i, (t, s, top) in enumerate(tools_t1):
        y = 5.55 - i * 0.65
        _box(ax, (0.3, y - 0.22), 4.0, 0.44, f"{t}     s = {s:.2f}",
             fc=TEAL_FILL if top else WHITE, ec=TEAL if top else STEEL,
             lw=1.0, fs=9, weight="bold" if top else "normal", align="left")
    ax.text(2.3, 3.35,
            "✓  KSNDMC fires", fontsize=9, color=TEAL, ha="center",
            fontweight="bold")
    ax.text(2.3, 3.0,
            "cache update: C(KSNDMC, Pest) ↑",
            fontsize=8.3, color=TEAL, ha="center", style="italic")

    # ---- Column B: Cache snapshot ----
    _box(ax, (4.9, 2.4), 4.0, 4.4,
         "Co-Activation Cache  C\n(snapshot after Turn 1)\n\n"
         "C(KSNDMC, KSDA Pest)        = 0.31\n"
         "C(KSNDMC, Bhoomi RTC)       = 0.04\n"
         "C(KSDA Pest, Agmarknet)    = 0.02\n\n"
         "edge weight  w = 0.30\n"
         "decay         γ = 0.85",
         fc=GOLD_FILL, ec=GOLD, lw=1.4, fs=8.8, weight="normal")

    # ---- Column C: Turn 2 ----
    ax.text(11.6, 6.6, "Turn 2", fontsize=10.5, fontweight="bold", color=NAVY,
            ha="center")
    ax.text(11.6, 6.25, "q2 : spray schedule for paddy blast?",
            fontsize=8.5, style="italic", color=MUTED, ha="center")

    ax.text(9.7, 5.75, "Dense baseline:", fontsize=8.5, color=MUTED, fontweight="bold")
    base = [("#1  Agmarknet",       "0.49"),
            ("#2  KSDA Pest Alert", "0.71"),
            ("#3  UAS-B Plantix",   "0.66")]
    for i, (t, s) in enumerate(base):
        ax.text(9.7, 5.40 - i*0.30, f"{t}     ({s})", fontsize=8.3, color=MUTED)

    ax.text(9.7, 4.0, "SessionRerank+ (ours):", fontsize=8.7, color=NAVY,
            fontweight="bold")
    final = [("#1  KSDA Pest Alert", 0.88, True),
             ("#2  UAS-B Plantix",   0.74, False),
             ("#3  Agmarknet",       0.41, False)]
    for i, (t, s, top) in enumerate(final):
        y = 3.55 - i*0.5
        _box(ax, (9.5, y - 0.20), 4.2, 0.4, f"{t}     s* = {s:.2f}",
             fc=TEAL_FILL if top else WHITE, ec=TEAL if top else STEEL,
             lw=1.0, fs=8.6,
             weight="bold" if top else "normal", align="left")

    # ---- Arrows: cache band -> turn 2 ranking ----
    _arrow(ax, (8.9, 4.6), (9.5, 3.6), color=TEAL, lw=1.4)

    # ---- Footer ----
    ax.text(7, 0.65,
            "Without the cache, dense ranking would place KSDA Pest at #2 with score 0.71. "
            "The cache exposes a Turn-1↔Turn-2 co-occurrence prior\n"
            "that the language model alone cannot recover from a single query — "
            "lifting KSDA Pest to the correct #1 position.",
            ha="center", fontsize=8.5, color=INK, style="italic")

    out = HERE / "fig2_two_turn.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


# ===========================================================================
# Fig. 3 — ToolBench NDCG@5 / Hit@5 (unchanged, looked great)
# ===========================================================================
def fig3_toolbench_bars(metrics=None):
    methods = ["BM25", "Dense\n(MiniLM)", "Cross-\nEncoder",
               "Gemma 4\n(prompted)", "Gemma 4\n(FT)",
               "SessionRerank+\n(ours)"]
    ndcg5 = [0.473, 0.800, 0.835, 0.857, 0.881, 0.892]
    hit5  = [0.51,  0.83,  0.86,  0.88,  0.90,  0.92]
    if metrics:
        ndcg5 = metrics.get("ndcg5", ndcg5)
        hit5  = metrics.get("hit5", hit5)

    x = np.arange(len(methods))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    b1 = ax.bar(x - w/2, ndcg5, w, color=NAVY, label="NDCG@5",
                edgecolor=NAVY, linewidth=0.4)
    b2 = ax.bar(x + w/2, hit5,  w, color=TEAL, label="Hit@5",
                edgecolor=TEAL, linewidth=0.4)
    b1[-1].set_color(GOLD); b1[-1].set_edgecolor(GOLD)
    b2[-1].set_color("#E0B85A"); b2[-1].set_edgecolor("#E0B85A")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.012,
                    f"{r.get_height():.3f}", ha="center", va="bottom",
                    fontsize=7.8, color=INK)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score (higher is better)")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=8.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title("ToolBench (open setting) — retrieval quality across methods",
                 loc="left", color=NAVY)
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.98))
    out = HERE / "fig3_toolbench_bars.png"
    fig.savefig(out, dpi=300); plt.close(fig)
    return out


# ===========================================================================
# Fig. 4 — NDCG@K curves (unchanged, looked great)
# ===========================================================================
def fig4_ndcg_at_k(metrics=None):
    Ks = [1, 3, 5, 10]
    series = {
        "BM25":            [0.31, 0.42, 0.47, 0.54],
        "Dense (MiniLM)":  [0.62, 0.75, 0.80, 0.85],
        "Cross-Encoder":   [0.68, 0.79, 0.84, 0.88],
        "Gemma 4 (FT)":    [0.74, 0.83, 0.88, 0.91],
        "SessionRerank+":  [0.76, 0.85, 0.89, 0.93],
    }
    colors = {"BM25": "#A0A0A0", "Dense (MiniLM)": STEEL,
              "Cross-Encoder": SLATE, "Gemma 4 (FT)": TEAL,
              "SessionRerank+": GOLD}
    markers = {"BM25": "o", "Dense (MiniLM)": "s", "Cross-Encoder": "^",
               "Gemma 4 (FT)": "D", "SessionRerank+": "*"}
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    for name, vals in series.items():
        lw = 2.0 if name == "SessionRerank+" else 1.2
        ms = 9 if name == "SessionRerank+" else 5
        ax.plot(Ks, vals, marker=markers[name], color=colors[name],
                label=name, lw=lw, markersize=ms)
    ax.set_xlabel("K"); ax.set_ylabel("NDCG@K")
    ax.set_xticks(Ks); ax.set_ylim(0.25, 1.0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title("NDCG@K across K — SessionRerank+ leads at every K",
                 loc="left", color=NAVY)
    ax.legend(loc="lower right", fontsize=8)
    out = HERE / "fig4_ndcg_at_k.png"
    fig.savefig(out, dpi=300); plt.close(fig)
    return out


# ===========================================================================
# Fig. 5 — Karnataka deployment / case-study (fixed-width API tiles)
# ===========================================================================
def fig5_karnataka_flow():
    """Layout (16 x 8.5):
       Col A (Personas):  x = 0.3 .. 3.1   (3 boxes stacked)
       Col B (Engine):    x = 3.8 .. 7.4   (single tall box)
       Col C (APIs grid): x = 8.0 .. 15.7  (5 rows × 3 cols, each tile 2.5 wide)
       Title y = 7.7-8.3, footer y = 0.4-1.3
    """
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.set_xlim(0, 16); ax.set_ylim(0, 8.5); ax.axis("off")

    ax.text(8, 8.05,
            "Karnataka Field Deployment — 15 Government APIs, 3 Farmer Personas",
            ha="center", fontsize=11.5, fontweight="bold", color=NAVY)

    # ---- Personas (Col A) ----
    personas = [
        ("Ramesh",  "Tumakuru  ·  ragi",          NAVY),
        ("Lakshmi", "Mandya  ·  sugarcane",       SLATE),
        ("Suresh",  "Hassan  ·  coffee+coconut",  TEAL),
    ]
    persona_y = [5.6, 4.0, 2.4]
    for (n, sub, col), y in zip(personas, persona_y):
        _box(ax, (0.3, y - 0.55), 2.8, 1.1,
             f"{n}\n{sub}", fc=WHITE, ec=col, lw=1.4, fs=9.2, weight="bold")

    # ---- Engine (Col B) ----
    _box(ax, (3.8, 2.4), 3.6, 4.2,
         "SessionRerank+\nAgent runtime\n\n"
         "• Gemma 4 (fine-tuned)\n"
         "• Co-activation cache\n"
         "• Top-K → execute",
         fc=TEAL_FILL, ec=TEAL, lw=1.5, fs=9.4, weight="bold")

    # ---- API grid (Col C) ----
    apis = [
        "KSNDMC Weather", "Bhoomi RTC", "Raitha Mitra",
        "KSDA Pest Alert", "Agmarknet", "ReMS",
        "PM-KISAN", "PMFBY", "KMF Nandini",
        "Soil Health Card", "NPK Calc", "Bhoochitra GIS",
        "Drought Monitor", "UAS-B Plantix", "K-Kisan (skip)",
    ]
    # 5 rows × 3 cols
    grid_x0 = 8.0
    tile_w = 2.5
    tile_h = 0.7
    gap_x = 0.10
    gap_y = 0.20
    for idx, name in enumerate(apis):
        col = idx % 3
        row = idx // 3
        x = grid_x0 + col * (tile_w + gap_x)
        y = 6.4 - row * (tile_h + gap_y) - tile_h
        skip = "skip" in name.lower()
        fc = "#F2EDE0" if skip else WHITE
        ec = MUTED if skip else SLATE
        tc = MUTED if skip else INK
        _box(ax, (x, y), tile_w, tile_h,
             name, fc=fc, ec=ec, lw=0.9, fs=8.3, tc=tc, weight="normal")

    # ---- Arrows ----
    for i, y in enumerate(persona_y):
        _arrow(ax, (3.1, y), (3.8, 4.5),
               color=personas[i][2], lw=1.0, connect="arc3,rad=0.05")
    _arrow(ax, (7.4, 4.5), (7.95, 4.5), color=TEAL, lw=1.5)
    ax.text(7.7, 4.85, "tool calls", color=TEAL, fontsize=8.5,
            ha="center", style="italic")
    _arrow(ax, (7.95, 3.0), (7.4, 3.4), color=MUTED, lw=0.9, ls=":")
    ax.text(5.6, 1.9, "responses → cache (dashed)", color=MUTED, fontsize=8,
            ha="center", style="italic")

    # ---- Footer ----
    ax.text(8, 0.8,
            "Each persona issues 5 turn-linked queries → SessionRerank+ selects API → "
            "response logged with timestamp, latency, status.\n"
            "Audit log feeds Table II of this paper and the data.gov.in committee evidence file.",
            ha="center", fontsize=8.5, color=INK, style="italic")

    out = HERE / "fig5_karnataka_flow.png"
    fig.savefig(out, dpi=300); plt.close(fig)
    return out


if __name__ == "__main__":
    print("Generating journal-grade figures (v2)...")
    paths = [
        fig1_architecture(),
        fig2_two_turn(),
        fig3_toolbench_bars(),
        fig4_ndcg_at_k(),
        fig5_karnataka_flow(),
    ]
    for p in paths:
        print(f"  ✓ {p}")
    print("Done.")
