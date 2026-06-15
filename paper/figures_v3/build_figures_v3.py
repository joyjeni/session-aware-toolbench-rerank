"""CRC-SPAST paper figures v3 (iteration 4).

DESIGN RULES (learned from iterations 1-3):
  * NO set_aspect('equal') — let matplotlib compute aspect from figsize.
  * Number stripe is a thin separate rectangle DRAWN FIRST, text box drawn
    SECOND OVER A NON-OVERLAPPING REGION; text uses clip_on=False but is
    anchored at the centre of the text region with short, hand-broken labels
    so it never overflows into the stripe.
  * Step boxes are taller (bh=1.25) so 2-line text fits comfortably.
  * Right column has its own non-overlapping y-band; the Step 7 box does not
    sit beside steps 1-2 (which would force horizontal arrows over text).
  * Legends live in a strict bottom band with NO text wider than its slot.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
HERE.mkdir(parents=True, exist_ok=True)

NAVY   = "#0B2545"
TEAL   = "#1B6E8A"
GOLD   = "#C49A2C"
INK    = "#1A1A1A"
MUTED  = "#5A5A5A"
WHITE  = "#FFFFFF"
TEAL_FILL = "#E6EEF1"
GOLD_FILL = "#FAF1DC"
NAVY_FILL = "#EAEEF4"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.0,
    "text.color": INK,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": WHITE,
    "figure.facecolor": WHITE,
})


def _box(ax, xy, w, h, text, fc=WHITE, ec=NAVY, tc=INK, fs=9.0,
         lw=1.0, weight="normal", ls="-"):
    x, y = xy
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.01,rounding_size=0.08",
                         linewidth=lw, edgecolor=ec, facecolor=fc,
                         linestyle=ls)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            color=tc, fontsize=fs, fontweight=weight, linespacing=1.35)


def _step_box(ax, xy, w, h, n, text, fc=WHITE, ec=NAVY, tc=INK, fs=9.5,
              weight="normal", num_fc=NAVY, num_tc=WHITE, nw=0.85):
    """Step box: numbered stripe on the left + text region on the right.

    The two regions are drawn as PLAIN rectangles (no rounded corners) sharing
    a common border, so the text region is a clean rectangle from (x+nw, y)
    to (x+w, y+h). The text is centred inside that region.
    """
    x, y = xy
    # Number stripe — plain rectangle
    ax.add_patch(Rectangle((x, y), nw, h, facecolor=num_fc, edgecolor=ec,
                            linewidth=1.0, zorder=2))
    ax.text(x + nw/2, y + h/2, str(n), ha="center", va="center",
            color=num_tc, fontsize=12.5, fontweight="bold", zorder=3)
    # Text region — plain rectangle
    ax.add_patch(Rectangle((x + nw, y), w - nw, h, facecolor=fc, edgecolor=ec,
                            linewidth=1.0, zorder=2))
    # Outer rounded frame for the whole step (cosmetic)
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                 boxstyle="round,pad=0.0,rounding_size=0.08",
                                 linewidth=1.2, edgecolor=ec, facecolor="none",
                                 zorder=4))
    ax.text(x + nw + (w - nw) / 2, y + h/2, text, ha="center", va="center",
            color=tc, fontsize=fs, fontweight=weight, linespacing=1.35,
            zorder=5)


def _arrow(ax, p1, p2, color=NAVY, lw=1.3, style="-", arrow="->", rad=0.0):
    arr = FancyArrowPatch(p1, p2, arrowstyle=arrow, mutation_scale=12,
                          color=color, linewidth=lw, linestyle=style,
                          connectionstyle=f"arc3,rad={rad}", zorder=6)
    ax.add_patch(arr)


# =============================================================================
# FIG 1 — ARCHITECTURE  (wider figsize, taller boxes, legend on one row only)
# =============================================================================
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 10.6)
    ax.axis("off")

    # Title
    ax.text(8.5, 10.05, "SessionRerank+   —   Session-Aware Re-Ranking for API Tool Retrieval",
            ha="center", va="center", color=NAVY, fontsize=12.5, fontweight="bold")
    ax.text(8.5, 9.55, "Three orthogonal signals fused into a single score",
            ha="center", va="center", color=MUTED, fontsize=9.5, style="italic")

    # Column 1 — inputs
    x1, w1 = 0.3, 4.0
    _box(ax, (x1, 7.4), w1, 1.2, "User query  q\n(turn t)",
         fc=NAVY_FILL, ec=NAVY, fs=10, weight="bold")
    _box(ax, (x1, 5.9), w1, 1.2, "Session history\nH = (h₁, …, hₙ)",
         fc=NAVY_FILL, ec=NAVY, fs=10, weight="bold")
    _box(ax, (x1, 4.4), w1, 1.2, "Farmer profile  p\n(crop, district, season)",
         fc=NAVY_FILL, ec=NAVY, fs=10, weight="bold")

    # Column 2 — three signal lanes
    x2, w2 = 5.6, 5.6
    _box(ax, (x2, 7.4), w2, 1.2,
         "Gemma-4 dense retriever\ns_d(q, v) = cos(emb_q, emb_v)",
         fc=TEAL_FILL, ec=TEAL, fs=10)
    _box(ax, (x2, 5.9), w2, 1.2,
         "Profile mask\nm(v, p) ∈ {0, 1}",
         fc=TEAL_FILL, ec=TEAL, fs=10)
    _box(ax, (x2, 4.4), w2, 1.2,
         "Co-activation cache  C\nφ_c(v | H),    w ∈ [0, 1]",
         fc=GOLD_FILL, ec=GOLD, fs=10, weight="bold")

    # Column 3 — fusion + output
    x3, w3 = 12.5, 4.2
    _box(ax, (x3, 5.9), w3, 2.7,
         "Score fusion\n\nscore(v) =\nwₐ·s_d + w_m·m + w_c·φ_c",
         fc=WHITE, ec=NAVY, fs=10, weight="bold")
    _box(ax, (x3, 4.4), w3, 1.2, "Top-K APIs\n→ executed by agent",
         fc=NAVY, ec=NAVY, tc=WHITE, fs=10, weight="bold")

    # Arrows column 1 -> column 2
    _arrow(ax, (x1 + w1, 8.0), (x2, 8.0))
    _arrow(ax, (x1 + w1, 6.5), (x2, 6.5))
    _arrow(ax, (x1 + w1, 5.0), (x2, 5.0))
    # Cross arrows (dotted): H -> cache, profile -> mask
    _arrow(ax, (x1 + w1, 6.5), (x2, 5.0), lw=0.8, style=":", color=MUTED, rad=-0.10)
    _arrow(ax, (x1 + w1, 5.0), (x2, 6.5), lw=0.8, style=":", color=MUTED, rad=0.10)

    # Arrows column 2 -> column 3 (fusion)
    _arrow(ax, (x2 + w2, 8.0), (x3, 7.6))
    _arrow(ax, (x2 + w2, 6.5), (x3, 7.0))
    _arrow(ax, (x2 + w2, 5.0), (x3, 6.4))

    # Fusion -> Top-K
    _arrow(ax, (x3 + w3/2, 5.9), (x3 + w3/2, 5.6), lw=1.5)

    # Feedback arc: Top-K -> cache (dashed gold), routed UNDER everything
    _arrow(ax, (x3, 4.6), (x2 + w2, 4.6),
           color=GOLD, lw=1.3, style="--", rad=-0.35)
    ax.text(11.6, 3.55, "successful  (a → b)  observed",
            color=GOLD, fontsize=9, style="italic", ha="center",
            bbox=dict(boxstyle="round,pad=0.25", fc=WHITE, ec="none"))

    # Bottom update-rule band (y: 1.6 .. 2.9) — own zone
    _box(ax, (0.3, 1.6), 16.4, 1.3,
         "Online update on success  (a → b):    "
         "w_{a,b} ← (1 − ρ)·w_{a,b} + δ·𝟙[success]          "
         "Between sessions:    W ← (1 − ρ)·W   (decay)",
         fc=GOLD_FILL, ec=GOLD, fs=10, weight="bold")

    # Legend — single bottom row, labels start right after each swatch, well-spaced
    leg_y = 0.45
    sw, sh = 0.45, 0.55
    slots = [
        (0.3,  NAVY_FILL, NAVY,  "inputs",                              False),
        (2.6,  TEAL_FILL, TEAL,  "static signals",                      False),
        (6.0,  GOLD_FILL, GOLD,  "session-varying signal (innovation)", True),
        (12.8, None,      None,  "feedback / online update",            False),
    ]
    for x_box, fc, ec, label, bold in slots:
        if fc is not None:
            ax.add_patch(Rectangle((x_box, leg_y), sw, sh, facecolor=fc, edgecolor=ec))
        else:
            ax.plot([x_box, x_box + sw], [leg_y + sh/2, leg_y + sh/2],
                    color=GOLD, linestyle="--", linewidth=1.5)
        ax.text(x_box + sw + 0.18, leg_y + sh/2, label, va="center",
                fontsize=9.5, fontweight="bold" if bold else "normal")

    out = HERE / "fig1_architecture.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"wrote {out}")


# =============================================================================
# FIG 2 — ALGORITHM FLOWCHART
# Left column: steps 1..6 stacked vertically (each tall enough for 2 lines).
# Right column: SINGLE tall box for Step 7 (online update) — NO second box,
# bullets moved BELOW (full-width band) so nothing competes for vertical space.
# =============================================================================
def fig2_algorithm():
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 17)
    ax.axis("off")

    # Title
    ax.text(8.5, 16.5, "SessionRerank+ Algorithm   —   one inference turn",
            ha="center", va="center", color=NAVY, fontsize=13, fontweight="bold")
    ax.text(8.5, 16.0, "Steps 1–6 run on every query.   Step 7 fires only on observed success.",
            ha="center", va="center", color=MUTED, fontsize=9.5, style="italic")

    # ---- Left lane: steps 1..6 (x: 0.3 .. 8.1, width 7.8) ----
    bx, bw = 0.3, 7.8
    bh = 1.30
    gap = 0.45
    y_top = 14.5
    ys = [y_top - i * (bh + gap) for i in range(6)]
    # y range: 14.5 down to 14.5 - 5*(1.75) = 5.75 → bottom of step 6 at 5.75

    steps = [
        (1, "Embed query\nemb_q ← Gemma-4(q)",                                       TEAL_FILL, TEAL,  NAVY, WHITE, INK,   "bold",   9.5),
        (2, "Retrieve top-100 candidates by\ns_d = cos(emb_q, emb_v)",                TEAL_FILL, TEAL,  NAVY, WHITE, INK,   "normal", 9.5),
        (3, "Apply profile mask m(v, p);\ndrop v where m = 0",                        TEAL_FILL, TEAL,  NAVY, WHITE, INK,   "normal", 9.5),
        (4, "Lookup co-activation prior:\nφ_c(v | H) = Σᵢ γⁿ⁻ⁱ · log(1 + w_{h_i,v})", GOLD_FILL, GOLD,  GOLD, WHITE, INK,   "bold",   9.5),
        (5, "Fuse:\nscore(v) = wₐ·s_d + w_m·m + w_c·φ_c",                             NAVY_FILL, NAVY,  NAVY, WHITE, INK,   "bold",   9.5),
        (6, "Return top-K  →  agent executes\nthe selected API",                       NAVY,      NAVY,  GOLD, NAVY,  WHITE, "bold",   9.5),
    ]
    for (n, txt, fc, ec, num_fc, num_tc, tc, w, fs), y in zip(steps, ys):
        _step_box(ax, (bx, y), bw, bh, n, txt, fc=fc, ec=ec, tc=tc,
                  num_fc=num_fc, num_tc=num_tc, fs=fs, weight=w)
        idx = n - 1
        if idx < 5:
            y_next = ys[idx + 1]
            _arrow(ax, (bx + bw/2, y), (bx + bw/2, y_next + bh), lw=1.4)

    # ---- Right lane: Step 7 — one tall box covering the full left-lane height ----
    rx, rw = 9.4, 7.3
    r_top = ys[0] + bh         # 14.5 + 1.30 = 15.80
    r_bot = ys[5]              # 5.75
    _step_box(ax, (rx, r_bot), rw, r_top - r_bot, 7,
              "Online cache update\n"
              "(SessionRerank+ innovation)\n\n"
              "For each observed transition (a → b)\nwith success = 1:\n\n"
              "w_{a,b} ← (1 − ρ)·w_{a,b} + δ·𝟙[success]\n\n"
              "Defaults:   ρ = 0.02    δ = 1.0    γ = 0.7\n\n"
              "Edges with w < 10⁻⁴ are evicted.\n\n"
              "Between sessions:\nW ← (1 − ρ)·W    (stale edges decay)",
              fc=GOLD_FILL, ec=GOLD, tc=INK,
              num_fc=GOLD, num_tc=WHITE, fs=10.0, weight="normal", nw=0.85)

    # Feedback arrows. Labels live in the inter-lane gutter; the gutter is narrow
    # (x: 8.1 .. 9.4), so labels are short and rendered with TIGHT bbox padding to
    # avoid bleeding into either lane.
    # step 6 -> step 7 (on success).
    _arrow(ax, (bx + bw, ys[5] + bh/2), (rx, r_bot + 1.2),
           color=GOLD, lw=1.6, style="--", rad=-0.20)
    ax.text(8.75, ys[4] + 0.10, "on\nsuccess",
            color=GOLD, fontsize=8.5, style="italic", ha="center", va="center",
            linespacing=1.2,
            bbox=dict(boxstyle="round,pad=0.15", fc=WHITE, ec=GOLD, lw=0.7))
    # step 7 -> step 4 (updated W used next turn).
    _arrow(ax, (rx, r_top - 1.2), (bx + bw, ys[3] + bh/2),
           color=GOLD, lw=1.4, style="--", rad=0.25)
    ax.text(8.75, ys[1] + 0.10, "updated\nW used\nnext turn",
            color=GOLD, fontsize=8.5, style="italic", ha="center", va="center",
            linespacing=1.2,
            bbox=dict(boxstyle="round,pad=0.15", fc=WHITE, ec=GOLD, lw=0.7))

    # ---- Bottom band: "Why this matters" (full-width, own y-zone) ----
    _box(ax, (0.3, 1.9), 16.4, 3.1,
         "Why this matters\n\n"
         "•  Edge weights adapt within a single session  (no offline retraining)\n"
         "•  Pure-Python, no GPU, ≈ 250 LOC  —  drop-in re-ranker over any dense retriever\n"
         "•  Bounded memory  (sparse storage;  eviction below 10⁻⁴)\n"
         "•  Falls back to static prior on cold start",
         fc=WHITE, ec=NAVY, fs=10, weight="normal")

    # Legend (own row at the very bottom). Each label starts right after its swatch,
    # and slots are spaced wide enough so no label can reach the next swatch.
    leg_y = 0.55
    sw, sh = 0.45, 0.55
    slots = [
        (0.3,  TEAL_FILL, TEAL, "static signal",                       False),
        (3.8,  GOLD_FILL, GOLD, "session-varying signal (innovation)", True),
        (11.5, NAVY,      NAVY, "output / decision",                   False),
    ]
    for x_box, fc, ec, label, bold in slots:
        ax.add_patch(Rectangle((x_box, leg_y), sw, sh, facecolor=fc, edgecolor=ec))
        ax.text(x_box + sw + 0.18, leg_y + sh/2, label, va="center",
                fontsize=9.5, fontweight="bold" if bold else "normal")

    out = HERE / "fig2_algorithm.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    fig1_architecture()
    fig2_algorithm()
    print("done")
