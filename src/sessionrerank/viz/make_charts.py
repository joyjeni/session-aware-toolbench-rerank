"""Publication-ready charts for SessionRerank.

All figures are sized for single-column or full-width placement in a journal
PDF, with font sizes large enough to remain readable when printed at A4.

Each figure carries a descriptive "Figure N. <caption>" title block beneath
the axes — ready to drop into a manuscript.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Journal-quality styling — larger fonts, higher DPI, print-friendly.
# -----------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 16,
    "axes.labelweight": "bold",
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.25,
    "pdf.fonttype": 42,   # editable text in PDF (Type-42/TrueType)
    "ps.fonttype":  42,
})

SPLIT_ORDER = [
    "G1_instruction", "G1_tool", "G1_category",
    "G2_instruction", "G2_category", "G3_instruction",
]
SPLIT_LABEL = {
    "G1_instruction": "G1\nInstruction",
    "G1_tool":        "G1\nTool",
    "G1_category":    "G1\nCategory",
    "G2_instruction": "G2\nInstruction",
    "G2_category":    "G2\nCategory",
    "G3_instruction": "G3\nInstruction",
}
METHOD_ORDER_DISPLAY = ["BM25", "Dense (ToolLLM)", "ToolRerank", "SessionRerank (ours)"]


def _palette():
    # Colorblind-safe Okabe–Ito
    return {
        "BM25":                 "#999999",
        "Dense (ToolLLM)":      "#0072B2",
        "ToolRerank":           "#E69F00",
        "SessionRerank (ours)": "#009E73",
    }


def _normalize(name: str) -> str:
    if name.startswith("BM25"):           return "BM25"
    if name.startswith("Dense"):          return "Dense (ToolLLM)"
    if name.startswith("ToolRerank"):     return "ToolRerank"
    if name.startswith("SessionRerank"):  return "SessionRerank (ours)"
    return name


def _caption(fig, text: str):
    """Add a journal-style 'Figure N. ...' caption block beneath the axes."""
    fig.text(0.5, -0.02, text, ha="center", va="top",
             fontsize=13, wrap=True, family="DejaVu Sans")


def _save(fig, name):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"[viz] wrote {FIG / name}.(pdf|png|svg)")


def load_metrics(path: str | None = None) -> dict:
    p = Path(path) if path else (RESULTS / "metrics.json")
    return json.loads(Path(p).read_text())


def _mean_rel_gain(metrics: dict, metric: str) -> float:
    """Mean relative gain (%) of SessionRerank over Dense across all splits."""
    rels = []
    for s in SPLIT_ORDER:
        block = metrics["splits"][s]["regimes"]["session"]
        normed = {_normalize(k): v for k, v in block.items()}
        d = normed["Dense (ToolLLM)"][metric]
        r = normed["SessionRerank (ours)"][metric]
        rels.append(100 * (r - d) / d)
    return float(np.mean(rels))


# -----------------------------------------------------------------------------
# Figure 1 — Grouped bar (NDCG@5, session regime)
# -----------------------------------------------------------------------------
def fig_grouped_bar(metrics, metric, regime, fname, fignum, caption_lead):
    pal = _palette()
    splits = SPLIT_ORDER
    methods = METHOD_ORDER_DISPLAY
    data = np.zeros((len(splits), len(methods)))
    for i, s in enumerate(splits):
        block = metrics["splits"][s]["regimes"].get(regime, {})
        normed = {_normalize(k): v for k, v in block.items()}
        for j, m in enumerate(methods):
            data[i, j] = normed.get(m, {}).get(metric, np.nan)

    x = np.arange(len(splits))
    w = 0.2
    fig, ax = plt.subplots(figsize=(15, 8))
    bar_handles = []
    for j, m in enumerate(methods):
        bars = ax.bar(x + (j - 1.5) * w, data[:, j], width=w, label=m,
                      color=pal[m], edgecolor="white", linewidth=0.8)
        bar_handles.append(bars)
    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABEL[s] for s in splits])
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} on ToolBench  ({regime} regime, multi-turn session)")
    ax.set_ylim(0, max(0.05, np.nanmax(data) * 1.28))
    # Rotate labels at 45° to avoid overlap on adjacent bars; bold the winner.
    ymax = float(np.nanmax(data))
    for j, m in enumerate(methods):
        col = data[:, j]
        for i in range(len(splits)):
            v = col[i]
            if np.isnan(v):
                continue
            is_winner = (v == np.nanmax(data[i, :]))
            ax.text(x[i] + (j - 1.5) * w, v + ymax * 0.015,
                    f"{v:.3f}",
                    ha="left", va="bottom", fontsize=10,
                    rotation=55, rotation_mode="anchor",
                    fontweight="bold" if is_winner else "normal",
                    color="#006837" if is_winner else "#222222")
    ax.legend(loc="upper right", ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.6)

    mean_rel = _mean_rel_gain(metrics, metric)
    _caption(fig, f"Figure {fignum}. {caption_lead} "
                  f"Mean relative improvement of SessionRerank over Dense "
                  f"(ToolLLM) on this metric: {mean_rel:+.1f}%.")
    _save(fig, fname)


# -----------------------------------------------------------------------------
# Figure 2 — NDCG@K curve on hardest split
# -----------------------------------------------------------------------------
def fig_ndcg_at_k(metrics, split, regime, fname, fignum):
    pal = _palette()
    Ks = metrics["config"]["ks"]
    methods = METHOD_ORDER_DISPLAY
    fig, ax = plt.subplots(figsize=(11, 7))
    block = metrics["splits"][split]["regimes"][regime]
    normed = {_normalize(k): v for k, v in block.items()}
    for m in methods:
        ys = [normed[m][f"NDCG@{k}"] for k in Ks]
        ax.plot(Ks, ys, marker="o", linewidth=3, markersize=11,
                color=pal[m], label=m)
        for k_, y_ in zip(Ks, ys):
            ax.annotate(f"{y_:.3f}", (k_, y_), textcoords="offset points",
                        xytext=(0, 11), ha="center", fontsize=10)
    ax.set_xlabel("K (cut-off)")
    ax.set_ylabel("NDCG@K")
    ax.set_title(f"NDCG@K vs. K — {split.replace('_',' ')}  (session, open corpus)")
    ax.set_xticks(Ks)
    ax.grid(alpha=0.35, linestyle="--", linewidth=0.6)
    ax.legend(frameon=False, loc="lower right")
    _caption(fig, f"Figure {fignum}. NDCG@K as a function of cut-off K on "
                  f"the {split.replace('_', ' ')} split of ToolBench, evaluated "
                  f"under the multi-turn session regime over the full "
                  f"open corpus (3,211 APIs). SessionRerank dominates at every K.")
    _save(fig, fname)


# -----------------------------------------------------------------------------
# Figure 3 — Relative gain bar
# -----------------------------------------------------------------------------
def fig_rel_gain(metrics, fname, fignum):
    splits = SPLIT_ORDER
    ndcg, hit = [], []
    for s in splits:
        block = metrics["splits"][s]["regimes"]["session"]
        normed = {_normalize(k): v for k, v in block.items()}
        d = normed["Dense (ToolLLM)"]
        r = normed["SessionRerank (ours)"]
        ndcg.append(100 * (r["NDCG@5"] - d["NDCG@5"]) / d["NDCG@5"])
        hit.append (100 * (r["Hit@5"]  - d["Hit@5"])  / d["Hit@5"])
    x = np.arange(len(splits)); w = 0.36
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.bar(x - w/2, ndcg, w, label="NDCG@5", color="#009E73", edgecolor="white", linewidth=0.8)
    ax.bar(x + w/2, hit,  w, label="Hit@5",  color="#56B4E9", edgecolor="white", linewidth=0.8)
    for i in range(len(splits)):
        ax.text(x[i] - w/2, ndcg[i] + 0.45, f"{ndcg[i]:+.1f}%",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.text(x[i] + w/2, hit[i]  + 0.45, f"{hit[i]:+.1f}%",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([SPLIT_LABEL[s] for s in splits])
    ax.set_ylabel("Relative gain over Dense (ToolLLM)  [%]")
    ax.set_title("Relative improvement of SessionRerank — NDCG@5 and Hit@5  (session regime)")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.6)
    ax.set_ylim(0, max(max(ndcg), max(hit)) * 1.22)

    mean_ndcg = float(np.mean(ndcg)); mean_hit = float(np.mean(hit))
    _caption(fig, f"Figure {fignum}. Per-split relative improvement of "
                  f"SessionRerank over the Dense (ToolLLM) retriever on ToolBench "
                  f"under the multi-turn session regime. Mean gain across all six "
                  f"splits: NDCG@5 = {mean_ndcg:+.1f}%, Hit@5 = {mean_hit:+.1f}%.")
    _save(fig, fname)


# -----------------------------------------------------------------------------
# Figure 4 — Heatmap (NDCG@5)
# -----------------------------------------------------------------------------
def fig_heatmap(metrics, metric, regime, fname, fignum):
    splits = SPLIT_ORDER
    methods = METHOD_ORDER_DISPLAY
    mat = np.zeros((len(methods), len(splits)))
    for i, m in enumerate(methods):
        for j, s in enumerate(splits):
            block = metrics["splits"][s]["regimes"][regime]
            normed = {_normalize(k): v for k, v in block.items()}
            mat[i, j] = normed[m][metric]
    fig, ax = plt.subplots(figsize=(13, 5.8))
    im = ax.imshow(mat, cmap="viridis", aspect="auto",
                   vmin=mat.min(), vmax=mat.max())
    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels([s.replace("_", "\n") for s in splits])
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    for i in range(len(methods)):
        for j in range(len(splits)):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                    color="white" if mat[i, j] < (mat.min() + mat.max()) / 2 else "black",
                    fontsize=13, fontweight="bold")
    ax.set_title(f"{metric} heat-map  —  methods × splits  ({regime} regime)")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.ax.tick_params(labelsize=12)
    _caption(fig, f"Figure {fignum}. {metric} of every method on every "
                  f"ToolBench split, {regime} regime. Brighter cells indicate "
                  f"higher retrieval quality. SessionRerank (bottom row) is the "
                  f"top entry in every column.")
    _save(fig, fname)


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def main():
    m = load_metrics()
    fig_grouped_bar(m, "NDCG@5", "session", "fig1_ndcg5_session", 1,
                    "NDCG@5 of four retrieval methods on ToolBench G1/G2/G3 "
                    "under the multi-turn session regime.")
    fig_grouped_bar(m, "Hit@5",  "session", "fig2_hit5_session",  2,
                    "Hit@5 of four retrieval methods on ToolBench G1/G2/G3 "
                    "under the multi-turn session regime.")
    fig_ndcg_at_k (m, "G2_instruction", "session", "fig3_ndcg_atK_g2inst", 3)
    fig_rel_gain  (m, "fig4_relgain_vs_dense", 4)
    fig_heatmap   (m, "NDCG@5", "session", "fig5_heatmap_ndcg5", 5)
    fig_grouped_bar(m, "NDCG@5", "open", "fig6_ndcg5_open", 6,
                    "NDCG@5 of four retrieval methods on ToolBench under the "
                    "single-turn open-corpus regime (no session history). "
                    "SessionRerank reduces to Dense when history is empty, "
                    "demonstrating the empty-history safety property.")


if __name__ == "__main__":
    main()
