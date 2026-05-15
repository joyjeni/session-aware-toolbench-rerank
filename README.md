# SessionRerank: Session-Aware Re-Ranking for ToolBench API Retrieval

Lightweight re-ranking layer on top of ToolBench's neural API retriever (ToolLLM).
Within a multi-turn session, successful tool-call outcomes are encoded as positive
reinforcement signals that **dynamically bias future retrieval** toward API categories,
schema structures, and endpoint patterns that matched the user's prior successful
queries — with **no explicit user preference input**.

## TL;DR results (full ToolBench G1/G2/G3 benchmark, session regime, NDCG@5)

| Method | G1-Inst | G1-Tool | G1-Cat | G2-Inst | G2-Cat | G3-Inst | Mean |
|---|---|---|---|---|---|---|---|
| BM25            | 0.627 | 0.518 | 0.481 | 0.487 | 0.398 | 0.326 | 0.473 |
| Dense (ToolLLM) | 0.573 | 0.529 | 0.553 | 0.412 | 0.334 | 0.304 | 0.451 |
| ToolRerank      | 0.573 | 0.563 | 0.587 | 0.417 | 0.343 | 0.306 | 0.465 |
| **SessionRerank (ours)** | **0.651** | **0.587** | **0.620** | **0.498** | **0.398** | **0.340** | **0.516** |
| **Δ vs Dense**  | +13.6% | +10.9% | +12.2% | **+20.7%** | **+19.2%** | +11.6% | **+14.7%** |

Hit@5 mean improvement over Dense: **+8.5%**. All NDCG@K gains on G1/G2 are
statistically significant under paired Wilcoxon (p < 1e-4). Numbers reproduced
verbatim from `results/metrics.json`.

## Layout
```
src/sessionrerank/
  data/         ToolBench loaders, multi-turn session extractor
  baselines/    BM25, dense (ToolLLM), ToolRerank reimplementation
  model/        SessionRerank: history-aware reranker
  eval/         Hit@K, NDCG@K, paired-t / Wilcoxon
  viz/          Publication-ready charts (Matplotlib + Seaborn)
notebooks/
  run_on_colab.ipynb       full T4 training + eval pipeline
results/
  metrics.json             machine-readable metrics
  figures/                 PDF + PNG + SVG charts
web/                       Next.js dashboard (Vercel)
paper/                     Method writeup with equations
```

## Quickstart (CPU, full benchmark, ~minutes)
```
pip install -r requirements.txt
python -m sessionrerank.data.download
python -m sessionrerank.eval.run_all --device cpu --encoder all-MiniLM-L6-v2
```

## Reproduction on Colab T4 (uses ToolBench's released IR encoder)
Open `notebooks/run_on_colab.ipynb`, run all cells.

## Web demo
See `web/` — deployed on Vercel.

## Method
See `paper/method.md` for full formulation. In one line:
> Compute base retriever score, then add a session-prior bias from successful
> tool-call traces over category, schema, and endpoint axes, combined via a
> learned gate trained on multi-turn ToolBench trajectories.

## Citation
```
@misc{sessionrerank2026,
  title  = {SessionRerank: Emergent Session-Level Personalization for Tool Retrieval},
  author = {Abigail Creations},
  year   = {2026},
  url    = {https://github.com/joyjeni/session-aware-toolbench-rerank}
}
```
