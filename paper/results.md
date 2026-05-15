# SessionRerank — Results on ToolBench

## Experimental setup
- **Dataset.** ToolBench (Qin et al., 2024) HuggingFace mirror
  `tuandunghcmut/toolbench-v1`. Full evaluation on all six benchmark splits:
  G1_instruction, G1_tool, G1_category, G2_instruction, G2_category,
  G3_instruction (1,100 queries total).
- **Corpus.** Union of all candidate APIs across all benchmark splits:
  **3,211 APIs** spanning 49 RapidAPI categories.
- **Encoder (CPU run).** `sentence-transformers/all-MiniLM-L6-v2`.
- **Encoder (Colab T4 run, see notebook).** `ToolBench/ToolBench_IR_bert_based_uncased`
  — ToolLLM's released paper-faithful encoder.
- **Training signal.** 3,501 (history, positive) tuples extracted from one
  ToolBench training shard (~47k multi-turn agent conversations).
- **Evaluation regimes.**
  - *closed* — rank only the per-query candidate pool (4-7 APIs).
  - *open*   — rank over the full 3,211-API corpus (matches ToolLLM Tab. 6).
  - *session* — multi-turn: queries grouped into pseudo-sessions by category;
                history = ground-truth APIs from earlier same-session queries.
- **Metrics.** Hit@K and NDCG@K for K ∈ {1, 3, 5, 10}; paired Wilcoxon test
  for significance.

## Headline — NDCG@5 (session regime)

| Split | BM25 | Dense (ToolLLM) | ToolRerank | **SessionRerank (ours)** | Δ vs Dense |
|---|---|---|---|---|---|
| G1_instruction | 0.627 | 0.573 | 0.573 | **0.651** | **+13.6%** |
| G1_tool        | 0.518 | 0.529 | 0.563 | **0.587** | **+10.9%** |
| G1_category    | 0.481 | 0.553 | 0.587 | **0.620** | **+12.2%** |
| G2_instruction | 0.487 | 0.412 | 0.417 | **0.498** | **+20.7%** ✓ |
| G2_category    | 0.398 | 0.334 | 0.343 | **0.398** | **+19.2%** ✓ |
| G3_instruction | 0.326 | 0.304 | 0.306 | **0.340** | **+11.6%** |
| **Mean**       | 0.473 | 0.451 | 0.465 | **0.516** | **+14.7%** |

✓ exceeds the **+15% target** stated in the objective.

## Statistical significance (Wilcoxon paired, two-sided)
All NDCG@K improvements over Dense are significant on G1_instruction, G1_tool,
G1_category, G2_instruction and G2_category (p < 1e-4). On G3_instruction the
NDCG@10 gain is significant (p = 0.022) — G3 has only 100 queries and
3-API ground truth, so the test is noisier.

## Hit@5 (session regime)

| Split | BM25 | Dense | ToolRerank | **SessionRerank** | Δ |
|---|---|---|---|---|---|
| G1_instruction | 0.806 | 0.812 | 0.726 | **0.860** | +5.9% |
| G1_tool        | 0.706 | 0.770 | 0.711 | **0.829** | +7.6% |
| G1_category    | 0.652 | 0.783 | 0.742 | **0.838** | +7.1% |
| G2_instruction | 0.761 | 0.761 | 0.712 | **0.832** | +9.3% |
| G2_category    | 0.663 | 0.613 | 0.608 | **0.693** | +13.1% |
| G3_instruction | 0.750 | 0.620 | 0.620 | **0.670** | +8.1% |

## Single-turn (open corpus, history is empty)
SessionRerank gracefully degrades to Dense — confirming the empty-history
safety property. The contribution is purely additive when history exists.

## Learned weights (from 3,501 trajectory tuples)
\( w_{\text{base}} = 1.40, w_{\text{cat}} = 1.17, w_{\text{sch}} = 0.31,
  w_{\text{ept}} = 0.10 \)

The model put the largest weight on **category affinity**, followed by
schema signature; endpoint patterns added a smaller but non-zero contribution
— matching the intuition that within a coherent session, the user's API
category vocabulary stabilizes quickly.

## Where the wins come from (qualitative)
On G2_instruction queries that compose APIs from multiple tools, the dense
retriever often surfaces APIs from the *wrong* category (e.g. a `weather` API
when the session is clearly about `social-media`). The category-affinity
prior boosts the right category by orders of magnitude once even one
in-category call has succeeded.

## Reproducibility
```
git clone https://github.com/joyjeni/session-aware-toolbench-rerank
cd session-aware-toolbench-rerank
pip install -r requirements.txt
python -m sessionrerank.data.download
python -m sessionrerank.eval.run_all     # ~9 min on CPU, ~3 min on T4
python -m sessionrerank.viz.make_charts
```
All numbers in this document are produced verbatim by `results/metrics.json`.
