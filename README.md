# SessionRerank+ : Co-activation Cached Tool Retrieval for Karnataka Smallholder Agriculture

> **Gemma 4 Good Hackathon · Main Track submission (May 2026)** — session-aware re-ranking that turns Gemma 4 into a field-deployable agricultural extension officer for 8.5M Karnataka farmers.
>
> 🚀 **[Live demo on Hugging Face (Gemma 4 in the runtime)](https://huggingface.co/spaces/abigailcreations/karnataka-agri-assistant)**  ·  📄 **[Kaggle Writeup](./WRITEUP.md)**  ·  🎬 **[Video script](./VIDEO_SCRIPT.md)**  ·  🐍 **[Co-activation cache](./src/sessionrerank/model/coactivation.py)**  ·  🌾 **[Karnataka API catalog](./agri_extension/karnataka_apis.py)**

**Two layers in one repo:**

1. **SessionRerank** — the underlying research contribution. Lightweight re-ranking layer on top of ToolBench's neural API retriever (ToolLLM). Within a multi-turn session, successful tool-call outcomes are encoded as positive reinforcement signals that **dynamically bias future retrieval** toward API categories, schema structures, endpoint patterns, and (new in this submission) **a co-activation cache** over (api_a → api_b) transitions.
2. **SessionRerank+ Karnataka agri** — the demo + deployment layer. 15 real Karnataka government and para-statal APIs (KSNDMC, Bhoomi, Raitha Mitra, KSDA, K-Kisan, Agmarknet KA, PM-KISAN, PMFBY, KMF Nandini, …) routed by SessionRerank+ with **real Gemma 4 embeddings** via the Hugging Face Inference API.

## What's new for the hackathon (May 16, 2026)

- **Real Gemma 4 in the live demo runtime** (`gemma_embed.py` on the HF Space) — every query and the catalog are embedded by `google/embeddinggemma-300m` through HF Inference. The response card surfaces which path served the request so judges can verify Gemma is in the loop. A MiniLM fallback keeps the UX responsive on cold starts.
- **`CoactivationCache`** — δ-increment + ρ-decay over successful tool transitions. Drop-in replacement for the static `CooccurrencePrior`, time-varying, gets sharper *during* a session. Pure NumPy, ~250 lines, no PyTorch.
- **Stage 4 Gemma cascade wiring** — fine-tuned Gemma 4 e2b LoRA model appears as a real method column in the eval table (`_GemmaCascade` in `eval/run_all.py`).
- **`agri_extension/`** — Karnataka API catalog (15 real services with Kannada labels), `SessionRerankPlus` glue layer, expert-seeded co-activation edges from KSDA SOP-2024.

## Run the live demo locally

```bash
pip install -r agri_extension/requirements.txt
# Optional: HF_TOKEN env var enables Gemma 4 via HF Inference (free, ~50ms/req)
# Without it, the demo falls back to MiniLM and labels the path accordingly.
python -m agri_extension.sessionrerank_plus   # smoke test
```

The canonical UI is the HF Space (`biorerank_space/app.py` in workspace). To launch a local Gradio: set your farmer profile (crop, district, language, season), ask in Kannada or English, watch the **Co-activation Edges** tab fill up as the system reinforces successful tool-chain transitions in real time.


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
  model/        SessionRerank + CoactivationCache (history-aware reranker)
  eval/         Hit@K, NDCG@K, paired-t / Wilcoxon
  viz/          Publication-ready charts (Matplotlib + Seaborn)
agri_extension/
  sessionrerank_plus.py    SessionRerank+ glue: dense + profile mask + co-activation
  gemma_embed.py           Gemma 4 via HF Inference API (with MiniLM fallback)
  karnataka_apis.py        15 real KA government APIs
notebooks/
  run_on_colab.ipynb       full T4 training + eval pipeline
results/
  metrics.json             machine-readable metrics
  figures/                 PDF + PNG + SVG charts
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
- **Karnataka agri demo (live, Gemma 4 in the runtime):** https://huggingface.co/spaces/abigailcreations/karnataka-agri-assistant
- **Space source mirror:** `biorerank_space/` in workspace (canonical) → pushed to HF Space at `app.py`.

## Method
See `paper/method.md` for the static formulation and `WRITEUP.md` for the co-activation extension. In one line:

$$\text{score}(v \mid q, H) = w_{\text{dense}}\,\text{dense}(q,v) + w_{\text{mask}}\,\text{profile}(v) + w_{\text{co}}\sum_{i=1}^{n}\gamma^{\,n-i}\log\!\bigl(1 + w_{h_i,v}\bigr)$$

where $w_{u,v}$ is a session-time-varying co-activation edge weight, incremented by $\delta$ on successful $(a\!\to\!b)$ transitions and decaying by $\rho$ each session.

## Citation
```
@misc{sessionrerank2026,
  title  = {SessionRerank+: Co-activation Cached Tool Retrieval for Multi-turn Agentic LLMs},
  author = {Jenisha T},
  year   = {2026},
  url    = {https://github.com/joyjeni/session-aware-toolbench-rerank}
}
```
