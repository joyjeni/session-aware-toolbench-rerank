# BioRerank · SessionRerank: Pheromone-Guided Tool Retrieval for Karnataka Smallholder Agriculture

> **Gemma 4 Good Hackathon · Main Track submission (May 2026)** — nature-inspired session-aware re-ranking that turns Gemma 4 into a field-deployable agricultural extension officer for 8.5M Karnataka farmers.
>
> 🚀 **[Live demo on Hugging Face](https://huggingface.co/spaces/abigailcreations/karnataka-agri-assistant)**  ·  📄 **[Kaggle Writeup](./WRITEUP.md)**  ·  🎬 **[Video script](./VIDEO_SCRIPT.md)**  ·  🐍 **[Pheromone prior](./src/sessionrerank/model/pheromone.py)**  ·  🌾 **[Karnataka API catalog](./agri_extension/karnataka_apis.py)**

**Two layers in one repo:**

1. **SessionRerank** — the underlying research contribution. Lightweight re-ranking layer on top of ToolBench's neural API retriever (ToolLLM). Within a multi-turn session, successful tool-call outcomes are encoded as positive reinforcement signals that **dynamically bias future retrieval** toward API categories, schema structures, endpoint patterns, and (new in this submission) **ant-colony pheromone trails** over (api_a → api_b) transitions.
2. **BioRerank · Karnataka agri** — the demo + deployment layer. 15 real Karnataka government and para-statal APIs (KSNDMC, Bhoomi, Raitha Mitra, KSDA, K-Kisan, Agmarknet KA, PM-KISAN, PMFBY, KMF Nandini, …) routed by SessionRerank + Gemma 4 e2b fine-tuned on ToolBench multi-turn trajectories with Unsloth.

## What's new for the hackathon (May 16, 2026)

- **`PheromoneCoocPrior`** — ACO-style τ-deposit + ρ-decay over successful tool transitions. Drop-in replacement for the static `CooccurrencePrior`, time-varying, gets sharper *during* a session. Pure NumPy, < 250 lines, no PyTorch.
- **Stage 4 Gemma cascade wiring** — fine-tuned Gemma 4 e2b LoRA model now appears as a real method column in the eval table (`_GemmaCascade` in `eval/run_all.py`).
- **`agri_extension/`** — Karnataka API catalog (15 real services with Kannada labels), `BioRerankAgri` glue layer, Gradio demo, expert-seeded pheromone chains from KSDA SOP-2024.

## Run the live demo locally

```bash
pip install -r agri_extension/requirements.txt
python agri_extension/gradio_demo.py   # http://localhost:7860
```

Set your farmer profile (crop, district, language, season), ask in Kannada or English, watch the **Pheromone Trails** tab fill up as the system reinforces successful tool-chain transitions in real time.


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
- **Karnataka agri demo (live):** https://huggingface.co/spaces/abigailcreations/karnataka-agri-assistant
- **Source:** `agri_extension/gradio_demo.py` (mirrored to the HF Space at `app.py`).
- **Benchmark dashboard:** `web/` — Next.js on Vercel.

## Method
See `paper/method.md` for the static formulation and `WRITEUP.md` for the pheromone extension. In one line:
> Score(v | q, H) = w·dense(q,v) + w·profile(v) + w·Σ γ^(n−i) log(1 + τ_{h_i, v}) — where τ is a session-time-varying ant-colony pheromone matrix laid down on successful (a→b) tool transitions and evaporating by ρ each session.

## Citation
```
@misc{sessionrerank2026,
  title  = {SessionRerank: Emergent Session-Level Personalization for Tool Retrieval},
  author = {Abigail Creations},
  year   = {2026},
  url    = {https://github.com/joyjeni/session-aware-toolbench-rerank}
}
```
