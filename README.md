# SessionRerank+ — Session-Aware Toolbench Reranking

> **PhD Objective 1** | Agricultural AI Assistant for Indian Farmers

[![Live Demo](https://img.shields.io/badge/🤗%20HF%20Space-LIVE-brightgreen)](https://huggingface.co/spaces/abigailcreations/karnataka-agri-assistant)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue)](https://www.kaggle.com/code/sessionrerank_gemma4_kaggle)
[![data.gov.in](https://img.shields.io/badge/data.gov.in-15%20Karnataka%20APIs-orange)](https://data.gov.in)
[![Languages](https://img.shields.io/badge/Languages-en%20%7C%20kn%20%7C%20hi%20(partial)-yellow)](#multilingual)

---

## What is SessionRerank+?

SessionRerank+ is a **session-aware API reranking system** that dynamically scores data.gov.in agricultural APIs based on a farmer's interaction history within a session. Unlike static one-shot reranking, it reinforces API co-activations: if a farmer queries price data (api_a) and then follows up with market listings (api_b), the transition weight w_{api_a → api_b} is strengthened — sharpening relevance scores for the remainder of the session.

**Novel Contribution**: The **co-activation cache** is the first session-aware (api_a → api_b) transition reinforcement applied to tool/API reranking for LLM agents. Dynamic δ-increment and ρ-decay ensure the cache remains sharp within a session and resets cleanly across sessions.

---

## Deployment

### HuggingFace Space (LIVE)

> **https://huggingface.co/spaces/abigailcreations/karnataka-agri-assistant**

- **Runtime model**: Gemma 4 (via HuggingFace Inference API)
- **Embedding model**: `google/embeddinggemma-300m`
- **Data source**: data.gov.in Karnataka APMC APIs (15 resources, resource ID `9ef84268-d588-465a-a308-a864a43d0070` + 14 others)
- **Languages**: English and Kannada fully supported; Hindi partial support
- **Interface**: Gradio chat interface with mandi price lookup and crop advisory

### Kaggle Notebook

Reproducible benchmark: `kaggle/sessionrerank_gemma4_kaggle.ipynb`

---

## Core Equations

### Final Ranking Score

$$\text{score}(v) = w_d \cdot s_d(v) + w_m \cdot m(v) + w_c \cdot \phi_c(v)$$

where:
- $s_d$ — dense embedding similarity (cosine, `google/embeddinggemma-300m`)
- $m$ — metadata score (recency, data freshness, coverage)
- $\phi_c$ — **co-activation score** (novel component)
- $w_d, w_m, w_c$ — learned weights (sum to 1)

### Co-Activation Score

$$\phi_c(v) = \sum_{i} \gamma^{(n-i)} \cdot \log\!\left(1 + w_{h_i, v}\right)$$

where $h_i$ is the $i$-th API called in the session history, $w_{h_i,v}$ is the co-activation edge weight, and $\gamma \in (0,1)$ is a recency decay.

### Edge Weight Update

$$w_{u,v} \leftarrow (1 - \rho) \cdot w_{u,v} + \delta \cdot \mathbf{1}[\text{success}]$$

where:
- $\rho$ — decay rate (prevents stale reinforcement across sessions)
- $\delta$ — increment on successful API call following $u$
- Both $\rho$ and $\delta$ are dynamic: $\delta$ scales with session confidence, $\rho$ increases at session boundary

---

## Results

### Benchmark: Karnataka APMC Query Set (G1/G2/G3 groups)

| Metric     | Dense Baseline | SessionRerank+ | Improvement  |
|------------|---------------|----------------|--------------|
| NDCG@5     | 0.450         | **0.516**      | +14.7%       |
| Hit@5      | —             | —              | +8.5%        |

- All three evaluation groups (G1, G2, G3) show statistically significant improvement.
- **Statistical test**: Wilcoxon signed-rank, p < 1×10⁻⁴ for all groups.

---

## Data Sources

### data.gov.in Integration

| Resource ID                              | Description                          |
|------------------------------------------|--------------------------------------|
| `9ef84268-d588-465a-a308-a864a43d0070`   | Current daily commodity prices (Karnataka) |
| + 14 additional Karnataka APIs           | Market listings, crop advisories, weather |

All 15 Karnataka APIs are indexed in the co-activation cache and reranked per session.

---

## Multilingual Support {#multilingual}

| Language  | ISO | Status          |
|-----------|-----|-----------------|
| English   | en  | Full support    |
| Kannada   | kn  | Full support    |
| Hindi     | hi  | Partial support |
| Tamil     | ta  | Planned         |
| Malayalam | ml  | Planned         |
| Marathi   | mr  | Planned         |

**FarmerProfile** schema includes a `language` field used for profile mask scoring. Language-specific API labels:
- `name_kn` — Kannada API label (exists)
- `name_ta`, `name_ml`, `name_hi`, `name_mr` — planned additions

See [`/docs/multilingual_integration.md`](./multilingual_integration.md) for the full multilingual design.

---

## Repository Structure

```
session-aware-toolbench-rerank/
├── src/
│   ├── reranker.py          # Core scoring: score = w_d·s_d + w_m·m + w_c·φ_c
│   ├── co_activation.py     # Co-activation cache, edge weight updates
│   ├── farmer_profile.py    # FarmerProfile dataclass (language field)
│   ├── api_index.py         # data.gov.in API catalogue + labels
│   └── embedding.py         # google/embeddinggemma-300m wrapper
├── experiments/
│   ├── benchmark_g1.py      # Group 1 evaluation
│   ├── benchmark_g2.py      # Group 2 evaluation
│   └── benchmark_g3.py      # Group 3 evaluation
├── kaggle/
│   └── sessionrerank_gemma4_kaggle.ipynb
├── hf_space/                # HuggingFace Space deployment files
│   └── app.py               # Gradio app (Gemma 4 runtime)
└── docs/
    └── README_OBJ1.md       # This file
```

---

## Integration with PhD Pipeline

SessionRerank+ is **Objective 1** in the four-component PhD pipeline:

```
[Obj1: SessionRerank+] ──priority scores──► [Obj2: APRR]
         ▲                                          │
         └──────────pruned context────── [Obj4: FCNP]
```

- **Feeds to Obj2 (APRR)**: Passes ranked API priority scores to the multi-agent router. APRR uses these to initialise the W-matrix for the current session, reducing cold-start routing errors.
- **Receives from Obj4 (FCNP)**: Receives the pruned context window back after FCNP compresses mandi data from 50+ records to top-5. The compressed context is used to update the co-activation cache (successful retrieval reinforces the edge).

### Data Flow

| Source       | Signal                      | Consumer     |
|--------------|-----------------------------|--------------|
| Obj1 → Obj2  | API priority ranking vector | APRR W-matrix initialisation |
| Obj4 → Obj1  | Pruned context + hit/miss   | Co-activation edge update    |

---

## FarmerProfile Schema

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class FarmerProfile:
    farmer_id: str
    state: str
    district: str
    language: str           # ISO code: en, kn, ta, ml, hi, mr
    crops: list[str]
    preferred_markets: list[str]
    session_id: str
```

---

## Running Locally

```bash
git clone https://github.com/joyjeni/session-aware-toolbench-rerank
cd session-aware-toolbench-rerank
pip install -r requirements.txt

# Run benchmark
python experiments/benchmark_g1.py

# Launch Gradio app locally
python hf_space/app.py
```

---

## Citation

```bibtex
@misc{sessionrerank2026,
  title  = {SessionRerank+: Session-Aware API Reranking for Agricultural AI},
  author = {Jeni, Joy},
  year   = {2026},
  note   = {PhD Objective 1, Karnataka Agri Assistant}
}
```

---

*Part of the PhD Agricultural AI pipeline. See also: [Obj2 APRR](./README_OBJ2.md) | [Obj3 MNCD](./README_OBJ3.md) | [Obj4 FCNP](./README_OBJ4.md) | [Multilingual Design](./multilingual_integration.md)*
