# Spec: Journal Paper (Taylor & Francis CRC Press Conference Proceedings)

## Goal
Write a publication-ready conference paper for **Taylor & Francis CRC Press** proceedings on **SessionRerank+** — a session-aware, co-activation-cached re-ranking layer for tool retrieval, powered by Gemma 4, evaluated on ToolBench + a Karnataka smallholder-agriculture deployment.

The author has a **draft 3-page version** already at `/home/user/workspace/sessionrerank/paper/sessionrerank_paper.docx`. The new task is to expand this into a **deeper, "professor-style" exposition** — every term defined, every formula derived, every design choice motivated — while staying within the CRC-SPAST template constraints.

## Template constraints (strict — CRC-SPAST T&F)
- **5 pages MAX, 10 refs MAX, 4000 words MAX, 12% plagiarism MAX**
- Times New Roman 10pt
- 2-column body (8.7cm column width, 1.06cm gutter)
- Abstract ≤150 words, single paragraph, indented 1cm
- Title ≤15 words; no "new/novel/the/a/an"; first word capitalized
- Numbered equations on a separate line, right-aligned (N)
- Tables: caption ABOVE, labeled "Table I", "Table II"
- Figures: caption BELOW, italic, labeled "Fig. 1", "Fig. 2"
- References: Brockman-style, year mandatory

## Author / affiliation
**Abigail Creations**, Kyndryl India / MS Ramaiah University of Applied Sciences, Bengaluru, India.
Email: `abigailinnovations@gmail.com`

## "Professor style" — what this means
The user explicitly asked for the paper to be written **as a professor would explain it** — pedagogical, not terse. Specifically:

- Define every technical term on first use, even if it would normally be assumed (e.g., "NDCG@5 (Normalized Discounted Cumulative Gain at rank 5) measures how well the top-5 retrieved items match the ideal ranking, weighted by position…").
- Motivate every design decision before stating it ("Because conventional rerankers treat each query independently, they cannot exploit the fact that farmers in a single session repeatedly query related APIs. We address this with…").
- Derive every formula step by step, not just present it. For the co-activation score, walk through why the recency discount γ^(n-i) is exponential, why the log(1+w) dampening is needed, what happens at the boundary cases.
- Use clear topic sentences. Each paragraph begins with the claim it proves.
- Prefer plain language over jargon. When jargon is unavoidable (e.g., LoRA, NDCG, BM25), define it inline.

Despite the pedagogical tone, the paper must remain dense — 5 pages is tight. Use the page budget for substance, not filler. Cut every sentence that doesn't earn its place.

## Required sections (in order, with target word counts)
1. **Title** (≤15 words)
2. **Abstract** (≤150 words, single paragraph)
3. **Keywords** (5-7 keywords)
4. **Introduction** (~500 words) — the tool-overload problem, why session-awareness matters, our three contributions
5. **Related work** (~400 words) — dense retrieval, learning-to-rank for tools, multi-turn dialog, co-occurrence priors. Cite Qin et al. (2023) ToolBench/ToolLLM, Reimers & Gurevych (2019) sentence-transformers, Wang et al. (2020) MiniLM, Gemma Team (2024), Daniel & Michael (2024) Unsloth, ToolRerank, cross-encoder rerankers
6. **Method** (~1000 words, 3 numbered equations)
   - 6.1 Problem formulation — formal definitions of query q, session history H, candidate API set V
   - 6.2 Score function (Eq. 1) — the three-term decomposition w_d·s_d + w_m·m + w_c·φ_c, derived term by term
   - 6.3 Co-activation prior (Eq. 2) — explain φ_c = Σ γ^(n-i) log(1+w_{h_i,v}), justify each design choice (exponential discount, log dampening, why we sum rather than max)
   - 6.4 Edge update rule (Eq. 3) — w_{u,v} ← (1-ρ)·w_{u,v} + δ·1[success], explain ρ=0.02 (50-session half-life), δ=1.0 (binary reinforcement), γ=0.7 (3-turn effective window), why these defaults
   - 6.5 Cascade architecture — BM25 → Dense → ToolRerank → Cross-Encoder → Gemma 4 prompted → Gemma 4 FT → SessionRerank+. Why this order, why each stage adds.
   - 6.6 Gemma 4 fine-tune — LoRA r=16 on 5000 ToolBench trajectories, 1 epoch, Unsloth, training loss curve
7. **Experimental setup** (~400 words) — ToolBench G1/G2/G3 open splits (Qin et al., 2023), Karnataka 100-query bank construction methodology (Tumakuru + Mandya extension officers), 5 random seeds {0, 1, 2, 42, 123}, 10000 paired bootstrap resamples
8. **Results** (~600 words, 1 table, 2 figures)
   - **Table I (placeholder until Kaggle finishes)** — full cascade with NDCG@5 and Karnataka first-hit
   - **Fig. 1** — architecture diagram (already at `paper/figures/fig1_architecture.png`)
   - **Fig. 2** — two-turn co-activation effect (already at `paper/figures/fig2_two_turn.png`)
   - Paired bootstrap p<0.01 against every baseline, on both metrics
   - Co-activation ablation: 0.892 → 0.881 NDCG, 0.91 → 0.86 first-hit
   - Profile mask ablation: 0.91 → 0.79 first-hit
9. **Impact and discussion** (~400 words) — 8.5M Karnataka smallholders, 12 queries/season average, reducing tool-chain length from 4.2 to 1.3 invocations saves ~5.2 MB/farmer/season (relevant for rural 2G/3G connectivity)
10. **Limitations and future work** (~200 words) — cold-start sessions, transfer to other state catalogs, Kannada query embedding quality
11. **Conclusion** (~150 words)
12. **References** (max 10)

## Dynamic metrics update
The Kaggle notebook at `https://www.kaggle.com/code/joyjeni/sessionrerank` is **currently running** (started ~10 AM IST June 15, 2026). When it completes:
- Output files appear at `https://www.kaggle.com/api/v1/kernels/output?userName=joyjeni&kernelSlug=sessionrerank`
- Look for `sr/results/metrics.json` and `metrics_summary.csv`
- **Replace placeholder Table I numbers** with the real ones, but keep the structure
- **Re-render the docx** via `python3 paper/build_paper_docx.py` after updating
- **Commit and push** to `https://github.com/joyjeni/session-aware-toolbench-rerank` (use `api_credentials=["github"]`)
- **Share final docx** via `share_file` with `name="sessionrerank_journal_paper"` (different from `sessionrerank_paper` so users can keep both versions)

Poll Kaggle every ~10 min until status transitions from `running` to `complete` or `error`.
Use `api_credentials=["custom-cred:www.kaggle.com"]` for Kaggle API calls.

## Existing artifacts to reuse / extend
- `/home/user/workspace/sessionrerank/paper/sessionrerank_paper.md` — 1,650-word source draft
- `/home/user/workspace/sessionrerank/paper/sessionrerank_paper.docx` — current 3-page output
- `/home/user/workspace/sessionrerank/paper/build_paper_docx.py` — python-docx builder (TNR 10pt, 2-col body)
- `/home/user/workspace/sessionrerank/paper/figures/fig1_architecture.png` — architecture diagram
- `/home/user/workspace/sessionrerank/paper/figures/fig2_two_turn.png` — co-activation effect
- `/home/user/workspace/crc_template.md` — pandoc-extracted CRC-SPAST template

## Key numbers (placeholder until Kaggle finishes — verbatim from existing draft)

| Method | ToolBench NDCG@5 | Karnataka first-hit |
|---|---:|---:|
| BM25 | 0.473 | 0.42 |
| Dense (MiniLM) | 0.800 | 0.71 |
| ToolRerank | 0.789 | 0.68 |
| Cross-Encoder | 0.835 | 0.77 |
| Gemma 4 prompted | 0.857 | 0.81 |
| Gemma 4 fine-tuned | 0.881 | 0.84 |
| **SessionRerank+** | **0.892** | **0.91** |

These will be **overwritten with real Kaggle results** before final delivery.

## Output format
- **DOCX** at `/home/user/workspace/sessionrerank/paper/sessionrerank_journal_paper.docx`
- Source MD at `/home/user/workspace/sessionrerank/paper/sessionrerank_journal_paper.md` (for diffs)
- Visual QA via `soffice --headless --convert-to pdf` and `pdftoppm` — read every page as a JPEG and verify no broken layouts, no clipped text, no overflowing tables before sharing
- Final share: `share_file(file_path=..., name="sessionrerank_journal_paper")`

## Constraints
- Stay ≤5 pages, ≤10 refs, ≤4000 words. Use `pandoc paper/sessionrerank_journal_paper.docx -o /tmp/check.md && wc -w /tmp/check.md` to validate.
- "Professor style" must NOT inflate beyond 4000 words. Choose depth, not breadth.
- Every cited reference must include the year (Brockman style).
- Use `api_credentials=["github"]` for git push.
- Use `api_credentials=["custom-cred:www.kaggle.com"]` for Kaggle API.
