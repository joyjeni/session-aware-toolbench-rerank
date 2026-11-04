# SATR — Session-aware ToolBench rerank (ACRS Objective 1)

> **Ph.D. CSE, M S Ramaiah University of Applied Sciences (FET).**  
> Scholar: Jenisha T (`24ETRP720001`). Supervisor: Dr. Padma Priya Darshini P.  
> Integrated loop: **SATR → APRR → MNCD → FCNP** (Adaptive Context Reasoning System).


> **Postdoctoral review (November 2026) — read first.** 2 critical issue(s) affect the results reported below; see [`docs/POSTDOC_REVIEW.md`](docs/POSTDOC_REVIEW.md) before citing any number from this repository.

This repository is the **retrieval chapter** of ACRS. Given query \(q\) and session
history \(H\), SATR returns a ranked list of ToolBench-schema tools fused with a
co-activation cache. The unit of a journal paper or thesis chapter is the **fusion
rule and the session cache**, not a leaderboard score.

**Proposal-stage stance:** no NDCG, Hit@k, or Wilcoxon figure in this README is a
commitment of the research proposal. Historical tables, if still present in
notebooks, are laboratory traces — not promised results.

## SOTA this chapter is written against

- Qin et al., *ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs*, **ICLR 2024**. SBERT over API documents, DFSDT planner, ToolEval. Turn-amnesic.
- Zheng et al., *ToolRerank: Adaptive and Hierarchy-Aware Reranking for Tool Retrieval*, **LREC-COLING 2024**. Contrastive rerank of ToolLLM candidates. Still turn-amnesic.

**Gap (G1).** Neither maintains a success-conditioned co-activation graph, nor
ingests pruned live citations from a later memory stage.

**Novelty for the community.** A session co-activation cache \(C[a,b]\) and a
convex fusion \(s = (1-\lambda)s_{\mathrm{sem}} + \lambda s_{\mathrm{sess}}\).
Hierarchy-aware truncation from ToolRerank is **kept**. Live Indian APIs are
**executed later** by MNCD; SATR does not fabricate mandi rows.

## Data policy (journal-publishable Indian OGD)

- **Ranking library:** ToolBench / ToolLLM (Apache-2.0). Not the execution corpus.
- **Live execution (in the integrated lab / MNCD):** verified `data.gov.in`
  Agriculture UUIDs only (GODL-India / NIC), e.g. AGMARKNET
  `9ef84268-d588-465a-a308-a864a43d0070`.
- **Out of scope:** StableToolBench dumps, Kaggle NPK, PlantVillage, simulated prices.
- HTTP 5xx/429 should be retried; snapshot files must not be served as prices.

## Core equations

### Ranking score

\[
s(a\mid q,H)=w_{\mathrm{base}}s_{\mathrm{base}}+w_{\mathrm{cat}}\mathrm{cat}+w_{\mathrm{sch}}\mathrm{sch}+w_{\mathrm{ept}}\mathrm{ept}+w_{\mathrm{cooc}}\phi_{\mathrm{coact}}+w_{\mathrm{rec}}\mathrm{rec}
\]

### Co-activation update

\[
w_{u,v}\leftarrow(1-\rho)\,w_{u,v}+\delta\cdot\mathbf{1}[\mathrm{success}]
\]

## Evaluation protocol after the postdoc review (November 2026)

```bash
# realistic: the model's own top-2 picks become the session history (default)
python -m sessionrerank.eval.run_all --session-history predicted --session-grouping category
# control: sessions not built by category — any gain must come from real session signal
python -m sessionrerank.eval.run_all --session-history predicted --session-grouping random
# upper bound only (original protocol): gold APIs of earlier turns as history
python -m sessionrerank.eval.run_all --session-history oracle
```

Earlier tables in `paper/results.md` were produced with oracle history and category-built sessions and
should be re-run before citation (see `docs/POSTDOC_REVIEW.md`).

## HTTP microservice

```bash
python service/app.py 43131
curl -X POST http://127.0.0.1:43131/rerank -H 'Content-Type: application/json' \
  -d '{"query":"mandi price of tomato in Punjab","history_uids":[],"top_k":5}'
```

## Integration

Master laboratory: `phd-agentic-ai-master` (SATR shortlist → APRR → MNCD live
OGD → FCNP write-back into this cache).
