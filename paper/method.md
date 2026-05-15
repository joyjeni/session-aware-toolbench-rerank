# SessionRerank — Method

## Problem
Tool retrieval in LLM agentic systems treats every user query in isolation, even
when the user is mid-session and has already successfully invoked several APIs.
ToolLLM's neural retriever (Qin et al., 2024) and ToolRerank (Zheng et al., 2024)
both operate on the *current* query only.

> **Question.** Can we use the trajectory of *previously successful tool calls*
> within the same session as a free signal to bias retrieval toward
> categorically / structurally / behaviourally similar APIs — without any
> explicit user preference input?

## Setup
Let
- \( q_t \) be the user query at turn \( t \),
- \( H_t = \{a^{*}_1, \dots, a^{*}_{t-1}\} \) be the set of APIs whose calls
  succeeded in turns \( 1, \dots, t-1 \) of the same session,
- \( s_{\text{base}}(a \mid q) \) be a base retriever score (BM25 or dense).

We define three *emergent* affinity functions over the candidate API \( a \):

\[
\begin{aligned}
\phi_{\text{cat}}(a, H) &= \frac{\sum_{a^{*} \in H} \gamma^{|H|-i}\,\mathbb{1}[\text{cat}(a^{*}) = \text{cat}(a)]}{Z_{\text{cat}}(H)} \\
\phi_{\text{sch}}(a, H) &= \frac{\sum_{a^{*} \in H} \gamma^{|H|-i}\,\mathbb{1}[\sigma(a^{*}) = \sigma(a)]}{Z_{\text{sch}}(H)} \\
\phi_{\text{ept}}(a, H) &= \frac{\sum_{a^{*} \in H} \gamma^{|H|-i}\,\mathbb{1}[\rho(a^{*}) = \rho(a)]}{Z_{\text{ept}}(H)}
\end{aligned}
\]

where
- \( \sigma(\cdot) \) is the **schema signature** — the sorted list of required-parameter names,
- \( \rho(\cdot) \) is the **endpoint pattern** — normalized api-name tokens,
- \( \gamma = 0.85 \) is a temporal decay, so more-recent successful calls
  dominate,
- \( Z \) normalizes each axis to a probability over observed values.

## Re-ranker
The SessionRerank score is the convex combination

\[
s(a \mid q, H) = w_{\text{base}}\,\tilde s_{\text{base}}(a \mid q)
              + w_{\text{cat}}\,\log(1+10\phi_{\text{cat}})
              + w_{\text{sch}}\,\log(1+10\phi_{\text{sch}})
              + w_{\text{ept}}\,\log(1+10\phi_{\text{ept}})
\]

with \( \tilde s_{\text{base}} \) z-standardized per query. The log-transform
turns the priors into log-odds, so they compose additively with the dense
log-similarity.

## Training the weights
We treat each multi-turn ToolBench trajectory with \( K \geq 2 \) successful
tool calls as \( K-1 \) training examples: given history \( H_k \), the
positive is the next successful API call \( a^{*}_{k+1} \), negatives are
random APIs from the corpus. With InfoNCE loss

\[
\mathcal{L} = -\log \frac{\exp(s(a^{*}; q, H_k))}{\sum_{a' \in \mathcal{N}_k \cup \{a^{*}\}} \exp(s(a'; q, H_k))}
\]

we learn \( (w_{\text{base}}, w_{\text{cat}}, w_{\text{sch}}, w_{\text{ept}}) \in \mathbb{R}_{+}^4 \)
via Adam (the parameters are stored in log-space to enforce positivity).

> **Note on emergent personalization.** No annotated preferences are used. The
> signal comes entirely from which APIs were *invoked and returned non-errors*
> in earlier turns. This is dataset-free personalization in the spirit of
> implicit-feedback recommender systems, but at the granularity of tool
> retrieval.

## Why this is novel
- **ToolRerank** (Zheng et al., 2024) re-ranks structurally (single- vs.
  multi-tool) but ignores per-session history.
- **PEToolLLM** (Sun et al., 2024) personalizes via synthetic preference
  profiles — explicit, not emergent.
- **SessionRerank** uses the trajectory itself as personalization signal — no
  new annotations, no profile collection, plug-and-play on top of any base
  retriever.

## Properties
- **Empty-history safety**: when \( H = \emptyset \), all \( \phi \to 0 \) and
  the score reduces to the base retriever. SessionRerank never hurts
  single-turn queries by construction.
- **Cost**: \( O(|H|) \) hash lookups per candidate at inference; no extra
  embeddings.
- **Reproducibility**: full benchmark in 9 minutes on CPU (`all-MiniLM-L6-v2`),
  ~3 minutes on T4 (`ToolBench_IR_bert_based_uncased`).
