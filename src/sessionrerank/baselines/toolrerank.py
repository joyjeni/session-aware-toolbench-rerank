"""ToolRerank baseline (Zheng et al., 2024).

ToolRerank introduces two ideas on top of dense retrieval:

  (a) Adaptive Truncation — for single-tool queries, only the top tool's APIs
      are kept; for multi-tool queries the top-N tools are kept. This down-
      weights "noisy" APIs from unrelated tools at the candidate pool.
  (b) Hierarchical-aware re-ranking — pairwise tool-vs-api scoring with a
      cross-encoder, encouraging that:
          * API description matches the query, AND
          * the API's *tool* description also matches.

We implement a faithful, lightweight version using:
  * tool-level mean-pooled score
  * api-level dense score
  * a mixing coefficient β tuned on G1_instruction's train-half (here we
    use the published β=0.6 from the ToolRerank paper).

Reference: Zheng, Y. et al. "ToolRerank: Adaptive and Hierarchy-Aware Reranking
for Tool Retrieval", LREC-COLING 2024.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np

from ..data.loader import API
from .dense import DenseRetriever


class ToolRerank:
    name = "ToolRerank"

    def __init__(self, dense: DenseRetriever, beta: float = 0.6):
        self.dense = dense
        self.beta = beta
        # Aggregate per-tool documents = concat of all API docs of that tool
        by_tool: dict[str, list[int]] = {}
        for i, a in enumerate(dense.corpus):
            by_tool.setdefault(a.tool_name, []).append(i)
        self.by_tool = by_tool
        self.tool_names = list(by_tool.keys())
        # tool embedding = mean of its api embeddings
        self.tool_emb = np.stack([
            dense.corpus_emb[idxs].mean(axis=0) for idxs in by_tool.values()
        ])
        # normalize
        n = np.linalg.norm(self.tool_emb, axis=1, keepdims=True) + 1e-9
        self.tool_emb = self.tool_emb / n
        self._tool2idx = {t: i for i, t in enumerate(self.tool_names)}

    def score(self, query: str, candidates: Sequence[API] | None = None) -> np.ndarray:
        q = self.dense.encode_query(query)
        api_scores = self.dense.score(query, candidates)
        cands = candidates if candidates is not None else self.dense.corpus
        # tool-level scores per candidate
        ts = np.zeros(len(cands), dtype=np.float32)
        for i, a in enumerate(cands):
            ti = self._tool2idx.get(a.tool_name, None)
            if ti is None:
                continue
            ts[i] = self.tool_emb[ti] @ q
        # adaptive truncation: penalize candidates whose tool isn't in top-K tools
        K = min(5, max(1, len(set(a.tool_name for a in cands)) // 2))
        top_tools = np.argsort(-ts)[:K] if len(ts) else np.array([])
        mask_set = {i for i in top_tools}
        # mixing
        final = self.beta * api_scores + (1.0 - self.beta) * ts
        # mild penalty for candidates whose tool is not in the top-K tool ranking
        # (Adaptive Truncation effect, implemented as a soft down-weight)
        for i in range(len(cands)):
            if i not in mask_set and ts[i] < (ts[top_tools].mean() if len(top_tools) else 0):
                final[i] -= 0.05
        return final
