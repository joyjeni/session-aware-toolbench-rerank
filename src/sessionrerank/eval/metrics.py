"""Hit@K, NDCG@K — standard IR metrics, ToolLLM-paper-compatible."""
from __future__ import annotations
import numpy as np
from typing import Sequence


def hit_at_k(ranked_uids: Sequence[str], relevant: set[str], k: int) -> float:
    return float(any(u in relevant for u in ranked_uids[:k]))


def recall_at_k(ranked_uids: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hit = sum(1 for u in ranked_uids[:k] if u in relevant)
    return hit / len(relevant)


def ndcg_at_k(ranked_uids: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, u in enumerate(ranked_uids[:k]):
        if u in relevant:
            dcg += 1.0 / np.log2(i + 2)
    # ideal DCG
    ideal = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg else 0.0


def aggregate(per_query: list[dict[str, float]]) -> dict[str, float]:
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: float(np.mean([d[k] for d in per_query])) for k in keys}
