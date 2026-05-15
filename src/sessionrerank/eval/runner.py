"""Evaluation runner.

Two regimes:
  * **closed** — rank only the per-query `candidate_apis` pool (small, 4-7).
  * **open**   — rank over the FULL corpus of ~3.2k APIs (harder, matches
                 ToolLLM Table 6 setup).

For each regime we compute Hit@{1,3,5,10} and NDCG@{1,3,5,10}.

Multi-turn session evaluation:
  Benchmark queries are grouped into pseudo-sessions sharing category. For
  query t in a session, the "history" passed to SessionRerank is the set of
  GROUND-TRUTH relevant APIs from queries 1..t-1. This emulates "previous
  successful tool calls" — i.e. the user did the right tool call earlier.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.stats import wilcoxon

from ..data.loader import (
    EvalQuery, API, load_all_eval, build_corpus, build_synthetic_sessions,
)
from ..model.session_rerank import SessionPrior
from .metrics import hit_at_k, ndcg_at_k, recall_at_k, aggregate

KS = (1, 3, 5, 10)


def _rank(scores: np.ndarray, items: Sequence) -> list:
    idx = np.argsort(-scores)
    return [items[i] for i in idx]


# --------------------------------------------------------------------------- #
# Closed-pool evaluation (single-turn)
# --------------------------------------------------------------------------- #
def eval_closed(model, eval_split: list[EvalQuery]) -> dict:
    per = []
    for q in eval_split:
        sc = model.score(q.query, candidates=q.candidate_apis)
        ranked = _rank(sc, [a.uid for a in q.candidate_apis])
        row = {}
        for k in KS:
            row[f"Hit@{k}"] = hit_at_k(ranked, q.relevant_uids, k)
            row[f"NDCG@{k}"] = ndcg_at_k(ranked, q.relevant_uids, k)
            row[f"Recall@{k}"] = recall_at_k(ranked, q.relevant_uids, k)
        per.append(row)
    out = aggregate(per)
    out["_per_query"] = per
    return out


# --------------------------------------------------------------------------- #
# Open-corpus evaluation (single-turn, ranks over full corpus)
# --------------------------------------------------------------------------- #
def eval_open(model, eval_split: list[EvalQuery], corpus: list[API]) -> dict:
    uids = [a.uid for a in corpus]
    per = []
    for q in eval_split:
        sc = model.score(q.query, candidates=corpus)
        ranked = _rank(sc, uids)
        row = {}
        for k in KS:
            row[f"Hit@{k}"] = hit_at_k(ranked, q.relevant_uids, k)
            row[f"NDCG@{k}"] = ndcg_at_k(ranked, q.relevant_uids, k)
            row[f"Recall@{k}"] = recall_at_k(ranked, q.relevant_uids, k)
        per.append(row)
    out = aggregate(per)
    out["_per_query"] = per
    return out


# --------------------------------------------------------------------------- #
# Session-aware (multi-turn) evaluation
# --------------------------------------------------------------------------- #
def eval_session(model, eval_split: list[EvalQuery], corpus: list[API], *,
                 use_session: bool = True, regime: str = "open") -> dict:
    """Evaluate with simulated multi-turn sessions.

    For each session, query t sees history = union of ground-truth APIs from
    queries 1..t-1 in the same session. This rewards methods that can leverage
    the history; baselines simply ignore the `prior` argument.
    """
    uid2api = {a.uid: a for a in corpus}
    sessions = build_synthetic_sessions(eval_split, session_len=3)
    per = []
    for sess in sessions:
        history_apis: list[API] = []
        for q in sess:
            cands = corpus if regime == "open" else q.candidate_apis
            prior = SessionPrior.from_apis(history_apis) if use_session else None
            if hasattr(model, "score") and "prior" in model.score.__code__.co_varnames:
                sc = model.score(q.query, candidates=cands, prior=prior)
            else:
                sc = model.score(q.query, candidates=cands)
            uids = [a.uid for a in cands]
            ranked = _rank(sc, uids)
            row = {}
            for k in KS:
                row[f"Hit@{k}"] = hit_at_k(ranked, q.relevant_uids, k)
                row[f"NDCG@{k}"] = ndcg_at_k(ranked, q.relevant_uids, k)
                row[f"Recall@{k}"] = recall_at_k(ranked, q.relevant_uids, k)
            per.append(row)
            # update history with this query's ground-truth APIs (simulated
            # "successful tool call" outcome).
            for ru in q.relevant_uids:
                if ru in uid2api:
                    history_apis.append(uid2api[ru])
    out = aggregate(per)
    out["_per_query"] = per
    return out


# --------------------------------------------------------------------------- #
# Statistical significance
# --------------------------------------------------------------------------- #
def paired_significance(per_a: list[dict], per_b: list[dict], metric: str) -> dict:
    a = np.array([r[metric] for r in per_a])
    b = np.array([r[metric] for r in per_b])
    if len(a) != len(b) or len(a) == 0:
        return {"p_value": None, "delta_mean": None, "delta_rel_pct": None}
    delta = b - a
    try:
        # Wilcoxon needs non-zero diffs
        if np.any(delta != 0):
            stat, p = wilcoxon(b, a, zero_method="zsplit", alternative="two-sided")
        else:
            p = 1.0
    except Exception:
        p = None
    return {
        "p_value": float(p) if p is not None else None,
        "delta_mean": float(delta.mean()),
        "delta_rel_pct": float(100.0 * delta.mean() / (a.mean() + 1e-9)),
    }
