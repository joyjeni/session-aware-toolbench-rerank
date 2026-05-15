"""Static BM25 baseline (ToolLLM paper Tab. 6)."""
from __future__ import annotations
import re
from typing import Sequence
import numpy as np
from rank_bm25 import BM25Okapi
from ..data.loader import API

_TOK = re.compile(r"[A-Za-z0-9]+")


def _tok(s: str) -> list[str]:
    return [t.lower() for t in _TOK.findall(s)]


class BM25Retriever:
    name = "BM25"

    def __init__(self, corpus: Sequence[API]):
        self.corpus = list(corpus)
        self.uids = [a.uid for a in self.corpus]
        self._uid2idx = {u: i for i, u in enumerate(self.uids)}
        self.bm25 = BM25Okapi([_tok(a.doc()) for a in self.corpus])

    def score(self, query: str, candidates: Sequence[API] | None = None) -> np.ndarray:
        toks = _tok(query)
        all_scores = self.bm25.get_scores(toks)
        if candidates is None:
            return all_scores
        # subset to candidates by uid
        idx = np.array([self._uid2idx[c.uid] for c in candidates], dtype=np.int64)
        return all_scores[idx]
