"""Cross-encoder reranker baseline — strong non-LLM neural reranker.

Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` (or any other CrossEncoder model
from sentence-transformers) to rescore the top-K candidates from a first-stage
retriever. This is the canonical retrieval-rerank cascade that the IR community
considers a hard baseline to beat — including SessionRerank.

Why this matters for our hackathon:
  - It isolates the contribution of *session-aware* signals: a strong
    cross-encoder can already exploit query/document semantics, so any further
    gain from SessionRerank is attributable to multi-turn history.
  - It runs CPU-friendly on tens of candidates per query; on Kaggle T4 it
    runs the full Stage 1 in ~3 minutes.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np

from ..data.loader import API
from ..device_utils import resolve_device


class CrossEncoderReranker:
    """First-stage retriever + sentence-transformers CrossEncoder rerank."""

    def __init__(self,
                 first_stage,
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 topk: int = 100,
                 device: str = "cpu",
                 batch_size: int = 64):
        try:
            from sentence_transformers import CrossEncoder
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker"
            ) from e
        self.first = first_stage
        self.topk = topk
        self.model_name = model_name
        # CrossEncoder uses the same device rules as DenseRetriever: no XLA, and
        # silent fallback to CPU when CUDA is requested but unavailable.
        self.device = resolve_device(device, for_sentence_transformer=True)
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, device=self.device)
        self.name = f"CrossEncoder ({model_name.split('/')[-1]}) top-{topk}"
        # Cache (query → cand_uids → score) to avoid re-running for the same
        # query under closed/open/session regimes within one process.
        self._cache: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}

    @property
    def corpus(self):
        return getattr(self.first, "corpus", None)

    # ------------------------------------------------------------------
    def score(self,
              query: str,
              candidates: Sequence[API] | None = None) -> np.ndarray:
        if candidates is None:
            candidates = self.first.corpus
        n = len(candidates)
        s1 = self.first.score(query, candidates)
        if n <= self.topk:
            # Rerank everything
            top_idx = np.arange(n)
        else:
            top_idx = np.argsort(-s1)[: self.topk]
        # Build (query, doc) pairs for top_idx
        cand_top = [candidates[i] for i in top_idx]
        key = (query, tuple(c.uid for c in cand_top))
        if key in self._cache:
            s2 = self._cache[key]
        else:
            pairs = [(query, c.doc()) for c in cand_top]
            s2 = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            self._cache[key] = np.asarray(s2, dtype=np.float32)
        if n <= self.topk:
            return s2.astype(np.float32)
        # Final scores: cross-encoder scores for top-K, -inf for the rest
        out = np.full(n, -1e9, dtype=np.float32)
        for j, i in enumerate(top_idx):
            out[i] = s2[j]
        return out
