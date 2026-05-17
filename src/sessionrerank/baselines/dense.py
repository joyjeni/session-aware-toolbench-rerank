"""Dense retrieval baseline — ToolLLM's neural API retriever.

By default uses `sentence-transformers/all-MiniLM-L6-v2` (CPU-friendly).
On Colab T4, swap to `ToolBench/ToolBench_IR_bert_based_uncased` for the
paper-faithful encoder.
"""
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ..data.loader import API
from ..device_utils import resolve_device


class DenseRetriever:
    name = "Dense"

    def __init__(self,
                 corpus: Sequence[API],
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu",
                 cache_dir: str | None = None):
        self.model_name = model_name
        self.name = f"Dense ({model_name.split('/')[-1]})"
        # SentenceTransformer does not understand "xla"/"tpu"; fall back to CPU
        # on TPU runtimes. Also fall back if "cuda" was requested but CUDA is
        # not available.
        self.device = resolve_device(device, for_sentence_transformer=True)
        self.model = SentenceTransformer(model_name, device=self.device)
        self.corpus = list(corpus)
        self.uids = [a.uid for a in self.corpus]
        self._uid2idx = {u: i for i, u in enumerate(self.uids)}
        cache = Path(cache_dir) if cache_dir else None
        emb_path = cache / f"corpus_emb__{model_name.replace('/', '_')}.npy" if cache else None
        if emb_path and emb_path.exists():
            self.corpus_emb = np.load(emb_path)
        else:
            self.corpus_emb = self.model.encode(
                [a.doc() for a in self.corpus],
                batch_size=64,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            if emb_path:
                emb_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(emb_path, self.corpus_emb)

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

    def score(self, query: str, candidates: Sequence[API] | None = None) -> np.ndarray:
        q = self.encode_query(query)
        if candidates is None:
            return self.corpus_emb @ q
        idx = np.array([self._uid2idx[c.uid] for c in candidates], dtype=np.int64)
        return self.corpus_emb[idx] @ q
