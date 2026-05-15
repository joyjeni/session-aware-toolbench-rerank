"""Gemma 4 as a bi-encoder via mean-pooled hidden states.

This wraps a Gemma 4 causal LM and uses the last-layer hidden states
(mean-pooled with attention mask, then L2-normalized) as a sentence
embedding. Drop-in replacement for `DenseRetriever`.

Why this works
--------------
Decoder-only LMs trained with causal masking still produce token-level
representations whose mean is a competitive sentence embedding,
especially for short, descriptive passages like API docs. See
GTR / Gecko / NV-Embed and the SGPT line of work.

Recommended model sizes
-----------------------
- ``google/gemma-4-270m`` — fast on T4, good baseline.
- ``google/gemma-4-1b``   — best speed/quality on A100.
- ``google/gemma-4-4b``   — strongest but heavier; needs ≥16 GB VRAM.

Note: at first publication the exact Gemma 4 HF model id may differ
(e.g. ``google/gemma-4-270m-it``). The ``model_name`` argument accepts
any HF causal-LM id, so you can point it at whatever the hackathon
publishes without code changes.

Usage
-----
>>> enc = GemmaEmbedder(corpus, model_name="google/gemma-4-270m", device="cuda")
>>> scores = enc.score(query, candidates)
"""
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from ..data.loader import API


class GemmaEmbedder:
    """Bi-encoder built on top of any Gemma-style causal LM."""

    name = "Gemma-Embed"

    def __init__(
        self,
        corpus: Sequence[API],
        model_name: str = "google/gemma-4-270m",
        device: str = "cuda",
        cache_dir: str | None = None,
        max_length: int = 256,
        batch_size: int = 16,
        torch_dtype: str = "auto",
    ):
        self.model_name = model_name
        short = model_name.split("/")[-1]
        self.name = f"Gemma-Embed ({short})"
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

        # HF token (Gemma is gated) — read silently from env so we never log it
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

        dtype = (
            torch.bfloat16 if torch_dtype == "auto" and device.startswith("cuda")
            else (torch.float32 if torch_dtype == "auto" else getattr(torch, torch_dtype))
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=dtype, token=hf_token
        ).to(device)
        self.model.eval()

        self.corpus = list(corpus)
        self.uids = [a.uid for a in self.corpus]
        self._uid2idx = {u: i for i, u in enumerate(self.uids)}

        # Cache corpus embeddings — keyed by model name so caches don't collide
        cache = Path(cache_dir) if cache_dir else None
        emb_path = (
            cache / f"corpus_emb__{model_name.replace('/', '_')}.npy"
            if cache else None
        )
        if emb_path and emb_path.exists():
            self.corpus_emb = np.load(emb_path)
        else:
            self.corpus_emb = self._encode([a.doc() for a in self.corpus])
            if emb_path:
                emb_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(emb_path, self.corpus_emb)

    @torch.no_grad()
    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        all_emb = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            out = self.model(**enc, output_hidden_states=False, return_dict=True)
            # last_hidden_state: [B, T, H]
            h = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()  # [B, T, 1]
            summed = (h * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            emb = summed / counts                              # mean pool
            emb = F.normalize(emb.float(), p=2, dim=1)         # L2 norm
            all_emb.append(emb.cpu().numpy())
        return np.concatenate(all_emb, axis=0).astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        return self._encode([query])[0]

    def score(self, query: str, candidates: Sequence[API] | None = None) -> np.ndarray:
        q = self.encode_query(query)
        if candidates is None:
            return self.corpus_emb @ q
        idx = np.array([self._uid2idx[c.uid] for c in candidates], dtype=np.int64)
        return self.corpus_emb[idx] @ q
