"""Gemma 4 as a prompted cross-encoder reranker.

Strategy: feed the LM a structured prompt of the form

    Query: <q>
    Candidate API: <tool/api/description>
    Is this API relevant? Answer YES or NO.

We then compute the log-likelihood of the "YES" token (vs "NO") and use
the YES-logit (after softmax) as the relevance score. This is the same
trick used by `monoT5` / Gemma-as-judge: prompt + first-token logit.

This is a *first-stage rerank* over a small candidate set (typically
top-100 from BM25 or Gemma-Embed), since LM forward passes are expensive.

Usage
-----
>>> rer = GemmaPromptedReranker(model_name="google/gemma-4-270m-it", device="cuda")
>>> scores = rer.score(query, candidates)  # candidates length ≤ 100
"""
from __future__ import annotations
import os
from typing import Sequence
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data.loader import API
from .gemma_embed import _maybe_patch_tokenizer_config


PROMPT_TEMPLATE = (
    "You are evaluating whether an API is relevant to a user query.\n"
    "Query: {query}\n"
    "API: {tool} / {api}\n"
    "Description: {desc}\n"
    "Answer YES if the API can help answer the query, otherwise NO.\n"
    "Answer:"
)


class GemmaPromptedReranker:
    """Cross-encoder reranker using a Gemma causal LM via YES/NO logits."""

    name = "GemmaRerank"

    def __init__(
        self,
        model_name: str = "google/gemma-4-270m-it",
        device: str = "cuda",
        max_length: int = 384,
        batch_size: int = 8,
        torch_dtype: str = "auto",
        load_in_4bit: bool = False,
    ):
        self.model_name = model_name
        short = model_name.split("/")[-1]
        self.name = f"GemmaRerank ({short})"
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        dtype = (
            torch.float16 if torch_dtype == "auto" and device.startswith("cuda")
            else (torch.float32 if torch_dtype == "auto" else getattr(torch, torch_dtype))
        )
        patched = _maybe_patch_tokenizer_config(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(patched, token=hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_name = patched  # use patched path for AutoModel below

        # Auto-enable 4-bit on large (>=1B) Gemma to fit on a single T4 (~14.5 GiB).
        if load_in_4bit or os.environ.get("GEMMA_RERANK_4BIT") == "1":
            try:
                from transformers import BitsAndBytesConfig
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, quantization_config=bnb, token=hf_token,
                    device_map={"": device} if device.startswith("cuda") else None,
                    low_cpu_mem_usage=True,
                )
            except Exception as e:
                print(f"[GemmaRerank] 4-bit load failed ({e}); falling back to fp16")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, dtype=dtype, token=hf_token, low_cpu_mem_usage=True,
                ).to(device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=dtype, token=hf_token, low_cpu_mem_usage=True,
            ).to(device)
        self.model.eval()

        # Pre-compute YES/NO token IDs. We try a few common encodings.
        def first_token(s: str) -> int:
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            return ids[0]

        self.yes_id = first_token(" YES")
        self.no_id = first_token(" NO")

    @torch.no_grad()
    def score(self, query: str, candidates: Sequence[API]) -> np.ndarray:
        prompts = [
            PROMPT_TEMPLATE.format(
                query=query, tool=c.tool, api=c.api,
                desc=(c.description or "")[:600],
            )
            for c in candidates
        ]
        out_scores = np.zeros(len(prompts), dtype=np.float32)
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i : i + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc, return_dict=True).logits  # [B, T, V]
            # Position of the last real token in each row
            lengths = enc["attention_mask"].sum(dim=1) - 1
            batch_idx = torch.arange(logits.size(0), device=logits.device)
            last_logits = logits[batch_idx, lengths]  # [B, V]
            # Softmax over {yes, no} to get a calibrated probability
            yes_no = last_logits[:, [self.yes_id, self.no_id]]
            probs = F.softmax(yes_no.float(), dim=-1)
            out_scores[i : i + len(batch)] = probs[:, 0].cpu().numpy()
        return out_scores
