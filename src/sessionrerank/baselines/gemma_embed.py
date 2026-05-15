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
import json
import shutil
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from ..data.loader import API


def _maybe_patch_tokenizer_config(model_path: str) -> str:
    """Patch Gemma 4 tokenizer_config.json so v4 transformers can load it.

    Gemma 4 ships extra_special_tokens as a v5-style list; v4 transformers
    expects a dict. If we detect that, we copy the model dir to a writable
    cache and rewrite the tokenizer_config.json. No-op for HF model ids,
    and a no-op on transformers >= 5 (which handles the list format natively).
    Returns the (possibly new) model path to use.
    """
    if not os.path.isdir(model_path):
        return model_path
    # Skip the patch entirely on transformers v5+: it understands the list format.
    try:
        import transformers
        major = int(transformers.__version__.split(".")[0])
        if major >= 5:
            return model_path
    except Exception:
        pass
    cfg_path = os.path.join(model_path, "tokenizer_config.json")
    if not os.path.isfile(cfg_path):
        return model_path
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except Exception:
        return model_path
    if not isinstance(cfg.get("extra_special_tokens"), list):
        return model_path
    # Need to patch. Build a writable mirror.
    cache_root = os.environ.get("GEMMA_PATCH_DIR", "/kaggle/working/gemma_patched")
    if not os.path.isdir("/kaggle"):
        cache_root = os.path.join(tempfile.gettempdir(), "gemma_patched")
    name = os.path.basename(model_path.rstrip("/")) or "model"
    patched_dir = os.path.join(cache_root, name)
    patched_cfg = os.path.join(patched_dir, "tokenizer_config.json")
    # Use exists+is-real-file to decide; symlinks to read-only file fail os.path.isfile differently
    already_patched = os.path.isfile(patched_cfg) and not os.path.islink(patched_cfg)
    if not already_patched:
        os.makedirs(patched_dir, exist_ok=True)
        # symlink everything; we'll delete the tokenizer_config.json symlink and write real file
        for entry in os.listdir(model_path):
            src = os.path.join(model_path, entry)
            dst = os.path.join(patched_dir, entry)
            if os.path.islink(dst) or os.path.exists(dst):
                # Remove existing symlink/file so we can replace it (especially tokenizer_config.json)
                try:
                    if os.path.islink(dst) or os.path.isfile(dst):
                        os.unlink(dst)
                    elif os.path.isdir(dst):
                        shutil.rmtree(dst)
                except Exception:
                    pass
            try:
                if os.path.isdir(src):
                    # Symlink the whole subdir (saves disk; nested files still resolve)
                    os.symlink(src, dst)
                else:
                    os.symlink(src, dst)
            except Exception:
                try:
                    shutil.copy2(src, dst) if os.path.isfile(src) else shutil.copytree(src, dst)
                except Exception:
                    pass
        # Now overwrite the tokenizer_config.json with patched dict-format content.
        # First make sure it's not still a symlink to the read-only source.
        if os.path.islink(patched_cfg):
            try:
                os.unlink(patched_cfg)
            except Exception:
                pass
        cfg["extra_special_tokens"] = {}
        with open(patched_cfg, "w") as f:
            json.dump(cfg, f)
        print(f"[gemma] patched tokenizer_config.json -> {patched_dir}")
    return patched_dir


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
            torch.float16 if torch_dtype == "auto" and device.startswith("cuda")
            else (torch.float32 if torch_dtype == "auto" else getattr(torch, torch_dtype))
        )
        patched = _maybe_patch_tokenizer_config(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(patched, token=hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Gemma 4 e2B is multimodal: AutoModel returns the joint encoder and
        # complains about MISSING/UNEXPECTED PLE + vision/audio tower keys.
        # AutoModelForCausalLM loads the full causal LM cleanly; .model is the
        # text-only inner Gemma4TextModel we actually want for mean pooling.
        try:
            causal = AutoModelForCausalLM.from_pretrained(
                patched, dtype=dtype, token=hf_token, low_cpu_mem_usage=True,
            )
            self.model = getattr(causal, "model", causal).to(device)
        except Exception as e:
            print(f"[Gemma-Embed] AutoModelForCausalLM failed ({e}); falling back to AutoModel")
            self.model = AutoModel.from_pretrained(
                patched, dtype=dtype, token=hf_token, low_cpu_mem_usage=True,
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

    def free_gpu(self):
        """Move model off GPU and clear cache. Query encoding will move it back lazily."""
        try:
            self.model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _ensure_on_device(self):
        try:
            p = next(self.model.parameters())
            if str(p.device) != self.device:
                self.model.to(self.device)
        except StopIteration:
            pass

    @torch.no_grad()
    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure_on_device()
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
