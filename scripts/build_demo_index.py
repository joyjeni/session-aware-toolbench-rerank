"""Precompute a tiny demo index for the Vercel /api/rerank endpoint.

Picks 60 APIs spread across 8 popular ToolBench categories, computes
MiniLM embeddings + metadata, and writes a single JSON file consumed by
`web/lib/demo-index.json`.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from sessionrerank.data.loader import build_corpus
from sessionrerank.baselines.dense import DenseRetriever

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "public" / "demo-index.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

CATS = ["Data", "Media", "Logistics", "Finance", "Tools", "eCommerce",
        "Sports", "Communication"]


def main():
    corpus = build_corpus()
    by_cat = {c: [] for c in CATS}
    for a in corpus:
        if a.category in by_cat and len(by_cat[a.category]) < 8:
            by_cat[a.category].append(a)
    demo = [a for v in by_cat.values() for a in v]
    print(f"[demo] {len(demo)} APIs across {len(by_cat)} categories")

    dense = DenseRetriever(demo, device="cpu",
                           cache_dir=str(ROOT / ".embeddings"))
    # ensure embeddings for demo subset
    apis = []
    for a, emb in zip(demo, dense.corpus_emb):
        apis.append({
            "uid": a.uid,
            "tool": a.tool_name,
            "api": a.api_name,
            "category": a.category,
            "description": a.description[:300],
            "schema": a.schema_signature,
            "endpoint": a.endpoint_pattern,
            "emb": [round(float(x), 5) for x in emb],
        })
    OUT.write_text(json.dumps({
        "apis": apis,
        "embedding_dim": int(dense.corpus_emb.shape[1]),
        "encoder": dense.model_name,
    }))
    print(f"[demo] wrote {OUT} ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
