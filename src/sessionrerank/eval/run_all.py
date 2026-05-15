"""Full benchmark: BM25 / Dense / ToolRerank / SessionRerank
on G1/G2/G3 closed + open + session regimes.

Writes results to results/metrics.json (machine-readable) and prints a table.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..data.loader import (
    load_eval_split, build_corpus, iter_train_sessions, BENCHMARK_FILES,
)
from ..baselines.bm25 import BM25Retriever
from ..baselines.dense import DenseRetriever
from ..baselines.toolrerank import ToolRerank
from ..model.session_rerank import SessionRerank, fit_weights
from .runner import eval_closed, eval_open, eval_session, paired_significance, KS

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Dense encoder. On Colab T4 swap to "
                         "'ToolBench/ToolBench_IR_bert_based_uncased'.")
    ap.add_argument("--fit-sessions", type=int, default=4000,
                    help="# multi-turn training tuples to fit SessionRerank weights.")
    ap.add_argument("--skip-open", action="store_true",
                    help="Skip open-corpus eval (faster).")
    # ---- Gemma 4 extensions (opt-in; require GPU) ------------------
    ap.add_argument("--gemma-embed", default=None,
                    help="HF id of a Gemma causal LM to use as a bi-encoder "
                         "(e.g. 'google/gemma-4-270m'). Adds a row to the table.")
    ap.add_argument("--gemma-rerank", default=None,
                    help="HF id of a Gemma instruct LM to use as a prompted "
                         "reranker (e.g. 'google/gemma-4-270m-it'). "
                         "Reranks top-100 from dense.")
    ap.add_argument("--gemma-finetuned", default=None,
                    help="Path to merged LoRA-finetuned Gemma model. "
                         "Used as the bi-encoder for SessionRerank-Gemma.")
    ap.add_argument("--rerank-topk", type=int, default=100,
                    help="Candidate pool size for Gemma cross-encoder rerank.")
    ap.add_argument("--gemma-rerank-device", default=None,
                    help="Device for Gemma reranker (e.g. 'cuda:1'). "
                         "Defaults to --device.")
    ap.add_argument("--gemma-4bit", action="store_true",
                    help="Load Gemma reranker / finetuned base in 4-bit (NF4) "
                         "to fit on a single 16GB GPU.")
    ap.add_argument("--free-embed-after-encode", action="store_true", default=True,
                    help="Move Gemma-Embed off GPU after corpus encoding to free memory.")
    ap.add_argument("--out", default=str(RESULTS / "metrics.json"))
    args = ap.parse_args()

    print(f"[run_all] device={args.device}  encoder={args.encoder}")
    corpus = build_corpus()
    print(f"[run_all] corpus={len(corpus)} APIs")

    cache_dir = ROOT / ".embeddings"
    cache_dir.mkdir(exist_ok=True)

    t0 = time.time()
    bm25  = BM25Retriever(corpus)
    dense = DenseRetriever(corpus, model_name=args.encoder, device=args.device,
                           cache_dir=str(cache_dir))
    trr   = ToolRerank(dense, beta=0.6)
    print(f"[run_all] retrievers ready in {time.time()-t0:.1f}s")

    # ---------------------------------------------------------------
    # Fit SessionRerank weights from training trajectories
    # ---------------------------------------------------------------
    print("[run_all] fitting SessionRerank weights …")
    sessions_iter = iter_train_sessions(max_sessions=20000)
    weights = fit_weights(
        base_retriever=dense,
        training_sessions=sessions_iter,
        corpus=corpus,
        max_sessions=args.fit_sessions,
        epochs=3,
        lr=0.2,
    )
    print("[run_all] learned:", weights)
    srr = SessionRerank(dense, **weights)

    methods = [bm25, dense, trr, srr]

    # ---------------------------------------------------------------
    # Optional Gemma 4 methods (require GPU / Colab)
    # ---------------------------------------------------------------
    gemma_embed = None
    gemma_rerank = None
    sr_gemma = None
    weights_gemma = None
    if args.gemma_embed:
        from ..baselines.gemma_embed import GemmaEmbedder
        print(f"[run_all] loading Gemma-Embed: {args.gemma_embed}")
        gemma_embed = GemmaEmbedder(corpus, model_name=args.gemma_embed,
                                     device=args.device, cache_dir=str(cache_dir))
        methods.append(gemma_embed)
        # Free corpus-encoder GPU memory; query encode will lazily reload.
        if args.free_embed_after_encode:
            print("[run_all] freeing Gemma-Embed GPU memory after corpus encode")
            try:
                gemma_embed.free_gpu()
            except Exception as e:
                print(f"[run_all] free_gpu failed: {e}")
    if args.gemma_rerank:
        from ..baselines.gemma_rerank import GemmaPromptedReranker
        rerank_device = args.gemma_rerank_device or args.device
        print(f"[run_all] loading GemmaRerank on {rerank_device}: {args.gemma_rerank}")
        import torch as _t
        if _t.cuda.is_available():
            _t.cuda.empty_cache()
        gemma_rerank = GemmaPromptedReranker(
            model_name=args.gemma_rerank,
            device=rerank_device,
            load_in_4bit=args.gemma_4bit,
            batch_size=4,
        )
        # Wrap as a first-stage cascade: dense top-K -> Gemma rerank
        class _GemmaCascade:
            def __init__(self, first, second, topk):
                self.first = first; self.second = second; self.topk = topk
                self.name = f"GemmaRerank cascade (top-{topk})"
            def score(self, query, candidates=None):
                # Used by closed-eval only; for open/session we rely on the runner
                # which calls score on a candidate list.
                return self.first.score(query, candidates)
            def rerank(self, query, candidates):
                # Take top-K by dense, then rescored by Gemma cross-encoder.
                s1 = self.first.score(query, candidates)
                top_idx = np.argsort(-s1)[: self.topk]
                top = [candidates[i] for i in top_idx]
                s2 = self.second.score(query, top)
                # Final scores: Gemma scores for top-K, -inf for the rest
                full = np.full(len(candidates), -1e9, dtype=np.float32)
                for j, i in enumerate(top_idx):
                    full[i] = s2[j]
                return full
        cascade = _GemmaCascade(dense, gemma_rerank, args.rerank_topk)
        # Patch score() to apply rerank when called with candidate list
        cascade.score = cascade.rerank
        methods.append(cascade)
    if args.gemma_finetuned:
        from ..baselines.gemma_embed import GemmaEmbedder
        # Free reranker GPU memory before loading another encoder
        if gemma_rerank is not None:
            try:
                gemma_rerank.model.to("cpu")
            except Exception:
                pass
        import torch as _t
        if _t.cuda.is_available():
            _t.cuda.empty_cache()
        print(f"[run_all] loading SessionRerank-Gemma base: {args.gemma_finetuned}")
        ft_base = GemmaEmbedder(corpus, model_name=args.gemma_finetuned,
                                 device=args.device, cache_dir=str(cache_dir))
        # Refit session weights against the new base
        print("[run_all] refitting weights on top of fine-tuned Gemma …")
        weights_g = fit_weights(
            base_retriever=ft_base,
            training_sessions=iter_train_sessions(max_sessions=20000),
            corpus=corpus, max_sessions=args.fit_sessions,
            epochs=3, lr=0.2,
        )
        sr_gemma = SessionRerank(ft_base, **weights_g)
        sr_gemma.name = "SessionRerank-Gemma (ours)"
        weights_gemma = weights_g
        methods.append(sr_gemma)

    results: dict = {
        "config": {
            "encoder": args.encoder,
            "device": args.device,
            "corpus_size": len(corpus),
            "weights": weights,
            "ks": list(KS),
            "gemma_embed": args.gemma_embed,
            "gemma_rerank": args.gemma_rerank,
            "gemma_finetuned": args.gemma_finetuned,
            "weights_gemma": weights_gemma,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "splits": {},
    }

    for split_name in BENCHMARK_FILES:
        print(f"\n=== {split_name} ===")
        qs = load_eval_split(split_name)
        split_block = {"n_queries": len(qs), "regimes": {}}

        # Closed
        closed = {}
        for m in methods:
            t1 = time.time()
            r = eval_closed(m, qs)
            print(f"  closed   | {m.name:30s} | NDCG@5={r['NDCG@5']:.3f}  "
                  f"Hit@5={r['Hit@5']:.3f}   ({time.time()-t1:.1f}s)")
            closed[m.name] = r
        split_block["regimes"]["closed"] = closed

        # Open
        if not args.skip_open:
            open_r = {}
            for m in methods:
                t1 = time.time()
                r = eval_open(m, qs, corpus)
                print(f"  open     | {m.name:30s} | NDCG@5={r['NDCG@5']:.3f}  "
                      f"Hit@5={r['Hit@5']:.3f}   ({time.time()-t1:.1f}s)")
                open_r[m.name] = r
            split_block["regimes"]["open"] = open_r

        # Session-aware (multi-turn, on open corpus)
        sess = {}
        for m in methods:
            t1 = time.time()
            r = eval_session(m, qs, corpus, use_session=isinstance(m, SessionRerank),
                             regime="open" if not args.skip_open else "closed")
            print(f"  session  | {m.name:30s} | NDCG@5={r['NDCG@5']:.3f}  "
                  f"Hit@5={r['Hit@5']:.3f}   ({time.time()-t1:.1f}s)")
            sess[m.name] = r
        split_block["regimes"]["session"] = sess

        results["splits"][split_name] = split_block

    # ---------------------------------------------------------------
    # Statistical significance: SessionRerank vs Dense (open + session)
    # ---------------------------------------------------------------
    sig = {}
    for split_name, blk in results["splits"].items():
        sig[split_name] = {}
        for regime in ("open", "session"):
            if regime not in blk["regimes"]:
                continue
            a = blk["regimes"][regime].get("Dense")
            # find Dense name dynamically
            dense_name = next((k for k in blk["regimes"][regime] if k.startswith("Dense")), None)
            sr_name = next((k for k in blk["regimes"][regime] if k.startswith("SessionRerank")), None)
            if not dense_name or not sr_name:
                continue
            per_a = blk["regimes"][regime][dense_name]["_per_query"]
            per_b = blk["regimes"][regime][sr_name]["_per_query"]
            sig[split_name][regime] = {
                m: paired_significance(per_a, per_b, m)
                for m in ("NDCG@5", "Hit@5", "NDCG@10", "Hit@10")
            }
    results["significance__SessionRerank_vs_Dense"] = sig

    # strip per-query before dump
    def _strip(o):
        if isinstance(o, dict):
            return {k: _strip(v) for k, v in o.items() if k != "_per_query"}
        return o
    out = _strip(results)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[run_all] wrote {args.out}")
    print(f"[run_all] total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
