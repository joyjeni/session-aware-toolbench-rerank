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

    results: dict = {
        "config": {
            "encoder": args.encoder,
            "device": args.device,
            "corpus_size": len(corpus),
            "weights": weights,
            "ks": list(KS),
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
