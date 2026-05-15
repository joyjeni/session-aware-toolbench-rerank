"""Full benchmark: BM25 / Dense / ToolRerank / CrossEncoder / SessionRerank
on G1/G2/G3 closed + open + session regimes.

Now supports multi-seed runs for variance estimation and a co-occurrence
prior + cross-encoder baseline as part of the Stage 1 method list.

Writes results to results/metrics.json (machine-readable) and prints a table.
"""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from ..data.loader import (
    load_eval_split, build_corpus, iter_train_sessions, BENCHMARK_FILES,
)
from ..baselines.bm25 import BM25Retriever
from ..baselines.dense import DenseRetriever
from ..baselines.toolrerank import ToolRerank
from ..model.session_rerank import SessionRerank, fit_weights, CooccurrencePrior
from .runner import eval_closed, eval_open, eval_session, paired_significance, KS

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def _set_global_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch as _t
        _t.manual_seed(seed)
        if _t.cuda.is_available():
            _t.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _strip_per_query(o):
    if isinstance(o, dict):
        return {k: _strip_per_query(v) for k, v in o.items() if k != "_per_query"}
    return o


def _eval_one_seed(seed: int, args, corpus, bm25, dense, trr, cross_enc,
                   cooc_prior) -> dict:
    """Run a single-seed Stage 1: refit SessionRerank weights + eval all methods."""
    _set_global_seed(seed)
    print(f"\n[run_all] ====== seed={seed} ======")

    # Refit SessionRerank weights for this seed
    sessions_iter = iter_train_sessions(max_sessions=20000)
    weights = fit_weights(
        base_retriever=dense,
        training_sessions=sessions_iter,
        corpus=corpus,
        max_sessions=args.fit_sessions,
        seed=seed,
        cooc_prior=cooc_prior,
    )
    feat_stats = weights.pop("_feat_stats", None)
    print(f"[seed={seed}] learned:", weights)
    srr = SessionRerank(dense, **weights, cooc_prior=cooc_prior,
                        feat_stats=feat_stats)

    methods = [bm25, dense, trr]
    if cross_enc is not None:
        methods.append(cross_enc)
    methods.append(srr)

    seed_result = {"config": {"seed": seed, "weights": weights}, "splits": {}}
    for split_name in BENCHMARK_FILES:
        print(f"  --- {split_name} ---")
        qs = load_eval_split(split_name)
        split_block = {"n_queries": len(qs), "regimes": {}}

        closed = {}
        for m in methods:
            t1 = time.time()
            r = eval_closed(m, qs)
            print(f"    closed   | {m.name:48s} | NDCG@5={r['NDCG@5']:.3f}  "
                  f"Hit@5={r['Hit@5']:.3f}   ({time.time()-t1:.1f}s)")
            closed[m.name] = r
        split_block["regimes"]["closed"] = closed

        if not args.skip_open:
            open_r = {}
            for m in methods:
                t1 = time.time()
                r = eval_open(m, qs, corpus)
                print(f"    open     | {m.name:48s} | NDCG@5={r['NDCG@5']:.3f}  "
                      f"Hit@5={r['Hit@5']:.3f}   ({time.time()-t1:.1f}s)")
                open_r[m.name] = r
            split_block["regimes"]["open"] = open_r

        sess = {}
        for m in methods:
            t1 = time.time()
            r = eval_session(m, qs, corpus,
                             use_session=isinstance(m, SessionRerank),
                             regime="open" if not args.skip_open else "closed")
            print(f"    session  | {m.name:48s} | NDCG@5={r['NDCG@5']:.3f}  "
                  f"Hit@5={r['Hit@5']:.3f}   ({time.time()-t1:.1f}s)")
            sess[m.name] = r
        split_block["regimes"]["session"] = sess

        seed_result["splits"][split_name] = split_block
    return seed_result


def _aggregate_seeds(per_seed_results: list[dict]) -> dict:
    """Aggregate per-seed metric numbers into mean ± std per
    (split, regime, method, metric)."""
    agg: dict = {"splits": {}}
    if not per_seed_results:
        return agg
    sample = per_seed_results[0]
    for split_name, blk in sample["splits"].items():
        agg["splits"][split_name] = {"n_queries": blk["n_queries"], "regimes": {}}
        for regime, mblock in blk["regimes"].items():
            agg["splits"][split_name]["regimes"][regime] = {}
            for method_name, metrics in mblock.items():
                agg_metrics: dict = {}
                for metric_name, _ in metrics.items():
                    if metric_name == "_per_query":
                        continue
                    vals: list[float] = []
                    for sr in per_seed_results:
                        try:
                            v = sr["splits"][split_name]["regimes"][regime][method_name][metric_name]
                            vals.append(float(v))
                        except (KeyError, TypeError):
                            pass
                    if not vals:
                        continue
                    mean = float(np.mean(vals))
                    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    agg_metrics[metric_name] = {
                        "mean": mean,
                        "std": std,
                        "values": vals,
                    }
                agg["splits"][split_name]["regimes"][regime][method_name] = agg_metrics
    return agg


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
    # ---- Multi-seed variance analysis -----------------------------------
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed for reproducible session-weight fitting.")
    ap.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated list of seeds (e.g. '0,1,2,42,123'). "
                         "If provided, supersedes --seed and reports mean±std.")
    # ---- New Stage 1 contributions (A + B) ------------------------------
    ap.add_argument("--cooc", action="store_true", default=True,
                    help="Enable co-occurrence prior P(api_b|api_a) (default on).")
    ap.add_argument("--no-cooc", dest="cooc", action="store_false",
                    help="Disable co-occurrence prior.")
    ap.add_argument("--cooc-sessions", type=int, default=20000,
                    help="# train sessions to harvest co-occurrence pairs from.")
    ap.add_argument("--cooc-window", type=int, default=3,
                    help="Sliding window size when counting successful pairs.")
    ap.add_argument("--cross-encoder", action="store_true", default=True,
                    help="Add cross-encoder reranker baseline (default on).")
    ap.add_argument("--no-cross-encoder", dest="cross_encoder", action="store_false",
                    help="Skip cross-encoder reranker baseline.")
    ap.add_argument("--cross-encoder-model",
                    default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                    help="HF id for the CrossEncoder rerank model.")
    ap.add_argument("--cross-encoder-topk", type=int, default=100,
                    help="Top-K from dense to feed into cross-encoder rerank.")
    # ---- Gemma 4 extensions (opt-in; require GPU) ----------------------
    ap.add_argument("--gemma-embed", default=None,
                    help="HF id of a Gemma causal LM to use as a bi-encoder.")
    ap.add_argument("--gemma-rerank", default=None,
                    help="HF id of a Gemma instruct LM as prompted reranker.")
    ap.add_argument("--gemma-finetuned", default=None,
                    help="Path to merged LoRA-finetuned Gemma model.")
    ap.add_argument("--rerank-topk", type=int, default=100,
                    help="Candidate pool size for Gemma cross-encoder rerank.")
    ap.add_argument("--gemma-rerank-device", default=None,
                    help="Device for Gemma reranker (e.g. 'cuda:1').")
    ap.add_argument("--gemma-4bit", action="store_true",
                    help="Load Gemma reranker / finetuned base in 4-bit (NF4).")
    ap.add_argument("--free-embed-after-encode", action="store_true", default=True,
                    help="Move Gemma-Embed off GPU after corpus encoding.")
    ap.add_argument("--out", default=str(RESULTS / "metrics.json"))
    args = ap.parse_args()

    # Resolve seeds: --seeds supersedes --seed
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    elif args.seed is not None:
        seeds = [int(args.seed)]
    else:
        seeds = [0]
    print(f"[run_all] seeds={seeds}")

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

    # Cross-encoder baseline (deterministic across seeds → load once)
    cross_enc = None
    if args.cross_encoder:
        from ..baselines.cross_encoder import CrossEncoderReranker
        print(f"[run_all] loading CrossEncoder: {args.cross_encoder_model}")
        cross_enc = CrossEncoderReranker(
            first_stage=dense,
            model_name=args.cross_encoder_model,
            topk=args.cross_encoder_topk,
            device=args.device,
        )

    # Co-occurrence prior (deterministic harvest → built once)
    cooc_prior = None
    if args.cooc:
        print(f"[run_all] building co-occurrence prior "
              f"({args.cooc_sessions} sessions, window={args.cooc_window}) …")
        t_co = time.time()
        cooc_prior = CooccurrencePrior.from_sessions(
            iter_train_sessions(max_sessions=args.cooc_sessions),
            max_sessions=args.cooc_sessions,
            window=args.cooc_window,
        )
        print(f"[run_all] cooc keys={cooc_prior.n_keys()} "
              f"sessions_seen={cooc_prior.n_sessions_seen} "
              f"({time.time()-t_co:.1f}s)")

    # ---------------------------------------------------------------
    # Run per-seed evaluations
    # ---------------------------------------------------------------
    per_seed_results: list[dict] = []
    for seed in seeds:
        sr = _eval_one_seed(seed, args, corpus, bm25, dense, trr,
                            cross_enc, cooc_prior)
        per_seed_results.append(sr)

    # ---------------------------------------------------------------
    # Aggregate (mean ± std across seeds) + write output
    # ---------------------------------------------------------------
    if len(per_seed_results) == 1:
        # Single-seed: keep legacy schema for backwards compatibility
        results = per_seed_results[0]
        sig: dict = {}
        for split_name, blk in results["splits"].items():
            sig[split_name] = {}
            for regime in ("open", "session"):
                if regime not in blk["regimes"]:
                    continue
                dense_name = next((k for k in blk["regimes"][regime]
                                   if k.startswith("Dense")), None)
                sr_name = next((k for k in blk["regimes"][regime]
                                if k.startswith("SessionRerank")), None)
                if not dense_name or not sr_name:
                    continue
                per_a = blk["regimes"][regime][dense_name]["_per_query"]
                per_b = blk["regimes"][regime][sr_name]["_per_query"]
                sig[split_name][regime] = {
                    m: paired_significance(per_a, per_b, m)
                    for m in ("NDCG@5", "Hit@5", "NDCG@10", "Hit@10")
                }
        results["significance__SessionRerank_vs_Dense"] = sig
        # Strip internal feat-stats payload if it sneaks in (it won't anymore
        # because we pop it above, but be defensive for older snapshots).
        existing_w = results.get("config", {}).get("weights")
        if isinstance(existing_w, dict):
            existing_w = {k: v for k, v in existing_w.items() if not k.startswith("_")}
        results["config"] = {
            "encoder": args.encoder,
            "device": args.device,
            "corpus_size": len(corpus),
            "seeds": seeds,
            "weights": existing_w,
            "ks": list(KS),
            "cooc_enabled": bool(args.cooc),
            "cooc_window": args.cooc_window if args.cooc else None,
            "cross_encoder": args.cross_encoder_model if args.cross_encoder else None,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        out = _strip_per_query(results)
    else:
        # Multi-seed: emit mean±std aggregate; keep per-seed weights for audit
        aggregate = _aggregate_seeds(per_seed_results)
        out = {
            "config": {
                "encoder": args.encoder,
                "device": args.device,
                "corpus_size": len(corpus),
                "seeds": seeds,
                "ks": list(KS),
                "cooc_enabled": bool(args.cooc),
                "cooc_window": args.cooc_window if args.cooc else None,
                "cross_encoder": args.cross_encoder_model if args.cross_encoder else None,
                "weights_per_seed": [
                    {"seed": s,
                     "weights": {k: v for k, v in r["config"]["weights"].items()
                                 if not k.startswith("_")}}
                    for s, r in zip(seeds, per_seed_results)
                ],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            "splits": aggregate["splits"],
        }
        # Paired significance — run on the first seed only (representative)
        sig: dict = {}
        first = per_seed_results[0]
        for split_name, blk in first["splits"].items():
            sig[split_name] = {}
            for regime in ("open", "session"):
                if regime not in blk["regimes"]:
                    continue
                dense_name = next((k for k in blk["regimes"][regime]
                                   if k.startswith("Dense")), None)
                sr_name = next((k for k in blk["regimes"][regime]
                                if k.startswith("SessionRerank")), None)
                if not dense_name or not sr_name:
                    continue
                per_a = blk["regimes"][regime][dense_name]["_per_query"]
                per_b = blk["regimes"][regime][sr_name]["_per_query"]
                sig[split_name][regime] = {
                    m: paired_significance(per_a, per_b, m)
                    for m in ("NDCG@5", "Hit@5", "NDCG@10", "Hit@10")
                }
        out["significance__SessionRerank_vs_Dense__seed0"] = sig

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[run_all] wrote {args.out}")
    print(f"[run_all] total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
