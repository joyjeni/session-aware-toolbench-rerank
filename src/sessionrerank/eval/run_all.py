"""Full benchmark: BM25 / Dense / ToolRerank / CrossEncoder / SessionRerank
on G1/G2/G3 closed + open + session regimes.

Now supports multi-seed runs for variance estimation and a co-occurrence
prior + cross-encoder baseline as part of the Stage 1 method list.

Writes results to results/metrics.json (machine-readable) and prints a table.

Resume / incremental checkpointing
----------------------------------
After every (seed, split, regime, method) evaluation we flush the live
``per_seed_results`` list to a sibling ``<out>.partial.json`` file. If that
file exists when ``run_all`` starts and ``--resume`` is on (the default),
already-computed entries are reused and skipped. This lets a Kaggle Stage 4
that OOM'd inside the Gemma reranker resume without recomputing the BM25 /
Dense / ToolRerank / CrossEncoder / Gemma-Embed baselines that already
finished. ``--force-rerun`` (or ``--no-resume``) disables this behaviour.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from ..data.loader import (
    load_eval_split, build_corpus, iter_train_sessions, BENCHMARK_FILES,
)
from ..baselines.bm25 import BM25Retriever
from ..baselines.dense import DenseRetriever
from ..baselines.toolrerank import ToolRerank
from ..device_utils import print_runtime_summary, resolve_device
from ..model.session_rerank import SessionRerank, fit_weights, CooccurrencePrior
from ..model.pheromone import PheromoneCoocPrior
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


# --------------------------------------------------------------------------- #
# Partial / resume checkpointing
# --------------------------------------------------------------------------- #
def _partial_path(out_path: str | os.PathLike) -> Path:
    out_path = Path(out_path)
    return out_path.with_suffix(out_path.suffix + ".partial") if out_path.suffix \
        else Path(str(out_path) + ".partial.json")


def _load_partial(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"[run_all] could not parse partial checkpoint {path}: {e}; "
              "starting fresh")
        return None


def _save_partial(path: Path, per_seed_results: list[dict],
                  meta: dict | None = None) -> None:
    payload = {
        "_schema": "run_all-partial-v1",
        "_meta": meta or {},
        "seeds": per_seed_results,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _is_oom(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in ("OutOfMemoryError",):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg


def _free_cuda(tag: str = "") -> None:
    try:
        import torch as _t
        if _t.cuda.is_available():
            _t.cuda.empty_cache()
            _t.cuda.ipc_collect()
            if tag:
                print(f"[run_all] cleared CUDA cache ({tag})")
    except Exception:
        pass


class _GemmaCascade:
    """Cascade wrapper: first_stage top-K → GemmaPromptedReranker rescore.

    Mirrors CrossEncoderReranker — exposes `.name`, `.corpus`, `.score()` so it
    plugs into the same eval loop without any special-casing.
    """
    def __init__(self, first_stage, inner, topk: int = 100, label: str | None = None):
        self.first = first_stage
        self.inner = inner
        self.topk = topk
        short = inner.model_name.split("/")[-1] if not label else label
        self.name = label or f"GemmaRerank ({short}) top-{topk}"
        self._cache: dict = {}

    @property
    def corpus(self):
        return getattr(self.first, "corpus", None)

    def score(self, query, candidates=None):
        if candidates is None:
            candidates = self.first.corpus
        n = len(candidates)
        s1 = self.first.score(query, candidates)
        if n <= self.topk:
            top_idx = np.arange(n)
        else:
            top_idx = np.argsort(-s1)[: self.topk]
        cand_top = [candidates[i] for i in top_idx]
        key = (query, tuple(c.uid for c in cand_top))
        if key in self._cache:
            s2 = self._cache[key]
        else:
            s2 = self.inner.score(query, cand_top)
            self._cache[key] = np.asarray(s2, dtype=np.float32)
        if n <= self.topk:
            return s2.astype(np.float32)
        out = np.full(n, -1e9, dtype=np.float32)
        for j, i in enumerate(top_idx):
            out[i] = s2[j]
        return out


def _is_gemma_rerank(m) -> bool:
    """Return True for Gemma reranker / fine-tuned cascade wrappers."""
    return isinstance(m, _GemmaCascade) or m.__class__.__name__ == "_GemmaCascade"


def _eval_one_seed(seed: int, args, corpus, bm25, dense, trr, cross_enc,
                   cooc_prior, gemma_embed=None, gemma_rerank=None,
                   gemma_finetuned=None,
                   existing_seed_result: dict | None = None,
                   save_cb=None) -> dict:
    """Run a single-seed Stage 1: refit SessionRerank weights + eval all methods.

    If ``existing_seed_result`` is supplied (from a partial checkpoint), each
    (split, regime, method) entry already present is skipped. After every
    method we invoke ``save_cb()`` so the partial file on disk stays current.
    """
    _set_global_seed(seed)
    print(f"\n[run_all] ====== seed={seed} ======")

    # Refit SessionRerank weights for this seed. We always refit (cheap
    # relative to Gemma reranker eval) rather than reuse from partial: the
    # partial JSON doesn't store the _feat_stats payload needed for
    # SessionRerank z-scoring, so refitting keeps SessionRerank scores
    # bit-identical across resume/fresh runs.
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
    if gemma_embed is not None:
        methods.append(gemma_embed)
    if gemma_rerank is not None:
        methods.append(gemma_rerank)
    if gemma_finetuned is not None:
        methods.append(gemma_finetuned)
    methods.append(srr)

    seed_result = existing_seed_result or {"config": {"seed": seed, "weights": weights},
                                            "splits": {}}
    # Always make sure config.weights reflects current weights (resume case
    # may have stripped feat_stats above).
    seed_result.setdefault("config", {})
    seed_result["config"]["seed"] = seed
    seed_result["config"]["weights"] = weights
    seed_result.setdefault("splits", {})

    def _flush():
        if save_cb is not None:
            try:
                save_cb()
            except Exception as e:
                print(f"[run_all] partial flush failed: {e}")

    def _eval_method(m, regime, qs, *, allow_oom: bool):
        t1 = time.time()
        try:
            if regime == "closed":
                r = eval_closed(m, qs)
            elif regime == "open":
                r = eval_open(m, qs, corpus)
            else:  # session
                r = eval_session(m, qs, corpus,
                                 use_session=isinstance(m, SessionRerank),
                                 regime="open" if not args.skip_open else "closed")
            print(f"    {regime:8s} | {m.name:48s} | NDCG@5={r['NDCG@5']:.3f}  "
                  f"Hit@5={r['Hit@5']:.3f}   ({time.time()-t1:.1f}s)")
            return r
        except BaseException as e:  # noqa: BLE001 - catch CUDA OOM too
            if allow_oom and _is_oom(e):
                print(f"    {regime:8s} | {m.name:48s} | CUDA OOM ({e}); "
                      "recording error marker and continuing")
                _free_cuda(tag=f"after OOM {m.name}/{regime}")
                return {
                    "_error": "cuda_oom",
                    "_error_message": str(e)[:500],
                    "_skipped": True,
                }
            raise

    for split_name in BENCHMARK_FILES:
        print(f"  --- {split_name} ---")
        qs = load_eval_split(split_name)
        split_block = seed_result["splits"].setdefault(
            split_name, {"n_queries": len(qs), "regimes": {}})
        split_block.setdefault("n_queries", len(qs))
        split_block.setdefault("regimes", {})

        regimes_to_run = ["closed"]
        if not args.skip_open:
            regimes_to_run.append("open")
        regimes_to_run.append("session")

        for regime in regimes_to_run:
            block = split_block["regimes"].setdefault(regime, {})
            for m in methods:
                # Skip if already in partial and not marked as an error to retry
                existing = block.get(m.name)
                if isinstance(existing, dict) and "NDCG@5" in existing:
                    print(f"    {regime:8s} | {m.name:48s} | "
                          f"NDCG@5={existing['NDCG@5']:.3f}  Hit@5={existing['Hit@5']:.3f}"
                          "   [resume: reused]")
                    continue
                # If existing is an error marker, only retry if not skipping gemma
                if isinstance(existing, dict) and existing.get("_error"):
                    if args.skip_gemma_rerank and _is_gemma_rerank(m):
                        print(f"    {regime:8s} | {m.name:48s} | "
                              "[resume: previously errored, --skip-gemma-rerank set]")
                        continue
                    print(f"    {regime:8s} | {m.name:48s} | "
                          "[resume: retrying after prior error]")
                if args.skip_gemma_rerank and _is_gemma_rerank(m):
                    print(f"    {regime:8s} | {m.name:48s} | "
                          "[--skip-gemma-rerank: marked skipped]")
                    block[m.name] = {"_error": "skipped",
                                      "_error_message": "--skip-gemma-rerank",
                                      "_skipped": True}
                    _flush()
                    continue
                r = _eval_method(m, regime, qs,
                                 allow_oom=_is_gemma_rerank(m))
                block[m.name] = r
                _flush()

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
                # Preserve error markers (CUDA OOM, --skip-gemma-rerank) as-is.
                if isinstance(metrics, dict) and metrics.get("_error"):
                    agg["splits"][split_name]["regimes"][regime][method_name] = dict(metrics)
                    continue
                agg_metrics: dict = {}
                for metric_name, _ in metrics.items():
                    if metric_name == "_per_query" or metric_name.startswith("_"):
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
    # ---- Pheromone (ant-colony) prior ----------------------------------
    ap.add_argument("--pheromone", action="store_true",
                    help="Use ACO-style pheromone prior instead of static cooc "
                         "(time-varying τ-deposit + ρ-decay). Nature-inspired.")
    ap.add_argument("--pheromone-rho", type=float, default=0.02,
                    help="Cross-session evaporation rate ρ ∈ [0, 1].")
    ap.add_argument("--pheromone-tau0", type=float, default=1.0,
                    help="Pheromone deposit τ₀ per successful transition.")
    ap.add_argument("--pheromone-decay", type=float, default=0.7,
                    help="In-session decay γ used at score time.")
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
    ap.add_argument("--gemma-rerank-topk", type=int, default=None,
                    help="Top-K candidates fed to Gemma reranker. Overrides "
                         "--rerank-topk for the zero-shot reranker (and the "
                         "fine-tuned cascade if --gemma-finetuned-topk is unset). "
                         "Use a smaller value (e.g. 50 or 32) to fit a T4.")
    ap.add_argument("--gemma-finetuned-topk", type=int, default=None,
                    help="Top-K for the fine-tuned Gemma cascade. Defaults to "
                         "--gemma-rerank-topk if set, else --rerank-topk.")
    ap.add_argument("--gemma-rerank-batch-size", type=int, default=None,
                    help="Batch size used inside GemmaPromptedReranker.score(). "
                         "Lower this (e.g. 2 or 1) to recover from CUDA OOM.")
    ap.add_argument("--skip-gemma-rerank", action="store_true",
                    help="Mark the Gemma reranker / fine-tuned cascade as "
                         "skipped without running them. Useful to finalise "
                         "metrics after an OOM crash.")
    # ---- Resume / incremental checkpointing ----------------------------
    ap.add_argument("--resume", dest="resume", action="store_true", default=True,
                    help="Reuse <out>.partial.json if present (default on).")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="Ignore any existing partial checkpoint.")
    ap.add_argument("--force-rerun", action="store_true",
                    help="Delete any existing partial checkpoint AND output file "
                         "before running (full rerun).")
    ap.add_argument("--partial-out", default=None,
                    help="Path to the partial checkpoint JSON. Defaults to "
                         "'<--out>.partial'.")
    ap.add_argument("--out", default=str(RESULTS / "metrics.json"))
    args = ap.parse_args()

    # Resolve Gemma reranker topk separately from cross-encoder.
    if args.gemma_rerank_topk is None:
        args.gemma_rerank_topk = args.rerank_topk
    if args.gemma_finetuned_topk is None:
        args.gemma_finetuned_topk = args.gemma_rerank_topk

    # Resolve seeds: --seeds supersedes --seed
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    elif args.seed is not None:
        seeds = [int(args.seed)]
    else:
        seeds = [0]
    print(f"[run_all] seeds={seeds}")

    # ---- Runtime / device safety ----------------------------------------
    # Print a one-shot summary so the user can confirm Kaggle/Colab picked
    # up the expected accelerator.
    print_runtime_summary("[run_all]")
    # Resolve dense-encoder device first: SentenceTransformer cannot use XLA
    # and must not receive 'cuda' on a TPU image (Torch+XLA wheel) where
    # torch.cuda.is_available() is False.
    dense_device = resolve_device(args.device, for_sentence_transformer=True)
    # General-purpose device for HF reranker stages (Gemma) — may use cuda if
    # available; falls back to cpu silently. We do NOT auto-pick xla here.
    general_device = resolve_device(args.device, for_sentence_transformer=False)
    if dense_device != args.device:
        print(f"[run_all] dense-encoder device adjusted: requested "
              f"{args.device!r} → using {dense_device!r}")
    if general_device != args.device:
        print(f"[run_all] reranker device adjusted: requested "
              f"{args.device!r} → using {general_device!r}")
    args.device = general_device  # keep downstream args consistent

    print(f"[run_all] device={args.device}  dense_device={dense_device}  "
          f"encoder={args.encoder}")
    corpus = build_corpus()
    print(f"[run_all] corpus={len(corpus)} APIs")

    cache_dir = ROOT / ".embeddings"
    cache_dir.mkdir(exist_ok=True)

    t0 = time.time()
    bm25  = BM25Retriever(corpus)
    dense = DenseRetriever(corpus, model_name=args.encoder, device=dense_device,
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
            device=dense_device,
        )

    # Co-occurrence prior (deterministic harvest → built once)
    cooc_prior = None
    if args.cooc:
        t_co = time.time()
        if args.pheromone:
            print(f"[run_all] building PHEROMONE prior "
                  f"({args.cooc_sessions} sessions, window={args.cooc_window}, "
                  f"ρ={args.pheromone_rho}, τ₀={args.pheromone_tau0}, "
                  f"γ={args.pheromone_decay}) …")
            cooc_prior = PheromoneCoocPrior.from_sessions(
                iter_train_sessions(max_sessions=args.cooc_sessions),
                max_sessions=args.cooc_sessions,
                window=args.cooc_window,
                rho=args.pheromone_rho,
                tau0=args.pheromone_tau0,
                in_session_decay=args.pheromone_decay,
            )
            print(f"[run_all] pheromone keys={cooc_prior.n_keys()} "
                  f"density={cooc_prior.density():.2f} "
                  f"mass={cooc_prior.total_mass():.1f} "
                  f"sessions_seen={cooc_prior.n_sessions_seen} "
                  f"({time.time()-t_co:.1f}s)")
        else:
            print(f"[run_all] building co-occurrence prior "
                  f"({args.cooc_sessions} sessions, window={args.cooc_window}) …")
            cooc_prior = CooccurrencePrior.from_sessions(
                iter_train_sessions(max_sessions=args.cooc_sessions),
                max_sessions=args.cooc_sessions,
                window=args.cooc_window,
            )
            print(f"[run_all] cooc keys={cooc_prior.n_keys()} "
                  f"sessions_seen={cooc_prior.n_sessions_seen} "
                  f"({time.time()-t_co:.1f}s)")

    # ---------------------------------------------------------------
    # Gemma 4 extensions (load once, reuse across seeds)
    # ---------------------------------------------------------------
    gemma_embed = None
    if args.gemma_embed:
        from ..baselines.gemma_embed import GemmaEmbedder
        print(f"[run_all] loading Gemma-Embed: {args.gemma_embed}")
        gemma_embed = GemmaEmbedder(
            corpus,
            model_name=args.gemma_embed,
            device=args.device,
            cache_dir=str(cache_dir),
        )
        # Free GPU after corpus encoding so the reranker fits alongside.
        if args.free_embed_after_encode and args.device.startswith("cuda"):
            try:
                gemma_embed.model.to("cpu")
                import torch as _t
                _t.cuda.empty_cache()
                print("[run_all] Gemma-Embed moved to CPU after encoding")
            except Exception as e:
                print(f"[run_all] could not free Gemma-Embed GPU: {e}")

    gemma_rerank = None
    if args.gemma_rerank and not args.skip_gemma_rerank:
        from ..baselines.gemma_rerank import GemmaPromptedReranker
        rdev = args.gemma_rerank_device or args.device
        print(f"[run_all] loading Gemma-Rerank: {args.gemma_rerank} on {rdev} "
              f"(topk={args.gemma_rerank_topk}, batch_size="
              f"{args.gemma_rerank_batch_size or 'default'})")
        rerank_kwargs = dict(
            model_name=args.gemma_rerank,
            device=rdev,
            load_in_4bit=args.gemma_4bit,
        )
        if args.gemma_rerank_batch_size is not None:
            rerank_kwargs["batch_size"] = args.gemma_rerank_batch_size
        inner = GemmaPromptedReranker(**rerank_kwargs)
        gemma_rerank = _GemmaCascade(dense, inner, topk=args.gemma_rerank_topk)
    elif args.gemma_rerank and args.skip_gemma_rerank:
        print("[run_all] --skip-gemma-rerank: NOT loading Gemma reranker")

    gemma_finetuned = None
    if args.gemma_finetuned and not args.skip_gemma_rerank:
        from ..baselines.gemma_rerank import GemmaPromptedReranker
        rdev = args.gemma_rerank_device or args.device
        print(f"[run_all] loading Gemma-Finetuned: {args.gemma_finetuned} on {rdev}")
        ft_kwargs = dict(
            model_name=args.gemma_finetuned,
            device=rdev,
            load_in_4bit=args.gemma_4bit,
        )
        if args.gemma_rerank_batch_size is not None:
            ft_kwargs["batch_size"] = args.gemma_rerank_batch_size
        inner_ft = GemmaPromptedReranker(**ft_kwargs)
        gemma_finetuned = _GemmaCascade(
            dense, inner_ft, topk=args.gemma_finetuned_topk,
            label=f"Gemma 4 finetuned top-{args.gemma_finetuned_topk}",
        )
    elif args.gemma_finetuned and args.skip_gemma_rerank:
        print("[run_all] --skip-gemma-rerank: NOT loading Gemma fine-tuned cascade")

    # ---------------------------------------------------------------
    # Resume from partial checkpoint
    # ---------------------------------------------------------------
    out_path = Path(args.out)
    partial_path = Path(args.partial_out) if args.partial_out else _partial_path(out_path)
    if args.force_rerun:
        for p in (out_path, partial_path):
            if p.exists():
                try:
                    p.unlink()
                    print(f"[run_all] --force-rerun: removed {p}")
                except Exception as e:
                    print(f"[run_all] could not remove {p}: {e}")

    partial_by_seed: dict[int, dict] = {}
    if args.resume and not args.force_rerun:
        existing = _load_partial(partial_path)
        if existing and isinstance(existing.get("seeds"), list):
            for sr in existing["seeds"]:
                try:
                    s = int(sr["config"]["seed"])
                    partial_by_seed[s] = sr
                except (KeyError, TypeError, ValueError):
                    continue
            if partial_by_seed:
                print(f"[run_all] resuming from {partial_path} "
                      f"(seeds={sorted(partial_by_seed)})")

    # ---------------------------------------------------------------
    # Run per-seed evaluations
    # ---------------------------------------------------------------
    per_seed_results: list[dict] = []
    partial_meta = {
        "out": str(out_path),
        "seeds": seeds,
        "encoder": args.encoder,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    def _save_now():
        _save_partial(partial_path, per_seed_results, meta=partial_meta)

    for seed in seeds:
        existing_sr = partial_by_seed.get(int(seed))
        # Reserve this seed's slot in per_seed_results BEFORE calling
        # _eval_one_seed so save_cb sees the in-progress result. If we have
        # no prior partial for this seed, we still pre-allocate an empty
        # stub dict that _eval_one_seed will mutate in place.
        if existing_sr is None:
            stub = {"config": {"seed": seed}, "splits": {}}
            per_seed_results.append(stub)
            existing_sr = stub
        else:
            per_seed_results.append(existing_sr)
        _eval_one_seed(seed, args, corpus, bm25, dense, trr,
                       cross_enc, cooc_prior,
                       gemma_embed=gemma_embed,
                       gemma_rerank=gemma_rerank,
                       gemma_finetuned=gemma_finetuned,
                       existing_seed_result=existing_sr,
                       save_cb=_save_now)
        _save_now()

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
    # Final write succeeded — keep the partial file alongside as an audit
    # trail (it includes _per_query) but also rewrite it with the latest
    # state so a follow-up resume is a no-op.
    try:
        _save_partial(partial_path, per_seed_results, meta=partial_meta)
    except Exception:
        pass
    print(f"[run_all] total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
