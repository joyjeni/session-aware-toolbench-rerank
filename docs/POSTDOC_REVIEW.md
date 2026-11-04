# Postdoctoral review — O1 — SATR (session-aware tool retrieval)

Reviewer stance: a postdoctoral researcher preparing this chapter for a Q1 journal (November 2026).
Severity: **Critical** blocks publication · **Major** a reviewer will require it · **Fixed** done in this commit.

| Severity | Finding | Evidence | Action |
|---|---|---|---|
| Critical | Session construction favours the method by design | `build_synthetic_sessions` groups queries that share an API category, and the largest learned weight is category affinity (w_cat = 1.17). The headline +14.7 % NDCG@5 therefore partly measures how sessions were built. | Added `--session-grouping random` control; report both. Build sessions from genuinely multi-turn data (BFCL v3 multi-turn, τ-bench, ToolSandbox, APIGen-MT). |
| Critical | Oracle history | History at turn t was the ground-truth APIs of turns 1..t−1, which a deployed system never sees. | Default is now `--session-history predicted` (the model's own top-2 picks); `oracle` remains as an upper bound. |
| Major | Leakage between session-weight fitting and evaluation | Weights are fitted on 3,501 training tuples; check that no evaluated query's API set appears in the fitting shard (94/100 did in the ACRS master benchmark). | Add a group-level split by tool and a leakage audit like `phd-agentic-ai-master/experiments/leakage_audit.py`. |
| Major | Dense baseline is MiniLM on CPU | ToolLLM's released retriever (ToolBench_IR_bert_based_uncased) is only run in the Colab notebook. | Report the paper-faithful retriever and a modern embedding model as primary baselines; add ToolRet. |
| Fixed | `run_all.py` did not parse | Two `--gemma`/`--ce` alias lines were mis-indented, one inside another call — the evaluation entry point raised IndentationError on master. | Fixed; all Python files now compile. |

## What this chapter needs to be journal-ready

1. Real LLM agents (open-weight 7–8B class) instead of simulated ones, with ≥ 300 evaluation items per arm.
2. At least two recognised public benchmarks (see the dataset plan in `phd-agentic-ai-master/docs/POSTDOC_ROADMAP.md`).
3. Pre-registered primary metric and one headline statistical test; all other comparisons labelled exploratory.
4. Leak-free splits and a reproducibility command that regenerates every table.
