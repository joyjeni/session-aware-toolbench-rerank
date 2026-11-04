"""ToolBench loaders.

Two views on the data:

1. **Benchmark eval splits**  (`benchmark/g{1,2,3}_{instruction,tool,category}.parquet`)
   Each row has:
     - query_id: int
     - query:    str
     - api_list: List[dict]  candidate APIs (already filtered to the relevant pool)
     - relevant_apis: List[[tool_name, api_name]]  ground truth
   Used to compute Hit@K and NDCG@K against a fixed candidate pool plus the full
   corpus (open retrieval) — exactly the ToolLLM retrieval evaluation.

2. **Multi-turn training conversations**  (`data/train-*.parquet`)
   Each row has 'conversations' = {from: [...], value: [...]} alternating
   system/user/assistant/function. We extract per-session:
     - the initial user query
     - the ordered list of (function name, function args, function response) tuples
       that were actually invoked (`from == 'function'`)
   These tuples are our "successful tool-call outcomes" — the emergent session
   signal that drives SessionRerank.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

ROOT = Path(__file__).resolve().parents[3] / "data" / "toolbench_hf"
BENCH = ROOT / "benchmark"
TRAIN = ROOT / "data"

BENCHMARK_FILES = {
    "G1_instruction": "g1_instruction-00000-of-00001.parquet",
    "G1_tool":        "g1_tool-00000-of-00001.parquet",
    "G1_category":    "g1_category-00000-of-00001.parquet",
    "G2_instruction": "g2_instruction-00000-of-00001.parquet",
    "G2_category":    "g2_category-00000-of-00001.parquet",
    "G3_instruction": "g3_instruction-00000-of-00001.parquet",
}


# --------------------------------------------------------------------------- #
# APIs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class API:
    tool_name: str
    api_name: str
    category: str
    description: str
    method: str = "GET"
    required_params: tuple = ()
    optional_params: tuple = ()

    @property
    def uid(self) -> str:
        return f"{self.tool_name}::{self.api_name}"

    def doc(self) -> str:
        """Text representation for embedding / BM25."""
        req = ", ".join(p.get("name", "") for p in self.required_params)
        opt = ", ".join(p.get("name", "") for p in self.optional_params)
        return (
            f"{self.category} | {self.tool_name} | {self.api_name} :: "
            f"{self.description} :: required[{req}] optional[{opt}]"
        ).strip()

    @property
    def schema_signature(self) -> str:
        """Sorted required-parameter names — used for schema-affinity bias."""
        names = sorted(p.get("name", "") for p in self.required_params if p.get("name"))
        return "|".join(names)

    @property
    def endpoint_pattern(self) -> str:
        """Snake-case-style API-name pattern (lower, alnum tokens)."""
        toks = re.findall(r"[a-z0-9]+", self.api_name.lower())
        return "_".join(toks) if toks else self.api_name.lower()


def _parse_api(d: dict) -> API:
    return API(
        tool_name=d.get("tool_name", ""),
        api_name=d.get("api_name", ""),
        category=d.get("category_name", ""),
        description=d.get("api_description", "") or "",
        method=d.get("method", "GET"),
        required_params=tuple(d.get("required_parameters") or ()),
        optional_params=tuple(d.get("optional_parameters") or ()),
    )


# --------------------------------------------------------------------------- #
# Benchmark eval rows
# --------------------------------------------------------------------------- #
@dataclass
class EvalQuery:
    split: str
    query_id: int
    query: str
    candidate_apis: list[API]
    relevant_keys: set[tuple[str, str]]  # set of (tool_name, api_name)

    @property
    def candidate_uids(self) -> list[str]:
        return [a.uid for a in self.candidate_apis]

    @property
    def relevant_uids(self) -> set[str]:
        return {f"{t}::{a}" for t, a in self.relevant_keys}


def _coerce_list(x):
    if isinstance(x, list):
        return x
    if hasattr(x, "tolist"):
        return list(x.tolist())
    return list(x) if x is not None else []


def load_eval_split(split: str) -> list[EvalQuery]:
    fname = BENCHMARK_FILES[split]
    df = pd.read_parquet(BENCH / fname)
    out: list[EvalQuery] = []
    for _, row in df.iterrows():
        raw_list = row["api_list"]
        if isinstance(raw_list, str):
            raw_list = json.loads(raw_list)
        else:
            raw_list = _coerce_list(raw_list)
        cands = [_parse_api(d) for d in raw_list]
        rel_raw = row["relevant_apis"]
        if isinstance(rel_raw, str):
            rel_raw = json.loads(rel_raw)
        else:
            rel_raw = _coerce_list(rel_raw)
        rel = set()
        for r in rel_raw:
            r = _coerce_list(r)
            if len(r) >= 2:
                rel.add((str(r[0]), str(r[1])))
        out.append(
            EvalQuery(
                split=split,
                query_id=int(row["query_id"]),
                query=str(row["query"]),
                candidate_apis=cands,
                relevant_keys=rel,
            )
        )
    return out


@lru_cache(maxsize=8)
def load_all_eval() -> dict[str, list[EvalQuery]]:
    return {s: load_eval_split(s) for s in BENCHMARK_FILES}


# --------------------------------------------------------------------------- #
# Global corpus (union of all candidate APIs across all benchmarks)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def build_corpus() -> list[API]:
    seen: dict[str, API] = {}
    for split in BENCHMARK_FILES:
        for q in load_eval_split(split):
            for a in q.candidate_apis:
                if a.uid not in seen:
                    seen[a.uid] = a
    return list(seen.values())


# --------------------------------------------------------------------------- #
# Multi-turn training sessions
# --------------------------------------------------------------------------- #
@dataclass
class SessionTurn:
    api_name: str | None
    tool_name: str | None  # may be None — extracted from function-call name
    arguments: dict = field(default_factory=dict)
    response_ok: bool = True


@dataclass
class Session:
    session_id: int
    initial_query: str
    turns: list[SessionTurn]
    # final tool/api uids of *successful* calls
    successful_uids: list[str] = field(default_factory=list)


_FN_NAME_RE = re.compile(r"^(?P<api>[A-Za-z0-9_]+)_for_(?P<tool>[A-Za-z0-9_]+)$")


def _extract_function_calls(conv) -> tuple[list[SessionTurn], str]:
    """Walk the assistant/function turn pairs."""
    froms = list(conv["from"])
    values = list(conv["value"])
    initial_query = ""
    for f, v in zip(froms, values):
        if f == "user":
            initial_query = str(v)[:1000]
            break

    turns: list[SessionTurn] = []
    for i, (f, v) in enumerate(zip(froms, values)):
        if f != "assistant":
            continue
        # ToolBench assistant outputs include lines like:
        #   "Action: search_for_xxx\nAction Input: { ... }"
        m_action = re.search(r"Action:\s*([A-Za-z0-9_]+)", v)
        m_input  = re.search(r"Action Input:\s*(\{.*\})", v, re.S)
        if not m_action:
            continue
        fn_name = m_action.group(1).strip()
        if fn_name in ("Finish", "finish"):
            continue
        try:
            args = json.loads(m_input.group(1)) if m_input else {}
        except Exception:
            args = {}
        # look ahead for the function response
        ok = True
        if i + 1 < len(froms) and froms[i + 1] == "function":
            resp = str(values[i + 1])
            # Heuristic: errors include "error" / "invalid" / non-200
            ok = not re.search(r'"error"\s*:\s*"[^"]+"|status_code"\s*:\s*[45]\d{2}', resp)
        m = _FN_NAME_RE.match(fn_name)
        if m:
            turns.append(SessionTurn(
                api_name=m.group("api"),
                tool_name=m.group("tool"),
                arguments=args if isinstance(args, dict) else {},
                response_ok=ok,
            ))
        else:
            turns.append(SessionTurn(api_name=fn_name, tool_name=None, arguments=args, response_ok=ok))
    return turns, initial_query


_NORM_RE = re.compile(r"[^a-z0-9]+")
def _norm(s: str) -> str:
    return _NORM_RE.sub("_", (s or "").lower()).strip("_")


def iter_train_sessions(max_sessions: int | None = None,
                        shards: Iterable[int] = (0,)) -> Iterator[Session]:
    """Yield Session objects from train shards (one shard ≈ 47k rows)."""
    count = 0
    # Quick lookup of normalized (tool, api) → uid
    name_idx: dict[tuple[str, str], str] = {}
    for a in build_corpus():
        name_idx[(_norm(a.tool_name), _norm(a.api_name))] = a.uid

    for s in shards:
        fp = TRAIN / f"train-{s:05d}-of-00004.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        for i, row in df.iterrows():
            conv = row["conversations"]
            turns, q = _extract_function_calls(conv)
            if not q or not turns:
                continue
            ok_uids: list[str] = []
            for t in turns:
                if not t.response_ok or not t.api_name or not t.tool_name:
                    continue
                key = (_norm(t.tool_name), _norm(t.api_name))
                if key in name_idx:
                    ok_uids.append(name_idx[key])
            yield Session(
                session_id=int(row["id"]) if str(row["id"]).isdigit() else hash(row["id"]) & 0x7fffffff,
                initial_query=q,
                turns=turns,
                successful_uids=ok_uids,
            )
            count += 1
            if max_sessions is not None and count >= max_sessions:
                return


# --------------------------------------------------------------------------- #
# Synthetic multi-turn eval sessions (built from benchmark splits)
# --------------------------------------------------------------------------- #
def build_synthetic_sessions(eval_split: list[EvalQuery],
                             *,
                             session_len: int = 3,
                             seed: int = 0,
                             grouping: str = "category") -> list[list[EvalQuery]]:
    """Group benchmark queries into pseudo-sessions of length `session_len`.

    ``grouping="category"`` (original): consecutive queries share an API
    category. Caution: SessionRerank's largest learned weight is category
    affinity, so this construction favours the method by design.
    ``grouping="random"`` (control, added after the 2026-11 postdoc review):
    sessions are random draws, so any gain must come from genuine session
    signal rather than from how sessions were built.
    """
    import random
    rng = random.Random(seed)
    if grouping == "random":
        qs = list(eval_split)
        rng.shuffle(qs)
        return [qs[i:i + session_len] for i in range(0, len(qs), session_len)]
    if grouping != "category":
        raise ValueError(f"unknown grouping {grouping!r}")
    by_cat: dict[str, list[EvalQuery]] = {}
    for q in eval_split:
        cats = {a.category for a in q.candidate_apis if a.category}
        for c in cats:
            by_cat.setdefault(c, []).append(q)
    used: set[int] = set()
    sessions: list[list[EvalQuery]] = []
    for cat, qs in by_cat.items():
        rng.shuffle(qs)
        buf: list[EvalQuery] = []
        for q in qs:
            if q.query_id in used:
                continue
            buf.append(q)
            used.add(q.query_id)
            if len(buf) == session_len:
                sessions.append(buf)
                buf = []
        if len(buf) >= 2:
            sessions.append(buf)
    # Singletons for queries not grouped
    for q in eval_split:
        if q.query_id not in used:
            sessions.append([q])
    return sessions
