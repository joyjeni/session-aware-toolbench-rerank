"""SessionRerank — session-aware re-ranking layer.

Setup
-----
At inference time the system observes, for the current user session, a list of
previously successful tool-call outcomes. Each successful call gives us three
*emergent* signals about what the user "likes":

    1. **Category affinity** — how often this category was successfully used.
    2. **Schema affinity**   — sorted-parameter-name signature of the previous
       successful APIs (captures "user keeps calling APIs that need user_id and
       date_range" style patterns).
    3. **Endpoint-pattern affinity** — normalized api-name tokens (e.g.
       `search`, `get_by_id`) — captures invocation-style preferences.

The final score for a candidate API a given query q and session history H is:

    s(a | q, H) = w_base · s_base(a | q)
                + w_cat  · cat_aff(a, H)
                + w_sch  · sch_aff(a, H)
                + w_ept  · ept_aff(a, H)

Weights are learned by a tiny logistic-regression-style fit on multi-turn
ToolBench trajectories — the loss treats *the next successful API call in the
trajectory* as the positive and the rest of the candidate pool as negatives.
No new annotations.

Crucially: when H is empty (first turn), the bias term vanishes and we fall
back exactly to the base retriever — so SessionRerank can never hurt single-
turn queries by construction.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence
import json
import numpy as np
import torch
import torch.nn.functional as F

from ..data.loader import API


# ---------------------------------------------------------------------------
# Session signal extraction
# ---------------------------------------------------------------------------
@dataclass
class SessionPrior:
    """Soft histograms over (category, schema, endpoint) from successful calls."""
    cat_w: dict[str, float] = field(default_factory=dict)
    sch_w: dict[str, float] = field(default_factory=dict)
    ept_w: dict[str, float] = field(default_factory=dict)
    n: int = 0

    @classmethod
    def from_apis(cls, apis: Sequence[API], decay: float = 0.85) -> "SessionPrior":
        sp = cls()
        for i, a in enumerate(apis):
            # most-recent calls weigh more
            w = decay ** (len(apis) - 1 - i)
            sp.cat_w[a.category] = sp.cat_w.get(a.category, 0.0) + w
            sp.sch_w[a.schema_signature] = sp.sch_w.get(a.schema_signature, 0.0) + w
            sp.ept_w[a.endpoint_pattern] = sp.ept_w.get(a.endpoint_pattern, 0.0) + w
            sp.n += 1
        # L1-normalize each axis
        for d in (sp.cat_w, sp.sch_w, sp.ept_w):
            tot = sum(d.values()) or 1.0
            for k in list(d):
                d[k] = d[k] / tot
        return sp

    def features(self, cand: API) -> tuple[float, float, float]:
        if self.n == 0:
            return 0.0, 0.0, 0.0
        return (
            self.cat_w.get(cand.category, 0.0),
            self.sch_w.get(cand.schema_signature, 0.0),
            self.ept_w.get(cand.endpoint_pattern, 0.0),
        )


# ---------------------------------------------------------------------------
# Re-ranker
# ---------------------------------------------------------------------------
class SessionRerank:
    """Wraps a base retriever and adds the session-prior bias."""

    name = "SessionRerank"

    def __init__(self,
                 base,
                 w_base: float = 1.0,
                 w_cat: float = 0.45,
                 w_sch: float = 0.25,
                 w_ept: float = 0.30,
                 temperature: float = 1.0):
        self.base = base
        self.w_base = w_base
        self.w_cat = w_cat
        self.w_sch = w_sch
        self.w_ept = w_ept
        self.temp = temperature
        self.name = f"SessionRerank({base.name})"

    # ------------------------------------------------------------------
    def score(self,
              query: str,
              candidates: Sequence[API] | None = None,
              prior: SessionPrior | None = None) -> np.ndarray:
        base = self.base.score(query, candidates)
        base = (base - base.mean()) / (base.std() + 1e-6)  # standardize
        if prior is None or prior.n == 0:
            return base * self.w_base
        cands = candidates if candidates is not None else self.base.corpus
        cat = np.array([prior.cat_w.get(c.category, 0.0) for c in cands])
        sch = np.array([prior.sch_w.get(c.schema_signature, 0.0) for c in cands])
        ept = np.array([prior.ept_w.get(c.endpoint_pattern, 0.0) for c in cands])
        # log-transform the priors so they behave additively in log-odds space
        cat = np.log1p(cat * 10.0)
        sch = np.log1p(sch * 10.0)
        ept = np.log1p(ept * 10.0)
        return (self.w_base * base
                + self.w_cat * cat
                + self.w_sch * sch
                + self.w_ept * ept) / self.temp


# ---------------------------------------------------------------------------
# Lightweight fit — learn the four weights from training trajectories
# ---------------------------------------------------------------------------
def fit_weights(base_retriever,
                training_sessions,           # iterable of Session
                corpus: Sequence[API],
                *,
                max_sessions: int = 8000,
                negatives_per_pos: int = 24,
                epochs: int = 4,
                lr: float = 0.25,
                device: str = "cpu",
                seed: int = 0) -> dict[str, float]:
    """Train (w_base, w_cat, w_sch, w_ept) by logistic regression.

    For each session with K successful calls, we generate K-1 training examples:
    given history H_k = first k successful calls, the next successful call's API
    is the positive; we sample `negatives_per_pos` random APIs from the corpus
    as negatives. We use **pre-cached base-retriever scores** so the loop is
    fast even on CPU.
    """
    rng = np.random.default_rng(seed)
    uid2idx = {a.uid: i for i, a in enumerate(corpus)}

    # Collect training tuples first
    tuples: list[tuple[str, list[str], str]] = []  # (query, history_uids, positive_uid)
    for sess in training_sessions:
        ok = [u for u in sess.successful_uids if u in uid2idx]
        if len(ok) < 2:
            continue
        for k in range(1, len(ok)):
            tuples.append((sess.initial_query, ok[:k], ok[k]))
        if len(tuples) >= max_sessions:
            break
    if not tuples:
        # Fallback: return reasonable defaults
        return dict(w_base=1.0, w_cat=0.45, w_sch=0.25, w_ept=0.30)

    print(f"[fit] training tuples: {len(tuples)}")

    # Pre-compute base scores per unique query (avoid repeated encoding).
    uniq_q = list({t[0] for t in tuples})
    q2idx = {q: i for i, q in enumerate(uniq_q)}
    base_scores = np.stack([base_retriever.score(q, candidates=corpus) for q in uniq_q])  # (Q, |C|)
    # standardize per query
    base_scores = (base_scores - base_scores.mean(axis=1, keepdims=True)) / (
        base_scores.std(axis=1, keepdims=True) + 1e-6)

    # Pre-compute candidate metadata arrays
    cat_arr = np.array([a.category for a in corpus])
    sch_arr = np.array([a.schema_signature for a in corpus])
    ept_arr = np.array([a.endpoint_pattern for a in corpus])

    # Parameters (in log-space to keep them positive)
    w = torch.tensor([0.0, np.log(0.45), np.log(0.25), np.log(0.30)],
                     dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr)

    rng = np.random.default_rng(seed)
    N = len(corpus)
    for epoch in range(epochs):
        total = 0.0
        rng_perm = rng.permutation(len(tuples))
        for step, idx in enumerate(rng_perm):
            q, hist_uids, pos_uid = tuples[idx]
            qi = q2idx[q]
            pos_i = uid2idx[pos_uid]
            neg_i = rng.integers(0, N, size=negatives_per_pos)
            # ensure positive not in negatives
            neg_i = np.where(neg_i == pos_i, (neg_i + 1) % N, neg_i)
            all_i = np.concatenate([[pos_i], neg_i])
            # base
            b = torch.from_numpy(base_scores[qi, all_i]).float()
            # priors from history
            hist_apis = [corpus[uid2idx[u]] for u in hist_uids]
            prior = SessionPrior.from_apis(hist_apis)
            cats = np.array([prior.cat_w.get(cat_arr[i], 0.0) for i in all_i])
            schs = np.array([prior.sch_w.get(sch_arr[i], 0.0) for i in all_i])
            epts = np.array([prior.ept_w.get(ept_arr[i], 0.0) for i in all_i])
            cats = torch.from_numpy(np.log1p(cats * 10.0)).float()
            schs = torch.from_numpy(np.log1p(schs * 10.0)).float()
            epts = torch.from_numpy(np.log1p(epts * 10.0)).float()
            wb, wc, ws, we = torch.exp(w[0]), torch.exp(w[1]), torch.exp(w[2]), torch.exp(w[3])
            logits = wb * b + wc * cats + ws * schs + we * epts
            # InfoNCE
            target = torch.zeros((), dtype=torch.long)  # positive is at index 0
            loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        with torch.no_grad():
            wb_, wc_, ws_, we_ = (float(torch.exp(w[i]).detach()) for i in range(4))
        print(f"[fit] epoch {epoch+1}/{epochs}  loss={total/len(tuples):.4f}  "
              f"w_base={wb_:.3f}  w_cat={wc_:.3f}  "
              f"w_sch={ws_:.3f}  w_ept={we_:.3f}")

    with torch.no_grad():
        wexp = torch.exp(w).detach()
    return dict(
        w_base=float(wexp[0]),
        w_cat=float(wexp[1]),
        w_sch=float(wexp[2]),
        w_ept=float(wexp[3]),
    )
