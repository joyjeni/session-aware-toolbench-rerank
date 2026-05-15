"""SessionRerank — session-aware re-ranking layer.

Setup
-----
At inference time the system observes, for the current user session, a list of
previously successful tool-call outcomes. Each successful call gives us four
*emergent* signals about what the user "likes":

    1. **Category affinity**       — how often this category was successfully used.
    2. **Schema affinity**         — sorted-parameter-name signature of previous
       successful APIs (captures "user keeps calling APIs that need user_id and
       date_range" style patterns).
    3. **Endpoint-pattern affinity** — normalized api-name tokens (e.g.
       `search`, `get_by_id`) — captures invocation-style preferences.
    4. **Co-occurrence prior**     — learned P(api_b | api_a) from successful
       sliding-window pairs in ToolBench train shards. Captures *which APIs
       tend to follow which* — the strongest session-aware signal.
    5. **Recency-weighted category** — same as (1) but restricted to the last-K
       turns with a sharper decay, isolating *short-horizon* drift from the
       long-horizon category prior.

The final score for a candidate API a given query q and session history H is:

    s(a | q, H) = w_base   · s_base(a | q)
                + w_cat    · cat_aff(a, H)
                + w_sch    · sch_aff(a, H)
                + w_ept    · ept_aff(a, H)
                + w_cooc   · cooc(a | last(H))
                + w_rec    · recent_cat_aff(a, H[-K:])

Weights are learned by a tiny logistic-regression-style fit on multi-turn
ToolBench trajectories — the loss treats *the next successful API call in the
trajectory* as the positive and the rest of the candidate pool as negatives.
No new annotations.

Crucially: when H is empty (first turn), the bias terms vanish and we fall
back exactly to the base retriever — so SessionRerank can never hurt single-
turn queries by construction.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence, Iterable
import json
import numpy as np
import torch
import torch.nn.functional as F

from ..data.loader import API


# ---------------------------------------------------------------------------
# Co-occurrence prior — learned once from training trajectories
# ---------------------------------------------------------------------------
@dataclass
class CooccurrencePrior:
    """P(api_b | api_a) estimated from successful sliding-window pairs.

    Stored sparsely as a nested dict: cooc[a_uid][b_uid] = conditional probability.
    Built once from `iter_train_sessions().successful_uids` with Laplace
    smoothing and a configurable window (default = 1, i.e. immediate next call).
    """
    cooc: dict[str, dict[str, float]] = field(default_factory=dict)
    n_sessions_seen: int = 0

    @classmethod
    def from_sessions(cls,
                      sessions: Iterable,
                      *,
                      max_sessions: int = 20000,
                      window: int = 3,
                      smoothing: float = 0.5) -> "CooccurrencePrior":
        """Harvest co-occurrence counts from successful sliding-window pairs.

        For each session's successful_uids = [u_0, u_1, ..., u_T], we increment
        count[u_i][u_j] for all j in (i, i+window]. Then normalize per-row to
        get P(b | a) and apply Laplace smoothing.
        """
        counts: dict[str, dict[str, float]] = {}
        n = 0
        for sess in sessions:
            uids = list(getattr(sess, "successful_uids", []) or [])
            if len(uids) < 2:
                continue
            for i, a in enumerate(uids):
                for j in range(i + 1, min(i + 1 + window, len(uids))):
                    b = uids[j]
                    if a == b:
                        continue
                    counts.setdefault(a, {})
                    counts[a][b] = counts[a].get(b, 0.0) + 1.0
            n += 1
            if n >= max_sessions:
                break
        # Row-normalize with Laplace smoothing
        cooc: dict[str, dict[str, float]] = {}
        for a, row in counts.items():
            tot = sum(row.values()) + smoothing * len(row)
            cooc[a] = {b: (c + smoothing) / tot for b, c in row.items()}
        return cls(cooc=cooc, n_sessions_seen=n)

    def score(self, history_uids: Sequence[str], cand_uid: str,
              *, decay: float = 0.7) -> float:
        """Score a candidate as decay-weighted sum of P(cand | h_i) for recent h_i."""
        if not history_uids:
            return 0.0
        s = 0.0
        for i, h in enumerate(history_uids):
            w = decay ** (len(history_uids) - 1 - i)
            row = self.cooc.get(h)
            if row is None:
                continue
            s += w * row.get(cand_uid, 0.0)
        return float(s)

    def n_keys(self) -> int:
        return len(self.cooc)


# ---------------------------------------------------------------------------
# Session signal extraction
# ---------------------------------------------------------------------------
@dataclass
class SessionPrior:
    """Soft histograms over (category, schema, endpoint) + recency window."""
    cat_w: dict[str, float] = field(default_factory=dict)
    sch_w: dict[str, float] = field(default_factory=dict)
    ept_w: dict[str, float] = field(default_factory=dict)
    rec_w: dict[str, float] = field(default_factory=dict)  # recency-restricted category
    history_uids: list[str] = field(default_factory=list)  # raw uid history for cooc
    n: int = 0

    @classmethod
    def from_apis(cls, apis: Sequence[API],
                  decay: float = 0.85,
                  recency_k: int = 3,
                  recency_decay: float = 0.5) -> "SessionPrior":
        sp = cls()
        sp.history_uids = [a.uid for a in apis]
        for i, a in enumerate(apis):
            # most-recent calls weigh more (long-horizon)
            w = decay ** (len(apis) - 1 - i)
            sp.cat_w[a.category] = sp.cat_w.get(a.category, 0.0) + w
            sp.sch_w[a.schema_signature] = sp.sch_w.get(a.schema_signature, 0.0) + w
            sp.ept_w[a.endpoint_pattern] = sp.ept_w.get(a.endpoint_pattern, 0.0) + w
            sp.n += 1
        # Recency window — last K turns only, sharper decay
        recent = apis[-recency_k:] if recency_k > 0 else []
        for i, a in enumerate(recent):
            w = recency_decay ** (len(recent) - 1 - i)
            sp.rec_w[a.category] = sp.rec_w.get(a.category, 0.0) + w
        # L1-normalize each axis
        for d in (sp.cat_w, sp.sch_w, sp.ept_w, sp.rec_w):
            tot = sum(d.values()) or 1.0
            for k in list(d):
                d[k] = d[k] / tot
        return sp

    def features(self, cand: API) -> tuple[float, float, float, float]:
        if self.n == 0:
            return 0.0, 0.0, 0.0, 0.0
        return (
            self.cat_w.get(cand.category, 0.0),
            self.sch_w.get(cand.schema_signature, 0.0),
            self.ept_w.get(cand.endpoint_pattern, 0.0),
            self.rec_w.get(cand.category, 0.0),
        )


# ---------------------------------------------------------------------------
# Re-ranker
# ---------------------------------------------------------------------------
class SessionRerank:
    """Wraps a base retriever and adds the session-prior bias.

    New in v2: optional co-occurrence prior and recency-restricted category
    feature. If `cooc_prior` is provided, the score includes a `w_cooc` term
    over recent history → candidate transitions. The `w_rec` term gives a
    separate weight to a short-horizon (last-K) category signal.
    """

    name = "SessionRerank"

    def __init__(self,
                 base,
                 w_base: float = 1.0,
                 w_cat: float = 0.45,
                 w_sch: float = 0.25,
                 w_ept: float = 0.30,
                 w_cooc: float = 0.0,
                 w_rec: float = 0.0,
                 cooc_prior: CooccurrencePrior | None = None,
                 temperature: float = 1.0):
        self.base = base
        self.w_base = w_base
        self.w_cat = w_cat
        self.w_sch = w_sch
        self.w_ept = w_ept
        self.w_cooc = w_cooc
        self.w_rec = w_rec
        self.cooc_prior = cooc_prior
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
        rec = np.array([prior.rec_w.get(c.category, 0.0) for c in cands])
        # log-transform the priors so they behave additively in log-odds space
        cat = np.log1p(cat * 10.0)
        sch = np.log1p(sch * 10.0)
        ept = np.log1p(ept * 10.0)
        rec = np.log1p(rec * 10.0)
        out = (self.w_base * base
               + self.w_cat * cat
               + self.w_sch * sch
               + self.w_ept * ept
               + self.w_rec * rec)
        if self.cooc_prior is not None and self.w_cooc != 0.0 and prior.history_uids:
            cooc_arr = np.array([
                self.cooc_prior.score(prior.history_uids, c.uid) for c in cands
            ])
            cooc_arr = np.log1p(cooc_arr * 10.0)
            out = out + self.w_cooc * cooc_arr
        return out / self.temp


# ---------------------------------------------------------------------------
# Lightweight fit — learn the weights from training trajectories
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
                seed: int = 0,
                cooc_prior: CooccurrencePrior | None = None) -> dict[str, float]:
    """Train (w_base, w_cat, w_sch, w_ept, w_cooc, w_rec) by logistic regression.

    For each session with K successful calls, we generate K-1 training examples:
    given history H_k = first k successful calls, the next successful call's API
    is the positive; we sample `negatives_per_pos` random APIs from the corpus
    as negatives. We use **pre-cached base-retriever scores** so the loop is
    fast even on CPU.

    If `cooc_prior` is provided, a `w_cooc` weight is also learned.
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
        return dict(w_base=1.0, w_cat=0.45, w_sch=0.25, w_ept=0.30,
                    w_cooc=0.0, w_rec=0.0)

    print(f"[fit] training tuples: {len(tuples)} (cooc={'on' if cooc_prior else 'off'})")

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
    uid_arr = np.array([a.uid for a in corpus])

    use_cooc = cooc_prior is not None
    # Parameters (in log-space to keep them positive)
    # Order: [base, cat, sch, ept, cooc, rec]
    init_log = [0.0, np.log(0.45), np.log(0.25), np.log(0.30),
                np.log(0.30), np.log(0.20)]
    w = torch.tensor(init_log, dtype=torch.float32, requires_grad=True)
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
            recs = np.array([prior.rec_w.get(cat_arr[i], 0.0) for i in all_i])
            cats = torch.from_numpy(np.log1p(cats * 10.0)).float()
            schs = torch.from_numpy(np.log1p(schs * 10.0)).float()
            epts = torch.from_numpy(np.log1p(epts * 10.0)).float()
            recs = torch.from_numpy(np.log1p(recs * 10.0)).float()
            if use_cooc:
                coocs = np.array([
                    cooc_prior.score(hist_uids, uid_arr[i]) for i in all_i
                ])
                coocs = torch.from_numpy(np.log1p(coocs * 10.0)).float()
            wb = torch.exp(w[0]); wc = torch.exp(w[1])
            ws_ = torch.exp(w[2]); we = torch.exp(w[3])
            wco = torch.exp(w[4]); wr = torch.exp(w[5])
            logits = wb * b + wc * cats + ws_ * schs + we * epts + wr * recs
            if use_cooc:
                logits = logits + wco * coocs
            # InfoNCE
            target = torch.zeros((), dtype=torch.long)  # positive is at index 0
            loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        with torch.no_grad():
            wb_, wc_, ws_v, we_, wco_, wr_ = (float(torch.exp(w[i]).detach()) for i in range(6))
        print(f"[fit] epoch {epoch+1}/{epochs}  loss={total/len(tuples):.4f}  "
              f"w_base={wb_:.3f}  w_cat={wc_:.3f}  "
              f"w_sch={ws_v:.3f}  w_ept={we_:.3f}  "
              f"w_cooc={wco_:.3f}  w_rec={wr_:.3f}")

    with torch.no_grad():
        wexp = torch.exp(w).detach()
    out = dict(
        w_base=float(wexp[0]),
        w_cat=float(wexp[1]),
        w_sch=float(wexp[2]),
        w_ept=float(wexp[3]),
        w_rec=float(wexp[5]),
    )
    if use_cooc:
        out["w_cooc"] = float(wexp[4])
    else:
        out["w_cooc"] = 0.0
    return out
