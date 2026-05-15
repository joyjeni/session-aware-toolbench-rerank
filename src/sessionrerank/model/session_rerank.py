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
                 feat_stats: dict | None = None,
                 temperature: float = 1.0,
                 **_ignored):
        self.base = base
        self.w_base = w_base
        self.w_cat = w_cat
        self.w_sch = w_sch
        self.w_ept = w_ept
        self.w_cooc = w_cooc
        self.w_rec = w_rec
        self.cooc_prior = cooc_prior
        # If feat_stats is supplied (from fit_weights), use the same z-score
        # normalization at inference. Otherwise default to identity (m=0,s=1).
        fs = feat_stats or {}
        self._cat_m, self._cat_s = fs.get("cat", [0.0, 1.0])
        self._sch_m, self._sch_s = fs.get("sch", [0.0, 1.0])
        self._ept_m, self._ept_s = fs.get("ept", [0.0, 1.0])
        self._rec_m, self._rec_s = fs.get("rec", [0.0, 1.0])
        self._coo_m, self._coo_s = fs.get("cooc", [0.0, 1.0])
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
        # Apply the same log1p + z-score transform used during fit_weights so
        # the learned weights are calibrated to unit-variance features.
        cat = (np.log1p(cat * 10.0) - self._cat_m) / (self._cat_s + 1e-6)
        sch = (np.log1p(sch * 10.0) - self._sch_m) / (self._sch_s + 1e-6)
        ept = (np.log1p(ept * 10.0) - self._ept_m) / (self._ept_s + 1e-6)
        rec = (np.log1p(rec * 10.0) - self._rec_m) / (self._rec_s + 1e-6)
        out = (self.w_base * base
               + self.w_cat * cat
               + self.w_sch * sch
               + self.w_ept * ept
               + self.w_rec * rec)
        if self.cooc_prior is not None and self.w_cooc != 0.0 and prior.history_uids:
            cooc_arr = np.array([
                self.cooc_prior.score(prior.history_uids, c.uid) for c in cands
            ])
            cooc_arr = (np.log1p(cooc_arr * 10.0) - self._coo_m) / (self._coo_s + 1e-6)
            out = out + self.w_cooc * cooc_arr
        return out / self.temp


# ---------------------------------------------------------------------------
# Lightweight fit — learn the weights from training trajectories
# ---------------------------------------------------------------------------
def _seed_everything(seed: int) -> None:
    """Deterministic seeding across numpy + torch (cpu & cuda)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def fit_weights(base_retriever,
                training_sessions,           # iterable of Session
                corpus: Sequence[API],
                *,
                max_sessions: int = 8000,
                negatives_per_pos: int = 32,
                epochs: int = 6,
                batch_size: int = 32,
                lr: float = 0.05,
                weight_decay: float = 1e-2,
                grad_clip: float = 1.0,
                avg_last_k_epochs: int = 2,
                device: str = "cpu",
                seed: int = 0,
                cooc_prior: CooccurrencePrior | None = None) -> dict[str, float]:
    """Train (w_base, w_cat, w_sch, w_ept, w_cooc, w_rec) by logistic regression.

    Stability fixes (v3):
      * Deterministic torch seeding (cuda + cudnn included).
      * Mini-batched InfoNCE (default batch=32) → lower gradient variance.
      * Per-feature z-score standardization (computed from training prior
        statistics) so each feature has unit scale before being scored —
        critical because the optimizer was previously seeing wildly different
        magnitudes for `cooc` (sparse, often 0) vs `cat` (dense, ~0.1).
      * AdamW with weight_decay anchors log-weights near a sensible prior,
        breaking the degenerate ridge between correlated features
        (w_cat ↔ w_rec ↔ w_cooc) that caused per-seed flipping.
      * Gradient clipping at L2 norm 1.0.
      * Final weights = mean of last `avg_last_k_epochs` epoch checkpoints,
        which suppresses end-of-training noise.

    For each session with K successful calls, we generate K-1 training examples:
    given history H_k = first k successful calls, the next successful call's API
    is the positive; we sample `negatives_per_pos` random APIs from the corpus
    as negatives.
    """
    _seed_everything(seed)
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
        return dict(w_base=1.0, w_cat=0.45, w_sch=0.25, w_ept=0.30,
                    w_cooc=0.0, w_rec=0.0)

    print(f"[fit] training tuples: {len(tuples)} (cooc={'on' if cooc_prior else 'off'})"
          f"  bs={batch_size}  lr={lr}  wd={weight_decay}  epochs={epochs}")

    # Pre-compute base scores per unique query (avoid repeated encoding).
    uniq_q = list({t[0] for t in tuples})
    q2idx = {q: i for i, q in enumerate(uniq_q)}
    base_scores = np.stack([base_retriever.score(q, candidates=corpus) for q in uniq_q])
    base_scores = (base_scores - base_scores.mean(axis=1, keepdims=True)) / (
        base_scores.std(axis=1, keepdims=True) + 1e-6)

    # Pre-compute candidate metadata arrays
    cat_arr = np.array([a.category for a in corpus])
    sch_arr = np.array([a.schema_signature for a in corpus])
    ept_arr = np.array([a.endpoint_pattern for a in corpus])
    uid_arr = np.array([a.uid for a in corpus])

    use_cooc = cooc_prior is not None

    # ------------------------------------------------------------------
    # Build per-example feature tensors ONCE (much faster, deterministic).
    # For each tuple i we cache: prior + history_uids for re-use in batches.
    # ------------------------------------------------------------------
    cached_priors: list[SessionPrior] = []
    cached_hist_uids: list[list[str]] = []
    for q, hist_uids, _pos in tuples:
        hist_apis = [corpus[uid2idx[u]] for u in hist_uids]
        cached_priors.append(SessionPrior.from_apis(hist_apis))
        cached_hist_uids.append(hist_uids)

    # ------------------------------------------------------------------
    # Per-feature statistics on a sample of training examples (used to
    # z-score features so the optimizer sees unit-variance signals).
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(tuples), size=min(2048, len(tuples)), replace=False)
    feat_samples = {"cat": [], "sch": [], "ept": [], "rec": [], "cooc": []}
    N = len(corpus)
    for idx in sample_idx:
        prior = cached_priors[idx]
        hu = cached_hist_uids[idx]
        # draw 16 random candidates per example to estimate marginal stats
        cand = rng.integers(0, N, size=16)
        feat_samples["cat"].extend(prior.cat_w.get(cat_arr[i], 0.0) for i in cand)
        feat_samples["sch"].extend(prior.sch_w.get(sch_arr[i], 0.0) for i in cand)
        feat_samples["ept"].extend(prior.ept_w.get(ept_arr[i], 0.0) for i in cand)
        feat_samples["rec"].extend(prior.rec_w.get(cat_arr[i], 0.0) for i in cand)
        if use_cooc:
            feat_samples["cooc"].extend(
                cooc_prior.score(hu, uid_arr[i]) for i in cand)

    def _norm_stats(vs: list[float]) -> tuple[float, float]:
        arr = np.log1p(np.asarray(vs, dtype=np.float32) * 10.0)
        m = float(arr.mean())
        s = float(arr.std() + 1e-6)
        return m, s

    cat_m, cat_s = _norm_stats(feat_samples["cat"])
    sch_m, sch_s = _norm_stats(feat_samples["sch"])
    ept_m, ept_s = _norm_stats(feat_samples["ept"])
    rec_m, rec_s = _norm_stats(feat_samples["rec"])
    coo_m, coo_s = (_norm_stats(feat_samples["cooc"]) if use_cooc else (0.0, 1.0))
    print(f"[fit] feat stats  cat=({cat_m:.3f},{cat_s:.3f})  "
          f"sch=({sch_m:.3f},{sch_s:.3f})  ept=({ept_m:.3f},{ept_s:.3f})  "
          f"rec=({rec_m:.3f},{rec_s:.3f})  cooc=({coo_m:.3f},{coo_s:.3f})")

    def _feat(vs: np.ndarray, m: float, s: float) -> torch.Tensor:
        return torch.from_numpy(((np.log1p(vs * 10.0) - m) / s)).float()

    # ------------------------------------------------------------------
    # Initialize log-weights tightly around a uniform prior
    # Order: [base, cat, sch, ept, cooc, rec]
    # ------------------------------------------------------------------
    init_log = np.array([np.log(1.0)] * 6, dtype=np.float32)
    w = torch.tensor(init_log, dtype=torch.float32, requires_grad=True)
    # AdamW with weight_decay anchors |log w| -> 0 (i.e. weights -> 1.0)
    opt = torch.optim.AdamW([w], lr=lr, weight_decay=weight_decay)

    # ------------------------------------------------------------------
    # Training loop with mini-batches
    # ------------------------------------------------------------------
    epoch_w_history: list[np.ndarray] = []
    for epoch in range(epochs):
        total = 0.0
        n_steps = 0
        rng_perm = rng.permutation(len(tuples))
        # Build mini-batches of size `batch_size`
        for b_start in range(0, len(rng_perm), batch_size):
            b_idx = rng_perm[b_start:b_start + batch_size]
            # Build (B, 1+neg) logits matrix
            B = len(b_idx)
            K = 1 + negatives_per_pos
            b_tensor = torch.zeros((B, K), dtype=torch.float32)
            f_cat = torch.zeros((B, K), dtype=torch.float32)
            f_sch = torch.zeros((B, K), dtype=torch.float32)
            f_ept = torch.zeros((B, K), dtype=torch.float32)
            f_rec = torch.zeros((B, K), dtype=torch.float32)
            f_coo = torch.zeros((B, K), dtype=torch.float32) if use_cooc else None
            for j, idx in enumerate(b_idx):
                q, hist_uids, pos_uid = tuples[idx]
                qi = q2idx[q]
                pos_i = uid2idx[pos_uid]
                neg_i = rng.integers(0, N, size=negatives_per_pos)
                neg_i = np.where(neg_i == pos_i, (neg_i + 1) % N, neg_i)
                all_i = np.concatenate([[pos_i], neg_i])
                b_tensor[j] = torch.from_numpy(base_scores[qi, all_i]).float()
                prior = cached_priors[idx]
                vc = np.array([prior.cat_w.get(cat_arr[i], 0.0) for i in all_i])
                vs = np.array([prior.sch_w.get(sch_arr[i], 0.0) for i in all_i])
                ve = np.array([prior.ept_w.get(ept_arr[i], 0.0) for i in all_i])
                vr = np.array([prior.rec_w.get(cat_arr[i], 0.0) for i in all_i])
                f_cat[j] = _feat(vc, cat_m, cat_s)
                f_sch[j] = _feat(vs, sch_m, sch_s)
                f_ept[j] = _feat(ve, ept_m, ept_s)
                f_rec[j] = _feat(vr, rec_m, rec_s)
                if use_cooc:
                    vco = np.array([cooc_prior.score(hist_uids, uid_arr[i]) for i in all_i])
                    f_coo[j] = _feat(vco, coo_m, coo_s)
            wb = torch.exp(w[0]); wc = torch.exp(w[1])
            ws_ = torch.exp(w[2]); we = torch.exp(w[3])
            wco = torch.exp(w[4]); wr = torch.exp(w[5])
            logits = (wb * b_tensor + wc * f_cat + ws_ * f_sch
                      + we * f_ept + wr * f_rec)
            if use_cooc:
                logits = logits + wco * f_coo
            target = torch.zeros(B, dtype=torch.long)
            loss = F.cross_entropy(logits, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([w], max_norm=grad_clip)
            opt.step()
            total += float(loss.item()) * B
            n_steps += B
        with torch.no_grad():
            wb_, wc_, ws_v, we_, wco_, wr_ = (float(torch.exp(w[i]).detach()) for i in range(6))
            epoch_w_history.append(torch.exp(w).detach().cpu().numpy().copy())
        print(f"[fit] epoch {epoch+1}/{epochs}  loss={total/max(1,n_steps):.4f}  "
              f"w_base={wb_:.3f}  w_cat={wc_:.3f}  "
              f"w_sch={ws_v:.3f}  w_ept={we_:.3f}  "
              f"w_cooc={wco_:.3f}  w_rec={wr_:.3f}")

    # Final weights = mean of last K epoch snapshots (suppresses tail noise)
    k = min(avg_last_k_epochs, len(epoch_w_history))
    final = np.mean(np.stack(epoch_w_history[-k:], axis=0), axis=0)
    out = dict(
        w_base=float(final[0]),
        w_cat=float(final[1]),
        w_sch=float(final[2]),
        w_ept=float(final[3]),
        w_rec=float(final[5]),
    )
    out["w_cooc"] = float(final[4]) if use_cooc else 0.0
    # Stash normalization stats so SessionRerank can apply the same transform
    # at inference time. (Optional; SessionRerank doesn't yet read these but
    # they are returned for downstream use.)
    out["_feat_stats"] = {
        "cat": [cat_m, cat_s], "sch": [sch_m, sch_s],
        "ept": [ept_m, ept_s], "rec": [rec_m, rec_s],
        "cooc": [coo_m, coo_s],
    }
    return out
