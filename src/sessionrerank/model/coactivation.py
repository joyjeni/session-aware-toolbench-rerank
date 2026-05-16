"""Co-activation Cache — session-aware learned cache of API→API transitions.

Drop-in replacement for a static co-occurrence prior. Maintains a sparse,
time-varying weighted directed graph of (api_a → api_b) edges, with:

  • edge-weight increment on every successful transition observed in a
    session (analogous to a TTL refresh on a write-through cache).
  • exponential weight decay per session step (cache eviction policy: stale
    edges fall below ε and get pruned, keeping the graph sparse).

Mathematically, for a successful transition u → v at session-step t:

    w_{u,v}(t)   =   (1 - rho) · w_{u,v}(t-1)   +   delta · 1[success]

Scoring a candidate v against the current history H = [h_1, …, h_n]:

    score(v | H)   =   Σ_i  gamma^{n-i}  ·  log(1 + w_{h_i, v})

The `log` compresses heavy-hitter pairs (some popular API chains dominate
raw counts by >100×) so the prior behaves like a *relevance* signal that
blends with dense retrieval scores rather than overpowering them.

This module has zero external dependencies beyond numpy.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Sequence
import math

import numpy as np


@dataclass
class CoactivationCache:
    """Adaptive cache of (api_a → api_b) transition weights.

    Attributes
    ----------
    edges : nested dict, edges[a_uid][b_uid] = w (>=0)
    rho : float, cross-session decay rate per step (default 0.02)
    delta : float, weight increment per observed successful transition
    in_session_decay : gamma in (0, 1] — sharper recency decay used at score time
    n_sessions_seen : how many training sessions were ingested
    """
    edges: dict[str, dict[str, float]] = field(default_factory=dict)
    rho: float = 0.02
    delta: float = 1.0
    in_session_decay: float = 0.7
    n_sessions_seen: int = 0

    # ------------------------------------------------------------------
    # Build from a stream of training sessions
    # ------------------------------------------------------------------
    @classmethod
    def from_sessions(
        cls,
        sessions: Iterable,
        *,
        max_sessions: int = 20000,
        window: int = 3,
        rho: float = 0.02,
        delta: float = 1.0,
        in_session_decay: float = 0.7,
    ) -> "CoactivationCache":
        """Populate the cache by replaying successful tool-call transitions."""
        edges: dict[str, dict[str, float]] = {}
        n = 0
        for sess in sessions:
            uids = list(getattr(sess, "successful_uids", []) or [])
            if len(uids) < 2:
                if edges:
                    cls._decay_inplace(edges, rho)
                n += 1
                if n >= max_sessions:
                    break
                continue

            for i, a in enumerate(uids):
                row = edges.setdefault(a, {})
                for j in range(i + 1, min(i + 1 + window, len(uids))):
                    b = uids[j]
                    if a == b:
                        continue
                    row[b] = row.get(b, 0.0) + delta

            cls._decay_inplace(edges, rho)
            n += 1
            if n >= max_sessions:
                break

        cls._evict_inplace(edges, eps=1e-4)
        return cls(
            edges=edges,
            rho=rho,
            delta=delta,
            in_session_decay=in_session_decay,
            n_sessions_seen=n,
        )

    @staticmethod
    def _decay_inplace(edges: dict[str, dict[str, float]], rho: float) -> None:
        factor = 1.0 - rho
        if factor >= 0.9999:
            return
        for a, row in edges.items():
            for b in list(row.keys()):
                row[b] *= factor

    @staticmethod
    def _evict_inplace(
        edges: dict[str, dict[str, float]], eps: float = 1e-4
    ) -> None:
        """Drop edges that fell below eps to keep the cache sparse."""
        dead_rows: list[str] = []
        for a, row in edges.items():
            for b in list(row.keys()):
                if row[b] < eps:
                    del row[b]
            if not row:
                dead_rows.append(a)
        for a in dead_rows:
            edges.pop(a, None)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def score(
        self,
        history_uids: Sequence[str],
        cand_uid: str,
        *,
        decay: float | None = None,
    ) -> float:
        """Decay-weighted log-edge score over the recent successful history."""
        if not history_uids:
            return 0.0
        gamma = self.in_session_decay if decay is None else float(decay)
        n_hist = len(history_uids)
        s = 0.0
        for i, h in enumerate(history_uids):
            row = self.edges.get(h)
            if row is None:
                continue
            w = row.get(cand_uid, 0.0)
            if w <= 0.0:
                continue
            s += (gamma ** (n_hist - 1 - i)) * math.log1p(w)
        return float(s)

    # ------------------------------------------------------------------
    # Online write path (used by the live demo to update during a session)
    # ------------------------------------------------------------------
    def update_edge(self, a_uid: str, b_uid: str, amount: float | None = None) -> None:
        """Increment the weight on a successful (a→b) transition."""
        if a_uid == b_uid:
            return
        amt = self.delta if amount is None else float(amount)
        row = self.edges.setdefault(a_uid, {})
        row[b_uid] = row.get(b_uid, 0.0) + amt

    def decay_step(self) -> None:
        """Apply one global decay step. Call between user sessions."""
        self._decay_inplace(self.edges, self.rho)
        if self.n_sessions_seen % 50 == 0:
            self._evict_inplace(self.edges)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def n_keys(self) -> int:
        return len(self.edges)

    def total_mass(self) -> float:
        return float(sum(sum(row.values()) for row in self.edges.values()))

    def density(self) -> float:
        if not self.edges:
            return 0.0
        return float(np.mean([len(row) for row in self.edges.values()]))

    def summary(self) -> dict:
        return {
            "n_keys": self.n_keys(),
            "total_mass": self.total_mass(),
            "avg_density": self.density(),
            "rho": self.rho,
            "delta": self.delta,
            "in_session_decay": self.in_session_decay,
            "n_sessions_seen": self.n_sessions_seen,
        }
