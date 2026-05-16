"""Pheromone-decay co-occurrence prior — ant-colony-optimization view of API call chains.

Why nature-inspired
-------------------
Foraging ants reinforce successful trails by depositing pheromone and let
unused trails *evaporate*. Two emergent properties matter for tool retrieval:

1. **Recency-of-success dominates** — pheromone is laid down ONLY on
   successful (api_a → api_b) transitions. Failed calls leave no trace.
2. **Stale paths fade** — every session step, all pheromones decay by ρ.
   So a co-occurrence pair that was hot in 2024 but unused since then
   contributes ~0 by 2026.

Mathematically, for a successful transition u → v at session-step t:

    τ_{u,v}(t)   =   (1 - ρ) · τ_{u,v}(t-1)  +  τ_0 · 1[success]

At inference time we score a candidate v given the current session's recent
successful uids H = [h_1, ..., h_n] (most recent last) as:

    score(v | H)  =  Σ_i  γ^{n-i} · log(1 + τ_{h_i, v})

where γ is an in-session decay (sharper than the cross-session ρ).

The log transform compresses heavy-hitter pairs (some popular API chains
dominate raw counts by 100×) so the prior behaves more like a relevance
signal than a frequency signal — important because we *blend* it with dense
scores, not replace them.

Drop-in API compatibility
-------------------------
``PheromoneCoocPrior`` exposes the same surface as ``CooccurrencePrior``:
``.from_sessions(...)``, ``.score(history_uids, cand_uid)``, ``.n_keys()``,
``.n_sessions_seen``. The existing SessionRerank model + ``fit_weights`` use
it without modification.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Sequence
import math

import numpy as np


@dataclass
class PheromoneCoocPrior:
    """ACO-style pheromone trails over successful API → API transitions.

    Attributes
    ----------
    pheromone : nested dict, pheromone[a_uid][b_uid] = τ (>=0)
    rho : float, cross-session evaporation rate per session (default 0.02)
    tau0 : float, deposit amount per observed successful transition
    in_session_decay : γ in (0, 1] — sharper recency decay used at score time
    n_sessions_seen : how many training sessions were ingested
    """
    pheromone: dict[str, dict[str, float]] = field(default_factory=dict)
    rho: float = 0.02
    tau0: float = 1.0
    in_session_decay: float = 0.7
    n_sessions_seen: int = 0

    # ------------------------------------------------------------------
    # Training-time harvest
    # ------------------------------------------------------------------
    @classmethod
    def from_sessions(
        cls,
        sessions: Iterable,
        *,
        max_sessions: int = 20000,
        window: int = 3,
        rho: float = 0.02,
        tau0: float = 1.0,
        in_session_decay: float = 0.7,
    ) -> "PheromoneCoocPrior":
        """Build pheromone matrix by simulating ant-colony deposit + evaporation.

        For each successful trajectory u_0, ..., u_T we lay down pheromone on
        every (u_i, u_j) with j in (i, i+window], then evaporate the *entire*
        matrix by ρ once per session. This produces a time-varying trail
        structure where newer training sessions exert more influence than
        older ones — exactly the property we want for a session-aware prior.
        """
        pher: dict[str, dict[str, float]] = {}
        n = 0
        for sess in sessions:
            uids = list(getattr(sess, "successful_uids", []) or [])
            if len(uids) < 2:
                # Still apply evaporation so global decay tracks session count
                if pher:
                    cls._evaporate_inplace(pher, rho)
                n += 1
                if n >= max_sessions:
                    break
                continue

            # 1) Deposit on every successful pair within the window
            for i, a in enumerate(uids):
                row = pher.setdefault(a, {})
                for j in range(i + 1, min(i + 1 + window, len(uids))):
                    b = uids[j]
                    if a == b:
                        continue
                    row[b] = row.get(b, 0.0) + tau0

            # 2) Evaporate once per session step
            cls._evaporate_inplace(pher, rho)

            n += 1
            if n >= max_sessions:
                break

        # Prune entries that have decayed below a numerical threshold
        cls._prune_inplace(pher, eps=1e-4)
        return cls(
            pheromone=pher,
            rho=rho,
            tau0=tau0,
            in_session_decay=in_session_decay,
            n_sessions_seen=n,
        )

    @staticmethod
    def _evaporate_inplace(pher: dict[str, dict[str, float]], rho: float) -> None:
        factor = 1.0 - rho
        if factor >= 0.9999:
            return
        for a, row in pher.items():
            for b in list(row.keys()):
                row[b] *= factor

    @staticmethod
    def _prune_inplace(
        pher: dict[str, dict[str, float]], eps: float = 1e-4
    ) -> None:
        """Drop pheromone entries that fell below eps to keep the matrix sparse."""
        dead_rows: list[str] = []
        for a, row in pher.items():
            for b in list(row.keys()):
                if row[b] < eps:
                    del row[b]
            if not row:
                dead_rows.append(a)
        for a in dead_rows:
            pher.pop(a, None)

    # ------------------------------------------------------------------
    # Inference-time scoring
    # ------------------------------------------------------------------
    def score(
        self,
        history_uids: Sequence[str],
        cand_uid: str,
        *,
        decay: float | None = None,
    ) -> float:
        """Score candidate using decay-weighted log-pheromone over history.

        ``decay`` (γ) defaults to ``self.in_session_decay`` but the SessionRerank
        weight-fitter passes its own value, so we accept the override.
        """
        if not history_uids:
            return 0.0
        gamma = self.in_session_decay if decay is None else float(decay)
        n_hist = len(history_uids)
        s = 0.0
        for i, h in enumerate(history_uids):
            row = self.pheromone.get(h)
            if row is None:
                continue
            tau = row.get(cand_uid, 0.0)
            if tau <= 0.0:
                continue
            w = gamma ** (n_hist - 1 - i)
            s += w * math.log1p(tau)
        return float(s)

    # ------------------------------------------------------------------
    # Live deposit (optional, for online/demo use)
    # ------------------------------------------------------------------
    def deposit(self, a_uid: str, b_uid: str, amount: float | None = None) -> None:
        """Lay down pheromone after observing a successful (a→b) transition.

        Used by the live demo to let the system get *smarter* during a session
        without retraining. Not called during the offline benchmark.
        """
        if a_uid == b_uid:
            return
        amt = self.tau0 if amount is None else float(amount)
        row = self.pheromone.setdefault(a_uid, {})
        row[b_uid] = row.get(b_uid, 0.0) + amt

    def evaporate_step(self) -> None:
        """Apply one global evaporation step. Call between user sessions."""
        self._evaporate_inplace(self.pheromone, self.rho)
        # Keep the matrix from growing without bound in long-running demos
        if self.n_sessions_seen % 50 == 0:
            self._prune_inplace(self.pheromone)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def n_keys(self) -> int:
        return len(self.pheromone)

    def total_mass(self) -> float:
        return float(sum(sum(row.values()) for row in self.pheromone.values()))

    def density(self) -> float:
        """Average # outgoing trails per source uid."""
        if not self.pheromone:
            return 0.0
        return float(np.mean([len(row) for row in self.pheromone.values()]))

    def summary(self) -> dict:
        return {
            "n_keys": self.n_keys(),
            "total_mass": self.total_mass(),
            "avg_density": self.density(),
            "rho": self.rho,
            "tau0": self.tau0,
            "in_session_decay": self.in_session_decay,
            "n_sessions_seen": self.n_sessions_seen,
        }
