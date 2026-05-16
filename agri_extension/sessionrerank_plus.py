"""SessionRerank+ — session-aware tool retrieval for Karnataka agriculture.

Demo-time glue layer that sits on top of the trained SessionRerank model:

1. Loads the 15-API Karnataka catalog (real services).
2. Calls **Gemma 4** (`google/embeddinggemma-300m`) via HF Inference for
   query and document embeddings.
3. Maintains a ``CoactivationCache`` of (api_a → api_b) transitions to bias
   the ranking toward tool chains that have worked in past sessions.
4. Adds a profile-based feature mask that demotes wrong-crop /
   wrong-district / wrong-season APIs.
5. Updates the cache during the live session so the system gets sharper
   the longer the farmer interacts with it.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Make ``sessionrerank.model.coactivation`` importable from agri_extension/
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

if __package__:
    from .karnataka_apis import KARNATAKA_APIS, KarnatakaAPI
    from . import gemma_embed
else:
    from karnataka_apis import KARNATAKA_APIS, KarnatakaAPI  # type: ignore
    import gemma_embed  # type: ignore

from sessionrerank.model.coactivation import CoactivationCache  # noqa: E402


@dataclass
class FarmerProfile:
    crop: str = "ragi"
    district: str = "Tumakuru"
    language: str = "kn"
    season: str = "kharif"
    aadhaar_linked: bool = True


@dataclass
class SessionState:
    profile: FarmerProfile
    history_uids: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    successes: list[bool] = field(default_factory=list)

    def record(self, query: str, api_uid: str, success: bool = True) -> None:
        self.queries.append(query)
        self.history_uids.append(api_uid)
        self.successes.append(success)


class SessionRerankPlus:
    """Session-aware reranker with Gemma 4 embeddings + co-activation cache."""

    def __init__(self, cache: CoactivationCache | None = None):
        self.apis: tuple[KarnatakaAPI, ...] = KARNATAKA_APIS
        self.cache = cache or CoactivationCache(
            rho=0.02, delta=1.0, in_session_decay=0.7
        )

        docs = [a.doc() for a in self.apis]
        self.doc_emb = gemma_embed.encode(docs)
        self.embedding_path = gemma_embed.last_path()
        self._seed_expert_chains()

    def _seed_expert_chains(self) -> None:
        """Expert-curated KSDA SOP-2024 chains seed the co-activation cache."""
        sops = [
            ["karnataka::uas_plantix_offline", "karnataka::ksda_pest_alert",
             "karnataka::raitha_mitra_advisory", "karnataka::pmfby_karnataka"],
            ["karnataka::ksndmc_weather", "karnataka::bhoomi_rtc",
             "karnataka::raitha_mitra_advisory", "karnataka::npk_calculator_kn"],
            ["karnataka::agmarknet_ka", "karnataka::rashtriya_e_market",
             "karnataka::bhoomi_rtc"],
            ["karnataka::bhoomi_rtc", "karnataka::pm_kisan_status"],
            ["karnataka::ksndmc_drought_monitor", "karnataka::ksndmc_weather",
             "karnataka::pmfby_karnataka", "karnataka::raitha_mitra_advisory"],
            ["karnataka::kmf_milk_price", "karnataka::shc_karnataka",
             "karnataka::npk_calculator_kn"],
            ["karnataka::shc_karnataka", "karnataka::npk_calculator_kn",
             "karnataka::raitha_mitra_advisory"],
        ]
        for chain in sops:
            for i, a in enumerate(chain):
                for b in chain[i + 1: i + 4]:
                    self.cache.update_edge(a, b, amount=3.0)

    def _profile_mask(self, profile: FarmerProfile) -> np.ndarray:
        mask = np.zeros(len(self.apis), dtype=np.float32)
        for i, a in enumerate(self.apis):
            score = 0.0
            if profile.season in a.seasons:
                score += 0.05
            if not a.crops or profile.crop in a.crops:
                score += 0.10
            else:
                score -= 0.20
            if not a.districts or profile.district in a.districts:
                score += 0.05
            if profile.language in a.languages:
                score += 0.05
            mask[i] = score
        return mask

    def rank(self, query: str, state: SessionState, top_k: int = 5):
        q_emb = gemma_embed.encode([query])[0]
        if q_emb.shape[-1] != self.doc_emb.shape[-1]:
            self.doc_emb = gemma_embed.encode([a.doc() for a in self.apis])
        self.embedding_path = gemma_embed.last_path()

        dense = self.doc_emb @ q_emb
        mask = self._profile_mask(state.profile)

        cache_score = np.zeros(len(self.apis), dtype=np.float32)
        if state.history_uids:
            for i, a in enumerate(self.apis):
                cache_score[i] = self.cache.score(state.history_uids, a.uid)

        total = 1.0 * dense + 1.0 * mask + 0.6 * cache_score
        order = np.argsort(-total)[:top_k]
        results = []
        for rank, idx in enumerate(order, start=1):
            results.append((self.apis[idx], {
                "rank": rank,
                "score_total": float(total[idx]),
                "score_dense": float(dense[idx]),
                "score_profile": float(mask[idx]),
                "score_cache": float(cache_score[idx]),
                "embedding_path": self.embedding_path,
            }))
        return results

    def record_success(self, state: SessionState, query: str, chosen_uid: str) -> None:
        for prev in state.history_uids:
            self.cache.update_edge(prev, chosen_uid)
        state.record(query, chosen_uid, success=True)
