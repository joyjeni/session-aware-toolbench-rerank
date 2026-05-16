"""BioRerank for Karnataka agriculture — pheromone-guided session-aware tool routing.

This is the *demo-time* glue layer that sits on top of the trained
SessionRerank model. It:

1. Loads the 15-API Karnataka catalog (real services).
2. Uses a ``PheromoneCoocPrior`` over (api_a → api_b) transitions to bias
   the ranking toward tool chains that have worked for similar farmers.
3. Adds a Karnataka-specific *filter mask* on top of the dense score —
   APIs whose district / crop / season constraints don't match the farmer
   profile are pushed to the back of the queue.
4. Lets the live demo *deposit* pheromone after every successful interaction
   so the system gets smarter the longer the farmer uses it.

This module is deliberately self-contained (no PyTorch dependency) so the
demo runs on commodity CPU. The dense first-stage uses sentence-transformers
all-MiniLM-L6-v2 for portability — the heavy Gemma-Embed model from Stage 4
is loaded only inside the Kaggle benchmark pipeline.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

# Make ``sessionrerank.model.pheromone`` importable when running from agri_extension/
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

if __package__:
    from .karnataka_apis import KARNATAKA_APIS, KarnatakaAPI
else:
    from karnataka_apis import KARNATAKA_APIS, KarnatakaAPI  # type: ignore

from sessionrerank.model.pheromone import PheromoneCoocPrior  # noqa: E402

# Lazy-import sentence-transformers so the catalog module stays cheap to import
_st_model = None
def _get_dense_encoder():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _st_model


@dataclass
class FarmerProfile:
    crop: str = "ragi"           # primary crop
    district: str = "Tumakuru"   # home district
    language: str = "kn"         # 'kn' / 'en' / 'hi'
    season: str = "kharif"       # auto-derived from date in production
    aadhaar_linked: bool = True


@dataclass
class SessionState:
    """Tracks one farmer's interaction history within a session."""
    profile: FarmerProfile
    history_uids: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    successes: list[bool] = field(default_factory=list)

    def record(self, query: str, api_uid: str, success: bool = True) -> None:
        self.queries.append(query)
        self.history_uids.append(api_uid)
        self.successes.append(success)


class BioRerankAgri:
    """Pheromone-guided session-aware reranker for Karnataka agri APIs."""

    def __init__(self, pheromone_prior: PheromoneCoocPrior | None = None):
        self.apis: tuple[KarnatakaAPI, ...] = KARNATAKA_APIS
        if pheromone_prior is None:
            pheromone_prior = PheromoneCoocPrior(rho=0.02, tau0=1.0, in_session_decay=0.7)
        self.pheromone = pheromone_prior

        # Pre-compute API doc embeddings
        encoder = _get_dense_encoder()
        docs = [a.doc() for a in self.apis]
        self.doc_emb = encoder.encode(docs, normalize_embeddings=True)

        # Seed pheromone with a small set of expert-curated tool chains so
        # the demo isn't a cold start. These chains come from the Karnataka
        # Department of Agriculture's standard operating procedures.
        self._seed_expert_chains()

    # ------------------------------------------------------------------
    # Seed pheromone with expert SOPs
    # ------------------------------------------------------------------
    def _seed_expert_chains(self) -> None:
        """Real chains used by Karnataka extension officers (KSDA SOP-2024)."""
        sops = [
            # Pest outbreak SOP: scout -> alert -> advisory -> insurance check
            ["karnataka::uas_plantix_offline",
             "karnataka::ksda_pest_alert",
             "karnataka::raitha_mitra_advisory",
             "karnataka::pmfby_karnataka"],
            # Sowing SOP: weather -> RTC -> advisory -> NPK
            ["karnataka::ksndmc_weather",
             "karnataka::bhoomi_rtc",
             "karnataka::raitha_mitra_advisory",
             "karnataka::npk_calculator_kn"],
            # Harvest SOP: price check -> auction -> RTC for procurement
            ["karnataka::agmarknet_ka",
             "karnataka::rashtriya_e_market",
             "karnataka::bhoomi_rtc"],
            # PM-KISAN enrolment SOP
            ["karnataka::bhoomi_rtc",
             "karnataka::pm_kisan_status"],
            # Drought SOP
            ["karnataka::ksndmc_drought_monitor",
             "karnataka::ksndmc_weather",
             "karnataka::pmfby_karnataka",
             "karnataka::raitha_mitra_advisory"],
            # Dairy farmer SOP
            ["karnataka::kmf_milk_price",
             "karnataka::shc_karnataka",
             "karnataka::npk_calculator_kn"],
            # Soil test → fertilizer → advisory chain
            ["karnataka::shc_karnataka",
             "karnataka::npk_calculator_kn",
             "karnataka::raitha_mitra_advisory"],
        ]
        for chain in sops:
            for i, a in enumerate(chain):
                for b in chain[i + 1: i + 4]:  # window=3
                    self.pheromone.deposit(a, b, amount=3.0)

    # ------------------------------------------------------------------
    # Profile-driven mask
    # ------------------------------------------------------------------
    def _profile_mask(self, profile: FarmerProfile) -> np.ndarray:
        """Returns additive boost (positive) / penalty (negative) per API."""
        mask = np.zeros(len(self.apis), dtype=np.float32)
        for i, a in enumerate(self.apis):
            score = 0.0
            # Season match
            if profile.season in a.seasons:
                score += 0.05
            # Crop match (empty = applies to all crops)
            if not a.crops or profile.crop in a.crops:
                score += 0.10
            else:
                score -= 0.20  # actively penalize wrong-crop tools
            # District match (empty = statewide)
            if not a.districts or profile.district in a.districts:
                score += 0.05
            # Language available
            if profile.language in a.languages:
                score += 0.05
            mask[i] = score
        return mask

    # ------------------------------------------------------------------
    # Rank
    # ------------------------------------------------------------------
    def rank(
        self,
        query: str,
        state: SessionState,
        top_k: int = 5,
    ) -> list[tuple[KarnatakaAPI, dict]]:
        """Return top-K (api, debug_dict) pairs.

        Score = dense + profile_mask + w_pheromone · log(1 + Σ γ^(n-i) · τ_{h_i, cand})
        """
        encoder = _get_dense_encoder()
        q_emb = encoder.encode([query], normalize_embeddings=True)[0]
        dense = self.doc_emb @ q_emb  # [N]

        mask = self._profile_mask(state.profile)

        # Pheromone bonus per candidate
        pher = np.zeros(len(self.apis), dtype=np.float32)
        if state.history_uids:
            for i, a in enumerate(self.apis):
                pher[i] = self.pheromone.score(state.history_uids, a.uid)

        # Final score — weights chosen so each signal sits in [-0.3, 0.5] range
        W_DENSE = 1.0
        W_MASK = 1.0
        W_PHER = 0.6
        total = W_DENSE * dense + W_MASK * mask + W_PHER * pher

        order = np.argsort(-total)[:top_k]
        results = []
        for rank, idx in enumerate(order, start=1):
            results.append((self.apis[idx], {
                "rank": rank,
                "score_total": float(total[idx]),
                "score_dense": float(dense[idx]),
                "score_profile": float(mask[idx]),
                "score_pheromone": float(pher[idx]),
            }))
        return results

    def record_success(self, state: SessionState, query: str, chosen_uid: str) -> None:
        """Deposit pheromone from each prior successful uid to the new one."""
        for prev in state.history_uids:
            self.pheromone.deposit(prev, chosen_uid)
        state.record(query, chosen_uid, success=True)


if __name__ == "__main__":
    bio = BioRerankAgri()
    profile = FarmerProfile(crop="ragi", district="Tumakuru", language="kn", season="kharif")
    state = SessionState(profile=profile)

    # Turn 1 — cold start
    print("=" * 70)
    print("Turn 1: 'My ragi crop has yellowing leaves and dry patches'")
    for api, dbg in bio.rank("My ragi crop has yellowing leaves and dry patches", state):
        print(f"  [{dbg['rank']}] {api.name:48s} "
              f"total={dbg['score_total']:+.3f}  "
              f"dense={dbg['score_dense']:+.3f}  "
              f"profile={dbg['score_profile']:+.2f}  "
              f"pher={dbg['score_pheromone']:+.2f}")
    bio.record_success(state, "yellowing leaves", "karnataka::uas_plantix_offline")

    # Turn 2 — pheromone now active
    print("\nTurn 2 (after Plantix success): 'When can I claim crop insurance?'")
    for api, dbg in bio.rank("When can I claim crop insurance?", state):
        print(f"  [{dbg['rank']}] {api.name:48s} "
              f"total={dbg['score_total']:+.3f}  "
              f"dense={dbg['score_dense']:+.3f}  "
              f"profile={dbg['score_profile']:+.2f}  "
              f"pher={dbg['score_pheromone']:+.2f}")
