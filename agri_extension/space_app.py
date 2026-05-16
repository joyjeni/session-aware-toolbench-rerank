"""SessionRerank+ · Karnataka — Hugging Face Space entry point.

Session-aware tool retrieval over 15 real Karnataka government / para-statal
agricultural APIs, with **Gemma 4** powering the query embedder and a
co-activation cache providing the session-time adaptation signal.

Built for the Gemma 4 Good Hackathon (Main Track), May 2026.
"""
from __future__ import annotations
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import datetime
import gradio as gr  # type: ignore

from sessionrerank_plus import SessionRerankPlus, FarmerProfile, SessionState
from karnataka_apis import KARNATAKA_APIS

# Single shared model instance (cold-loaded once)
_RANKER = SessionRerankPlus()

# --- Season auto-detection (India / Karnataka calendar) -------------------
#   Kharif : Jun – Oct  (monsoon sowing → autumn harvest)
#   Rabi   : Nov – Mar  (post-monsoon sowing → spring harvest)
#   Zaid   : Apr – May  (summer, irrigation-only short cycle)
# Labels match what KSDA / Raitha Mitra extension officers print on field cards.

SEASON_CHOICES = [
    ("Kharif — Monsoon Sowing (Jun–Oct)", "kharif"),
    ("Rabi — Winter / Harvest Season (Nov–Mar)", "rabi"),
    ("Zaid — Summer Short Cycle (Apr–May)", "zaid"),
]


def detect_season(today: datetime.date | None = None) -> str:
    """Return 'kharif' | 'rabi' | 'zaid' for the given date (defaults to today, IST)."""
    if today is None:
        # IST = UTC + 5:30; use naive IST for stability inside the Space.
        today = (datetime.datetime.utcnow() + datetime.timedelta(hours=5, minutes=30)).date()
    m = today.month
    if 6 <= m <= 10:
        return "kharif"  # Jun–Oct: monsoon sowing
    if m >= 11 or m <= 3:
        return "rabi"    # Nov–Mar: winter / harvest
    return "zaid"        # Apr–May: summer short cycle


def season_banner(season: str) -> str:
    today = (datetime.datetime.utcnow() + datetime.timedelta(hours=5, minutes=30)).date()
    blurbs = {
        "kharif": (
            "🌧️ **Kharif — Monsoon Sowing Season.** Southwest monsoon brings rain; "
            "farmers sow ragi, paddy, jowar, maize, cotton, tur. Harvest in Oct–Nov."
        ),
        "rabi": (
            "❄️ **Rabi — Winter & Harvest Season.** Sown on residual soil moisture + irrigation. "
            "Wheat, chickpea, mustard, rabi jowar. Harvest in Feb–Apr."
        ),
        "zaid": (
            "☀️ **Zaid — Summer Short Cycle.** 100% irrigation; only borewell / canal-fed farms. "
            "Watermelon, cucumber, summer moong, fodder crops."
        ),
    }
    detected = detect_season(today)
    suffix = "" if season == detected else (
        f"  \n_Today is **{today.strftime('%d %b %Y')}** → auto-detected season is "
        f"**{detected}**. You have manually overridden to **{season}**._"
    )
    return blurbs[season] + suffix

# In-memory single-user demo session
_SESSION: dict = {"state": None}


def setup_profile(crop: str, district: str, language: str, season: str):
    profile = FarmerProfile(crop=crop, district=district, language=language, season=season)
    _SESSION["state"] = SessionState(profile=profile)
    return (
        f"✓ Profile saved: {crop} farmer in {district} "
        f"({season}, language={language}). Session history cleared."
    )


def _embedder_badge(path: str) -> str:
    if path == "gemma":
        return "🟢 Gemma 4 (embeddinggemma-300m)"
    if path == "minilm-fallback":
        return "🟡 MiniLM fallback (HF Gemma endpoint cold — Gemma still used for doc embeddings)"
    return "⚪ embedder not yet called"


def _format_ranked(ranked) -> str:
    lines = [
        "| Rank | API | Total | Dense (Gemma 4) | Profile | Co-activation |",
        "|---:|:---|---:|---:|---:|---:|",
    ]
    for api, dbg in ranked:
        lines.append(
            f"| {dbg['rank']} | **{api.name}** ({api.name_kn}) — "
            f"_{api.provider}_ | {dbg['score_total']:+.3f} "
            f"| {dbg['score_dense']:+.3f} "
            f"| {dbg['score_profile']:+.2f} "
            f"| {dbg['score_cache']:+.2f} |"
        )
    return "\n".join(lines)


def _format_cache_summary(state: SessionState) -> str:
    if not state.history_uids:
        return "_No session edges yet — Turn 1 is a cold start._"
    summary = _RANKER.cache.summary()
    lines = [
        f"**Total edge weight:** {summary['total_mass']:.2f}",
        f"**Unique source APIs with edges:** {summary['n_keys']}",
        f"**Avg outgoing edges per source:** {summary['avg_density']:.2f}",
        "",
        "**Edges reinforced this session (from the most recent API):**",
    ]
    last_uid = state.history_uids[-1]
    row = _RANKER.cache.edges.get(last_uid, {})
    if row:
        for dest_uid, w in sorted(row.items(), key=lambda x: -x[1])[:6]:
            dest_api = next((a for a in KARNATAKA_APIS if a.uid == dest_uid), None)
            label = dest_api.name if dest_api else dest_uid
            lines.append(
                f"- `{last_uid.split('::')[1]} → {dest_uid.split('::')[1]}` "
                f"  w = {w:.2f}  ({label})"
            )
    else:
        lines.append("- _No outgoing edges from the most recent API yet._")
    return "\n".join(lines)


def query_handler(message: str, chat_history):
    if _SESSION["state"] is None:
        chat_history = chat_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "⚠️ Please set up your farmer profile first."},
        ]
        return chat_history, "", ""

    state: SessionState = _SESSION["state"]
    ranked = _RANKER.rank(message, state, top_k=5)
    top_api, top_dbg = ranked[0]
    _RANKER.record_success(state, message, top_api.uid)

    badge = _embedder_badge(top_dbg["embedding_path"])
    bot_md = (
        f"**Embedder used:** {badge}\n\n"
        f"**Top recommendation:** {top_api.name} ({top_api.name_kn})\n\n"
        f"_{top_api.description}_\n\n"
        f"→ [{top_api.provider}]({top_api.url})\n\n"
        f"---\n\n**Full ranking:**\n\n{_format_ranked(ranked)}"
    )
    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": bot_md},
    ]
    return chat_history, "", _format_cache_summary(state)


def reset_session():
    if _SESSION["state"] is not None:
        profile = _SESSION["state"].profile
        _SESSION["state"] = SessionState(profile=profile)
    return [], "_Session history cleared. Profile retained._"


KANNADA_EXAMPLES = [
    "ನನ್ನ ರಾಗಿ ಬೆಳೆಯಲ್ಲಿ ಎಲೆಗಳು ಹಳದಿಯಾಗುತ್ತಿವೆ",
    "ಮುಂದಿನ ವಾರ ಮಳೆಯಾಗಲಿದೆಯೇ?",
    "ಪಿಎಂ-ಕಿಸಾನ್ ಪಾವತಿ ಸ್ಥಿತಿ ಪರಿಶೀಲಿಸಿ",
    "ಟೊಮೆಟೊ ಎಪಿಎಂಸಿ ಬೆಲೆ ಬೆಂಗಳೂರು",
    "ಫಸಲ್ ಬಿಮಾ ನೋಂದಣಿ ಗಡುವು ಯಾವಾಗ?",
]

ENGLISH_EXAMPLES = [
    "My ragi crop has yellowing leaves and dry patches",
    "When is the right sowing window for kharif paddy in Mandya?",
    "Check my PM-KISAN payment status",
    "What is today's tomato modal price in Kolar APMC?",
    "Soil test shows low nitrogen — how much urea per acre for ragi?",
    "Whitefly outbreak in cotton — what should I do?",
]


with gr.Blocks(
    title="SessionRerank+ · Karnataka Agri",
    theme=gr.themes.Soft(primary_hue="green"),
) as demo:
    gr.Markdown(
        """
        # SessionRerank+ · Karnataka Smallholder Agri Assistant
        ### Session-aware tool retrieval — powered by **Gemma 4** + a learned co-activation cache

        Gemma 4 (`google/embeddinggemma-300m`) embeds the farmer query and the
        15 Karnataka API documents. A learned **co-activation cache** of
        (api_a → api_b) transitions then reinforces successful tool chains
        during the session, so the ranking gets sharper the more a farmer
        interacts with the system.
        """
    )

    with gr.Tab("1 · Set Profile"):
        with gr.Row():
            crop = gr.Dropdown(
                ["ragi", "paddy", "jowar", "tomato", "chilli", "cotton",
                 "areca", "groundnut", "tur", "onion", "potato"],
                value="ragi", label="Primary crop",
            )
            district = gr.Dropdown(
                ["Tumakuru", "Mandya", "Mysuru", "Kolar", "Chikkaballapur",
                 "Chitradurga", "Hassan", "Belagavi", "Dharwad", "Koppal",
                 "Raichur", "Ballari", "Shivamogga", "Udupi",
                 "Dakshina Kannada", "Uttara Kannada", "Bengaluru Rural"],
                value="Tumakuru", label="District",
            )
            language = gr.Dropdown(
                [("Kannada (ಕನ್ನಡ)", "kn"), ("English", "en"), ("Hindi (हिंदी)", "hi")],
                value="kn", label="Preferred language",
            )
            season = gr.Dropdown(
                SEASON_CHOICES,
                value=detect_season(),
                label=f"Season (auto-detected from today's date — IST)",
            )
        season_info = gr.Markdown(season_banner(detect_season()))
        season.change(season_banner, inputs=season, outputs=season_info)
        setup_btn = gr.Button("Save Profile", variant="primary")
        setup_status = gr.Textbox(label="Status", interactive=False)
        setup_btn.click(setup_profile,
                        inputs=[crop, district, language, season],
                        outputs=setup_status)

    with gr.Tab("2 · Ask the Assistant"):
        chat = gr.Chatbot(height=420, label="Conversation", type="messages")
        msg = gr.Textbox(placeholder="Ask in Kannada, Hindi, or English…",
                         label="Your question", lines=2)
        with gr.Row():
            send_btn = gr.Button("Ask", variant="primary")
            reset_btn = gr.Button("Reset session")
        gr.Examples(KANNADA_EXAMPLES, msg, label="Kannada examples")
        gr.Examples(ENGLISH_EXAMPLES, msg, label="English examples")

    with gr.Tab("3 · Co-activation edges (live)"):
        cache_md = gr.Markdown("_Run a few queries to see edges form._")

    send_btn.click(query_handler, [msg, chat], [chat, msg, cache_md])
    msg.submit(query_handler, [msg, chat], [chat, msg, cache_md])
    reset_btn.click(reset_session, outputs=[chat, cache_md])

    gr.Markdown(
        """
        ---
        Built for the **Gemma 4 Good Hackathon · Main Track** (May 2026) ·
        [GitHub](https://github.com/joyjeni/session-aware-toolbench-rerank) ·
        Apache 2.0
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
