"""Gradio demo for BioRerank Karnataka — pheromone-guided agri tool routing.

Run locally
-----------
    pip install gradio sentence-transformers
    python agri_extension/gradio_demo.py
    # Then open http://localhost:7860 (or use share=True for a public URL)

Deploy
------
This file is also the entry point for the Hugging Face Space deployment.
Set ``share=True`` and Hugging Face will issue a permanent ``*.hf.space`` URL.

What the judges see
-------------------
- A farmer profile setup (crop / district / language / season).
- A chat surface that takes Kannada or English queries.
- After each turn, a *Pheromone Trails* panel showing the (api_a → api_b)
  pairs that just got reinforced. This is the visible "session learning"
  proof — it's the panel the video should zoom in on.
"""
from __future__ import annotations
import sys
from pathlib import Path

# Make agri_extension importable when run as a script
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import gradio as gr  # type: ignore

from agri_extension.biorerank_agri import (
    BioRerankAgri, FarmerProfile, SessionState,
)
from agri_extension.karnataka_apis import KARNATAKA_APIS

# Single shared model instance (cold-loaded once)
_BIO = BioRerankAgri()

# In-memory session per user (single-user demo; multi-user would need IDs)
_SESSION: dict = {"state": None}


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
def setup_profile(crop: str, district: str, language: str, season: str):
    profile = FarmerProfile(crop=crop, district=district, language=language, season=season)
    _SESSION["state"] = SessionState(profile=profile)
    return (
        f"✓ Profile set: {crop} farmer in {district} "
        f"({season}, language={language}). Session history cleared."
    )


def _format_ranked(ranked) -> str:
    lines = ["| Rank | API | Total | Dense | Profile | Pheromone |",
             "|---:|:---|---:|---:|---:|---:|"]
    for api, dbg in ranked:
        lines.append(
            f"| {dbg['rank']} | **{api.name}** ({api.name_kn}) — "
            f"_{api.provider}_ | {dbg['score_total']:+.3f} "
            f"| {dbg['score_dense']:+.3f} "
            f"| {dbg['score_profile']:+.2f} "
            f"| {dbg['score_pheromone']:+.2f} |"
        )
    return "\n".join(lines)


def _format_pheromone_summary(state: SessionState) -> str:
    if not state.history_uids:
        return "_No pheromone trails yet — first turn is a cold start._"
    summary = _BIO.pheromone.summary()
    lines = [
        f"**Total pheromone mass:** {summary['total_mass']:.2f}",
        f"**Unique source APIs with trails:** {summary['n_keys']}",
        f"**Avg outgoing trails per source:** {summary['avg_density']:.2f}",
        "",
        "**Recently reinforced transitions (this session):**",
    ]
    last_uid = state.history_uids[-1]
    row = _BIO.pheromone.pheromone.get(last_uid, {})
    if row:
        sorted_dests = sorted(row.items(), key=lambda x: -x[1])[:6]
        for dest_uid, tau in sorted_dests:
            dest_api = next((a for a in KARNATAKA_APIS if a.uid == dest_uid), None)
            label = dest_api.name if dest_api else dest_uid
            lines.append(f"- `{last_uid.split('::')[1]} → {dest_uid.split('::')[1]}` "
                         f"  τ = {tau:.2f}  ({label})")
    else:
        lines.append("- _No outgoing trails from the most recent API yet._")
    return "\n".join(lines)


def query_handler(message: str, chat_history):
    if _SESSION["state"] is None:
        return chat_history + [(message, "⚠️ Please set up your farmer profile first.")], "", ""

    state: SessionState = _SESSION["state"]
    ranked = _BIO.rank(message, state, top_k=5)

    # Auto-record the top-1 as the chosen tool (simulating a successful pick).
    # In a production deployment this would wait for the user to click "Use this".
    top_api, _ = ranked[0]
    _BIO.record_success(state, message, top_api.uid)

    bot_md = (
        f"**Top recommendation:** {top_api.name} ({top_api.name_kn})\n\n"
        f"_{top_api.description}_\n\n"
        f"→ [{top_api.provider}]({top_api.url})\n\n"
        f"---\n\n**Full ranking:**\n\n{_format_ranked(ranked)}"
    )
    chat_history = chat_history + [(message, bot_md)]
    return chat_history, "", _format_pheromone_summary(state)


def reset_session():
    if _SESSION["state"] is not None:
        profile = _SESSION["state"].profile
        _SESSION["state"] = SessionState(profile=profile)
    return [], "_Session history cleared. Profile retained._"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
KANNADA_EXAMPLES = [
    "ನನ್ನ ರಾಗಿ ಬೆಳೆಯಲ್ಲಿ ಎಲೆಗಳು ಹಳದಿಯಾಗುತ್ತಿವೆ",  # My ragi leaves are turning yellow
    "ಮುಂದಿನ ವಾರ ಮಳೆಯಾಗಲಿದೆಯೇ?",                   # Will it rain next week?
    "ಪಿಎಂ-ಕಿಸಾನ್ ಪಾವತಿ ಸ್ಥಿತಿ ಪರಿಶೀಲಿಸಿ",          # Check PM-KISAN payment status
    "ಟೊಮೆಟೊ ಎಪಿಎಂಸಿ ಬೆಲೆ ಬೆಂಗಳೂರು",                # Tomato APMC price Bangalore
    "ಫಸಲ್ ಬಿಮಾ ನೋಂದಣಿ ಗಡುವು ಯಾವಾಗ?",            # PMFBY enrolment cut-off
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
    title="BioRerank · Karnataka Agri",
    theme=gr.themes.Soft(primary_hue="green"),
) as demo:
    gr.Markdown(
        """
        # 🐝 BioRerank · Karnataka Smallholder Agri Assistant
        ### Pheromone-guided session-aware tool retrieval — powered by Gemma 4

        Ant-colony pheromone trails reinforce successful (api → api) chains; the
        ranking gets sharper the more a farmer interacts with the system.
        15 real Karnataka government and para-statal APIs are routed.
        """
    )

    with gr.Tab("1 · Set Profile"):
        with gr.Row():
            crop = gr.Dropdown(
                ["ragi", "paddy", "jowar", "tomato", "chilli", "cotton", "areca", "groundnut", "tur", "onion", "potato"],
                value="ragi", label="Primary crop",
            )
            district = gr.Dropdown(
                ["Tumakuru", "Mandya", "Mysuru", "Kolar", "Chikkaballapur", "Chitradurga",
                 "Hassan", "Belagavi", "Dharwad", "Koppal", "Raichur", "Ballari",
                 "Shivamogga", "Udupi", "Dakshina Kannada", "Uttara Kannada", "Bengaluru Rural"],
                value="Tumakuru", label="District",
            )
            language = gr.Dropdown(
                [("Kannada (ಕನ್ನಡ)", "kn"), ("English", "en"), ("Hindi (हिंदी)", "hi")],
                value="kn", label="Preferred language",
            )
            season = gr.Dropdown(["kharif", "rabi", "zaid"], value="kharif", label="Season")
        setup_btn = gr.Button("Save Profile", variant="primary")
        setup_status = gr.Textbox(label="Status", interactive=False)
        setup_btn.click(setup_profile,
                        inputs=[crop, district, language, season],
                        outputs=setup_status)

    with gr.Tab("2 · Ask the Assistant"):
        chat = gr.Chatbot(height=420, label="Conversation")
        msg = gr.Textbox(placeholder="Ask in Kannada, Hindi, or English…",
                         label="Your question", lines=2)
        with gr.Row():
            send_btn = gr.Button("Ask", variant="primary")
            reset_btn = gr.Button("Reset session")
        gr.Examples(KANNADA_EXAMPLES, msg, label="Kannada examples")
        gr.Examples(ENGLISH_EXAMPLES, msg, label="English examples")

    with gr.Tab("3 · Pheromone Trails (live)"):
        pher_md = gr.Markdown("_Run a few queries to see trails form._")

    # Wire events — chat updates the trails tab too
    send_btn.click(query_handler, [msg, chat], [chat, msg, pher_md])
    msg.submit(query_handler, [msg, chat], [chat, msg, pher_md])
    reset_btn.click(reset_session, outputs=[chat, pher_md])

    gr.Markdown(
        """
        ---
        Built for the **Gemma 4 Good Hackathon · Main Track** (May 2026) ·
        [GitHub](https://github.com/joyjeni/session-aware-toolbench-rerank) ·
        Apache 2.0
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
