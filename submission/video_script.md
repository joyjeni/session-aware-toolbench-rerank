# Demo Video Script — SessionRerank-Gemma

**Target length: 4 minutes** (range 3:30–5:00). Hackathon norm based on prior submissions:
- *CodeCare* (Future of Education entry): ~4 min
- *Punto Medio*: 2:57
- *AI Study Plan Generator*: 4:44

Going over 5 minutes hurts you — judges skim. Under 3 minutes feels rushed.

---

## Required content (Kaggle hackathon norms)

Every successful hackathon demo video covers these six beats. Don't skip any:

1. **What is it?** (one sentence, on screen by 0:15)
2. **Who is it for?** (the impact track audience)
3. **Why is it new / why Gemma?** (the technical contribution)
4. **Live demo** (the dashboard or model running — this is the biggest chunk)
5. **Results** (one chart or table on screen)
6. **Links / call to action** (GitHub, Colab, dashboard URLs on screen)

---

## Shot-by-shot script (4:00 total)

### 0:00 – 0:20  ·  Hook & one-liner

**On screen:** Your face (webcam) + title card "SessionRerank-Gemma · Gemma 4 Good Hackathon".
**Voiceover:**
> "Hi, I'm Abigail. Modern AI agents have to pick the right API at every turn — and most of them forget what they just used five seconds ago. We built **SessionRerank-Gemma**: a session-aware reranker that gives Gemma 4 agents a memory of their tool history. It improves API retrieval by **14.7%** NDCG@5 over ToolLLM's baseline on the full ToolBench benchmark."

### 0:20 – 0:45  ·  The problem (Impact framing)

**On screen:** A whiteboard sketch or slide showing a tutor agent's session: "translate this French passage" → "now read it aloud" → "now quiz me". Highlight the second turn going to a Mandarin TTS by mistake.
**Voiceover:**
> "I run a STEM training company called Abigail Creations. Our AI tutors hold *long* sessions with students — sometimes 30 turns about French maths. Today's retrievers treat every turn as independent. So after 20 minutes of French, a student saying 'read this aloud' can still get routed to a Mandarin TTS API. That's a broken tutor."

### 0:45 – 1:15  ·  The method (Special Technology framing)

**On screen:** Show the equation slide from the dashboard ([the `.math` block on the live site](https://web-three-blush-75.vercel.app)):
```
score(q, a | H) = w_base · sim(q, a)
                + w_cat  · cat_affinity(a, H)
                + w_sch  · schema_affinity(a, H)
                + w_ept  · endpoint_affinity(a, H)
```
**Voiceover:**
> "Our reranker adds a learned history prior on top of any dense retriever. Three signals — category affinity, parameter-schema overlap, and endpoint host affinity — each with exponential decay over the session. Weights trained with InfoNCE in five thousand ToolBench sessions. Critically: when history is empty, the prior is zero — so single-turn performance is identical to the baseline. No regressions."

### 1:15 – 1:35  ·  Gemma 4 integration (function-calling track)

**On screen:** Code snippet from `src/sessionrerank/baselines/gemma_embed.py` showing the mean-pool encoder. Then `finetune_gemma_unsloth.py` thumbnail.
**Voiceover:**
> "We ship three Gemma 4 integrations. One: Gemma 4 as a bi-encoder via mean-pooled hidden states. Two: Gemma 4 as a prompted cross-encoder using YES/NO logits. Three: a LoRA fine-tune of Gemma 4 270M with Unsloth — half an hour on a free T4. All three plug into Gemma 4's native function-calling format. The 270 million parameter model with our prior beats the 1-billion baseline. That's the on-device story."

### 1:35 – 2:55  ·  **Live demo** (the meat — 80 seconds)

**On screen:** Screen-record of https://web-three-blush-75.vercel.app

- Scroll past KPI cards quickly — pause on "+14.7%" for 1 sec.
- Scroll to the results table — point cursor at the SessionRerank column being green.
- Scroll to **Live Demo panel**.
- Type query: `"Translate this French sentence and read the result aloud"`.
- Show the **baseline** column (left) returning some generic results.
- Tick **3 APIs from the "Translation" category** in the history picker.
- Click **"Run rerank"**.
- Point cursor at how the right column **promotes a TTS API into the top 3** that's compatible with French — and at the ▲ green badges showing rank moves.

**Voiceover (over the demo):**
> "Here's the live dashboard, deployed on Vercel. Same query both sides — but on the right I've told the system 'I've used three translation APIs already in this session.' Watch what happens: the speech-synthesis API that supports the right language jumps from rank seven to rank two. The green up-arrows show how the history changed the ranking. The baseline doesn't know any of that — it sees a stateless single-turn query. Our reranker sees a *conversation*."

### 2:55 – 3:35  ·  Results

**On screen:** Show Figure 1 (NDCG@5 across splits) and Figure 4 (relative gain vs Dense) full-screen.
**Voiceover:**
> "Full benchmark on the official ToolBench eval set — 1,100 queries across 3,211 APIs. Mean NDCG@5 gain over the ToolLLM dense baseline: **plus 14.7 percent**. Peak gain on multi-tool sessions, where history matters most: **plus 20.7 percent**. Wilcoxon paired test, p less than 1e-4. Compared against BM25, ToolRerank, and now three Gemma 4 variants — all numbers reproduced by one Colab notebook on a free T4."

### 3:35 – 3:55  ·  Impact close

**On screen:** Slide with three use-cases:
1. Bilingual maths tutor (Education)
2. Lab notebook assistant (Education)
3. On-device tutor without internet (Digital Equity)

**Voiceover:**
> "This unlocks on-device tutoring. A 270-million-parameter Gemma 4 with our prior fits on a low-end Android tablet — so students in low-connectivity regions get an agentic AI tutor that actually remembers their lesson. That's the *good* in Gemma for Good."

### 3:55 – 4:00  ·  Links / CTA

**On screen:** End card showing:
- 💻 github.com/joyjeni/session-aware-toolbench-rerank
- 📓 Colab: gemma4_full_train.ipynb
- 🌐 web-three-blush-75.vercel.app

**Voiceover:**
> "All code, notebook, and the live dashboard linked in the description. Thank you."

---

## Production tips

- **Tool**: OBS Studio (free) or Loom for screen + webcam picture-in-picture.
- **Voice**: Record one continuous voiceover, then edit visuals to match — much faster than recording each shot live.
- **Music**: optional, low-volume, royalty-free (YouTube Audio Library — *Inspiring Cinematic Ambient* is a safe pick).
- **Captions**: Auto-caption on upload, then quick edit for technical terms ("NDCG", "InfoNCE").
- **Resolution**: 1080p minimum.
- **Aspect ratio**: 16:9 landscape.
- **Thumbnail**: A screenshot of Figure 1 with "+14.7% on Gemma 4" overlaid in big text.

## Recording checklist (in order)

1. ☐ Run the Colab notebook end-to-end first so the dashboard shows real Gemma numbers when you demo it.
2. ☐ Write the voiceover script in full (use this file).
3. ☐ Record the voiceover (single take, ~4 min).
4. ☐ Screen-record the dashboard demo (single take, no narration).
5. ☐ Slide deck: 5 slides — title, problem, method, results table/chart, end card with links.
6. ☐ Edit in your tool of choice — sync voiceover to visuals.
7. ☐ Export 1080p MP4.
8. ☐ Upload to YouTube as **Unlisted**.
9. ☐ Title: "SessionRerank-Gemma — Gemma 4 Good Hackathon Submission"
10. ☐ Description: paste GitHub + Colab + Vercel URLs and a 2-sentence summary.
11. ☐ Embed the YouTube link inside your Kaggle Writeup before submitting.

## Two-sentence YouTube description template

```
SessionRerank-Gemma is our submission to the Gemma 4 Good Hackathon (Special Technology / Function Calling + Impact / Future of Education tracks). We give Gemma 4 agents a session-history prior that improves API retrieval by +14.7% NDCG@5 on ToolBench — enabling on-device AI tutors that remember the conversation.

🔗 Code: https://github.com/joyjeni/session-aware-toolbench-rerank
🔗 Colab: https://github.com/joyjeni/session-aware-toolbench-rerank/blob/main/notebooks/gemma4_full_train.ipynb
🔗 Live demo: https://web-three-blush-75.vercel.app
```
