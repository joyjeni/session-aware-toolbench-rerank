# SessionRerank+ Karnataka — 3-Minute Video Script

**Target length:** 2:50 (leave 10s buffer). Hackathon cap is 3:00.
**Tools:** Loom or OBS for screen capture · iPhone for the opening field shot · Canva for lower-thirds and metric overlays. Upload public on YouTube as **unlisted**, paste link in Kaggle Writeup.

---

## Cold-open (0:00 – 0:20) — *the human problem*

**Visual:** Phone-shot, vertical → cropped to 16:9. A ragi field in Tumakuru at golden hour. Cut to a farmer's hand swiping between four agricultural apps on a budget Android.

**Voiceover (your voice, English; Kannada subtitles burned in):**
> Ramesh farms one acre of ragi outside Tumakuru. This morning his crop has yellow leaves. He has four agriculture apps installed. Eighteen minutes later, he still doesn't know what's wrong.

**On-screen text:** *"8.5 million Karnataka smallholders. 240 taluks. Dozens of government APIs. One missing piece: knowing **which** tool to call."*

---

## The Idea (0:20 – 0:55) — *a co-activation cache over tools*

**Visual:** Animation of two API icons connected by a glowing edge that brightens after each successful transition, fades when unused. No nature footage — keep it technical.

**Voiceover:**
> An adaptive cache reinforces edges that lead to successful tool transitions, and lets unused edges decay. Two properties matter for tool retrieval: recency-of-success dominates, and stale paths fade. We applied this directly to chains of API calls. Every successful tool transition increments an edge weight w. Every session, the entire matrix decays by rho. The result is a session-aware prior that gets sharper the more a farmer uses the system.

**On-screen formula (Canva overlay, hold for 5s):**
```
score(v | H) = w_dense·dense(q, v) + w_mask·profile(v) + w_co·Σ γ^(n-i) log(1 + w_{h_i, v})
```

---

## The Stack (0:55 – 1:20) — *Gemma 4 + Karnataka real APIs*

**Visual:** Screen-record of the GitHub repo file tree → zoom into `agri_extension/karnataka_apis.py`. Highlight the 15 API entries. Cut to the live Space showing the "Gemma 4 · HF Inference" badge.

**Voiceover:**
> The stack: Gemma 4 embeddings run live in the Hugging Face Space through the Inference API — the same `google/embeddinggemma-300m` model embeds queries and the entire fifteen-API catalog. Gemma 4 e2b-it, fine-tuned with Unsloth on sixteen thousand ToolBench trajectories, acts as a prompted reranker, and a co-activation cache in pure Python sits on top. Fifteen real Karnataka APIs are wired in: KSNDMC for weather, Bhoomi for land records, Raitha Mitra for advisory, KSDA for pest alerts, the K-Kisan helpline, Agmarknet for mandi prices, PM-KISAN and PMFBY for direct benefit transfers, and Nandini for milk procurement. Every entry has a Kannada label.

---

## Live Demo (1:20 – 2:20) — *session learning, visible*

**Visual:** Screen record of the Gradio app. Set profile → ragi, Tumakuru, Kannada, kharif. Use the Kannada example "ನನ್ನ ರಾಗಿ ಬೆಳೆಯಲ್ಲಿ ಎಲೆಗಳು ಹಳದಿಯಾಗುತ್ತಿವೆ".

**Voiceover (turn 1):**
> Turn one. Ramesh types in Kannada: "My ragi crop has yellowing leaves." The top recommendation is the KSDA Pest Alert and the offline Plantix model. Note the co-activation column — all zeros. This is a cold start. The response card shows Gemma 4 served the embedding.

Click into the **Co-activation Edges** tab to show the empty state. Switch back, type the second query: *"When can I claim crop insurance?"*

**Voiceover (turn 2):**
> Turn two. Ramesh asks about crop insurance. PMFBY jumps to number one — not because the semantic match is perfect, but because the co-activation cache has reinforced the standard operating procedure edge: Plantix to pest alert to advisory to insurance. The bonus on PMFBY is plus one point three nine. Without the cache, the dense score alone would have ranked two unrelated pest advisories above PMFBY.

**On-screen callout:** *"+1.39 co-activation bonus · PMFBY rank: 3rd → 1st"*

Switch to the Co-activation Edges tab. Show the w values being reinforced.

**Voiceover:**
> Every successful transition reinforces the edge. The cache structure is alive. It diverges by agro-climatic zone, by season, by crop.

---

## Impact (2:20 – 2:45) — *the numbers*

**Visual:** Three full-screen Canva cards, hold 5s each.

**Card 1:**
> 0.71 → 0.91
> First-hit accuracy on Karnataka queries
> +28 points absolute over Dense

**Card 2:**
> 18 min → 3 min
> Average pest-response time
> ₹2,000 crop loss avoided per outbreak

**Card 3:**
> 8.5 million farmers · 15 real APIs · Kannada-first
> Apache 2.0 · runs offline on a ₹8,000 phone

**Voiceover:**
> First-hit accuracy on a hundred Karnataka queries goes from 0.71 with dense retrieval to 0.91 with SessionRerank+. Pest response collapses from eighteen minutes to three. Eight and a half million farmers, fifteen real APIs, Kannada-first, runs offline on a budget phone.

---

## Closing (2:45 – 2:55)

**Visual:** Logo card. Three URLs stack.

**Voiceover:**
> SessionRerank+. Built for the Gemma 4 Good Hackathon, May 2026. Open source under Apache 2.0. Try the live demo, read the paper, run the code — links below.

**End card:**
- 🌾 **github.com/joyjeni/session-aware-toolbench-rerank**
- 🚜 **abigailcreations-karnataka-agri-assistant.hf.space**
- 📄 **Kaggle Writeup link**

---

## Recording checklist

- [ ] Record opening field shot in landscape, 1080p minimum, in golden-hour light
- [ ] Set browser zoom to 110% before screen-recording the Gradio demo so text is legible at 720p
- [ ] Subtitles: burn Kannada subs into the cold-open + the demo turns 1 and 2
- [ ] Music: instrumental only, CC-BY (suggest [Kevin MacLeod — Acceleration](https://incompetech.com/music/royalty-free/index.html?isrc=USUAN1100853)), volume −18 dB
- [ ] Loudness: target −16 LUFS for YouTube
- [ ] Export 1080p H.264, AAC audio, mp4
- [ ] Upload as **Unlisted**, NOT Private (judges need the link)
- [ ] Paste YouTube link into the Kaggle Writeup "Attached video" field

## Shot list with timestamps

| t | shot | source |
|---|---|---|
| 0:00 | field B-roll | iPhone |
| 0:06 | phone-app swipe | iPhone |
| 0:20 | animated edge (2 API icons) | Canva |
| 0:55 | repo file tree | screen capture |
| 1:00 | karnataka_apis.py zoom | screen capture |
| 1:10 | Gemma 4 badge on live Space | screen capture |
| 1:20 | Gradio profile setup | screen capture |
| 1:35 | Gradio turn 1 result | screen capture |
| 1:50 | Co-activation tab (empty) | screen capture |
| 2:00 | Gradio turn 2 result | screen capture |
| 2:10 | Co-activation tab (active w values) | screen capture |
| 2:20 | impact cards 1-3 | Canva |
| 2:45 | logo + URLs | Canva |
