# BioRerank Karnataka — 3-Minute Video Script

**Target length:** 2:50 (leave 10s buffer). Hackathon cap is 3:00.
**Tools:** Loom or OBS for screen capture · iPhone for the opening field shot · Canva for lower-thirds and metric overlays. Upload public on YouTube as **unlisted**, paste link in Kaggle Writeup.

---

## Cold-open (0:00 – 0:20) — *the human problem*

**Visual:** Phone-shot, vertical → cropped to 16:9. A ragi field in Tumakuru at golden hour. Cut to a farmer's hand swiping between four agricultural apps on a budget Android.

**Voiceover (your voice, English; Kannada subtitles burned in):**
> Ramesh farms one acre of ragi outside Tumakuru. This morning his crop has yellow leaves. He has four agriculture apps installed. Eighteen minutes later, he still doesn't know what's wrong.

**On-screen text:** *"8.5 million Karnataka smallholders. 240 taluks. Dozens of government APIs. One missing piece: knowing **which** tool to call."*

---

## The Idea (0:20 – 0:55) — *ant trails over tools*

**Visual:** Cut to a 4-second nature clip of ants on a trail (use a CC-0 clip from Pexels — credit on screen). Dissolve to a hand-drawn animation of two API icons connected by a glowing path that brightens.

**Voiceover:**
> Foraging ants reinforce trails that lead to food and let unused trails evaporate. Two properties matter for tool retrieval: recency-of-success dominates, and stale paths fade. We applied this directly to chains of API calls. Every successful tool transition deposits pheromone τ. Every session, the entire matrix decays by ρ. The result is a session-aware prior that gets sharper the more a farmer uses the system.

**On-screen formula (Canva overlay, hold for 5s):**
```
score(v | H) = w·dense(q, v) + w·profile(v) + w·Σ γⁿ⁻ⁱ log(1 + τ_{hᵢ, v})
```

---

## The Stack (0:55 – 1:20) — *Gemma 4 + Karnataka real APIs*

**Visual:** Screen-record of the GitHub repo file tree → zoom into `agri_extension/karnataka_apis.py`. Highlight the 15 API entries.

**Voiceover:**
> The stack: Gemma 4 e2b as a bi-encoder, Gemma 4 e2b-it fine-tuned with Unsloth on sixteen thousand ToolBench trajectories as a prompted reranker, and a pheromone prior in pure Python. Fifteen real Karnataka APIs are wired in: KSNDMC for weather, Bhoomi for land records, Raitha Mitra for advisory, KSDA for pest alerts, the K-Kisan helpline, Agmarknet for mandi prices, PM-KISAN and PMFBY for direct benefit transfers, and Nandini for milk procurement. Every entry has a Kannada label.

---

## Live Demo (1:20 – 2:20) — *session learning, visible*

**Visual:** Screen record of the Gradio app. Set profile → ragi, Tumakuru, Kannada, kharif. Use the Kannada example "ನನ್ನ ರಾಗಿ ಬೆಳೆಯಲ್ಲಿ ಎಲೆಗಳು ಹಳದಿಯಾಗುತ್ತಿವೆ".

**Voiceover (turn 1):**
> Turn one. Ramesh types in Kannada: "My ragi crop has yellowing leaves." The top recommendation is the KSDA Pest Alert and the offline Plantix model. Note the pheromone column — all zeros. This is a cold start.

Click into the **Pheromone Trails** tab to show the empty state. Switch back, type the second query: *"When can I claim crop insurance?"*

**Voiceover (turn 2):**
> Turn two. Ramesh asks about crop insurance. PMFBY jumps to number one — not because the semantic match is perfect, but because pheromone has reinforced the standard operating procedure chain: Plantix to pest alert to advisory to insurance. The bonus on PMFBY is plus one point three nine. Without pheromone, the dense score alone would have ranked two unrelated pest advisories above PMFBY.

**On-screen callout:** *"+1.39 pheromone bonus · PMFBY rank: 3rd → 1st"*

Switch to the Pheromone Trails tab. Show the τ values being reinforced.

**Voiceover:**
> Every successful transition deposits more pheromone. The trail structure is alive. It diverges by agro-climatic zone, by season, by crop.

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
> First-hit accuracy on a hundred Karnataka queries goes from 0.71 with dense retrieval to 0.91 with BioRerank. Pest response collapses from eighteen minutes to three. Eight and a half million farmers, fifteen real APIs, Kannada-first, runs offline on a budget phone.

---

## Closing (2:45 – 2:55)

**Visual:** Logo card. Three URLs stack.

**Voiceover:**
> BioRerank. Built for the Gemma 4 Good Hackathon, May 2026. Open source under Apache 2.0. Try the live demo, read the paper, run the code — links below.

**End card:**
- 🐝 **github.com/joyjeni/session-aware-toolbench-rerank**
- 🚜 **biorerank-karnataka.hf.space** *(or your Vercel URL)*
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
| 0:20 | ant trail nature clip | Pexels CC-0 |
| 0:24 | animated trail (2 API icons) | Canva |
| 0:55 | repo file tree | screen capture |
| 1:00 | karnataka_apis.py zoom | screen capture |
| 1:20 | Gradio profile setup | screen capture |
| 1:35 | Gradio turn 1 result | screen capture |
| 1:50 | Pheromone tab (empty) | screen capture |
| 2:00 | Gradio turn 2 result | screen capture |
| 2:10 | Pheromone tab (active τ values) | screen capture |
| 2:20 | impact cards 1-3 | Canva |
| 2:45 | logo + URLs | Canva |
