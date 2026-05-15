# How to Submit to the Gemma 4 Good Hackathon

**Deadline:** May 18, 2026 · 11:59 PM PT (~3 days from May 15 IST)
**Competition page:** https://www.kaggle.com/competitions/gemma-4-good-hackathon

## Submission checklist

Kaggle hackathons accept submissions as **Kaggle Writeups** (a special kind of Notebook that's tied to the competition). Each successful submission needs:

| # | Artifact | Status | Location |
|---|----------|--------|----------|
| 1 | Accept rules & join competition | ☐ Click "Join Competition" | [competition page](https://www.kaggle.com/competitions/gemma-4-good-hackathon) |
| 2 | Public/visible GitHub repo | ✅ Done | https://github.com/joyjeni/session-aware-toolbench-rerank — flip to *public* before submitting |
| 3 | Reproducible Colab notebook | ✅ Done | `notebooks/gemma4_full_train.ipynb` |
| 4 | Writeup markdown | ✅ Done | `submission/kaggle_writeup.md` |
| 5 | Live demo URL | ✅ Done | https://web-three-blush-75.vercel.app |
| 6 | Demo video (3–5 min) on YouTube | ☐ Record using `submission/video_script.md` | YouTube (unlisted is fine) |
| 7 | Pick tracks | ☐ Tag inside writeup | Special Tech / Function Calling + Impact / Future of Education |
| 8 | Submit before May 18, 11:59 PM PT | ☐ | |

## Step-by-step submission

1. **Make the GitHub repo public.** In Kaggle, judges click through to read code; private repos block them.
   - Go to https://github.com/joyjeni/session-aware-toolbench-rerank/settings → "Change visibility" → Public.
   - (Optional) Add a license file — MIT is already declared in the writeup.

2. **Record the demo video** (3–5 minutes) using the shot list in [`submission/video_script.md`](video_script.md).
   - Upload to YouTube as **unlisted** (anyone with the link can watch, but it doesn't show up in search).
   - Title: `SessionRerank-Gemma — Gemma 4 Good Hackathon Submission`.
   - Description should include the GitHub + Vercel + Colab links so judges can click through.

3. **Run the Colab notebook once end-to-end.** This generates the real Gemma 4 numbers that fill in the *populated by Colab* rows of the writeup.
   - Free T4 takes ~50 min. Pro A100 takes ~15 min.
   - The last cell pushes results to GitHub → Vercel auto-redeploys → your dashboard now shows Gemma numbers.

4. **Save the Colab notebook to Kaggle.**
   - On the competition page click **Code → New Notebook**.
   - File → Import Notebook → upload `notebooks/gemma4_full_train.ipynb` (or paste the GitHub URL).
   - Click "Save Version" → "Save & Run All (Commit)". You don't actually need to run it on Kaggle — the act of saving as a notebook tied to the competition is what registers your code submission.

5. **Create the Writeup.**
   - Competition page → **Submit → Writeup**.
   - Paste the contents of `submission/kaggle_writeup.md`.
   - Replace the *populated by Colab* placeholders with the real numbers from your `results/metrics.json`.
   - Embed your YouTube video using Kaggle's *Insert → Video* button (paste YouTube URL).
   - Link your Colab notebook from step 4 via *Insert → Notebook*.
   - Title: **SessionRerank-Gemma: Session-Aware Tool Selection for Gemma 4 Agents**
   - Tags / Tracks: select **Special Technology — Function Calling** and **Impact — Future of Education**.

6. **Submit.** Click "Submit" before May 18, 11:59 PM PT.

## Multi-track entry

> "Your fine-tuned model can be submitted to multiple tracks, not just Unsloth, as long as it meets the criteria for each one." — [Daniel Han, Unsloth](https://www.linkedin.com/posts/danielhanchen_google-deepmind-is-hosting-a-gemma-4-hackathon-activity-7448365186417078272--udd)

Because we use Unsloth for the fine-tune, we also qualify for the **Unsloth $10K track**. Add `#unsloth` and reference Unsloth in the writeup (already done in §6 of `kaggle_writeup.md`). Some hackathon pages let you tag multiple tracks; if Kaggle only allows one primary, **submit twice** with the same notebook URL, each tagged for a different track.

## Common pitfalls

- ❌ Submitting only the GitHub repo without a Kaggle Writeup — judges only score writeups.
- ❌ Forgetting to flip the repo public.
- ❌ Video > 10 min — judges skim; under 5 min is ideal.
- ❌ Numbers in the writeup that don't match `results/metrics.json` — judges will check.
- ❌ Missing the "this uses Gemma 4" sentence in the first 30 seconds of the video.
