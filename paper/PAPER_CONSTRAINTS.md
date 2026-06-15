# CRC-SPAST Paper — Locked Constraints (from user, 15 Jun 2026)

## Authors (final, do not change)
1. **Jenisha T** — first author
   - PhD Student, Dept. of Computer Science and Engineering
   - Faculty of Engineering and Technology, MS Ramaiah University of Applied Sciences
   - Email: 24ETRP720001@msruas.ac.in
2. **Dr. Jyothi A P** — second author
   - Associate Professor and Programme Head (M&C)
   - Dept. of Computer Science and Engineering
   - Faculty of Engineering and Technology, MS Ramaiah University of Applied Sciences
   - Email: jyothi.cs.et@msruas.ac.in

## Title (locked — must match submitted abstract verbatim)
**SessionRerank+: A Session-Aware Re-Ranking for API Tool Retrieval in Agentic Large Language Model Systems**

(Note: spelled "plus" in the docx, but standard ASCII "+" is what CRC reviewers expect on title page.)

## Abstract (submitted — keep substantively unchanged; LIMITED edits only)
Source: paper/submitted_abstract.docx paragraphs 2-5.

Must keep these stated numbers if they are real:
- 1,100 queries × 3,211 candidate APIs across 49 RapidAPI categories
- Mean NDCG@5 = 0.516 (SessionRerank+) vs 0.451 (Dense baseline) vs 0.465 (ToolRerank)
- 14.7% relative gain
- Hit@5: 0.726 -> 0.771
- p < 1e-4 on 5 of 6 splits
- O(|H|) lookup complexity

⚠️ THESE NUMBERS MUST BE BACKED BY THE REAL v15 RUN. If v15 produces
different numbers, we must reconcile — update either the paper OR the
abstract footnote with a note "numbers refreshed in camera-ready vs
submitted abstract". Keep changes MINIMAL.

## Keywords (locked, from abstract)
Tool Retrieval, Agentic LLMs, Session Context, Re-Ranking, Co-activation Cache, ToolBench, Multi-Turn Interaction, Implicit Personalization, Information Retrieval

## References (already in submitted version: 10 entries on KV-cache /
## pruning topics)
The submitted reference list is about KV-cache / pruning — not aligned with
the tool-retrieval theme of the paper. We will ADD references on:
- ToolBench / ToolLLM (Qin et al.)
- ToolRerank
- Gemma 4 technical report (or Gemma 2)
- Cross-encoder reranking (SBERT)
- BM25 / Robertson
- Session-aware IR (Zamani / Bennett)
- Agentic benchmarks (OSWorld, WebArena, BrowseComp)

Keep ALL 10 originally cited entries (user may have submitted them for a
reason), append the additional topically-relevant ones → likely 18-20 refs
total. >10 satisfied; reviewers will see the original 10 plus topical ones.

## CRC template constraints (inherited)
- 5 pages MAX (final)
- 2-column, 8.7cm width, 1.06cm gutter
- Times New Roman 10pt, line spacing 1.1
- Author-year citations: (Qin et al., 2024)
- Figures: Fig. 1, Fig. 2, ... — embedded inline, cannot span columns
- Table I caption ABOVE table
- Equations numbered (1), (2)
- No "new" or "novel" in title (we use "+" — fine)
- Paragraph indent 0.3cm
- Abstract: 1cm indent, ≤150 words, NO citations, single paragraph
  -> The current abstract paragraphs 2-5 are ~400 words. We must COMPRESS
     to ≤150 words for CRC. Keep the headline framing + numbers; drop the
     methodological detail (it goes in Methods anyway).
