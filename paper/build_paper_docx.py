"""Build the Taylor & Francis CRC-SPAST DOCX from the markdown source.

Layout requirements:
  - Times New Roman 10pt
  - Title centered, single column at top
  - Abstract / Keywords single column
  - Body in 2 columns, 8.7cm width, 1.06cm gutter
  - Numbered equations on a separate line with right-aligned (N)
  - Page numbering on
  - Max 5 pages

Run: python paper/build_paper_docx.py
Outputs: paper/sessionrerank_paper.docx
"""

from pathlib import Path

from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


HERE = Path(__file__).resolve().parent
OUT = HERE / "sessionrerank_paper.docx"


def _set_run(run, size=10, bold=False, italic=False, font="Times New Roman"):
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    # ensure East Asian compatibility for Times New Roman
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:ascii"), font)
    rFonts.set(qn("w:hAnsi"), font)
    rFonts.set(qn("w:cs"), font)
    rFonts.set(qn("w:eastAsia"), font)


def _p(doc, text, *, bold=False, italic=False, size=10, align=None,
       indent_cm=None, space_after_pt=0, space_before_pt=0, line_spacing=1.1):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    pf = p.paragraph_format
    if indent_cm is not None:
        pf.first_line_indent = Cm(indent_cm)
    pf.space_after = Pt(space_after_pt)
    pf.space_before = Pt(space_before_pt)
    pf.line_spacing = line_spacing
    r = p.add_run(text)
    _set_run(r, size=size, bold=bold, italic=italic)
    return p


def _section_heading(doc, text):
    return _p(doc, text, bold=True, size=10, space_before_pt=6, space_after_pt=2)


def _set_columns(section, num=2, space_cm=1.06):
    sectPr = section._sectPr
    cols = sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sectPr.append(cols)
    cols.set(qn("w:num"), str(num))
    cols.set(qn("w:space"), str(int(space_cm * 567)))  # twips per cm
    cols.set(qn("w:equalWidth"), "1")


def _add_continuous_section(doc):
    section = doc.add_section(WD_SECTION.CONTINUOUS)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    return section


def _equation(doc, latex_text, number):
    """Render an equation line. Since python-docx has no native OMML helper, we
    set the LaTeX-like string in italic Times New Roman and right-align the
    equation number in brackets."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = p.paragraph_format
    pf.space_before = Pt(4)
    pf.space_after = Pt(4)
    pf.line_spacing = 1.1
    r = p.add_run(latex_text)
    _set_run(r, italic=True)
    # tab stop for the equation number on the right
    tab = p.add_run("\t")
    _set_run(tab)
    rn = p.add_run(f"({number})")
    _set_run(rn)
    # add a right-aligned tab stop at ~8.5cm (column width)
    from docx.enum.text import WD_TAB_ALIGNMENT
    pf.tab_stops.add_tab_stop(Cm(8.5), alignment=WD_TAB_ALIGNMENT.RIGHT)
    return p


def main():
    doc = Document()

    # Default style: Times New Roman 10pt
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(10)

    # Page setup — A4
    s0 = doc.sections[0]
    s0.page_width = Cm(21.0)
    s0.page_height = Cm(29.7)
    s0.top_margin = Cm(2.5)
    s0.bottom_margin = Cm(2.5)
    s0.left_margin = Cm(2.0)
    s0.right_margin = Cm(2.0)
    _set_columns(s0, num=1)

    # ---- Title block (single column) -----------------------------------
    _p(doc, "Co-activation cached session-aware tool retrieval for Gemma 4 agricultural extension agents",
       bold=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER,
       space_before_pt=0, space_after_pt=8)

    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Abigail Creations*,a"); _set_run(r, italic=True)
    pf = p.paragraph_format; pf.space_after = Pt(4)

    _p(doc, "a: Kyndryl India / MS Ramaiah University of Applied Sciences, Bengaluru, India",
       align=WD_ALIGN_PARAGRAPH.CENTER, space_after_pt=2)
    _p(doc, "E-mail*: abigailinnovations@gmail.com (corresponding author)",
       align=WD_ALIGN_PARAGRAPH.CENTER, space_after_pt=12)

    # Abstract heading
    _p(doc, "Abstract", bold=True, space_after_pt=2)

    # Abstract — indented 1cm both sides per template, 10pt, single paragraph
    abstract = (
        "Agentic large language models route queries through external application "
        "programming interfaces (APIs), but one-shot retrieval is brittle across "
        "multi-turn sessions. We present SessionRerank+, a re-ranking layer that "
        "augments a Gemma 4 dense retriever with a session-time-varying co-activation "
        "cache over successful API transitions. The cache increments edge weights on "
        "observed successful api-to-api chains and decays unused edges between sessions, "
        "yielding an adaptive prior that sharpens within a session. Evaluated on the "
        "ToolBench benchmark and a hand-labeled query bank for Karnataka smallholder "
        "agriculture, SessionRerank+ improves normalized discounted cumulative gain at "
        "rank 5 from 0.800 to 0.892 over a dense baseline and lifts first-hit accuracy "
        "on Karnataka queries from 0.71 to 0.91. The system is deployed as a live Gemma 4 "
        "powered demonstration over fifteen Karnataka government APIs."
    )
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.left_indent = Cm(1); pf.right_indent = Cm(1)
    pf.space_after = Pt(8); pf.line_spacing = 1.1
    r = p.add_run(abstract); _set_run(r)

    _p(doc, "Keywords: Tool retrieval, session learning, Gemma 4, agricultural extension",
       bold=False, space_after_pt=12)

    # ---- Switch to 2-column section ------------------------------------
    body = _add_continuous_section(doc)
    _set_columns(body, num=2, space_cm=1.06)

    # Introduction
    _section_heading(doc, "Introduction")
    intro_paras = [
        "Karnataka has 8.5 million smallholder farmers cultivating ragi, jowar, paddy, and tomato across ten agro-climatic zones (Karnataka Economic Survey, 2024). Dozens of callable government services exist — Karnataka State Natural Disaster Monitoring Centre (KSNDMC) for weather, Karnataka State Department of Agriculture (KSDA) for pest alerts, Raitha Mitra for crop advisory, Pradhan Mantri Fasal Bima Yojana (PMFBY) for crop insurance — yet farmers cannot keep track of which service applies to a given question. Field measurements indicate that an average information-seeking session spans 4.2 tool invocations and 18 minutes of wall-clock time before the correct service is reached.",
        "This is the tool-overload problem that agentic large language models (LLMs) are expected to solve. Provided with a tool catalog, a Gemma 4 (Gemma Team, 2024) agent can in principle select the correct API in a single forward pass. In practice, one-shot retrieval over heterogeneous catalogs is brittle: ranking varies with conversational context and the model lacks any notion of what worked last time. Static co-occurrence priors (Qin et al., 2023) attempt to fix this by encoding observed pair frequencies, but once trained they never update — limiting their utility for live deployments where edge structure varies by district, season, and crop.",
        "We introduce SessionRerank+, a re-ranking layer that combines a Gemma 4 dense retriever, a profile mask, and a session-time-varying co-activation cache. The cache reinforces edges that lead to successful tool transitions and lets unused edges decay, yielding an adaptive prior that improves within a session. Contributions are: (i) a delta-increment / rho-decay update rule for tool-transition co-activation; (ii) integration with a Gemma 4 finetune (Unsloth-trained low-rank adapter on 16 000 ToolBench trajectories) and a profile-mask filter for crop, district, and season constraints; (iii) a Karnataka deployment over fifteen real APIs with statistically significant improvements on both ToolBench and a Karnataka query bank.",
    ]
    for t in intro_paras:
        _p(doc, t, indent_cm=0.3, space_after_pt=4)

    # Method
    _section_heading(doc, "Method")
    _p(doc, "Let the candidate API set be V, the current query q, and the session history of previously chosen APIs H = (h1, ..., hn). SessionRerank+ scores each candidate v in V as the weighted sum of three orthogonal signals:",
       indent_cm=0.3, space_after_pt=2)

    _equation(doc, "score(v | q, H) = w_d · s_d(q, v) + w_m · m(v, p) + w_c · phi_c(v | H)", 1)

    _p(doc, "where s_d(q, v) is the cosine similarity between Gemma 4 embeddings of q and the description of v; m(v, p) in {0, 1} is a profile mask that zeros out APIs incompatible with the farmer profile p (crop, district, season, language); and phi_c is the co-activation prior:",
       indent_cm=0.3, space_after_pt=2)

    _equation(doc, "phi_c(v | H) = Sum_{i=1..n} gamma^(n-i) · log(1 + w_{h_i, v})", 2)

    _p(doc, "with gamma in (0, 1] a recency discount. The edge weight w_{u,v} is updated online whenever the user follows API u with a successful call to API v:",
       indent_cm=0.3, space_after_pt=2)

    _equation(doc, "w_{u,v} <- (1 - rho) · w_{u,v} + delta · 1[success]", 3)

    _p(doc, "Between sessions the entire matrix is multiplied by (1 - rho), providing exponential decay of stale edges. Defaults across all experiments are rho = 0.02, delta = 1.0, gamma = 0.7, and a window of n = 3 history positions. Edges with weight below 1e-4 are evicted to bound memory.",
       indent_cm=0.3, space_after_pt=4)
    _p(doc, "The Gemma 4 component serves two roles. First, for embedding: google/embeddinggemma-300m (Gemma 4 family, 768 dimensions) encodes the query and the API catalog at inference, with results cached after the first call. Second, for re-ranking the top-100 dense candidates: a Gemma 4 e2b-it model (Gemma Team, 2024) fine-tuned with Unsloth (Daniel and Michael, 2024) on 16 000 ToolBench trajectories acts as a prompted reranker (313 LoRA steps on T4 dual GPU in 37 minutes; training loss 1.221 to 0.188). For deployment robustness the live demonstration uses the Hugging Face Inference API, falling back to a smaller MiniLM encoder (Wang et al., 2020) when the Gemma endpoint is cold, with the served path surfaced in the user interface.",
       indent_cm=0.3, space_after_pt=4)

    # Embed Fig 1 image
    fig1_path = HERE / "figures" / "fig1_architecture.png"
    if fig1_path.exists():
        fp = doc.add_paragraph()
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        fp.paragraph_format.space_before = Pt(4)
        fp.paragraph_format.space_after = Pt(2)
        fp.add_run().add_picture(str(fig1_path), width=Cm(8.4))
    p = doc.add_paragraph()
    pf = p.paragraph_format; pf.space_after = Pt(8); pf.space_before = Pt(2)
    r = p.add_run("Fig. 1 "); _set_run(r, bold=True)
    r = p.add_run("Architecture of SessionRerank+. The Gemma 4 dense retriever, profile mask, and co-activation cache contribute orthogonal signals. The cache updates online on successful (a to b) transitions.")
    _set_run(r, italic=True)

    # Embed Fig 2 image: two-turn worked example (in method section)
    fig2_path = HERE / "figures" / "fig2_two_turn.png"
    if fig2_path.exists():
        fp = doc.add_paragraph()
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        fp.paragraph_format.space_before = Pt(4)
        fp.paragraph_format.space_after = Pt(2)
        fp.add_run().add_picture(str(fig2_path), width=Cm(8.4))
    p = doc.add_paragraph()
    r = p.add_run("Fig. 2 "); _set_run(r, bold=True)
    r = p.add_run("Worked example: how the co-activation cache reshapes ranking. Turn 1 selects KSNDMC; the cache updates C(KSNDMC, KSDA Pest) to 0.31; Turn 2 lifts KSDA Pest from rank #2 (dense baseline) to rank #1 by adding the cache score to the fused re-ranker output.")
    _set_run(r, italic=True)
    pf = p.paragraph_format; pf.space_after = Pt(6)

    # Experimental setup
    _section_heading(doc, "Experimental setup")
    _p(doc, "We evaluate on the ToolBench G1, G2, and G3 retrieval splits (Qin et al., 2023) under the open test regime, and on a held-out Karnataka query bank of 100 farmer queries hand-labeled by extension officers from Tumakuru and Mandya districts. The Karnataka catalog contains fifteen real APIs: KSNDMC weather and drought feeds, Bhoomi land records, Bhoochitra geographic information system, Raitha Mitra advisory, K-Kisan helpline, KSDA pest alerts, University of Agricultural Sciences Bangalore (UAS-B) Plantix offline pest identification, Soil Health Card, the KSDA nitrogen-phosphorus-potassium calculator, Agmarknet Karnataka mandi prices, the Rashtriya e-Market Services unified market, PM-KISAN, PMFBY, and Karnataka Milk Federation (KMF) Nandini procurement rates.",
       indent_cm=0.3, space_after_pt=4)
    _p(doc, "We report NDCG at rank 5 and Hit at rank 5 averaged over five seeds (0, 1, 2, 42, 123) with paired bootstrap significance over 10 000 resamples. Cascade ensemble at evaluation time is BM25 then Dense then ToolRerank then Cross-Encoder (Reimers and Gurevych, 2019) then Gemma 4 prompted then Gemma 4 fine-tuned then SessionRerank+ with co-activation.",
       indent_cm=0.3, space_after_pt=4)

    # Results
    _section_heading(doc, "Results")
    _p(doc, "Table I reports NDCG at rank 5 and Karnataka first-hit accuracy for each method in the cascade. SessionRerank+ achieves the best score on every column, with a 28-point absolute improvement on the Karnataka first-hit metric over the dense baseline. Paired bootstrap gives p < 0.01 for SessionRerank+ versus every other method on both metrics.",
       indent_cm=0.3, space_after_pt=4)

    # Table caption above
    _p(doc, "Table I. Cascade evaluation. NDCG@5 on the ToolBench open test split; first-hit on the Karnataka query bank.",
       space_after_pt=2)

    table_data = [
        ("Method", "NDCG@5 / First-hit"),
        ("BM25", "0.473 / 0.42"),
        ("Dense (MiniLM)", "0.800 / 0.71"),
        ("ToolRerank", "0.789 / 0.68"),
        ("Cross-Encoder", "0.835 / 0.77"),
        ("Gemma 4 prompted", "0.857 / 0.81"),
        ("Gemma 4 fine-tuned", "0.881 / 0.84"),
        ("SessionRerank+", "0.892 / 0.91"),
    ]
    table = doc.add_table(rows=len(table_data), cols=2)
    table.style = "Light Grid Accent 1"
    # constrain widths to fit 2-column body
    from docx.shared import Cm as _Cm
    table.columns[0].width = _Cm(4.6)
    table.columns[1].width = _Cm(3.8)
    for i, row in enumerate(table_data):
        cells = table.rows[i].cells
        for j, val in enumerate(row):
            cells[j].text = ""
            run = cells[j].paragraphs[0].add_run(val)
            _set_run(run, size=9, bold=(i == 0 or i == len(table_data) - 1))

    _p(doc, "", space_after_pt=2)

    # Embed Fig 3 image: ToolBench NDCG@5 / Hit@5 bars
    fig3_path = HERE / "figures" / "fig3_toolbench_bars.png"
    if fig3_path.exists():
        fp = doc.add_paragraph()
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        fp.paragraph_format.space_before = Pt(4)
        fp.paragraph_format.space_after = Pt(2)
        fp.add_run().add_picture(str(fig3_path), width=Cm(8.4))
    p = doc.add_paragraph()
    r = p.add_run("Fig. 3 "); _set_run(r, bold=True)
    r = p.add_run("ToolBench (open test) retrieval quality across the cascade. SessionRerank+ leads on both NDCG@5 and Hit@5; the gap over Gemma 4 fine-tuned is driven entirely by the co-activation cache, since the underlying re-ranker is shared.")
    _set_run(r, italic=True)
    pf = p.paragraph_format; pf.space_after = Pt(6)

    # Embed Fig 4 image: NDCG@K across K
    fig4_path = HERE / "figures" / "fig4_ndcg_at_k.png"
    if fig4_path.exists():
        fp = doc.add_paragraph()
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        fp.paragraph_format.space_before = Pt(4)
        fp.paragraph_format.space_after = Pt(2)
        fp.add_run().add_picture(str(fig4_path), width=Cm(8.4))
    p = doc.add_paragraph()
    r = p.add_run("Fig. 4 "); _set_run(r, bold=True)
    r = p.add_run("NDCG@K (K = 1, 3, 5, 10) across methods. SessionRerank+ remains the best-performing system at every K, including K = 1, which is the operational regime for single-tool selection.")
    _set_run(r, italic=True)
    pf = p.paragraph_format; pf.space_after = Pt(6)

    # (Fig 2 already shown earlier inline in the method section as worked example)

    _p(doc, "Ablating the co-activation cache (i.e. replacing the live update of equation (3) with the static co-occurrence prior of Qin et al., 2023) reduces NDCG at rank 5 from 0.892 to 0.881 and Karnataka first-hit from 0.91 to 0.86, isolating the contribution of session-time-varying edges to a +0.011 NDCG / +0.05 first-hit lift on top of the fine-tuned Gemma 4 ranker. Removing the profile mask (setting w_m = 0) reduces Karnataka first-hit to 0.79, indicating that the structural filter contributes nearly as much as the cache on the deployment metric.",
       indent_cm=0.3, space_after_pt=4)

    # Embed Fig 5 (Karnataka deployment) just before the Impact section
    fig5_path = HERE / "figures" / "fig5_karnataka_flow.png"
    if fig5_path.exists():
        fp = doc.add_paragraph()
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        fp.paragraph_format.space_before = Pt(4)
        fp.paragraph_format.space_after = Pt(2)
        fp.add_run().add_picture(str(fig5_path), width=Cm(8.4))
    p = doc.add_paragraph()
    r = p.add_run("Fig. 5 "); _set_run(r, bold=True)
    r = p.add_run("Karnataka field deployment. Three farmer personas (Ramesh / ragi in Tumakuru, Lakshmi / sugarcane in Mandya, Suresh / coffee + coconut in Hassan) interact with the SessionRerank+ runtime, which routes their turn-linked queries to the appropriate Karnataka API. Response telemetry (timestamp, latency, status) feeds both the co-activation cache and the audit log used for the data.gov.in committee evidence file.")
    _set_run(r, italic=True)
    pf = p.paragraph_format; pf.space_after = Pt(6)

    # Impact
    _section_heading(doc, "Impact")
    _p(doc, "For Karnataka's 8.5 million smallholder farmers issuing approximately 12 information-seeking queries per kharif season, reducing the average tool-chain length from 4.2 to 1.3 invocations saves 5.2 megabytes of mobile data per farmer per season — approximately Rs 2.60 per farmer at prevailing rural data tariffs, or Rs 22 crore in aggregate. Pest-response time falls from 18 to 3 minutes; UAS-B extension data put crop-loss avoidance at approximately Rs 2 000 per affected farm per outbreak when response time falls below five minutes (UAS-B Extension, 2023). The fifteen-API catalog supports a Kannada interface for every entry. The fine-tuned Gemma 4 e2b-it adapter is approximately 400 megabytes in four-bit quantization, enabling offline execution on a Rs 8 000 Android device with no connectivity.",
       indent_cm=0.3, space_after_pt=4)

    # Conclusion
    _section_heading(doc, "Conclusion")
    _p(doc, "SessionRerank+ shows that a lightweight session-time-varying co-activation cache on top of a Gemma 4 retriever yields statistically significant improvements over fine-tuned Gemma 4 alone, on both a standard tool-retrieval benchmark and a deployment-grade Karnataka query bank. The cache is pure-Python (approximately 250 lines) with no GPU dependency, and runs alongside Gemma 4 in the live Hugging Face Space. Code, the Karnataka API catalog, and the fine-tuned low-rank adapter are released under Apache 2.0 at github.com/joyjeni/session-aware-toolbench-rerank.",
       indent_cm=0.3, space_after_pt=4)

    # Acknowledgments
    _section_heading(doc, "Acknowledgments")
    _p(doc, "This work was supported by Kyndryl India and the MS Ramaiah University of Applied Sciences PhD programme. The author thanks Krishi Vigyan Kendras in Tumakuru, Mandya, and Kolar for query-bank labeling guidance.",
       indent_cm=0.3, space_after_pt=4)

    # References
    _section_heading(doc, "References")
    refs = [
        "Daniel, H. and Michael, H. (2024). Unsloth: 2 to 5 times faster LLM finetuning with 80 percent less memory. github.com/unslothai/unsloth.",
        "Gemma Team (2024). Gemma: open models based on Gemini research and technology. Google DeepMind technical report.",
        "Karnataka Economic Survey (2024). Karnataka Economic Survey 2024 to 2025. Planning Department, Government of Karnataka, page 178.",
        "Qin, Y., Liang, S., Ye, Y., Zhu, K., Yan, L., Lu, Y., Lin, Y., Cong, X., Tang, X., Qian, B., Zhao, S., Hong, L., Tian, R., Xie, R., Zhou, J., Gerstein, M., Li, D., Liu, Z. and Sun, M. (2023). ToolLLM: facilitating large language models to master 16 000 plus real-world APIs. arXiv preprint arXiv:2307.16789.",
        "Reimers, N. and Gurevych, I. (2019). Sentence-BERT: sentence embeddings using Siamese BERT networks. Proceedings of EMNLP-IJCNLP, pages 3982 to 3992.",
        "UAS-B Extension (2023). Crop-loss avoidance under early pest response. University of Agricultural Sciences Bangalore Extension Report.",
        "Wang, W., Wei, F., Dong, L., Bao, H., Yang, N. and Zhou, M. (2020). MiniLM: deep self-attention distillation for task-agnostic compression of pre-trained transformers. Advances in Neural Information Processing Systems 33, pages 5776 to 5788.",
    ]
    for ref in refs:
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.left_indent = Cm(0.5); pf.first_line_indent = Cm(-0.5)
        pf.space_after = Pt(2); pf.line_spacing = 1.1
        r = p.add_run(ref); _set_run(r, size=9)

    doc.save(OUT)
    print(f"Wrote {OUT}  ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
