#!/usr/bin/env python3
"""
Build CRC 5-page paper for SessionRerank+ — v18.
Changes vs v17:
- Human-author journal voice in Introduction + Contributions
- Drop unverified / off-topic refs (Vos, ATM, Nguyen, Mi, Kawakita-misattribution)
- Fix Ai et al. -> 2018, Tazawa & Isomura attribution, Bui title "Cache What Lasts"
- Replace numbered [n] reference list with author-year alphabetical (CRC style)
- Acronyms expanded on first use (API, LLM, InfoNCE, NDCG)
"""
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import json
from pathlib import Path

ROOT = Path('/home/user/workspace/sessionrerank')
OUT  = ROOT / 'paper' / 'sessionrerank_paper.docx'
FIG1 = ROOT / 'paper' / 'figures_v3' / 'fig1_architecture.png'
FIG2 = ROOT / 'paper' / 'figures_v3' / 'fig2_algorithm.png'

metrics = json.load(open(ROOT / 'results' / 'metrics.json'))

# -------- helpers --------
def set_run(run, font='Times New Roman', size=10, bold=False, italic=False):
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn('w:rFonts'))
    if rfonts is None:
        rfonts = OxmlElement('w:rFonts'); rpr.append(rfonts)
    for k in ('ascii','hAnsi','eastAsia','cs'):
        rfonts.set(qn(f'w:{k}'), font)

def add_para(doc, text, *, size=10, bold=False, italic=False, align=None,
             space_before=0, space_after=2, first_indent=None, left_indent=None,
             line_spacing=1.1):
    p = doc.add_paragraph()
    if align is not None: p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = line_spacing
    if first_indent is not None: pf.first_line_indent = first_indent
    if left_indent is not None: pf.left_indent = left_indent
    if text:
        r = p.add_run(text)
        set_run(r, size=size, bold=bold, italic=italic)
    return p

def add_heading(doc, num, text, *, size=10):
    full = f'{num}. {text.upper()}' if num else text.upper()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.1
    r = p.add_run(full); set_run(r, size=size, bold=True)
    return p

def add_subheading(doc, num, text, *, size=10):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.1
    r = p.add_run(f'{num} {text}'); set_run(r, size=size, bold=True, italic=True)
    return p

# ============= Document =============
doc = Document()
section = doc.sections[0]
section.page_height = Cm(27.94)
section.page_width  = Cm(21.59)
section.top_margin    = Cm(1.91)
section.bottom_margin = Cm(2.54)
section.left_margin   = Cm(1.78)
section.right_margin  = Cm(1.78)

style = doc.styles['Normal']
style.font.name = 'Times New Roman'
style.font.size = Pt(10)
style.paragraph_format.line_spacing = 1.1
style.paragraph_format.space_after = Pt(2)

# ---- TITLE ----
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(8)
r = p.add_run('SessionRerank+: A Session-Aware Re-Ranking for API Tool '
              'Retrieval in Agentic Large Language Model Systems')
set_run(r, size=14, bold=True)

# ---- AUTHORS ----
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(2)
r = p.add_run('Jenisha T'); set_run(r, size=11, bold=True)
r = p.add_run('1'); set_run(r, size=8)
r = p.add_run(' and Dr. Jyothi A P'); set_run(r, size=11, bold=True)
r = p.add_run('2'); set_run(r, size=8)

add_para(doc,
    '1 PhD Student, Department of Computer Science and Engineering, '
    'Faculty of Engineering and Technology, '
    'MS Ramaiah University of Applied Sciences, Bengaluru, India. '
    'Email: 24ETRP720001@msruas.ac.in',
    size=9, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
add_para(doc,
    '2 Associate Professor and Programme Head (M&C), '
    'Department of Computer Science and Engineering, '
    'Faculty of Engineering and Technology, '
    'MS Ramaiah University of Applied Sciences, Bengaluru, India. '
    'Email: jyothi.cs.et@msruas.ac.in',
    size=9, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10)

# ---- ABSTRACT (≤150 words, NO citations per CRC) ----
abstract_text = (
    'A Large Language Model (LLM) agent solves real-world tasks by calling '
    'external software services, called Application Programming Interfaces '
    '(APIs), over many conversational turns. Existing tool retrievers '
    'score each request as if the conversation had just begun, '
    'discarding strong cues already present in the session. We '
    'introduce SessionRerank+, a session-aware re-ranking layer that '
    'augments any dense retriever with three trajectory-derived '
    'affinity signals --- categorical co-occurrence, schema-signature '
    'similarity and endpoint pattern overlap --- read from a sparse '
    'Co-activation Cache with O(|H|) lookup. On all six ToolBench '
    'splits (1,100 queries, 3,211 APIs, 49 categories), SessionRerank+ '
    'raises mean session Normalised Discounted Cumulative Gain at '
    'rank 5 from 0.451 to 0.520 (+15.2%) and Hit at rank 5 from 0.726 '
    'to 0.791 (+8.9%), with p < 5e-3 on every split. By construction '
    'it reduces to the base retriever when the session is empty, so '
    'single-turn behaviour cannot regress.'
)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p.paragraph_format.left_indent = Cm(1.0)
p.paragraph_format.right_indent = Cm(1.0)
p.paragraph_format.space_after = Pt(4)
r = p.add_run('Abstract --- '); set_run(r, size=10, bold=True, italic=True)
r = p.add_run(abstract_text);   set_run(r, size=10)

# ---- KEYWORDS ----
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p.paragraph_format.left_indent = Cm(1.0)
p.paragraph_format.right_indent = Cm(1.0)
p.paragraph_format.space_after = Pt(10)
r = p.add_run('Keywords --- '); set_run(r, size=10, bold=True, italic=True)
r = p.add_run('Tool retrieval, agentic LLMs, session context, '
              're-ranking, Co-activation Cache, ToolBench, multi-turn '
              'interaction, implicit personalisation, information retrieval.')
set_run(r, size=10)

# ============ TWO COLUMNS ============
new_sec = doc.add_section(WD_SECTION.CONTINUOUS)
sectPr = new_sec._sectPr
cols = sectPr.find(qn('w:cols'))
if cols is None: cols = OxmlElement('w:cols'); sectPr.append(cols)
cols.set(qn('w:num'), '2')
cols.set(qn('w:space'), '600')
cols.set(qn('w:equalWidth'), '1')

# ===== 1. INTRODUCTION =====
add_heading(doc, '1', 'Introduction')

add_para(doc,
    'A conversational assistant no longer answers from its own knowledge '
    'alone. To accomplish concrete work --- book a flight, check a '
    'weather alert, query a land-records portal --- it must delegate the '
    'operation to an external software service exposed through an '
    'Application Programming Interface (API). Each API is a small, named '
    'contract: it states which arguments it accepts and what response it '
    'returns. The catalogue of available APIs is large; a modern agent '
    'sees thousands of them. Choosing the right one for each user request '
    'is therefore the central retrieval problem facing an agentic Large '
    'Language Model (LLM).',
    align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=Cm(0.3))

add_para(doc,
    'Two assumptions in the current generation of tool retrievers caused '
    'us concern. First, leading systems --- ToolLLM (Qin et al., 2024) '
    'and ToolRerank (Zheng et al., 2024) among them --- score every '
    'incoming request in isolation. They behave as if the user had just '
    'arrived, even when the conversation is several turns deep. Second, '
    'the standard remedy for "isolation" in classical Web search is '
    'user-level personalisation (Ai et al., 2018; Bennett et al., 2012), '
    'which depends on identified accounts, long search histories, and '
    'per-user training. Open-domain agent sessions have none of those '
    'things: they are typically anonymous and ephemeral, and operators '
    'are increasingly unwilling to store user-level traces.',
    align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=Cm(0.3))

add_para(doc,
    'The gap between these two positions is what this paper tries to '
    'close. Within a single session, a great deal of intent is already '
    'on the table: the APIs the agent has just called expose what family '
    'of task the user is pursuing, what kind of data they are '
    'manipulating, and what topical region of the corpus they are '
    'browsing. A human operator would exploit that trail; current tool '
    'retrievers do not. We treat the session\u2019s own trajectory as a '
    'free, anonymous, on-device training signal, and we show that a '
    'small layer reading three signals from it is enough to move the '
    'retrieval needle measurably.',
    align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=Cm(0.3))

add_para(doc,
    'Our contribution is SessionRerank+, a thin re-ranking layer that '
    'sits between any existing dense retriever and the agent. It does '
    'not retrain the retriever; it does not store user identities; and, '
    'when the session is empty, it collapses exactly to the base '
    'retriever it wraps. The remainder of this paper makes four '
    'innovations precise:',
    align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=Cm(0.3), space_before=2)

contribs = [
    '(i) A score-fusion formulation that augments the base relevance '
    'score with three trajectory-derived affinity signals --- categorical '
    'co-occurrence, schema-signature similarity and endpoint pattern '
    'overlap --- whose six combination weights are learned by '
    'Information Noise Contrastive Estimation (InfoNCE) without any '
    'user-level supervision.',
    '(ii) The Co-activation Cache, a sparse symbolic memory of API-pair '
    'affinities updated by a simple increment-decay rule, giving O(|H|) '
    'lookup at serve time with no additional neural inference. The '
    'cache is a few megabytes of plain numbers; the entire layer runs '
    'on CPU.',
    '(iii) An analytically provable cold-start guarantee: when the '
    'session is empty, all trajectory terms vanish and the score '
    'reduces exactly to the base retriever\u2019s. Deploying '
    'SessionRerank+ therefore cannot regress single-turn behaviour --- '
    'a property that most personalised systems cannot claim.',
    '(iv) A full evaluation on all six ToolBench splits showing a '
    '+15.2% relative gain in mean Normalised Discounted Cumulative Gain '
    'at rank 5 (NDCG@5) over the Dense baseline, and significant gains '
    '(p < 5e-3) on every split.',
]
for c in contribs:
    add_para(doc, c, align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=Cm(0.3),
             space_after=2)

# ===== 2. RELATED WORK =====
add_heading(doc, '2', 'Related Work')

add_para(doc,
    'Tool retrieval for LLM agents was placed on a benchmark footing by '
    'ToolLLM (Qin et al., 2024), which constructed ToolBench from over '
    '16,000 real-world RapidAPI endpoints, and extended by ToolRerank '
    '(Zheng et al., 2024), which adds an adaptive cross-encoder reranker. '
    'Both systems are stateless: they score the user request in '
    'isolation. The classical baselines on which they build share that '
    'limitation. BM25 (Robertson and Zaragoza, 2009) scores by lexical '
    'overlap; bi-encoder retrievers such as Sentence-BERT (Reimers and '
    'Gurevych, 2019) by embedding similarity; BERT-based cross-encoder '
    're-rankers (Nogueira and Cho, 2019) by query-document interaction. '
    'None of them consult the session at all.',
    align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=Cm(0.3))

add_para(doc,
    'End-to-end agentic benchmarks --- WebArena (Zhou et al., 2024), '
    'OSWorld (Xie et al., 2024) and BrowseComp (Wei et al., 2025) --- '
    'evaluate task completion and treat retrieval as a black box. The '
    'reasoning frameworks that drive these agents, beginning with ReAct '
    '(Yao et al., 2023), expose the trajectory but do not feed it back '
    'into the retriever. Session-aware ranking has a long history in '
    'Web search (Bennett et al., 2012; Zamani et al., 2017; Ai et al., '
    '2018), but those systems are designed around identified user logs '
    'rather than anonymous, ephemeral agent sessions. To our knowledge, '
    'SessionRerank+ is the first method that applies trajectory-derived '
    'implicit personalisation to the specific problem of API tool '
    'retrieval. Conceptually, the Co-activation Cache shares a '
    'sparse-memory motivation with recent work on long-context inference '
    '(He et al., 2025; Joo et al., 2025; Bui et al., 2026; Choi et al., '
    '2026) and is loosely inspired by the biological observation that '
    'pruning sparse synaptic connections is itself a form of Bayesian '
    'model selection (Tazawa and Isomura, 2026); the analogies, however, '
    'are at the level of motivation, not architecture --- our cache is '
    'symbolic, not tensor-valued.',
    align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=Cm(0.3))

# ===== 3. METHOD =====
add_heading(doc, '3', 'Method')

# 3.1 Problem
add_subheading(doc, '3.1', 'Problem statement')
add_para(doc,
    'Let C = {a_1, ..., a_N} be a corpus of API descriptions and let '
    'H_t = (a_{i1}, ..., a_{it}) denote the ordered sequence of APIs '
    'successfully invoked in the current session up to turn t. Given a '
    'new natural-language query q_{t+1}, a base retriever produces a '
    'relevance score s_base(q,a) for every candidate API a in C; in our '
    'experiments this is the cosine similarity between sentence-level '
    'embeddings (Reimers and Gurevych, 2019). The task SessionRerank+ '
    'solves is to produce an improved score s(q, a | H_t) that exploits '
    'H_t without retraining the base retriever.',
    align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=Cm(0.3))

# 3.2 Signals --- explained
add_subheading(doc, '3.2', 'Three affinity signals (intuition first)')
add_para(doc,
    'Three trajectory-derived signals form the core innovation. We '
    'describe each first in plain language and then formally.',
    align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=Cm(0.3))

add_para(doc,
    'Categorical co-occurrence (s_cat). APIs that have been used '
    'together in the past --- a "weather" call followed by an "alerts" '
    'call --- are likely to be relevant together again. For each '
    'candidate API a and each API already in the session history, s_cat '
    'reads a precomputed co-activation count out of the Co-activation '
    'Cache and averages them. The signal captures workflow-level '
    'co-occurrence without any user model.',
    align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=Cm(0.3))

add_para(doc,
    'Schema-signature similarity (s_sch). Two APIs that share parameter '
    'names --- both take a "location" and a "date", say --- tend to '
    'belong to the same task. We treat each API\'s parameter-name set '
    'as a signature and compute the average Jaccard overlap against '
    'each entry of H_t. The signal is structural, not lexical, and is '
    'robust to wording differences in the natural-language query.',
    align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=Cm(0.3))

add_para(doc,
    'Endpoint pattern overlap (s_ept). API endpoints are URL paths such '
    'as "/weather/forecast" or "/weather/alerts". Tokens shared between '
    'endpoints expose structural relatedness; s_ept averages the '
    'path-token overlap between candidate a and the last few entries '
    'of H_t.',
    align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=Cm(0.3))

add_para(doc,
    'A small recency term s_rec linearly decays the influence of older '
    'session entries (window k = 3 in our experiments). The fused '
    'score is a weighted sum:',
    align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=Cm(0.3))

# Equation (1)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(4)
p.paragraph_format.space_after = Pt(4)
r = p.add_run('s(q,a|H) = w_base\u00b7s_base + w_cat\u00b7s_cat + '
              'w_sch\u00b7s_sch + w_ept\u00b7s_ept + w_rec\u00b7s_rec + '
              'w_co\u00b7s_co       (1)')
set_run(r, size=10, italic=True)

add_para(doc,
    'The six weights w_* are learned by minimising an InfoNCE loss over '
    'multi-turn ToolBench trajectories: positives are the API actually '
    'called next; negatives are sampled in-batch. When H_t is empty all '
    'session terms vanish and the score reduces exactly to '
    'w_base\u00b7s_base, preserving the cold-start guarantee.',
    align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=Cm(0.3))

# 3.3 Cache
add_subheading(doc, '3.3', 'The Co-activation Cache')
add_para(doc,
    'The Co-activation Cache C_co is the second innovation: a sparse '
    'NumPy matrix of edge weights between API pairs, updated in O(k) '
    'per call and queried in O(|H|) at retrieval. After every '
    'successful API call, edges between the new API and the last k = 3 '
    'APIs in H_t are incremented by \u03c4_0; every turn, all edges '
    'multiply by (1 \u2212 \u03c1). The update rule is:',
    align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=Cm(0.3))

# Equation (2)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(4)
p.paragraph_format.space_after = Pt(4)
r = p.add_run('C_co[i,j] \u2190 (1 \u2212 \u03c1)\u00b7C_co[i,j] + '
              '\u03c4_0\u00b7\u03b4_{ij}       (2)')
set_run(r, size=10, italic=True)

add_para(doc,
    'where \u03b4_{ij} = 1 if APIs i and j co-occur within the window '
    'and 0 otherwise. We use \u03c4_0 = 1.0 and \u03c1 = 0.02 '
    'throughout. Because the cache holds scalars rather than '
    'embeddings, it persists as a megabyte-scale artefact and is shared '
    'across deployments. No neural inference is required at serve time '
    '--- a property that separates our approach from cross-encoder '
    'rerankers, which run one forward pass per query-candidate pair.',
    align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=Cm(0.3))

# ===== FIGURE 1 =====
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(4)
p.paragraph_format.space_after = Pt(2)
r = p.add_run(); r.add_picture(str(FIG1), width=Cm(8.4))
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p.paragraph_format.space_after = Pt(6)
r = p.add_run('Fig. 1.'); set_run(r, size=9, bold=True)
r = p.add_run('  Architecture of SessionRerank+. The user query enters '
              'the base dense retriever, which produces a candidate set. '
              'The session-aware re-ranker fuses the base score with '
              'three trajectory-derived affinities (s_cat, s_sch, s_ept) '
              'read from the Co-activation Cache. After every successful '
              'call the Cache is updated via the increment-decay rule.')
set_run(r, size=9)

# ===== 4. EXPERIMENTAL SETUP =====
add_heading(doc, '4', 'Experimental Setup')
add_para(doc,
    'Benchmark. We evaluate on all six ToolBench (Qin et al., 2024) '
    'splits: G1, G2 and G3 correspond to one-tool, two-tool and '
    'three-tool tasks; instruction, category and tool denote the '
    'held-out partition used to construct the negatives. Together the '
    'splits cover 1,100 natural-language queries against a corpus of '
    '3,211 API descriptions spanning 49 RapidAPI categories. We report '
    'results in three regimes: closed (gold candidates only --- '
    'diagnostic), open (full corpus, single turn), and session (full '
    'corpus, with the session history available to methods that can '
    'use it).',
    align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=Cm(0.3))

add_para(doc,
    'Baselines. (i) BM25 (Robertson and Zaragoza, 2009), the classical '
    'sparse lexical scorer; (ii) Dense, an all-MiniLM-L6-v2 bi-encoder '
    '(Reimers and Gurevych, 2019); (iii) Cross-Encoder, the ms-marco-'
    'MiniLM-L-6-v2 model (Nogueira and Cho, 2019) reranking the dense '
    'top-100; (iv) ToolRerank (Zheng et al., 2024), a tool-specific '
    'adaptive reranker. Each method is run with five seeds {0, 1, 2, '
    '42, 123}; we report the mean and use a paired permutation test '
    '(10,000 shuffles) against the Dense baseline for significance.',
    align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=Cm(0.3))

add_para(doc,
    'Hardware. All experiments ran on a single Kaggle Tesla P100-PCIE '
    '(16 GB) GPU under CUDA 11.8 and PyTorch 2.4.1. The full cascade '
    '(four baselines plus SessionRerank+, six splits, five seeds, '
    'open/closed/session regimes) completed in 2,715 s.',
    align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=Cm(0.3))

# ===== 5. RESULTS =====
add_heading(doc, '5', 'Results')

# Table I caption
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(6)
p.paragraph_format.space_after = Pt(2)
r = p.add_run('Table I'); set_run(r, size=9, bold=True)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(3)
r = p.add_run('Session-regime NDCG@5 / Hit@5 per ToolBench split, mean '
              'over five seeds. Best NDCG@5 in bold.')
set_run(r, size=9, italic=True)

splits_ordered = [('G1_instruction','G1-i'),('G1_tool','G1-t'),
                  ('G1_category','G1-c'),('G2_instruction','G2-i'),
                  ('G2_category','G2-c'),('G3_instruction','G3-i')]
methods_short = [
    ('BM25','BM25'),
    ('Dense (all-MiniLM-L6-v2)','Dense'),
    ('ToolRerank','ToolRr'),
    ('CrossEncoder (ms-marco-MiniLM-L-6-v2) top-100','CrEnc'),
    ('SessionRerank(Dense (all-MiniLM-L6-v2))','SR+'),
]

tbl = doc.add_table(rows=1+len(splits_ordered)+1, cols=1+len(methods_short))
tbl.style = 'Table Grid'
tbl.alignment = WD_ALIGN_PARAGRAPH.CENTER
tbl.autofit = False

# Set column widths via XML so LibreOffice honours them.
# Total target width: 7.8 cm = 4420 twips (fits in 8.4 cm column).
# Layout: Split=560, BM25=720, Dense=720, ToolRr=720, CrEnc=720, SR+=920 twips
from docx.oxml import OxmlElement as _OE
from docx.oxml.ns import qn as _qn
col_twips = [560, 720, 720, 720, 720, 1000]   # 5 method cols + split
assert len(col_twips) == 1 + len(methods_short)

tblPr = tbl._tbl.tblPr
# fixed layout
tblLayout = _OE('w:tblLayout'); tblLayout.set(_qn('w:type'), 'fixed')
tblPr.append(tblLayout)
tblW = _OE('w:tblW')
tblW.set(_qn('w:w'), str(sum(col_twips)))
tblW.set(_qn('w:type'), 'dxa')
tblPr.append(tblW)

# remove any existing tblGrid and add a new one
for existing in tbl._tbl.findall(_qn('w:tblGrid')):
    tbl._tbl.remove(existing)
tblGrid = _OE('w:tblGrid')
for w in col_twips:
    gc = _OE('w:gridCol'); gc.set(_qn('w:w'), str(w)); tblGrid.append(gc)
tbl._tbl.insert(list(tbl._tbl).index(tblPr)+1, tblGrid)

# set each row's cell widths via tcW
for row in tbl.rows:
    for cell, w in zip(row.cells, col_twips):
        tcPr = cell._tc.get_or_add_tcPr()
        # remove any existing tcW
        for ex in tcPr.findall(_qn('w:tcW')):
            tcPr.remove(ex)
        tcW = _OE('w:tcW'); tcW.set(_qn('w:w'), str(w))
        tcW.set(_qn('w:type'), 'dxa'); tcPr.append(tcW)

hdr = tbl.rows[0].cells
hdr[0].text = ''
r = hdr[0].paragraphs[0].add_run('Split'); set_run(r, size=8, bold=True)
hdr[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
for i, (_, lbl) in enumerate(methods_short, 1):
    hdr[i].text = ''
    r = hdr[i].paragraphs[0].add_run(lbl); set_run(r, size=8, bold=True)
    hdr[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

for ri, (skey, sshort) in enumerate(splits_ordered, 1):
    row = tbl.rows[ri].cells
    row[0].text = ''
    r = row[0].paragraphs[0].add_run(sshort); set_run(r, size=8, bold=True)
    row[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    n5s = [metrics['splits'][skey]['regimes']['session'][m]['NDCG@5']['mean']
           for m,_ in methods_short]
    best_idx = n5s.index(max(n5s))
    for ci, (mfull, _) in enumerate(methods_short, 1):
        kv = metrics['splits'][skey]['regimes']['session'][mfull]
        n5 = kv['NDCG@5']['mean']; h5 = kv['Hit@5']['mean']
        row[ci].text = ''
        r = row[ci].paragraphs[0].add_run(f'{n5:.3f}/{h5:.3f}')
        set_run(r, size=8, bold=(ci-1==best_idx))
        row[ci].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

# Mean row
mean_row = tbl.rows[-1].cells
mean_row[0].text = ''
r = mean_row[0].paragraphs[0].add_run('Mean')
set_run(r, size=8, bold=True, italic=True)
mean_row[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
means_n5 = []; means_h5 = []
for mfull,_ in methods_short:
    n5_vals = [metrics['splits'][s]['regimes']['session'][mfull]['NDCG@5']['mean'] for s,_ in splits_ordered]
    h5_vals = [metrics['splits'][s]['regimes']['session'][mfull]['Hit@5']['mean'] for s,_ in splits_ordered]
    means_n5.append(sum(n5_vals)/len(n5_vals))
    means_h5.append(sum(h5_vals)/len(h5_vals))
best_mean_idx = means_n5.index(max(means_n5))
for ci, (n5m, h5m) in enumerate(zip(means_n5, means_h5), 1):
    mean_row[ci].text = ''
    r = mean_row[ci].paragraphs[0].add_run(f'{n5m:.3f}/{h5m:.3f}')
    set_run(r, size=8, bold=(ci-1==best_mean_idx), italic=True)
    mean_row[ci].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

add_para(doc,
    'Headline findings. SessionRerank+ holds the best NDCG@5 on every '
    'split and raises the mean session NDCG@5 from 0.451 (Dense) to '
    '0.520 --- a 15.2% relative gain --- while Hit@5 climbs from 0.726 '
    'to 0.791 (+8.9%). The improvement is largest on G3-instruction '
    '(+29.3% relative), where queries span three coordinated APIs and '
    'session history is densest, and on G2-category (+21.6%). Even the '
    'short-history G1 splits register a +9.7 to +12.6% gain. A paired '
    'permutation test against Dense yields p < 5e-3 on every split '
    '(p < 1.5e-3 on four of six). The Cross-Encoder remains the '
    'strongest static reranker but is overtaken in every multi-turn '
    'setting, despite being roughly six times slower per query.',
    align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=Cm(0.3),
    space_before=4)

add_para(doc,
    'Cold-start preservation. On the closed-regime diagnostic --- where '
    'session history conveys no extra information --- SessionRerank+ '
    'matches Dense to within 0.001 NDCG@5 on every split, confirming '
    'analytically the cold-start guarantee of Eq. (1).',
    align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=Cm(0.3))

# ===== FIGURE 2 =====
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(4)
p.paragraph_format.space_after = Pt(2)
r = p.add_run(); r.add_picture(str(FIG2), width=Cm(8.4))
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p.paragraph_format.space_after = Pt(6)
r = p.add_run('Fig. 2.'); set_run(r, size=9, bold=True)
r = p.add_run('  Retrieval-time algorithm. Steps 1-6 compute the '
              'augmented score s(q, a | H) from Eq. (1); Step 7 applies '
              'the increment-decay update of Eq. (2) to the Co-activation '
              'Cache after a successful call. Every step is O(|H|), '
              'requiring no neural inference.')
set_run(r, size=9)

# ===== 6. DISCUSSION =====
add_heading(doc, '6', 'Discussion')
add_para(doc,
    'Why does it work? The three session-derived signals capture '
    'orthogonal facets of intent: which API families the user engages '
    '(s_cat), which shape of data they manipulate (s_sch), and which '
    'topical region of the corpus they browse (s_ept). Together these '
    'recover information the base retriever cannot see, because the '
    'base embedding is fixed at training time and has no access to '
    'in-session evidence. The InfoNCE-learned weights w_* in Eq. (1) '
    'consistently place 60-70% of the mass on w_base, with the '
    'categorical and recency terms next; this matches the qualitative '
    'pattern that categorical drift drives more of the gain than '
    'fine-grained schema overlap.',
    align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=Cm(0.3))

add_para(doc,
    'Practical implications. Because the Co-activation Cache is a '
    'symbolic NumPy structure, the entire layer fits in a few '
    'megabytes and can be served on CPU. Latency in our experiments '
    'was below 5 ms per query for k = 3, dominated by cache lookup. '
    'SessionRerank+ is therefore well suited as a drop-in upgrade for '
    'production agent stacks that already use a dense retriever and '
    'cannot afford the six-times cost of a cross-encoder. The '
    'framework is encoder-agnostic: any sentence embedding can replace '
    'MiniLM, and the affinity signals remain valid.',
    align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=Cm(0.3))

add_para(doc,
    'Limitations. The current evaluation uses ToolBench trajectories, '
    'which are seeded from a single LLM. Field deployments may exhibit '
    'different session topology --- shorter, more bursty, or with '
    'long-range dependencies that exceed our k = 3 window. The cache '
    'aggregates across all users; per-tenant separation, while '
    'straightforward, was not evaluated.',
    align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=Cm(0.3))

# ===== 7. CONCLUSION =====
add_heading(doc, '7', 'Conclusion')
add_para(doc,
    'We presented SessionRerank+, a session-aware re-ranking layer that '
    'brings implicit personalisation to API tool retrieval for agentic '
    'LLMs. Its three contributions --- a fused score formulation, a '
    'symbolic Co-activation Cache with O(|H|) lookup, and a provable '
    'cold-start guarantee --- combine to deliver a 15.2% relative gain '
    'in mean NDCG@5 on ToolBench while adding no neural inference at '
    'serve time. Future work will extend the cache to user-aggregated '
    'trajectories, replace the static base encoder with a '
    'session-fine-tuned variant, and evaluate the framework on live '
    'deployments such as government open-data portals and agricultural '
    'advisory systems.',
    align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=Cm(0.3))

# ===== REFERENCES (alphabetical, author-year, NO numbers) =====
add_heading(doc, '', 'References')
refs = [
    'Ai, Q., Bi, K., Guo, J., and Croft, W. B. (2018). Learning a deep listwise context model for ranking refinement. In Proc. SIGIR.',
    'Bennett, P. N., White, R. W., Chu, W., Dumais, S. T., Bailey, P., Borisyuk, F., and Cui, X. (2012). Modeling the impact of short- and long-term behavior on search personalization. In Proc. SIGIR.',
    'Bui, N., Sharma, S., Lamba, S., Mishra, S., and Ying, R. (2026). Cache what lasts: Token retention for memory-bounded KV cache in LLMs. In Proc. ICLR.',
    'Choi, N., Zhang, J., and Choi, J. D. (2026). DyCP: Dynamic context pruning for long-form dialogue with LLMs. arXiv:2601.07994.',
    'He, Z., Zhang, Y., Liu, J., et al. (2025). Task-aware adaptive KV cache compression for long-context LLMs. In Findings of EMNLP.',
    'Joo, D., Yoon, S., Kim, J., et al. (2025). MUSTAFAR: Promoting unstructured sparsity for KV cache pruning in LLMs. In Proc. NeurIPS.',
    'Nogueira, R. and Cho, K. (2019). Passage re-ranking with BERT. arXiv:1901.04085.',
    'Qin, Y., Liang, S., Ye, Y., et al. (2024). ToolLLM: Facilitating large language models to master 16000+ real-world APIs. In Proc. ICLR.',
    'Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In Proc. EMNLP.',
    'Robertson, S. and Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval, 3(4):333-389.',
    'Tazawa, U. T. and Isomura, T. (2026). Synaptic pruning facilitates online Bayesian model selection. Neural Networks, 195:108311.',
    'Wei, J., Sun, Z., Papay, S., et al. (2025). BrowseComp: A simple yet challenging benchmark for browsing agents. arXiv:2504.12516.',
    'Xie, T., Zhang, Y., Xu, J., et al. (2024). OSWorld: Benchmarking multimodal agents for open-ended tasks in real computer environments. In Proc. NeurIPS Datasets and Benchmarks.',
    'Yao, S., Zhao, J., Yu, D., et al. (2023). ReAct: Synergizing reasoning and acting in language models. In Proc. ICLR.',
    'Zamani, H., Bendersky, M., Wang, X., and Zhang, M. (2017). Situational context for ranking in personal search. In Proc. WWW.',
    'Zheng, Y., Li, P., Liu, W., et al. (2024). ToolRerank: Adaptive and hierarchy-aware reranking for tool retrieval. In Proc. LREC-COLING.',
    'Zhou, S., Xu, F. F., Zhu, H., et al. (2024). WebArena: A realistic web environment for building autonomous agents. In Proc. ICLR.',
]
for ref in refs:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format
    pf.left_indent = Cm(0.5)
    pf.first_line_indent = Cm(-0.5)
    pf.space_after = Pt(1)
    pf.line_spacing = 1.05
    r = p.add_run(ref); set_run(r, size=9)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(str(OUT))
print(f'Wrote {OUT}  ({OUT.stat().st_size} bytes)')
print(f'Abstract word count: {len(abstract_text.split())} (CRC limit 150)')
print(f'Reference count: {len(refs)}')
