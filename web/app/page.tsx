"use client";

import { useEffect, useMemo, useState } from "react";

type ApiItem = {
  uid: string;
  tool: string;
  api: string;
  category: string;
  description: string;
};

type RankedItem = ApiItem & { rank: number; score: number };

type RerankResponse = {
  query: string;
  history_size: number;
  encoder: string;
  corpus_size: number;
  baseline: RankedItem[];
  session_rerank: RankedItem[];
  weights: { w_base: number; w_cat: number; w_sch: number; w_ept: number };
};

const SPLITS = [
  "G1_instruction",
  "G1_tool",
  "G1_category",
  "G2_instruction",
  "G2_category",
  "G3_instruction",
];
const METHODS = ["BM25", "Dense (all-MiniLM-L6-v2)", "ToolRerank", "SessionRerank (ours)"];
const METHOD_SHORT: Record<string, string> = {
  BM25: "BM25",
  "Dense (all-MiniLM-L6-v2)": "Dense",
  ToolRerank: "ToolRerank",
  "SessionRerank (ours)": "SessionRerank",
};

const FIGURES_BASE = [
  {
    file: "fig1_ndcg5_session.png",
    caption:
      "Figure 1. NDCG@5 across all six ToolBench evaluation splits in the session regime. SessionRerank (green) outperforms BM25, Dense, and ToolRerank on every split.",
  },
  {
    file: "fig2_hit5_session.png",
    caption:
      "Figure 2. Hit@5 across all six evaluation splits in the session regime. Gains are largest on multi-tool (G2) splits where session history is most informative.",
  },
  {
    file: "fig3_ndcg_atK_g2inst.png",
    caption:
      "Figure 3. NDCG@K for K∈{1,3,5,10} on G2_instruction. SessionRerank dominates across all cutoffs.",
  },
  {
    file: "fig4_relgain_vs_dense.png",
    caption:
      "Figure 4. Relative improvement of SessionRerank over the Dense retrieval baseline (ToolLLM IR). Mean Δ across splits: +14.7% NDCG@5, +8.5% Hit@5.",
  },
  {
    file: "fig5_heatmap_ndcg5.png",
    caption:
      "Figure 5. Heatmap of NDCG@5 (methods × splits). Warmer cells = higher performance. SessionRerank row is consistently the warmest.",
  },
  {
    file: "fig6_ndcg5_open.png",
    caption:
      "Figure 6. Open-corpus single-turn regime (3,211 APIs, no session history). SessionRerank degrades gracefully to Dense performance — no harm from the added prior when history is empty.",
  },
  {
    file: "fig7_gemma_vs_minilm.png",
    caption:
      "Figure 7. Gemma 4 vs MiniLM head-to-head — three Gemma variants (Embed, Rerank, fine-tuned) compared against the MiniLM-based SessionRerank on NDCG@5 (session regime).",
  },
];

function num(x: number, digits = 3) {
  if (typeof x !== "number" || !isFinite(x)) return "—";
  return x.toFixed(digits);
}

export default function Page() {
  const [sample, setSample] = useState<ApiItem[]>([]);
  const [corpusSize, setCorpusSize] = useState<number>(0);
  const [query, setQuery] = useState<string>(
    "Find a translation API and then convert the result to speech audio"
  );
  const [history, setHistory] = useState<Set<string>>(new Set());
  const [filter, setFilter] = useState<string>("");
  const [result, setResult] = useState<RerankResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [metrics, setMetrics] = useState<any | null>(null);
  const [figuresAvailable, setFiguresAvailable] = useState<string[]>([]);

  // Load API sample for the history picker
  useEffect(() => {
    fetch("/api/rerank")
      .then((r) => r.json())
      .then((d) => {
        setSample(d.sample || []);
        setCorpusSize(d.corpus_size || 0);
      })
      .catch(() => {});
    fetch("/metrics.json")
      .then((r) => r.json())
      .then(setMetrics)
      .catch(() => {});
    // Probe each figure with HEAD so we only render those that exist
    Promise.all(
      FIGURES_BASE.map(async (f) => {
        try {
          const r = await fetch(`/figures/${f.file}`, { method: "HEAD" });
          return r.ok ? f.file : null;
        } catch {
          return null;
        }
      })
    ).then((files) => setFiguresAvailable(files.filter(Boolean) as string[]));
  }, []);

  const figures = useMemo(
    () => FIGURES_BASE.filter((f) => figuresAvailable.includes(f.file)),
    [figuresAvailable]
  );

  const filteredSample = useMemo(() => {
    const f = filter.toLowerCase().trim();
    if (!f) return sample;
    return sample.filter(
      (a) =>
        a.tool.toLowerCase().includes(f) ||
        a.api.toLowerCase().includes(f) ||
        a.category.toLowerCase().includes(f)
    );
  }, [sample, filter]);

  function toggleHistory(uid: string) {
    setHistory((prev) => {
      const next = new Set(prev);
      if (next.has(uid)) next.delete(uid);
      else next.add(uid);
      return next;
    });
  }

  async function runDemo() {
    setLoading(true);
    try {
      const res = await fetch("/api/rerank", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          history: Array.from(history),
          top_k: 8,
        }),
      });
      const data = await res.json();
      setResult(data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }

  // Compute rank-change badges for the session column
  const rankDelta = useMemo(() => {
    if (!result) return new Map<string, number>();
    const baseRank = new Map(result.baseline.map((b) => [b.uid, b.rank]));
    const m = new Map<string, number>();
    for (const s of result.session_rerank) {
      const br = baseRank.get(s.uid);
      if (br == null) m.set(s.uid, 99); // new entry
      else m.set(s.uid, br - s.rank);
    }
    return m;
  }, [result]);

  return (
    <main>
      {/* Hero */}
      <header className="hero">
        <div className="container">
          <div className="eyebrow">
            ToolBench · Session-Aware Re-Ranking ·{" "}
            <a
              href="https://www.kaggle.com/competitions/gemma-4-good-hackathon"
              target="_blank"
              rel="noreferrer"
              style={{ color: "inherit", textDecoration: "underline" }}
            >
              Gemma 4 Good Hackathon submission
            </a>
          </div>
          <h1>SessionRerank</h1>
          <p className="tagline">
            History-aware tool re-ranking for LLM agents — outperforming BM25,
            ToolLLM Dense, and ToolRerank on the full ToolBench benchmark.
          </p>
          <div className="hero-cta">
            <a
              className="btn btn-primary"
              href="https://github.com/joyjeni/session-aware-toolbench-rerank"
              target="_blank"
              rel="noreferrer"
            >
              View on GitHub
            </a>
            <a className="btn btn-ghost" href="#demo">
              Try the live demo ↓
            </a>
          </div>
          {metrics?.config?.timestamp && (
            <div className="hero-timestamp">
              Results last updated:{" "}
              <strong>
                {new Date(metrics.config.timestamp).toLocaleString()}
              </strong>
              {" · encoder: "}
              <code>{metrics.config.encoder}</code>
            </div>
          )}
        </div>
      </header>

      {/* KPI cards */}
      <section className="container">
        <div className="kpi-grid">
          <div className="kpi">
            <div className="kpi-value">+14.7%</div>
            <div className="kpi-label">Mean NDCG@5 gain vs Dense</div>
            <div className="kpi-sub">Averaged over 6 ToolBench splits</div>
          </div>
          <div className="kpi">
            <div className="kpi-value">+20.7%</div>
            <div className="kpi-label">Peak NDCG@5 gain (G2_instruction)</div>
            <div className="kpi-sub">Multi-tool agentic workflows</div>
          </div>
          <div className="kpi">
            <div className="kpi-value">p &lt; 1e-4</div>
            <div className="kpi-label">Wilcoxon signed-rank significance</div>
            <div className="kpi-sub">vs Dense, on G1 &amp; G2 NDCG@5</div>
          </div>
          <div className="kpi">
            <div className="kpi-value">3,211</div>
            <div className="kpi-label">APIs evaluated</div>
            <div className="kpi-sub">49 categories · 1,100 eval queries</div>
          </div>
        </div>
      </section>

      {/* Method */}
      <section className="container section">
        <h2>Method</h2>
        <p className="body">
          SessionRerank augments a dense bi-encoder retriever with an{" "}
          <strong>interaction-history prior</strong> learned from previous
          tool-call traces in the same agent session. Given a query{" "}
          <code>q</code>, candidate API <code>a</code>, and a session history{" "}
          <code>H = (a₁, a₂, …, aₜ)</code> with exponential decay{" "}
          <code>γ=0.85</code>, we score:
        </p>
        <pre className="math">
{`score(q, a | H) = w_base · sim(q, a)
                + w_cat  · cat_affinity(a, H)
                + w_sch  · schema_affinity(a, H)
                + w_ept  · endpoint_affinity(a, H)`}
        </pre>
        <p className="body">
          Weights are learned end-to-end with an InfoNCE objective over
          ToolBench G2/G3 training sessions. Trained values:{" "}
          <code>w_base = 1.40</code>, <code>w_cat = 1.17</code>,{" "}
          <code>w_sch = 0.31</code>, <code>w_ept = 0.10</code>. With empty
          history the prior collapses to zero and the model reduces exactly to
          the Dense baseline — guaranteeing no regression on single-turn
          retrieval.
        </p>
      </section>

      {/* Results table */}
      <section className="container section">
        <h2>Results — NDCG@5 (session regime)</h2>
        <p className="body">
          Full evaluation on the official ToolBench benchmark splits. Best per
          row in <strong className="winner">bold green</strong>.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Split</th>
                {METHODS.map((m) => (
                  <th key={m}>{METHOD_SHORT[m]}</th>
                ))}
                <th>Δ vs Dense</th>
              </tr>
            </thead>
            <tbody>
              {SPLITS.map((split) => {
                const row = metrics?.splits?.[split]?.regimes?.session;
                if (!row) {
                  return (
                    <tr key={split}>
                      <td>{split}</td>
                      {METHODS.map((m) => (
                        <td key={m}>—</td>
                      ))}
                      <td>—</td>
                    </tr>
                  );
                }
                const vals = METHODS.map((m) => row[m]?.["NDCG@5"]);
                const best = Math.max(...vals.filter((v) => typeof v === "number"));
                const dense = row["Dense (all-MiniLM-L6-v2)"]?.["NDCG@5"];
                const ours = row["SessionRerank (ours)"]?.["NDCG@5"];
                const delta = dense ? ((ours - dense) / dense) * 100 : 0;
                return (
                  <tr key={split}>
                    <td className="mono">{split}</td>
                    {METHODS.map((m, i) => {
                      const v = vals[i];
                      const isBest = typeof v === "number" && Math.abs(v - best) < 1e-9;
                      return (
                        <td key={m} className={isBest ? "winner" : ""}>
                          {num(v)}
                        </td>
                      );
                    })}
                    <td className={delta >= 15 ? "delta-strong" : "delta"}>
                      {delta >= 0 ? "+" : ""}
                      {delta.toFixed(1)}%
                    </td>
                  </tr>
                );
              })}
              {/* Mean row */}
              {metrics && (
                <tr className="mean-row">
                  <td className="mono"><strong>Mean</strong></td>
                  {METHODS.map((m) => {
                    const vs = SPLITS.map(
                      (s) => metrics.splits?.[s]?.regimes?.session?.[m]?.["NDCG@5"]
                    ).filter((v) => typeof v === "number");
                    const mean = vs.reduce((a, b) => a + b, 0) / Math.max(1, vs.length);
                    return (
                      <td key={m}>
                        <strong>{num(mean)}</strong>
                      </td>
                    );
                  })}
                  <td className="delta-strong"><strong>+14.7%</strong></td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      {/* Figures grid */}
      <section className="container section">
        <h2>Publication-Ready Figures</h2>
        <p className="body">
          All figures rendered at 300 DPI (PDF/PNG/SVG) for journal print.
          Captions follow IEEE / ACM convention.
        </p>
        <div className="figure-grid">
          {figures.map((f) => (
            <figure key={f.file} className="figure-card">
              <img src={`/figures/${f.file}`} alt={f.caption} />
              <figcaption>{f.caption}</figcaption>
            </figure>
          ))}
        </div>
      </section>

      {/* Demo */}
      <section className="container section" id="demo">
        <h2>Live Demo — Baseline vs SessionRerank</h2>
        <p className="body">
          Enter a query, tick a few APIs the agent has already used in this
          session, and watch the right column re-rank in response to that
          history. The serverless endpoint <code>/api/rerank</code> runs the
          same scoring rule used in the paper.
        </p>

        <div className="demo-panel">
          <div className="demo-controls">
            <label className="field">
              <span className="field-label">Query</span>
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                rows={3}
              />
            </label>

            <label className="field">
              <span className="field-label">
                Session history — pick APIs already used ({history.size} selected)
              </span>
              <input
                className="filter"
                type="text"
                placeholder="Filter by tool / category…"
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
              />
              <div className="api-picker">
                {filteredSample.map((a) => (
                  <label key={a.uid} className="api-chip">
                    <input
                      type="checkbox"
                      checked={history.has(a.uid)}
                      onChange={() => toggleHistory(a.uid)}
                    />
                    <span>
                      <strong>{a.tool}</strong> / {a.api}
                      <em className="cat">{a.category}</em>
                    </span>
                  </label>
                ))}
                {filteredSample.length === 0 && (
                  <div className="muted">No APIs match that filter.</div>
                )}
              </div>
              <div className="muted small">
                Showing {filteredSample.length} of {sample.length} sample APIs
                · full benchmark uses {corpusSize.toLocaleString()} APIs
              </div>
            </label>

            <button
              className="btn btn-primary"
              onClick={runDemo}
              disabled={loading || !query.trim()}
            >
              {loading ? "Ranking…" : "Run rerank"}
            </button>
          </div>

          {result && (
            <div className="demo-results">
              <div className="result-col">
                <h3>Baseline (Dense, no history)</h3>
                <ol className="rerank-list">
                  {result.baseline.map((r) => (
                    <li key={r.uid}>
                      <div className="r-head">
                        <span className="r-rank">#{r.rank}</span>
                        <span className="r-name">
                          <strong>{r.tool}</strong> / {r.api}
                        </span>
                        <span className="r-score">{r.score.toFixed(3)}</span>
                      </div>
                      <div className="r-meta">
                        <span className="cat">{r.category}</span>
                        <span className="desc">{r.description}</span>
                      </div>
                    </li>
                  ))}
                </ol>
              </div>

              <div className="result-col">
                <h3>
                  SessionRerank{" "}
                  <span className="muted small">
                    · history size {result.history_size}
                  </span>
                </h3>
                <ol className="rerank-list">
                  {result.session_rerank.map((r) => {
                    const delta = rankDelta.get(r.uid) ?? 0;
                    return (
                      <li key={r.uid}>
                        <div className="r-head">
                          <span className="r-rank">#{r.rank}</span>
                          <span className="r-name">
                            <strong>{r.tool}</strong> / {r.api}
                          </span>
                          {delta > 0 && (
                            <span className="badge up">▲ {delta}</span>
                          )}
                          {delta < 0 && delta > -50 && (
                            <span className="badge down">▼ {Math.abs(delta)}</span>
                          )}
                          {delta >= 50 && (
                            <span className="badge new">NEW</span>
                          )}
                          <span className="r-score">{r.score.toFixed(3)}</span>
                        </div>
                        <div className="r-meta">
                          <span className="cat">{r.category}</span>
                          <span className="desc">{r.description}</span>
                        </div>
                      </li>
                    );
                  })}
                </ol>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <div className="container">
          <div>
            SessionRerank · ToolBench session-aware re-ranking · MIT License
          </div>
          <div className="muted small">
            Implementation, full benchmark, and Colab notebook on{" "}
            <a
              href="https://github.com/joyjeni/session-aware-toolbench-rerank"
              target="_blank"
              rel="noreferrer"
            >
              GitHub
            </a>
            .
          </div>
        </div>
      </footer>
    </main>
  );
}
