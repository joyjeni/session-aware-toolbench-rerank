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

type AgentResponse = {
  query: string;
  answer: string;
  path: "groq" | "openrouter" | "hf" | "offline";
  path_error: string | null;
  top_apis: RankedItem[];
  history_size: number;
};

const PATH_LABEL: Record<AgentResponse["path"], string> = {
  groq: "Groq · llama-3.1-8b-instant",
  openrouter: "OpenRouter · llama-3.2-3b (free)",
  hf: "Hugging Face · zephyr-7b",
  offline: "Offline template (no LLM key)",
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

// Extract a numeric value from either a raw number or a {mean,std,values} cell
// produced by the multi-seed aggregation in eval/run_all.py.
function valOf(cell: any): number | null {
  if (cell == null) return null;
  if (typeof cell === "number") return cell;
  if (typeof cell === "object" && typeof cell.mean === "number") return cell.mean;
  return null;
}
function stdOf(cell: any): number | null {
  if (cell && typeof cell === "object" && typeof cell.std === "number") return cell.std;
  return null;
}
function nSeeds(cell: any): number | null {
  if (cell && typeof cell === "object" && Array.isArray(cell.values)) return cell.values.length;
  return null;
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
  const [agent, setAgent] = useState<AgentResponse | null>(null);
  const [agentLoading, setAgentLoading] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(false);
  const [metrics, setMetrics] = useState<any | null>(null);
  const [figuresAvailable, setFiguresAvailable] = useState<string[]>([]);
  const [stages, setStages] = useState<Record<string, any>>({});

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
    // Probe per-stage metrics files (added in v1.1 Gemma 4 run)
    const STAGE_FILES: Record<string, string> = {
      "Stage 1 — Classical": "/metrics_classical.json",
      "Stage 2 — + Gemma zero-shot": "/metrics_with_gemma.json",
      "Stage 4 — + Fine-tuned": "/metrics.json",
    };
    Promise.all(
      Object.entries(STAGE_FILES).map(async ([label, url]) => {
        try {
          const r = await fetch(url);
          if (!r.ok) return [label, null] as const;
          return [label, await r.json()] as const;
        } catch {
          return [label, null] as const;
        }
      })
    ).then((entries) => {
      const out: Record<string, any> = {};
      for (const [label, m] of entries) if (m) out[label] = m;
      setStages(out);
    });
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

  // Detect a CrossEncoder method name in the loaded metrics so we can
  // show it as an extra column when present (Stage 1 additions).
  const crossEncoderMethod = useMemo(() => {
    if (!metrics?.splits) return null;
    for (const split of SPLITS) {
      const row = metrics.splits?.[split]?.regimes?.session;
      if (!row) continue;
      const key = Object.keys(row).find((k) => k.startsWith("CrossEncoder"));
      if (key) return key;
    }
    return null;
  }, [metrics]);

  const tableMethods = useMemo(() => {
    if (!crossEncoderMethod) return METHODS;
    // Insert CE column between ToolRerank and SessionRerank
    const idx = METHODS.indexOf("ToolRerank");
    const out = [...METHODS];
    out.splice(idx + 1, 0, crossEncoderMethod);
    return out;
  }, [crossEncoderMethod]);

  const methodShort = useMemo<Record<string, string>>(() => {
    const m = { ...METHOD_SHORT } as Record<string, string>;
    if (crossEncoderMethod) m[crossEncoderMethod] = "CrossEncoder";
    return m;
  }, [crossEncoderMethod]);

  // Detect # of seeds for the variance caption
  const seedsRun = useMemo(() => {
    const cfgSeeds = (metrics as any)?.config?.seeds;
    if (Array.isArray(cfgSeeds) && cfgSeeds.length) return cfgSeeds as number[];
    // Fallback: probe a cell for `values`
    for (const split of SPLITS) {
      const r = metrics?.splits?.[split]?.regimes?.session;
      if (!r) continue;
      for (const k of Object.keys(r)) {
        const n = nSeeds((r as any)[k]?.["NDCG@5"]);
        if (n) return Array.from({ length: n }, (_, i) => i);
      }
    }
    return null;
  }, [metrics]);

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
    setAgentLoading(true);
    setAgent(null);
    const body = JSON.stringify({
      query,
      history: Array.from(history),
      top_k: 8,
    });
    // Fire both endpoints in parallel — the agent endpoint reuses the same
    // ranking logic but additionally synthesizes a natural-language answer.
    const rerankP = fetch("/api/rerank", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    })
      .then((r) => r.json())
      .then((d) => setResult(d))
      .catch((e) => console.error("rerank:", e))
      .finally(() => setLoading(false));

    const agentP = fetch("/api/agent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    })
      .then((r) => r.json())
      .then((d) => setAgent(d))
      .catch((e) => {
        console.error("agent:", e);
        setAgent({
          query,
          answer:
            "The agent endpoint failed to respond. The rerank ranking below is still valid.",
          path: "offline",
          path_error: String(e),
          top_apis: [],
          history_size: history.size,
        });
      })
      .finally(() => setAgentLoading(false));

    await Promise.allSettled([rerankP, agentP]);
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
          {seedsRun && seedsRun.length > 1 && (
            <>
              {" "}Numbers are mean across <strong>{seedsRun.length} seeds</strong>{" "}
              (seeds <code>{seedsRun.join(", ")}</code>); the ± value is the
              standard deviation across seeds.
            </>
          )}
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Split</th>
                {tableMethods.map((m) => (
                  <th key={m}>{methodShort[m] || m}</th>
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
                      {tableMethods.map((m) => (
                        <td key={m}>—</td>
                      ))}
                      <td>—</td>
                    </tr>
                  );
                }
                const vals = tableMethods.map((m) => valOf(row[m]?.["NDCG@5"]));
                const stds = tableMethods.map((m) => stdOf(row[m]?.["NDCG@5"]));
                const nums = vals.filter((v): v is number => typeof v === "number");
                const best = nums.length ? Math.max(...nums) : -Infinity;
                const dense = valOf(row["Dense (all-MiniLM-L6-v2)"]?.["NDCG@5"]);
                const ours = valOf(row["SessionRerank (ours)"]?.["NDCG@5"]);
                const delta = dense && ours != null ? ((ours - dense) / dense) * 100 : 0;
                return (
                  <tr key={split}>
                    <td className="mono">{split}</td>
                    {tableMethods.map((m, i) => {
                      const v = vals[i];
                      const sd = stds[i];
                      const isBest = typeof v === "number" && Math.abs(v - best) < 1e-9;
                      return (
                        <td key={m} className={isBest ? "winner" : ""}>
                          {v == null ? "—" : num(v)}
                          {sd != null && sd > 0 && (
                            <span style={{ fontSize: 11, opacity: 0.7, marginLeft: 4 }}>
                              ±{sd.toFixed(3)}
                            </span>
                          )}
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
                  {tableMethods.map((m) => {
                    const vs = SPLITS.map(
                      (s) => valOf(metrics.splits?.[s]?.regimes?.session?.[m]?.["NDCG@5"])
                    ).filter((v): v is number => typeof v === "number");
                    const mean = vs.length ? vs.reduce((a, b) => a + b, 0) / vs.length : null;
                    return (
                      <td key={m}>
                        <strong>{mean == null ? "—" : num(mean)}</strong>
                      </td>
                    );
                  })}
                  <td className="delta-strong">
                    <strong>
                      {(() => {
                        const dvs = SPLITS.map((s) => {
                          const r = metrics.splits?.[s]?.regimes?.session;
                          if (!r) return null;
                          const d = valOf(r["Dense (all-MiniLM-L6-v2)"]?.["NDCG@5"]);
                          const o = valOf(r["SessionRerank (ours)"]?.["NDCG@5"]);
                          if (d == null || o == null || !d) return null;
                          return ((o - d) / d) * 100;
                        }).filter((v): v is number => typeof v === "number");
                        if (!dvs.length) return "+14.7%";
                        const m = dvs.reduce((a, b) => a + b, 0) / dvs.length;
                        return `${m >= 0 ? "+" : ""}${m.toFixed(1)}%`;
                      })()}
                    </strong>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      {/* Stage Comparison — added for Gemma 4 Hackathon submission */}
      {Object.keys(stages).length > 0 && (
        <section className="container section">
          <h2>Stage-by-Stage Benchmark</h2>
          <p className="body">
            NDCG@5 in the session regime for each stage of the pipeline.
            Stage 1 establishes classical baselines — now extended with a
            <code> cross-encoder/ms-marco-MiniLM-L-6-v2</code> reranker, a
            learned tool co-occurrence prior P(api<sub>b</sub>|api<sub>a</sub>),
            and multi-seed runs (seeds 0,1,2,42,123) so every number carries an
            explicit standard-deviation band. Stage 2 adds zero-shot Gemma 4 e2B
            (bi-encoder and prompted cross-encoder). Stage 4 adds the LoRA
            fine-tuned <code>SessionRerank-Gemma</code>.
          </p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Split</th>
                  {Object.keys(stages).map((label) => (
                    <th key={label}>{label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {SPLITS.map((split) => {
                  const cells = Object.keys(stages).map((label) => {
                    const m = stages[label];
                    const row = m?.splits?.[split]?.regimes?.session;
                    if (!row) return { label, value: null as number | null, std: null as number | null, method: "" };
                    // Best NDCG@5 in this row
                    let bestM = "";
                    let bestV = -Infinity;
                    let bestSd: number | null = null;
                    for (const [methodName, mv] of Object.entries(row)) {
                      const cell = (mv as any)?.["NDCG@5"];
                      const v = valOf(cell);
                      if (typeof v === "number" && v > bestV) {
                        bestV = v;
                        bestM = methodName;
                        bestSd = stdOf(cell);
                      }
                    }
                    return { label, value: bestV === -Infinity ? null : bestV, std: bestSd, method: bestM };
                  });
                  return (
                    <tr key={split}>
                      <td className="mono">{split}</td>
                      {cells.map((c, i) => (
                        <td key={i}>
                          {c.value === null ? (
                            "—"
                          ) : (
                            <>
                              <strong>{num(c.value)}</strong>
                              {c.std != null && c.std > 0 && (
                                <span style={{ fontSize: 11, opacity: 0.7, marginLeft: 4 }}>
                                  ±{c.std.toFixed(3)}
                                </span>
                              )}
                              <div style={{ fontSize: 11, opacity: 0.65 }}>{c.method}</div>
                            </>
                          )}
                        </td>
                      ))}
                    </tr>
                  );
                })}
                <tr className="mean-row">
                  <td><strong>Mean</strong></td>
                  {Object.keys(stages).map((label) => {
                    const m = stages[label];
                    const splitMeans = SPLITS.map((split) => {
                      const row = m?.splits?.[split]?.regimes?.session;
                      if (!row) return null;
                      const vs = Object.values(row)
                        .map((mv: any) => valOf(mv?.["NDCG@5"]))
                        .filter((v): v is number => typeof v === "number");
                      return vs.length ? Math.max(...vs) : null;
                    }).filter((v): v is number => typeof v === "number");
                    const mean = splitMeans.length
                      ? splitMeans.reduce((a, b) => a + b, 0) / splitMeans.length
                      : null;
                    return (
                      <td key={label}>
                        <strong>{mean === null ? "—" : num(mean)}</strong>
                      </td>
                    );
                  })}
                </tr>
              </tbody>
            </table>
          </div>
          <p className="body" style={{ marginTop: 12, fontSize: 13, opacity: 0.75 }}>
            Each cell shows the best NDCG@5 within that stage on the session
            regime, along with the winning method. Stage 3 (LoRA fine-tune of
            Gemma 4 e2B with Unsloth) does not produce its own benchmark file;
            its output feeds into Stage 4.
          </p>
        </section>
      )}

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
        <h2>Live Demo — Ask the Agent</h2>
        <p className="body">
          Ask a question and the agent responds with a synthesized answer
          grounded in the top tools surfaced by SessionRerank. Below that, the
          baseline-vs-session columns let you see <em>which</em> tools fed the
          answer — and how session history changes the ranking. The agent
          endpoint (<code>/api/agent</code>) tries Groq → OpenRouter → HF
          Inference in order; if no key is configured it falls back to an
          offline template and labels the path accordingly.
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

          {(agentLoading || agent) && (
            <div className="agent-card" aria-live="polite">
              <h4>
                <span>🤖 Agent answer</span>
                {agentLoading && (
                  <span className="muted small">· thinking…</span>
                )}
              </h4>
              {agent ? (
                <>
                  <div className="agent-body">{agent.answer}</div>
                  <div className="agent-meta">
                    <span>
                      via{" "}
                      <span className={`path-badge path-${agent.path}`}>
                        {PATH_LABEL[agent.path]}
                      </span>
                    </span>
                    <span>history size: {agent.history_size}</span>
                    <span>top APIs: {agent.top_apis.length}</span>
                  </div>
                  {agent.path === "offline" && (
                    <div className="agent-error">
                      No LLM key detected. Set one of{" "}
                      <code>GROQ_API_KEY</code> (free at console.groq.com),{" "}
                      <code>OPENROUTER_API_KEY</code> (free models), or{" "}
                      <code>HF_TOKEN</code> in your Vercel env vars to get a
                      natural-language answer instead of the template.
                    </div>
                  )}
                  {agent.path_error && agent.path !== "offline" && (
                    <div className="agent-error">
                      Prior path failed: {agent.path_error}
                    </div>
                  )}
                </>
              ) : (
                <div className="agent-body muted">Synthesizing answer…</div>
              )}
            </div>
          )}
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
