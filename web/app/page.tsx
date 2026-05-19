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

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  path?: AgentResponse["path"];
  top_apis?: RankedItem[];
  pending?: boolean;
  error?: string;
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

const BASE_METHODS = [
  "BM25",
  "Dense (all-MiniLM-L6-v2)",
  "ToolRerank",
  "SessionRerank(Dense (all-MiniLM-L6-v2))",
];

const METHOD_SHORT: Record<string, string> = {
  BM25: "BM25",
  "Dense (all-MiniLM-L6-v2)": "Dense",
  ToolRerank: "ToolRerank",
  "SessionRerank(Dense (all-MiniLM-L6-v2))": "SessionRerank",
};

const FIGURES_BASE = [
  {
    file: "fig1_ndcg5_session.png",
    caption:
      "Figure 1. NDCG@5 across all six ToolBench evaluation splits in the session regime. SessionRerank outperforms BM25, Dense, and ToolRerank on every split.",
  },
  {
    file: "fig2_hit5_session.png",
    caption:
      "Figure 2. Hit@5 across all six evaluation splits in the session regime. Gains are largest on multi-tool G2 splits where session history is most informative.",
  },
  {
    file: "fig3_ndcg_atK_g2inst.png",
    caption:
      "Figure 3. NDCG@K for K={1,3,5,10} on G2_instruction. SessionRerank dominates across all cutoffs.",
  },
  {
    file: "fig4_relgain_vs_dense.png",
    caption:
      "Figure 4. Relative improvement of SessionRerank over the Dense retrieval baseline. Mean Δ across splits is computed from the loaded metrics.",
  },
  {
    file: "fig5_heatmap_ndcg5.png",
    caption:
      "Figure 5. Heatmap of NDCG@5 across methods and splits. Warmer cells indicate higher performance.",
  },
  {
    file: "fig6_ndcg5_open.png",
    caption:
      "Figure 6. Open-corpus single-turn regime with no session history. SessionRerank degrades gracefully toward the Dense baseline when history is empty.",
  },
  {
    file: "fig7_gemma_vs_minilm.png",
    caption:
      "Figure 7. Gemma-vs-MiniLM comparison figure, rendered when available in the public figures directory.",
  },
];

function num(x: number | null | undefined, digits = 3) {
  if (typeof x !== "number" || !isFinite(x)) return "—";
  return x.toFixed(digits);
}

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

function labelSplit(split: string) {
  return split.replaceAll("_", " ");
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
  const [activeTab, setActiveTab] = useState<"chat" | "answer" | "retrieval">("chat");
  const [chat, setChat] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState<string>("");
  const [chatSending, setChatSending] = useState<boolean>(false);

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

    const STAGE_FILES: Record<string, string> = {
      "Stage 1 — Classical": "/metrics_classical.json",
      "Stage 2 — + Gemma zero-shot": "/metrics_with_gemma.json",
      "Stage 4 — current metrics": "/metrics.json",
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
      for (const [label, m] of entries) {
        if (m) out[label] = m;
      }
      setStages(out);
    });

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
    if (!crossEncoderMethod) return BASE_METHODS;
    const idx = BASE_METHODS.indexOf("ToolRerank");
    const out = [...BASE_METHODS];
    out.splice(idx + 1, 0, crossEncoderMethod);
    return out;
  }, [crossEncoderMethod]);

  const methodShort = useMemo<Record<string, string>>(() => {
    const m = { ...METHOD_SHORT };
    if (crossEncoderMethod) m[crossEncoderMethod] = "CrossEncoder";
    return m;
  }, [crossEncoderMethod]);

  const seedsRun = useMemo(() => {
    const cfgSeeds = metrics?.config?.seeds;
    if (Array.isArray(cfgSeeds) && cfgSeeds.length) return cfgSeeds as number[];
    for (const split of SPLITS) {
      const r = metrics?.splits?.[split]?.regimes?.session;
      if (!r) continue;
      for (const k of Object.keys(r)) {
        const n = nSeeds(r[k]?.["NDCG@5"]);
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

  const meanDeltaPct = useMemo(() => {
    if (!metrics?.splits) return null;
    const deltas = SPLITS.map((split) => {
      const row = metrics.splits?.[split]?.regimes?.session;
      if (!row) return null;
      const dense = valOf(row["Dense (all-MiniLM-L6-v2)"]?.["NDCG@5"]);
      const ours = valOf(row["SessionRerank(Dense (all-MiniLM-L6-v2))"]?.["NDCG@5"]);
      if (dense == null || ours == null || dense === 0) return null;
      return ((ours - dense) / dense) * 100;
    }).filter((v): v is number => typeof v === "number");

    if (!deltas.length) return null;
    return deltas.reduce((a, b) => a + b, 0) / deltas.length;
  }, [metrics]);

  const peakDeltaPct = useMemo(() => {
    if (!metrics?.splits) return null;
    const deltas = SPLITS.map((split) => {
      const row = metrics.splits?.[split]?.regimes?.session;
      if (!row) return null;
      const dense = valOf(row["Dense (all-MiniLM-L6-v2)"]?.["NDCG@5"]);
      const ours = valOf(row["SessionRerank(Dense (all-MiniLM-L6-v2))"]?.["NDCG@5"]);
      if (dense == null || ours == null || dense === 0) return null;
      return { split, pct: ((ours - dense) / dense) * 100 };
    }).filter((v): v is { split: string; pct: number } => !!v);

    if (!deltas.length) return null;
    return deltas.reduce((best, cur) => (cur.pct > best.pct ? cur : best));
  }, [metrics]);

  const sessionPValue = useMemo(() => {
    return metrics?.significance?.G2_instruction?.session?.["NDCG@5"]?.p_value ?? null;
  }, [metrics]);

  function toggleHistory(uid: string) {
    setHistory((prev) => {
      const next = new Set(prev);
      if (next.has(uid)) next.delete(uid);
      else next.add(uid);
      return next;
    });
  }

  async function sendChat(text?: string) {
    const content = (text ?? chatInput).trim();
    if (!content || chatSending) return;

    const userMsg: ChatMessage = {
      id: `u-${Date.now()}`,
      role: "user",
      content,
    };
    const assistantId = `a-${Date.now()}`;
    const pendingMsg: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      pending: true,
    };

    const prior = chat;
    setChat([...prior, userMsg, pendingMsg]);
    setChatInput("");
    setChatSending(true);

    const transcript = [...prior, userMsg].map((m) => ({
      role: m.role,
      content: m.content,
    }));

    try {
      const r = await fetch("/api/agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: content,
          history: Array.from(history),
          chat: transcript.slice(0, -1),
          top_k: 8,
        }),
      });
      const d: AgentResponse = await r.json();
      setChat((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                id: assistantId,
                role: "assistant",
                content: d.answer || "(empty response)",
                path: d.path,
                top_apis: d.top_apis,
                pending: false,
              }
            : m
        )
      );
    } catch (e: any) {
      setChat((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                id: assistantId,
                role: "assistant",
                content:
                  "The chat endpoint failed to respond. Try again, or use Run rerank below to inspect retrieval directly.",
                path: "offline",
                pending: false,
                error: String(e?.message || e),
              }
            : m
        )
      );
    } finally {
      setChatSending(false);
    }
  }

  function clearChat() {
    setChat([]);
    setChatInput("");
  }

  async function runDemo() {
    setLoading(true);
    setAgentLoading(true);
    setAgent(null);
    setActiveTab("answer");

    const body = JSON.stringify({
      query,
      history: Array.from(history),
      top_k: 8,
    });

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

  const rankDelta = useMemo(() => {
    if (!result) return new Map<string, number>();
    const baseRank = new Map(result.baseline.map((b) => [b.uid, b.rank]));
    const m = new Map<string, number>();
    for (const s of result.session_rerank) {
      const br = baseRank.get(s.uid);
      if (br == null) m.set(s.uid, 99);
      else m.set(s.uid, br - s.rank);
    }
    return m;
  }, [result]);

  return (
    <main>
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
            History-aware tool re-ranking for LLM agents, with benchmark and demo results loaded directly from metrics.json.
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
          {metrics?.config?.encoder && (
            <div className="hero-timestamp">
              Encoder: <code>{metrics.config.encoder}</code>
              {metrics?.config?.corpus_size ? (
                <>
                  {" · corpus: "}
                  <strong>{metrics.config.corpus_size.toLocaleString()}</strong>
                </>
              ) : null}
            </div>
          )}
        </div>
      </header>

      <section className="container">
        <div className="kpi-grid">
          <div className="kpi">
            <div className="kpi-value">
              {meanDeltaPct == null ? "—" : `${meanDeltaPct >= 0 ? "+" : ""}${meanDeltaPct.toFixed(1)}%`}
            </div>
            <div className="kpi-label">Mean NDCG@5 gain vs Dense</div>
            <div className="kpi-sub">Averaged over 6 ToolBench splits</div>
          </div>
          <div className="kpi">
            <div className="kpi-value">
              {peakDeltaPct ? `${peakDeltaPct.pct >= 0 ? "+" : ""}${peakDeltaPct.pct.toFixed(1)}%` : "—"}
            </div>
            <div className="kpi-label">
              Peak NDCG@5 gain {peakDeltaPct ? `(${peakDeltaPct.split})` : ""}
            </div>
            <div className="kpi-sub">Largest relative session-regime improvement</div>
          </div>
          <div className="kpi">
            <div className="kpi-value">
              {sessionPValue == null ? "—" : `p < ${sessionPValue.toExponential(1)}`}
            </div>
            <div className="kpi-label">Session significance sample</div>
            <div className="kpi-sub">G2_instruction · NDCG@5</div>
          </div>
          <div className="kpi">
            <div className="kpi-value">
              {metrics?.config?.corpus_size?.toLocaleString?.() ?? corpusSize.toLocaleString()}
            </div>
            <div className="kpi-label">APIs evaluated</div>
            <div className="kpi-sub">Loaded from metrics and demo index</div>
          </div>
        </div>
      </section>

      <section className="container section">
        <h2>Method</h2>
        <p className="body">
          SessionRerank augments a dense bi-encoder retriever with an <strong>interaction-history prior</strong> learned from previous tool-call traces in the same agent session.
        </p>
        <pre className="math">{`score(q, a | H) = w_base · sim(q, a)
                + w_cat  · cat_affinity(a, H)
                + w_sch  · schema_affinity(a, H)
                + w_ept  · endpoint_affinity(a, H)`}</pre>
        <p className="body">
          With empty history, the prior collapses to zero and the model reduces to the Dense baseline.
          {metrics?.config?.weights && (
            <>
              {" "}Loaded weights: <code>w_base={num(metrics.config.weights.w_base, 2)}</code>,{" "}
              <code>w_cat={num(metrics.config.weights.w_cat, 2)}</code>,{" "}
              <code>w_sch={num(metrics.config.weights.w_sch, 2)}</code>,{" "}
              <code>w_ept={num(metrics.config.weights.w_ept, 2)}</code>.
            </>
          )}
        </p>
      </section>

      <section className="container section">
        <h2>Results — NDCG@5 (session regime)</h2>
        <p className="body">
          Full evaluation on the ToolBench benchmark splits. Best per row appears in <strong className="winner">bold green</strong>.
          {seedsRun && seedsRun.length > 1 && (
            <>
              {" "}Numbers are mean across <strong>{seedsRun.length} seeds</strong>; the ± value is the standard deviation across seeds.
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
                      <td>{labelSplit(split)}</td>
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
                const ours = valOf(row["SessionRerank(Dense (all-MiniLM-L6-v2))"]?.["NDCG@5"]);
                const delta =
                  dense != null && ours != null && dense !== 0 ? ((ours - dense) / dense) * 100 : null;

                return (
                  <tr key={split}>
                    <td className="mono">{labelSplit(split)}</td>
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
                    <td className={typeof delta === "number" && delta >= 15 ? "delta-strong" : "delta"}>
                      {delta == null ? "—" : `${delta >= 0 ? "+" : ""}${delta.toFixed(1)}%`}
                    </td>
                  </tr>
                );
              })}
              {metrics && (
                <tr className="mean-row">
                  <td className="mono">
                    <strong>Mean</strong>
                  </td>
                  {tableMethods.map((m) => {
                    const vs = SPLITS.map((s) =>
                      valOf(metrics.splits?.[s]?.regimes?.session?.[m]?.["NDCG@5"])
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
                      {meanDeltaPct == null ? "—" : `${meanDeltaPct >= 0 ? "+" : ""}${meanDeltaPct.toFixed(1)}%`}
                    </strong>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      {Object.keys(stages).length > 0 && (
        <section className="container section">
          <h2>Stage-by-Stage Benchmark</h2>
          <p className="body">
            Each cell shows the best NDCG@5 found within that stage’s session-regime row, plus the winning method name.
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
                    if (!row) {
                      return { label, value: null as number | null, std: null as number | null, method: "" };
                    }

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

                    return {
                      label,
                      value: bestV === -Infinity ? null : bestV,
                      std: bestSd,
                      method: bestM,
                    };
                  });

                  return (
                    <tr key={split}>
                      <td className="mono">{labelSplit(split)}</td>
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
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section className="container section">
        <h2>Publication-Ready Figures</h2>
        <p className="body">Only figures that exist in the public folder are rendered below.</p>
        <div className="figure-grid">
          {figures.map((f) => (
            <figure key={f.file} className="figure-card">
              <img src={`/figures/${f.file}`} alt={f.caption} />
              <figcaption>{f.caption}</figcaption>
            </figure>
          ))}
        </div>
      </section>

      <section className="container section" id="demo">
        <h2>Live Demo — Ask the Agent</h2>
        <p className="body">
          Chat with the agent, or run rerank to compare baseline retrieval against session-aware reranking.
        </p>

        <div className="demo-panel">
          <div className="demo-controls">
            <label className="field">
              <span className="field-label">Query</span>
              <textarea value={query} onChange={(e) => setQuery(e.target.value)} rows={3} />
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
                {filteredSample.length === 0 && <div className="muted">No APIs match that filter.</div>}
              </div>
              <div className="muted small">
                Showing {filteredSample.length} of {sample.length} sample APIs · full benchmark uses{" "}
                {corpusSize.toLocaleString()} APIs
              </div>
            </label>

            <button className="btn btn-primary" onClick={runDemo} disabled={loading || !query.trim()}>
              {loading ? "Ranking…" : "Run rerank"}
            </button>
          </div>

          <div className="output-panel" aria-live="polite">
            <div className="output-tabs" role="tablist" aria-label="Demo output">
              <button
                role="tab"
                aria-selected={activeTab === "chat"}
                className={`output-tab ${activeTab === "chat" ? "active" : ""}`}
                onClick={() => setActiveTab("chat")}
              >
                <span className="tab-icon" aria-hidden="true">💭</span>
                <span className="tab-label">Free chat</span>
                {chat.length > 0 && (
                  <span className="tab-count">{chat.filter((m) => m.role === "user").length}</span>
                )}
              </button>

              <button
                role="tab"
                aria-selected={activeTab === "answer"}
                className={`output-tab ${activeTab === "answer" ? "active" : ""}`}
                onClick={() => setActiveTab("answer")}
                disabled={!agent && !agentLoading}
              >
                <span className="tab-icon" aria-hidden="true">💬</span>
                <span className="tab-label">Single-shot answer</span>
              </button>

              <button
                role="tab"
                aria-selected={activeTab === "retrieval"}
                className={`output-tab ${activeTab === "retrieval" ? "active" : ""}`}
                onClick={() => setActiveTab("retrieval")}
                disabled={!result}
              >
                <span className="tab-icon" aria-hidden="true">📊</span>
                <span className="tab-label">Retrieval &amp; re-ranking</span>
              </button>
            </div>

            {activeTab === "chat" && (
              <div className="output-pane chat-pane" role="tabpanel">
                <div className="chat-header">
                  <div>
                    <h3 className="chat-title">Conversation</h3>
                    <p className="chat-sub">Free-form follow-ups grounded in SessionRerank.</p>
                  </div>
                  {chat.length > 0 && (
                    <button className="link-btn" onClick={clearChat} disabled={chatSending} type="button">
                      Clear chat
                    </button>
                  )}
                </div>

                <div className="chat-log" role="log">
                  {chat.length === 0 && (
                    <div className="chat-empty">
                      <p>
                        Ask anything — for example, <em>I need to translate text then convert it to speech.</em>
                      </p>
                    </div>
                  )}
                  {chat.map((m) => (
                    <div key={m.id} className={`chat-msg chat-msg-${m.role}`}>
                      <div className="chat-bubble">
                        {m.pending ? (
                          <div className="chat-typing" aria-label="thinking">
                            <span />
                            <span />
                            <span />
                          </div>
                        ) : (
                          <div className="chat-body">{m.content}</div>
                        )}
                        {!m.pending && m.role === "assistant" && (
                          <div className="chat-meta">
                            {m.path && (
                              <span className={`path-badge path-${m.path}`}>{PATH_LABEL[m.path]}</span>
                            )}
                            {m.top_apis && m.top_apis.length > 0 && (
                              <span className="chat-grounded">
                                grounded in <strong>{m.top_apis.length}</strong> top APIs
                              </span>
                            )}
                            {m.error && <span className="chat-error">· {m.error}</span>}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>

                <form
                  className="chat-composer"
                  onSubmit={(e) => {
                    e.preventDefault();
                    sendChat();
                  }}
                >
                  <textarea
                    className="chat-input"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    placeholder="Ask a follow-up…"
                    rows={2}
                    disabled={chatSending}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        sendChat();
                      }
                    }}
                  />
                  <button
                    type="submit"
                    className="btn btn-primary chat-send"
                    disabled={chatSending || !chatInput.trim()}
                  >
                    {chatSending ? "…" : "Send"}
                  </button>
                </form>
              </div>
            )}

            {activeTab === "answer" && (agent || agentLoading) && (
              <div className="output-pane" role="tabpanel">
                {agent ? (
                  <div className="answer-card">
                    <div className="answer-header">
                      <h3>Agent response</h3>
                      <span className={`path-badge path-${agent.path}`}>{PATH_LABEL[agent.path]}</span>
                    </div>
                    <div className="answer-body">{agent.answer}</div>
                    <div className="answer-meta">
                      <span>
                        history size: <strong>{agent.history_size}</strong>
                      </span>
                      <span>·</span>
                      <span>
                        grounded in <strong>{agent.top_apis.length}</strong> top APIs
                      </span>
                    </div>
                    {agent.path_error && <div className="agent-error">{agent.path_error}</div>}
                  </div>
                ) : (
                  <div className="answer-card answer-loading">
                    <div className="answer-header">
                      <h3>Agent response</h3>
                      <span className="muted small">thinking…</span>
                    </div>
                    <div className="skeleton-lines" aria-hidden="true">
                      <span />
                      <span />
                      <span />
                    </div>
                  </div>
                )}
              </div>
            )}

            {activeTab === "retrieval" && result && (
              <div className="output-pane" role="tabpanel">
                <p className="body small muted" style={{ marginBottom: 14 }}>
                  Side-by-side comparison of the Dense baseline and SessionRerank for the same query.
                </p>
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
                      SessionRerank <span className="muted small">· history size {result.history_size}</span>
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
                              {delta > 0 && <span className="badge up">▲ {delta}</span>}
                              {delta < 0 && delta > -50 && (
                                <span className="badge down">▼ {Math.abs(delta)}</span>
                              )}
                              {delta >= 50 && <span className="badge new">NEW</span>}
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
              </div>
            )}
          </div>
        </div>
      </section>

      <footer className="footer">
        <div className="container">
          <div>SessionRerank · ToolBench session-aware re-ranking · MIT License</div>
          <div className="muted small">
            Implementation, benchmark, and notebook on{" "}
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
