"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";

type SessionRow = {
  split: string;
  bm25: number | null;
  dense: number | null;
  tool: number | null;
  session: number | null;
};

type StageRow = {
  split: string;
  stage?: string;
  value?: number;
  winner?: string;
};

type FigureDef = {
  id: string;
  title: string;
  file: string;
  alt: string;
};

type MetricsData = {
  corpus_size?: number;
  encoder?: string;
  weights?: {
    w_base?: number;
    w_cat?: number;
    w_sch?: number;
    w_ept?: number;
  };
  session_regime?: {
    rows?: Array<Record<string, any>>;
    hit5_rows?: Array<Record<string, any>>;
  };
  stage_benchmark?: {
    rows?: Array<Record<string, any>>;
  };
  open_corpus_single_turn?: Record<string, any>;
  figures?: Record<string, any>;
  significance_sample?: {
    split?: string;
    metric?: string;
    apis_evaluated?: number;
  };
};

const FIGURES: FigureDef[] = [
  {
    id: "fig1",
    title: "Figure 1. NDCG@5 across all six ToolBench evaluation splits in the session regime.",
    file: "/figures/figure1-ndcg5-session-regime.png",
    alt: "NDCG at 5 across ToolBench session-regime splits",
  },
  {
    id: "fig2",
    title: "Figure 2. Hit@5 across all six evaluation splits in the session regime.",
    file: "/figures/figure2-hit5-session-regime.png",
    alt: "Hit at 5 across ToolBench session-regime splits",
  },
  {
    id: "fig3",
    title: "Figure 3. NDCG@K for K={1,3,5,10} on G2_instruction.",
    file: "/figures/figure3-g2-instruction-ndcgk.png",
    alt: "NDCG at K on G2 instruction",
  },
  {
    id: "fig4",
    title: "Figure 4. Relative improvement of SessionRerank over the Dense retrieval baseline.",
    file: "/figures/figure4-delta-vs-dense.png",
    alt: "Relative improvement versus Dense baseline",
  },
  {
    id: "fig5",
    title: "Figure 5. Heatmap of NDCG@5 across methods and splits.",
    file: "/figures/figure5-heatmap-ndcg5.png",
    alt: "Heatmap of NDCG at 5 across methods and splits",
  },
  {
    id: "fig6",
    title: "Figure 6. Open-corpus single-turn regime with no session history.",
    file: "/figures/figure6-open-corpus-single-turn.png",
    alt: "Open-corpus single-turn benchmark without session history",
  },
];

function asNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function mean(values: Array<number | null>): number | null {
  const valid = values.filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  if (!valid.length) return null;
  return valid.reduce((a, b) => a + b, 0) / valid.length;
}

function relGainPct(improved: number | null, baseline: number | null): number | null {
  if (improved == null || baseline == null || baseline === 0) return null;
  return ((improved - baseline) / baseline) * 100;
}

function fmtPct(value: number | null, digits = 1): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}%`;
}

function fmtNum(value: number | null, digits = 3): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

function fmtInt(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return new Intl.NumberFormat("en-US").format(value);
}

function extractSessionRows(metrics: MetricsData): SessionRow[] {
  const raw = metrics.session_regime?.rows ?? [];
  return raw.map((row) => ({
    split: String(row.split ?? row.name ?? "Unknown split"),
    bm25: asNumber(row.bm25_ndcg5 ?? row.bm25),
    dense: asNumber(row.dense_ndcg5 ?? row.dense),
    tool: asNumber(row.toolrerank_ndcg5 ?? row.toolrerank ?? row.tool_rerank),
    session: asNumber(row.sessionrerank_ndcg5 ?? row.sessionrerank ?? row.session_rerank),
  }));
}

function extractStageRows(metrics: MetricsData): StageRow[] {
  const raw = metrics.stage_benchmark?.rows ?? [];
  return raw.map((row) => ({
    split: String(row.split ?? row.name ?? "Unknown split"),
    stage: row.stage ? String(row.stage) : "Stage 4 — current metrics",
    value: asNumber(row.value ?? row.best_ndcg5 ?? row.ndcg5),
    winner: row.winner ? String(row.winner) : undefined,
  }));
}

function winnerName(row: SessionRow): string {
  const pairs = [
    { name: "BM25", value: row.bm25 },
    { name: "Dense", value: row.dense },
    { name: "ToolRerank", value: row.tool },
    { name: "SessionRerank", value: row.session },
  ].filter((x): x is { name: string; value: number } => x.value != null);

  if (!pairs.length) return "—";
  const best = pairs.reduce((a, b) => (b.value > a.value ? b : a));
  return best.name;
}

function maxBy<T>(items: T[], getValue: (item: T) => number | null): T | null {
  let best: T | null = null;
  let bestValue = -Infinity;
  for (const item of items) {
    const value = getValue(item);
    if (value == null || !Number.isFinite(value)) continue;
    if (best == null || value > bestValue) {
      best = item;
      bestValue = value;
    }
  }
  return best;
}

export default function Page() {
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [figureAvailability, setFigureAvailability] = useState<Record<string, boolean>>({});

  useEffect(() => {
    async function load() {
      try {
        const res = await fetch("/metrics.json", { cache: "no-store" });
        const data = await res.json();
        setMetrics(data);
      } catch (error) {
        console.error("Failed to load metrics.json", error);
        setMetrics({});
      }
    }
    load();
  }, []);

  useEffect(() => {
    async function checkFigures() {
      const entries = await Promise.all(
        FIGURES.map(async (fig) => {
          try {
            const res = await fetch(fig.file, { method: "HEAD" });
            return [fig.id, res.ok] as const;
          } catch {
            return [fig.id, false] as const;
          }
        })
      );
      setFigureAvailability(Object.fromEntries(entries));
    }
    checkFigures();
  }, []);

  const sessionRows = useMemo(() => (metrics ? extractSessionRows(metrics) : []), [metrics]);
  const stageRows = useMemo(() => (metrics ? extractStageRows(metrics) : []), [metrics]);

  const ndcgRows = useMemo(() => {
    return sessionRows.map((row) => ({
      ...row,
      deltaDense: relGainPct(row.session, row.dense),
    }));
  }, [sessionRows]);

  const means = useMemo(() => {
    return {
      bm25: mean(sessionRows.map((r) => r.bm25)),
      dense: mean(sessionRows.map((r) => r.dense)),
      tool: mean(sessionRows.map((r) => r.tool)),
      session: mean(sessionRows.map((r) => r.session)),
      deltaDense: mean(ndcgRows.map((r) => r.deltaDense)),
    };
  }, [sessionRows, ndcgRows]);

  const peakDeltaRow = useMemo(
    () => maxBy(ndcgRows, (r) => r.deltaDense),
    [ndcgRows]
  );

  const significanceSample = metrics?.significance_sample;
  const corpusSize = metrics?.corpus_size;
  const encoder = metrics?.encoder;
  const weights = metrics?.weights;

  const figureInsights = useMemo(() => {
    if (!metrics) return {};

    const validRows = ndcgRows.filter(
      (r) => r.bm25 != null && r.dense != null && r.tool != null && r.session != null
    );

    const sessionWinsAll = validRows.length
      ? validRows.every(
          (r) =>
            (r.session ?? -Infinity) >= (r.bm25 ?? -Infinity) &&
            (r.session ?? -Infinity) >= (r.dense ?? -Infinity) &&
            (r.session ?? -Infinity) >= (r.tool ?? -Infinity)
        )
      : false;

    const hit5Rows = metrics.session_regime?.hit5_rows ?? [];
    const hit5Peak = maxBy(
      hit5Rows
        .map((row) => ({
          split: String(row.split ?? row.name ?? "Unknown split"),
          deltaDense: relGainPct(
            asNumber(row.sessionrerank_hit5 ?? row.sessionrerank ?? row.session_rerank),
            asNumber(row.dense_hit5 ?? row.dense)
          ),
        })),
      (r) => r.deltaDense
    );

    const g2Curve =
      metrics.figures?.g2_instruction_ndcgk ??
      metrics.figures?.figure3 ??
      null;

    let dominatesAllCutoffs = false;
    if (
      g2Curve?.k &&
      g2Curve?.sessionrerank &&
      g2Curve?.dense &&
      g2Curve?.bm25 &&
      g2Curve?.toolrerank
    ) {
      dominatesAllCutoffs = g2Curve.k.every((_: unknown, i: number) => {
        const s = asNumber(g2Curve.sessionrerank[i]);
        const d = asNumber(g2Curve.dense[i]);
        const b = asNumber(g2Curve.bm25[i]);
        const t = asNumber(g2Curve.toolrerank[i]);
        if (s == null || d == null || b == null || t == null) return false;
        return s >= d && s >= b && s >= t;
      });
    }

    const singleTurn = metrics.open_corpus_single_turn;
    const singleTurnDelta = singleTurn
      ? relGainPct(
          asNumber(singleTurn.sessionrerank_ndcg5 ?? singleTurn.sessionrerank),
          asNumber(singleTurn.dense_ndcg5 ?? singleTurn.dense)
        )
      : null;

    return {
      fig1: [
        sessionWinsAll
          ? "SessionRerank outperforms BM25, Dense, and ToolRerank on every loaded session-regime split."
          : "SessionRerank is strongest on most loaded session-regime splits.",
        peakDeltaRow
          ? `Its largest relative gain over Dense appears on ${peakDeltaRow.split} at ${fmtPct(
              peakDeltaRow.deltaDense
            )}.`
          : null,
      ]
        .filter(Boolean)
        .join(" "),

      fig2: [
        "Hit@5 insight is generated from the loaded session-regime results.",
        hit5Peak
          ? `The largest relative gain over Dense appears on ${hit5Peak.split} at ${fmtPct(
              hit5Peak.deltaDense
            )}.`
          : null,
      ]
        .filter(Boolean)
        .join(" "),

      fig3: [
        "The plotted NDCG@K comparison is read from the loaded cutoff metrics.",
        g2Curve
          ? dominatesAllCutoffs
            ? "SessionRerank leads across every shown cutoff."
            : "The best method varies across cutoffs in the loaded data."
          : "Cutoff-specific insight will appear when curve data is present in metrics.json.",
      ].join(" "),

      fig4: [
        "Δ (delta) vs Dense is computed as the relative percentage improvement of SessionRerank over Dense.",
        means.deltaDense != null
          ? `The mean split-wise gain from the loaded metrics is ${fmtPct(means.deltaDense)}${
              peakDeltaRow
                ? `, with the largest gain on ${peakDeltaRow.split} at ${fmtPct(
                    peakDeltaRow.deltaDense
                  )}.`
                : "."
            }`
          : "Mean delta will appear once split metrics are loaded.",
      ].join(" "),

      fig5: [
        "The heatmap is rendered from the same NDCG@5 matrix used in the benchmark table.",
        peakDeltaRow
          ? `The warmest SessionRerank advantage should align with ${peakDeltaRow.split}, where the gain over Dense is ${fmtPct(
              peakDeltaRow.deltaDense
            )}.`
          : null,
      ]
        .filter(Boolean)
        .join(" "),

      fig6: [
        "This open-corpus single-turn figure uses the loaded no-history benchmark.",
        singleTurnDelta != null
          ? `SessionRerank changes by ${fmtPct(
              singleTurnDelta
            )} relative to Dense, showing how closely it tracks the baseline when history is empty.`
          : "SessionRerank should reduce toward the Dense baseline when session history is absent.",
      ].join(" "),
    };
  }, [metrics, ndcgRows, means.deltaDense, peakDeltaRow]);

  const visibleFigures = FIGURES.filter((fig) => figureAvailability[fig.id]);

  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-100">
      <section className="mx-auto max-w-6xl px-6 py-14 md:px-8">
        <div className="mb-4 text-xs font-medium uppercase tracking-[0.2em] text-emerald-400/80">
          ToolBench · Session-Aware Re-Ranking · Gemma 4 Good Hackathon Submission
        </div>

        <div className="flex flex-col gap-6 md:flex-row md:items-end md:justify-between">
          <div className="max-w-3xl">
            <h1 className="text-4xl font-semibold tracking-tight text-white md:text-6xl">
              SessionRerank
            </h1>
            <p className="mt-4 text-base leading-7 text-neutral-300 md:text-lg">
              History-aware tool re-ranking for LLM agents, with benchmark and demo results loaded directly from <code className="rounded bg-white/5 px-1.5 py-0.5 text-sm">metrics.json</code>.
            </p>

            <div className="mt-6 flex flex-wrap gap-3">
              <Link
                href="https://github.com/joyjeni/session-aware-toolbench-rerank"
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-white transition hover:bg-white/10"
              >
                View on GitHub
              </Link>
              <a
                href="#demo"
                className="rounded-full bg-emerald-400 px-4 py-2 text-sm font-medium text-neutral-950 transition hover:bg-emerald-300"
              >
                Try the live demo ↓
              </a>
            </div>
          </div>

          <div className="grid w-full max-w-xl grid-cols-1 gap-3 sm:grid-cols-2">
            <StatCard
              label="Encoder"
              value={encoder ?? "—"}
              subtext={`Corpus: ${fmtInt(corpusSize)}`}
            />
            <StatCard
              label="Mean NDCG@5 gain vs Dense"
              value={fmtPct(means.deltaDense)}
              subtext={`Averaged over ${sessionRows.length || "—"} ToolBench splits`}
            />
            <StatCard
              label="Peak NDCG@5 gain"
              value={peakDeltaRow ? fmtPct(peakDeltaRow.deltaDense) : "—"}
              subtext={peakDeltaRow ? `(${peakDeltaRow.split})` : "Largest relative session-regime improvement"}
            />
            <StatCard
              label="Session significance sample"
              value={
                significanceSample?.split
                  ? `${significanceSample.split} · ${significanceSample.metric ?? "NDCG@5"}`
                  : "—"
              }
              subtext={`${fmtInt(significanceSample?.apis_evaluated)} APIs evaluated`}
            />
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 pb-8 md:px-8">
        <div className="rounded-3xl border border-white/10 bg-white/[0.03] p-6 md:p-8">
          <h2 className="text-2xl font-semibold text-white">Method</h2>
          <p className="mt-4 max-w-4xl text-neutral-300">
            SessionRerank augments a dense bi-encoder retriever with an interaction-history prior learned from previous tool-call traces in the same agent session.
          </p>

          <div className="mt-6 overflow-x-auto rounded-2xl border border-white/10 bg-black/20 p-4">
            <pre className="text-sm leading-7 text-emerald-200/90">
{`score(q, a | H) = w_base · sim(q, a) + w_cat · cat_affinity(a, H) + w_sch · schema_affinity(a, H) + w_ept · endpoint_affinity(a, H)`}
            </pre>
          </div>

          <p className="mt-4 text-neutral-300">
            With empty history, the prior collapses to zero and the model reduces to the Dense baseline.
          </p>

          <div className="mt-5 flex flex-wrap gap-3 text-sm text-neutral-300">
            <Pill label="w_base" value={weights?.w_base} />
            <Pill label="w_cat" value={weights?.w_cat} />
            <Pill label="w_sch" value={weights?.w_sch} />
            <Pill label="w_ept" value={weights?.w_ept} />
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 py-8 md:px-8">
        <h2 className="text-2xl font-semibold text-white">Results — NDCG@5 (session regime)</h2>
        <p className="mt-3 text-neutral-300">
          Full evaluation on the ToolBench benchmark splits. Δ (delta) vs Dense is the relative percentage improvement of SessionRerank over the Dense baseline.
        </p>

        <div className="mt-6 overflow-hidden rounded-3xl border border-white/10 bg-white/[0.03]">
          <div className="overflow-x-auto">
            <table className="min-w-full text-left text-sm">
              <thead className="bg-white/5 text-neutral-200">
                <tr>
                  <Th>Split</Th>
                  <Th>BM25</Th>
                  <Th>Dense</Th>
                  <Th>ToolRerank</Th>
                  <Th>SessionRerank</Th>
                  <Th>Δ (delta) vs Dense</Th>
                </tr>
              </thead>
              <tbody>
                {ndcgRows.map((row) => {
                  const best = Math.max(
                    row.bm25 ?? -Infinity,
                    row.dense ?? -Infinity,
                    row.tool ?? -Infinity,
                    row.session ?? -Infinity
                  );

                  return (
                    <tr key={row.split} className="border-t border-white/10">
                      <Td>{row.split}</Td>
                      <Td highlight={row.bm25 === best}>{fmtNum(row.bm25)}</Td>
                      <Td highlight={row.dense === best}>{fmtNum(row.dense)}</Td>
                      <Td highlight={row.tool === best}>{fmtNum(row.tool)}</Td>
                      <Td highlight={row.session === best}>{fmtNum(row.session)}</Td>
                      <Td className="text-emerald-300">{fmtPct(row.deltaDense)}</Td>
                    </tr>
                  );
                })}

                <tr className="border-t border-white/10 bg-white/5 font-semibold">
                  <Td>Mean</Td>
                  <Td>{fmtNum(means.bm25)}</Td>
                  <Td>{fmtNum(means.dense)}</Td>
                  <Td>{fmtNum(means.tool)}</Td>
                  <Td>{fmtNum(means.session)}</Td>
                  <Td className="text-emerald-300">{fmtPct(means.deltaDense)}</Td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 py-8 md:px-8">
        <h2 className="text-2xl font-semibold text-white">Stage-by-Stage Benchmark</h2>
        <p className="mt-3 text-neutral-300">
          Each cell shows the best NDCG@5 found within that stage row, plus the winning method.
        </p>

        <div className="mt-6 overflow-hidden rounded-3xl border border-white/10 bg-white/[0.03]">
          <div className="overflow-x-auto">
            <table className="min-w-full text-left text-sm">
              <thead className="bg-white/5 text-neutral-200">
                <tr>
                  <Th>Split</Th>
                  <Th>{stageRows[0]?.stage ?? "Stage 4 — current metrics"}</Th>
                </tr>
              </thead>
              <tbody>
                {(stageRows.length ? stageRows : sessionRows.map((row) => ({
                  split: row.split,
                  stage: "Stage 4 — current metrics",
                  value: Math.max(
                    row.bm25 ?? -Infinity,
                    row.dense ?? -Infinity,
                    row.tool ?? -Infinity,
                    row.session ?? -Infinity
                  ),
                  winner: winnerName(row),
                }))).map((row) => (
                  <tr key={row.split} className="border-t border-white/10">
                    <Td>{row.split}</Td>
                    <Td>
                      <span className="font-semibold text-emerald-300">{fmtNum(row.value ?? null)}</span>
                      <span className="ml-2 text-neutral-300">{row.winner ?? "—"}</span>
                    </Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 py-8 md:px-8">
        <h2 className="text-2xl font-semibold text-white">Publication-Ready Figures</h2>
        <p className="mt-3 text-neutral-300">
          Only figures that exist in the public folder are rendered below, and the insights beneath each figure are generated dynamically from the loaded metrics.
        </p>

        <div className="mt-8 grid gap-6">
          {visibleFigures.map((fig) => (
            <figure
              key={fig.id}
              className="overflow-hidden rounded-3xl border border-white/10 bg-white/[0.03]"
            >
              <img
                src={fig.file}
                alt={fig.alt}
                className="w-full object-cover"
              />
              <figcaption className="border-t border-white/10 px-5 py-4 text-sm leading-7 text-neutral-300">
                <span className="font-medium text-white">{fig.title}</span>{" "}
                {String((figureInsights as Record<string, string>)[fig.id] ?? "")}
              </figcaption>
            </figure>
          ))}

          {!visibleFigures.length && (
            <div className="rounded-3xl border border-dashed border-white/10 bg-white/[0.03] p-6 text-neutral-400">
              No figure images were found in the public folder.
            </div>
          )}
        </div>
      </section>

      <section id="demo" className="mx-auto max-w-6xl px-6 py-8 md:px-8">
        <div className="rounded-3xl border border-white/10 bg-white/[0.03] p-6 md:p-8">
          <h2 className="text-2xl font-semibold text-white">Live Demo — Ask the Agent</h2>
          <p className="mt-3 max-w-3xl text-neutral-300">
            Chat with the agent, or run rerank to compare baseline retrieval against session-aware reranking.
          </p>

          <div className="mt-6 rounded-2xl border border-white/10 bg-black/20 p-4 text-sm text-neutral-300">
            Keep your existing demo UI here. This page structure is focused on removing hard-coded benchmark values and generating captions, deltas, and summary stats dynamically from <code className="rounded bg-white/5 px-1 py-0.5">metrics.json</code>.
          </div>
        </div>
      </section>

      <footer className="mx-auto max-w-6xl px-6 py-12 text-sm text-neutral-400 md:px-8">
        SessionRerank · ToolBench session-aware re-ranking · MIT License ·{" "}
        <Link
          href="https://github.com/joyjeni/session-aware-toolbench-rerank"
          target="_blank"
          rel="noopener noreferrer"
          className="text-neutral-200 underline underline-offset-4"
        >
          GitHub
        </Link>
      </footer>
    </main>
  );
}

function StatCard({
  label,
  value,
  subtext,
}: {
  label: string;
  value: string;
  subtext: string;
}) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
      <div className="text-xs uppercase tracking-[0.16em] text-neutral-400">{label}</div>
      <div className="mt-2 text-2xl font-semibold text-white">{value}</div>
      <div className="mt-1 text-sm text-neutral-400">{subtext}</div>
    </div>
  );
}

function Pill({ label, value }: { label: string; value?: number }) {
  return (
    <div className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5">
      <span className="text-neutral-400">{label}=</span>
      <span className="ml-1 text-white">{value != null ? value.toFixed(2) : "—"}</span>
    </div>
  );
}

function Th({ children }: { children: React.ReactNode }) {
  return <th className="px-4 py-3 font-medium">{children}</th>;
}

function Td({
  children,
  highlight = false,
  className = "",
}: {
  children: React.ReactNode;
  highlight?: boolean;
  className?: string;
}) {
  return (
    <td
      className={`px-4 py-3 ${highlight ? "font-semibold text-emerald-300" : "text-neutral-200"} ${className}`}
    >
      {children}
    </td>
  );
}