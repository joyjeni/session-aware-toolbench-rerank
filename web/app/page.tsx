"use client";

import React, { useEffect, useMemo, useState } from "react";

type MetricMap = Record<string, number>;
type MethodMetrics = Record<string, MetricMap>;

type SplitData = {
  n_queries?: number;
  regimes?: {
    closed?: MethodMetrics;
    open?: MethodMetrics;
    session?: MethodMetrics;
  };
};

type SignificanceEntry = {
  p_value?: number;
  delta_mean?: number;
  delta_rel_pct?: number;
};

type MetricsData = {
  config?: {
    encoder?: string;
    device?: string;
    corpus_size?: number;
    weights?: {
      w_base?: number;
      w_cat?: number;
      w_sch?: number;
      w_ept?: number;
    };
    ks?: number[];
  };
  splits?: Record<string, SplitData>;
  significance__SessionRerank_vs_Dense?: Record<
    string,
    Record<string, Record<string, SignificanceEntry>>
  >;
};

type SessionRow = {
  split: string;
  nQueries: number | null;
  bm25: number | null;
  dense: number | null;
  tool: number | null;
  session: number | null;
  deltaDense: number | null;
  deltaTool: number | null;
};

type FigureCard = {
  id: string;
  title: string;
  src: string;
};

const FIGURES: FigureCard[] = [
  { id: "fig1", title: "Figure 1 · Session-regime NDCG@5 by split", src: "/figures/figure1.png" },
  { id: "fig2", title: "Figure 2 · Significance summary", src: "/figures/figure2.png" },
  { id: "fig3", title: "Figure 3 · Best absolute session performance", src: "/figures/figure3.png" },
  { id: "fig4", title: "Figure 4 · Δ (delta) vs Dense", src: "/figures/figure4.png" },
  { id: "fig5", title: "Figure 5 · Hit@5 session-aware gains", src: "/figures/figure5.png" },
  { id: "fig6", title: "Figure 6 · Open vs session comparison", src: "/figures/figure6.png" },
];

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function fmtNum(value: number | null | undefined, digits = 3): string {
  return value == null ? "—" : value.toFixed(digits);
}

function fmtPct(value: number | null | undefined, digits = 1): string {
  return value == null ? "—" : `${value >= 0 ? "+" : ""}${value.toFixed(digits)}%`;
}

function fmtP(value: number | null | undefined): string {
  if (value == null) return "—";
  if (value === 0) return "0";
  if (value < 0.001) return value.toExponential(2);
  return value.toFixed(4);
}

function relativePct(base: number | null, value: number | null): number | null {
  if (base == null || value == null || base === 0) return null;
  return ((value - base) / base) * 100;
}

function mean(values: Array<number | null | undefined>): number | null {
  const nums = values.filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  if (!nums.length) return null;
  return nums.reduce((a, b) => a + b, 0) / nums.length;
}

function maxBy<T>(items: T[], getValue: (item: T) => number | null): T | null {
  let best: T | null = null;
  let bestValue = -Infinity;

  for (const item of items) {
    const value = getValue(item);
    if (value == null) continue;
    if (best == null || value > bestValue) {
      best = item;
      bestValue = value;
    }
  }
  return best;
}

function splitLabel(split: string): string {
  return split.replaceAll("_", " · ");
}

function extractSessionRows(metrics: MetricsData): SessionRow[] {
  const splits = metrics.splits ?? {};

  return Object.entries(splits)
    .map(([splitName, splitData]) => {
      const session = splitData.regimes?.session ?? {};

      const bm25 = asNumber(session["BM25"]?.["NDCG@5"]);
      const dense = asNumber(session["Dense (all-MiniLM-L6-v2)"]?.["NDCG@5"]);
      const tool = asNumber(session["ToolRerank"]?.["NDCG@5"]);
      const sessionRerank = asNumber(
        session["SessionRerank(Dense (all-MiniLM-L6-v2))"]?.["NDCG@5"]
      );

      return {
        split: splitName,
        nQueries: asNumber(splitData.n_queries),
        bm25,
        dense,
        tool,
        session: sessionRerank,
        deltaDense: relativePct(dense, sessionRerank),
        deltaTool: relativePct(tool, sessionRerank),
      };
    })
    .sort((a, b) => a.split.localeCompare(b.split));
}

function buildFigureInsights(metrics: MetricsData | null, rows: SessionRow[]) {
  if (!metrics) {
    return {
      fig1: "Loading metrics…",
      fig2: "Loading significance results…",
      fig3: "Loading best split…",
      fig4: "Loading delta summary…",
      fig5: "Loading Hit@5 gains…",
      fig6: "Loading regime comparison…",
    };
  }

  const sig = metrics.significance__SessionRerank_vs_Dense ?? {};
  const peakDense = maxBy(rows, (r) => r.deltaDense);
  const bestSession = maxBy(rows, (r) => r.session);

  const ndcg5SigRows = Object.entries(sig)
    .map(([split, regimes]) => ({
      split,
      p: asNumber(regimes?.session?.["NDCG@5"]?.p_value),
      gain: asNumber(regimes?.session?.["NDCG@5"]?.delta_rel_pct),
      meanDelta: asNumber(regimes?.session?.["NDCG@5"]?.delta_mean),
    }))
    .filter((x) => x.gain != null);

  const hit5SigRows = Object.entries(sig)
    .map(([split, regimes]) => ({
      split,
      p: asNumber(regimes?.session?.["Hit@5"]?.p_value),
      gain: asNumber(regimes?.session?.["Hit@5"]?.delta_rel_pct),
      meanDelta: asNumber(regimes?.session?.["Hit@5"]?.delta_mean),
    }))
    .filter((x) => x.gain != null);

  const ndcg10SigRows = Object.entries(sig)
    .map(([split, regimes]) => ({
      split,
      p: asNumber(regimes?.session?.["NDCG@10"]?.p_value),
      gain: asNumber(regimes?.session?.["NDCG@10"]?.delta_rel_pct),
      meanDelta: asNumber(regimes?.session?.["NDCG@10"]?.delta_mean),
    }))
    .filter((x) => x.gain != null);

  const topNdcg5 = maxBy(ndcg5SigRows, (r) => r.gain);
  const topHit5 = maxBy(hit5SigRows, (r) => r.gain);
  const topNdcg10 = maxBy(ndcg10SigRows, (r) => r.gain);

  const meanDeltaDense = mean(rows.map((r) => r.deltaDense));
  const openVsSession = Object.entries(metrics.splits ?? {})
    .map(([split, splitData]) => {
      const openDense = asNumber(splitData.regimes?.open?.["Dense (all-MiniLM-L6-v2)"]?.["NDCG@5"]);
      const sessionRerank = asNumber(
        splitData.regimes?.session?.["SessionRerank(Dense (all-MiniLM-L6-v2))"]?.["NDCG@5"]
      );
      return {
        split,
        improvement: relativePct(openDense, sessionRerank),
      };
    })
    .filter((x) => x.improvement != null);

  const bestOpenToSession = maxBy(openVsSession, (r) => r.improvement);

  return {
    fig1: peakDense
      ? `${splitLabel(peakDense.split)} has the largest relative NDCG@5 gain over Dense in the session regime at ${fmtPct(peakDense.deltaDense)}.`
      : "No session NDCG@5 gain found.",

    fig2: topNdcg5
      ? `The strongest significant NDCG@5 session gain appears on ${splitLabel(topNdcg5.split)} with ${fmtPct(topNdcg5.gain)} relative improvement and p=${fmtP(topNdcg5.p)}.`
      : "No significance summary available.",

    fig3: bestSession
      ? `${splitLabel(bestSession.split)} has the highest absolute SessionRerank NDCG@5 at ${fmtNum(bestSession.session)}.`
      : "No best-performing session split found.",

    fig4:
      meanDeltaDense != null
        ? `Δ (delta) vs Dense is computed as ((SessionRerank - Dense) / Dense) × 100. The mean split-wise NDCG@5 gain is ${fmtPct(meanDeltaDense)}.`
        : "No delta summary available.",

    fig5: topHit5
      ? `For Hit@5, the largest session gain over Dense occurs on ${splitLabel(topHit5.split)} at ${fmtPct(topHit5.gain)} with p=${fmtP(topHit5.p)}.`
      : "No Hit@5 gain data available.",

    fig6: bestOpenToSession
      ? `${splitLabel(bestOpenToSession.split)} shows the largest jump from open Dense to session SessionRerank in NDCG@5 at ${fmtPct(bestOpenToSession.improvement)}.`
      : "No open-to-session comparison available.",
  };
}

function StatCard({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-sm">
      <div className="text-xs uppercase tracking-[0.18em] text-white/55">{label}</div>
      <div className="mt-2 text-2xl font-semibold text-white">{value}</div>
      {hint ? <div className="mt-2 text-sm text-white/60">{hint}</div> : null}
    </div>
  );
}

function Badge({
  children,
  tone = "neutral",
}: {
  children: React.ReactNode;
  tone?: "neutral" | "green" | "blue";
}) {
  const styles =
    tone === "green"
      ? "border-emerald-400/20 bg-emerald-500/15 text-emerald-300"
      : tone === "blue"
      ? "border-cyan-400/20 bg-cyan-500/15 text-cyan-300"
      : "border-white/10 bg-white/5 text-white/70";

  return (
    <span className={`inline-flex rounded-full border px-2.5 py-1 text-xs font-medium ${styles}`}>
      {children}
    </span>
  );
}

function FigurePanel({
  figure,
  insight,
}: {
  figure: FigureCard;
  insight: string;
}) {
  const [failed, setFailed] = useState(false);

  return (
    <article className="overflow-hidden rounded-3xl border border-white/10 bg-white/[0.03]">
      <div className="border-b border-white/10 px-5 py-4">
        <h3 className="text-base font-semibold text-white">{figure.title}</h3>
      </div>

      <div className="aspect-[16/9] bg-black/20">
        {!failed ? (
          <img
            src={figure.src}
            alt={figure.title}
            className="h-full w-full object-contain"
            loading="lazy"
            onError={() => setFailed(true)}
          />
        ) : (
          <div className="flex h-full items-center justify-center px-6 text-center text-sm text-white/45">
            Image missing at {figure.src}
          </div>
        )}
      </div>

      <div className="px-5 py-4 text-sm leading-6 text-white/70">{insight}</div>
    </article>
  );
}

export default function Page() {
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const res = await fetch("/metrics.json", { cache: "no-store" });
        if (!res.ok) throw new Error(`Failed to load metrics.json (${res.status})`);
        const json = (await res.json()) as MetricsData;
        setMetrics(json);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Unknown error");
      }
    }
    load();
  }, []);

  const sessionRows = useMemo(() => (metrics ? extractSessionRows(metrics) : []), [metrics]);

  const meanDeltaDense = useMemo(
    () => mean(sessionRows.map((r) => r.deltaDense)),
    [sessionRows]
  );

  const peakDelta = useMemo(
    () => maxBy(sessionRows, (r) => r.deltaDense),
    [sessionRows]
  );

  const bestSession = useMemo(
    () => maxBy(sessionRows, (r) => r.session),
    [sessionRows]
  );

  const figureInsights = useMemo(
    () => buildFigureInsights(metrics, sessionRows),
    [metrics, sessionRows]
  );

  return (
    <main className="min-h-screen bg-[#07111a] text-white">
      <div className="mx-auto max-w-7xl px-6 py-10">
        <header className="mb-8">
          <p className="mb-2 text-xs uppercase tracking-[0.24em] text-cyan-300/80">
            ToolBench · Session-aware retrieval
          </p>
          <h1 className="text-4xl font-semibold tracking-tight">SessionRerank</h1>
          <p className="mt-3 max-w-3xl text-white/70">
            Tables, summary cards, and figure explanations are rendered directly from metrics.json.
          </p>
        </header>

        {error ? (
          <div className="mb-8 rounded-2xl border border-rose-400/20 bg-rose-500/10 p-4 text-sm text-rose-200">
            Failed to load metrics: {error}
          </div>
        ) : null}

        <section className="grid gap-4 md:grid-cols-3">
          <StatCard
            label="Mean Δ (delta) vs Dense"
            value={fmtPct(meanDeltaDense)}
            hint="Average split-wise session NDCG@5 gain over Dense"
          />
          <StatCard
            label="Peak Δ (delta) vs Dense"
            value={peakDelta ? fmtPct(peakDelta.deltaDense) : "—"}
            hint={peakDelta ? splitLabel(peakDelta.split) : "Largest relative gain"}
          />
          <StatCard
            label="Best session split"
            value={bestSession ? splitLabel(bestSession.split) : "—"}
            hint={bestSession ? `NDCG@5 ${fmtNum(bestSession.session)}` : "Highest absolute session score"}
          />
        </section>

        <section className="mt-10 rounded-3xl border border-white/10 bg-white/[0.03] p-5">
          <div className="mb-4 flex flex-wrap items-center gap-3">
            <h2 className="text-xl font-semibold">Session-regime table</h2>
            <Badge>Δ = relative gain over Dense</Badge>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full min-w-[900px] border-collapse">
              <thead>
                <tr className="border-b border-white/10 text-left text-xs uppercase tracking-[0.16em] text-white/55">
                  <th className="px-4 py-3">Split</th>
                  <th className="px-4 py-3">Queries</th>
                  <th className="px-4 py-3">BM25</th>
                  <th className="px-4 py-3">Dense</th>
                  <th className="px-4 py-3">ToolRerank</th>
                  <th className="px-4 py-3">SessionRerank</th>
                  <th className="px-4 py-3">Δ vs Dense</th>
                  <th className="px-4 py-3">Δ vs Tool</th>
                </tr>
              </thead>
              <tbody>
                {sessionRows.map((row) => (
                  <tr key={row.split} className="border-b border-white/6 hover:bg-white/[0.03]">
                    <td className="px-4 py-3 font-medium">{splitLabel(row.split)}</td>
                    <td className="px-4 py-3">{row.nQueries ?? "—"}</td>
                    <td className="px-4 py-3">{fmtNum(row.bm25)}</td>
                    <td className="px-4 py-3">{fmtNum(row.dense)}</td>
                    <td className="px-4 py-3">{fmtNum(row.tool)}</td>
                    <td className="px-4 py-3">{fmtNum(row.session)}</td>
                    <td className="px-4 py-3">
                      <Badge tone="green">{fmtPct(row.deltaDense)}</Badge>
                    </td>
                    <td className="px-4 py-3">
                      <Badge tone="blue">{fmtPct(row.deltaTool)}</Badge>
                    </td>
                  </tr>
                ))}
                {!sessionRows.length ? (
                  <tr>
                    <td className="px-4 py-5 text-white/50" colSpan={8}>
                      Loading rows from metrics.json…
                    </td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
        </section>

        <section className="mt-10">
          <div className="mb-4">
            <h2 className="text-xl font-semibold">Figures</h2>
            <p className="mt-1 text-sm text-white/60">
              Each figure panel stays visible and explains itself from JSON metrics even if the image file is missing.
            </p>
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            {FIGURES.map((figure) => (
              <FigurePanel
                key={figure.id}
                figure={figure}
                insight={figureInsights[figure.id as keyof typeof figureInsights]}
              />
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}
