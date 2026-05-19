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

const FIGURES = [
  { id: "fig1", src: "/figures/figure1.png", title: "Figure 1 · Session-regime NDCG@5 by split" },
  { id: "fig2", src: "/figures/figure2.png", title: "Figure 2 · Significance summary" },
  { id: "fig3", src: "/figures/figure3.png", title: "Figure 3 · Retrieval overview" },
  { id: "fig4", src: "/figures/figure4.png", title: "Figure 4 · Δ (delta) vs Dense" },
  { id: "fig5", src: "/figures/figure5.png", title: "Figure 5 · Session-aware gains" },
  { id: "fig6", src: "/figures/figure6.png", title: "Figure 6 · Open-corpus view" },
] as const;

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
    <div className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-sm backdrop-blur">
      <div className="text-xs uppercase tracking-[0.18em] text-white/55">{label}</div>
      <div className="mt-2 text-2xl font-semibold text-white">{value}</div>
      {hint ? <div className="mt-2 text-sm text-white/60">{hint}</div> : null}
    </div>
  );
}

function Pill({
  children,
  tone = "neutral",
  title,
}: {
  children: React.ReactNode;
  tone?: "neutral" | "green" | "blue" | "amber";
  title?: string;
}) {
  const toneClass =
    tone === "green"
      ? "bg-emerald-500/15 text-emerald-300 border-emerald-400/20"
      : tone === "blue"
      ? "bg-cyan-500/15 text-cyan-300 border-cyan-400/20"
      : tone === "amber"
      ? "bg-amber-500/15 text-amber-300 border-amber-400/20"
      : "bg-white/5 text-white/70 border-white/10";

  return (
    <span
      title={title}
      className={`inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-medium ${toneClass}`}
    >
      {children}
    </span>
  );
}

function Th({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <th
      className={`border-b border-white/10 px-4 py-3 text-left text-xs font-semibold uppercase tracking-[0.16em] text-white/55 ${className}`}
    >
      {children}
    </th>
  );
}

function Td({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <td className={`border-b border-white/6 px-4 py-3 align-top text-sm text-white/85 ${className}`}>{children}</td>;
}

export default function Page() {
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    async function load() {
      try {
        const res = await fetch("/metrics.json", { cache: "no-store" });
        if (!res.ok) throw new Error(`Failed to load metrics.json (${res.status})`);
        const json = (await res.json()) as MetricsData;
        if (active) {
          setMetrics(json);
          setError(null);
        }
      } catch (err) {
        if (active) {
          setError(err instanceof Error ? err.message : "Unknown error");
        }
      }
    }

    load();
    return () => {
      active = false;
    };
  }, []);

  const sessionRows = useMemo(() => (metrics ? extractSessionRows(metrics) : []), [metrics]);

  const means = useMemo(() => {
    return {
      deltaDense: mean(sessionRows.map((r) => r.deltaDense)),
      deltaTool: mean(sessionRows.map((r) => r.deltaTool)),
      sessionNdcg5: mean(sessionRows.map((r) => r.session)),
      denseNdcg5: mean(sessionRows.map((r) => r.dense)),
    };
  }, [sessionRows]);

  const peakDeltaRow = useMemo(
    () => maxBy(sessionRows, (row) => row.deltaDense),
    [sessionRows]
  );

  const bestSessionRow = useMemo(
    () => maxBy(sessionRows, (row) => row.session),
    [sessionRows]
  );

  const significanceSample = useMemo(() => {
    const sig = metrics?.significance__SessionRerank_vs_Dense;
    if (!sig) return null;

    const candidates = Object.entries(sig)
      .map(([split, regimes]) => {
        const entry = regimes?.session?.["NDCG@5"];
        return {
          split,
          metric: "NDCG@5",
          p_value: asNumber(entry?.p_value),
          delta_mean: asNumber(entry?.delta_mean),
          delta_rel_pct: asNumber(entry?.delta_rel_pct),
        };
      })
      .filter((x) => x.delta_rel_pct != null);

    return maxBy(candidates, (x) => x.delta_rel_pct);
  }, [metrics]);

  const figureInsights = useMemo(() => {
    const sig = metrics?.significance__SessionRerank_vs_Dense ?? {};

    const hit5Peak = maxBy(
      Object.entries(sig).map(([split, regimes]) => ({
        split,
        deltaDense: asNumber(regimes?.session?.["Hit@5"]?.delta_rel_pct),
      })),
      (r) => r.deltaDense
    );

    return {
      fig1: peakDeltaRow
        ? `Computed from loaded session-regime NDCG@5 values. ${splitLabel(
            peakDeltaRow.split
          )} shows the largest relative gain over Dense at ${fmtPct(peakDeltaRow.deltaDense)}.`
        : "Session-regime NDCG@5 insight will appear after metrics load.",

      fig2: significanceSample
        ? `Significance is read from significance__SessionRerank_vs_Dense. The strongest session NDCG@5 gain appears on ${splitLabel(
            significanceSample.split
          )} with p=${fmtP(significanceSample.p_value)} and ${fmtPct(
            significanceSample.delta_rel_pct
          )} relative gain.`
        : "Significance insight will appear after metrics load.",

      fig3: bestSessionRow
        ? `The best absolute session-regime NDCG@5 appears on ${splitLabel(
            bestSessionRow.split
          )} at ${fmtNum(bestSessionRow.session)}.`
        : "Overview insight will appear after metrics load.",

      fig4:
        means.deltaDense != null
          ? `Δ (delta) vs Dense is computed as relative percentage improvement: (SessionRerank - Dense) / Dense × 100. The mean split-wise gain is ${fmtPct(
              means.deltaDense
            )}.`
          : "Delta insight will appear after metrics load.",

      fig5: hit5Peak
        ? `Hit@5 significance metadata is loaded from the JSON. ${splitLabel(
            hit5Peak.split
          )} has the strongest relative Hit@5 gain over Dense at ${fmtPct(hit5Peak.deltaDense)}.`
        : "Session-aware gain insight will appear after metrics load.",

      fig6:
        "This panel stays descriptive because the current JSON does not expose a separate open-corpus summary block beyond split/regime metrics.",
    };
  }, [metrics, means.deltaDense, peakDeltaRow, significanceSample, bestSessionRow]);

  const corpusSize = metrics?.config?.corpus_size ?? null;
  const encoder = metrics?.config?.encoder ?? "—";
  const weights = metrics?.config?.weights;
  const visibleFigures = FIGURES;

  return (
    <main className="min-h-screen bg-[#07111a] text-white">
      <div className="mx-auto max-w-7xl px-6 py-10">
        <div className="mb-8 flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
          <div>
            <p className="mb-2 text-xs uppercase tracking-[0.24em] text-cyan-300/80">
              ToolBench · Session-aware re-ranking
            </p>
            <h1 className="text-4xl font-semibold tracking-tight text-white">
              SessionRerank
            </h1>
            <p className="mt-3 max-w-3xl text-base text-white/70">
              History-aware tool re-ranking with metrics loaded directly from{" "}
              <code className="rounded bg-white/10 px-1.5 py-0.5 text-white">metrics.json</code>.
              Delta labels are computed dynamically from the loaded benchmark values.
            </p>
          </div>

          <div className="flex flex-wrap gap-2">
            <Pill tone="blue">Encoder: {encoder}</Pill>
            <Pill tone="neutral">Corpus: {corpusSize ?? "—"}</Pill>
            <Pill
              tone="green"
              title="Δ (delta) vs Dense = relative percentage gain of SessionRerank over the Dense baseline."
            >
              Δ (delta) vs Dense
            </Pill>
          </div>
        </div>

        {error ? (
          <div className="mb-8 rounded-2xl border border-rose-400/20 bg-rose-500/10 p-4 text-sm text-rose-200">
            Failed to load metrics: {error}
          </div>
        ) : null}

        <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <StatCard
            label="Mean Δ (delta) vs Dense"
            value={fmtPct(means.deltaDense)}
            hint="Average split-wise relative NDCG@5 gain in the session regime."
          />
          <StatCard
            label="Peak Δ (delta) vs Dense"
            value={peakDeltaRow ? fmtPct(peakDeltaRow.deltaDense) : "—"}
            hint={peakDeltaRow ? splitLabel(peakDeltaRow.split) : "Peak split will appear after load."}
          />
          <StatCard
            label="Mean Session NDCG@5"
            value={fmtNum(means.sessionNdcg5)}
            hint="Average session-regime NDCG@5 for SessionRerank."
          />
          <StatCard
            label="Best session split"
            value={bestSessionRow ? splitLabel(bestSessionRow.split) : "—"}
            hint={bestSessionRow ? `NDCG@5 ${fmtNum(bestSessionRow.session)}` : "Best split will appear after load."}
          />
        </section>

        <section className="mt-10 grid gap-6 xl:grid-cols-[1.4fr_0.9fr]">
          <div className="rounded-3xl border border-white/10 bg-white/[0.03] p-5">
            <div className="mb-4 flex items-center justify-between gap-4">
              <div>
                <h2 className="text-xl font-semibold">Session-regime table</h2>
                <p className="mt-1 text-sm text-white/60">
                  NDCG@5 values are read from <code>splits.*.regimes.session</code>. Δ is computed live from loaded values.
                </p>
              </div>
              <Pill
                tone="neutral"
                title="Δ (delta) = ((SessionRerank - Dense) / Dense) × 100"
              >
                Δ (delta) formula
              </Pill>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full min-w-[920px] border-collapse">
                <thead>
                  <tr>
                    <Th>Split</Th>
                    <Th>N queries</Th>
                    <Th>BM25</Th>
                    <Th>Dense</Th>
                    <Th>ToolRerank</Th>
                    <Th>SessionRerank</Th>
                    <Th>Δ (delta) vs Dense</Th>
                    <Th>Δ (delta) vs Tool</Th>
                  </tr>
                </thead>
                <tbody>
                  {sessionRows.map((row) => (
                    <tr key={row.split} className="hover:bg-white/[0.03]">
                      <Td className="font-medium text-white">{splitLabel(row.split)}</Td>
                      <Td>{row.nQueries ?? "—"}</Td>
                      <Td>{fmtNum(row.bm25)}</Td>
                      <Td>{fmtNum(row.dense)}</Td>
                      <Td>{fmtNum(row.tool)}</Td>
                      <Td>{fmtNum(row.session)}</Td>
                      <Td>
                        <Pill tone={row.deltaDense != null && row.deltaDense > 0 ? "green" : "neutral"}>
                          {fmtPct(row.deltaDense)}
                        </Pill>
                      </Td>
                      <Td>
                        <Pill tone={row.deltaTool != null && row.deltaTool > 0 ? "blue" : "neutral"}>
                          {fmtPct(row.deltaTool)}
                        </Pill>
                      </Td>
                    </tr>
                  ))}
                  {!sessionRows.length ? (
                    <tr>
                      <Td className="text-white/50" colSpan={8}>
                        Loading rows from metrics.json…
                      </Td>
                    </tr>
                  ) : null}
                </tbody>
              </table>
            </div>
          </div>

          <div className="space-y-6">
            <div className="rounded-3xl border border-white/10 bg-white/[0.03] p-5">
              <h2 className="text-xl font-semibold">Config</h2>
              <div className="mt-4 grid gap-3">
                <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
                  <div className="text-xs uppercase tracking-[0.16em] text-white/50">Encoder</div>
                  <div className="mt-2 text-sm text-white">{encoder}</div>
                </div>
                <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
                  <div className="text-xs uppercase tracking-[0.16em] text-white/50">Corpus size</div>
                  <div className="mt-2 text-sm text-white">{corpusSize ?? "—"}</div>
                </div>
                <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
                  <div className="text-xs uppercase tracking-[0.16em] text-white/50">Weights</div>
                  <div className="mt-2 grid gap-2 text-sm text-white/85">
                    <div>w_base: {fmtNum(asNumber(weights?.w_base), 4)}</div>
                    <div>w_cat: {fmtNum(asNumber(weights?.w_cat), 4)}</div>
                    <div>w_sch: {fmtNum(asNumber(weights?.w_sch), 4)}</div>
                    <div>w_ept: {fmtNum(asNumber(weights?.w_ept), 4)}</div>
                  </div>
                </div>
              </div>
            </div>

            <div className="rounded-3xl border border-white/10 bg-white/[0.03] p-5">
              <h2 className="text-xl font-semibold">Significance sample</h2>
              {significanceSample ? (
                <div className="mt-4 space-y-3 text-sm text-white/80">
                  <div>
                    <span className="text-white/50">Split:</span>{" "}
                    {splitLabel(significanceSample.split)}
                  </div>
                  <div>
                    <span className="text-white/50">Metric:</span>{" "}
                    {significanceSample.metric}
                  </div>
                  <div>
                    <span className="text-white/50">p-value:</span>{" "}
                    {fmtP(significanceSample.p_value)}
                  </div>
                  <div>
                    <span className="text-white/50">Delta mean:</span>{" "}
                    {fmtNum(significanceSample.delta_mean, 4)}
                  </div>
                  <div>
                    <span className="text-white/50">Δ (delta) vs Dense:</span>{" "}
                    {fmtPct(significanceSample.delta_rel_pct)}
                  </div>
                </div>
              ) : (
                <p className="mt-4 text-sm text-white/55">
                  Waiting for significance metadata from the loaded JSON.
                </p>
              )}
            </div>
          </div>
        </section>

        <section className="mt-10">
          <div className="mb-4">
            <h2 className="text-xl font-semibold">Figures</h2>
            <p className="mt-1 text-sm text-white/60">
              Figures stay visible, and captions are derived from the live JSON where possible.
            </p>
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            {visibleFigures.map((fig) => (
              <article
                key={fig.id}
                className="overflow-hidden rounded-3xl border border-white/10 bg-white/[0.03]"
              >
                <div className="border-b border-white/10 px-5 py-4">
                  <h3 className="text-base font-semibold text-white">{fig.title}</h3>
                </div>
                <div className="aspect-[16/9] bg-black/20">
                  <img
                    src={fig.src}
                    alt={fig.title}
                    className="h-full w-full object-contain"
                    loading="lazy"
                  />
                </div>
                <div className="px-5 py-4 text-sm leading-6 text-white/70">
                  {figureInsights[fig.id as keyof typeof figureInsights]}
                </div>
              </article>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}
