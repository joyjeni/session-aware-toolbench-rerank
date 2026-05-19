"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";

type AnyObj = Record<string, any>;

type MetricsData = {
  encoder?: string;
  corpus_size?: number;
  weights?: {
    w_base?: number;
    w_cat?: number;
    w_sch?: number;
    w_ept?: number;
  };
  session_regime?: {
    rows?: AnyObj[];
    hit5_rows?: AnyObj[];
  };
  stage_benchmark?: {
    rows?: AnyObj[];
  };
  open_corpus_single_turn?: AnyObj;
  figures?: Record<string, any>;
  significance_sample?: {
    split?: string;
    metric?: string;
    apis_evaluated?: number;
    p_value?: number;
    ci_low?: number;
    ci_high?: number;
    note?: string;
  };
};

type SessionRow = {
  split: string;
  bm25: number | null;
  dense: number | null;
  tool: number | null;
  session: number | null;
};

type FigureDef = {
  id: string;
  title: string;
  file: string;
  alt: string;
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
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

function fmtNum(value: number | null, digits = 3) {
  if (value == null || !Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

function fmtPct(value: number | null, digits = 1) {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}%`;
}

function fmtInt(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value)) return "—";
  return new Intl.NumberFormat("en-US").format(value);
}

function mean(values: Array<number | null>) {
  const valid = values.filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  if (!valid.length) return null;
  return valid.reduce((a, b) => a + b, 0) / valid.length;
}

function relGainPct(improved: number | null, baseline: number | null) {
  if (improved == null || baseline == null || baseline === 0) return null;
  return ((improved - baseline) / baseline) * 100;
}

function maxBy<T>(items: T[], getValue: (item: T) => number | null): T | null {
  let best: T | null = null;
  let bestVal = -Infinity;
  for (const item of items) {
    const val = getValue(item);
    if (val == null || !Number.isFinite(val)) continue;
    if (best == null || val > bestVal) {
      best = item;
      bestVal = val;
    }
  }
  return best;
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

function getSignificanceSample(metrics: MetricsData, sessionRows: SessionRow[]) {
  if (metrics.significance_sample) {
    return {
      split: metrics.significance_sample.split ?? "—",
      metric: metrics.significance_sample.metric ?? "NDCG@5",
      apisEvaluated:
        metrics.significance_sample.apis_evaluated ?? metrics.corpus_size ?? null,
      pValue: asNumber(metrics.significance_sample.p_value),
      ciLow: asNumber(metrics.significance_sample.ci_low),
      ciHigh: asNumber(metrics.significance_sample.ci_high),
      note: metrics.significance_sample.note ?? null,
      source: "json",
    };
  }

  const bestRow = maxBy(
    sessionRows.map((r) => ({
      split: r.split,
      deltaDense: relGainPct(r.session, r.dense),
    })),
    (r) => r.deltaDense
  );

  return {
    split: bestRow?.split ?? "—",
    metric: "NDCG@5",
    apisEvaluated: metrics.corpus_size ?? null,
    pValue: null,
    ciLow: null,
    ciHigh: null,
    note: "Derived from loaded metrics: split is chosen as the largest relative NDCG@5 gain over Dense; this is a sample descriptor, not a statistical test result.",
    source: "derived",
  };
}

function winnerName(row: SessionRow) {
  const items = [
    { name: "BM25", value: row.bm25 },
    { name: "Dense", value: row.dense },
    { name: "ToolRerank", value: row.tool },
    { name: "SessionRerank", value: row.session },
  ].filter((x): x is { name: string; value: number } => x.value != null);

  if (!items.length) return "—";
  return items.reduce((best, item) => (item.value > best.value ? item : best)).name;
}

function Tooltip({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <span className="group relative inline-flex items-center">
      <button
        type="button"
        aria-label={label}
        className="ml-2 inline-flex h-5 w-5 items-center justify-center rounded-full border border-white/15 bg-white/5 text-[11px] font-semibold text-neutral-300 hover:bg-white/10"
      >
        ?
      </button>
      <span className="pointer-events-none absolute left-1/2 top-full z-20 mt-2 hidden w-72 -translate-x-1/2 rounded-xl border border-white/10 bg-neutral-900 p-3 text-left text-xs leading-5 text-neutral-200 shadow-2xl group-hover:block group-focus-within:block">
        {children}
      </span>
    </span>
  );
}

function StatCard({
  label,
  value,
  subtext,
  tooltip,
}: {
  label: string;
  value: string;
  subtext: string;
  tooltip: React.ReactNode;
}) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
      <div className="flex items-center text-xs uppercase tracking-[0.16em] text-neutral-400">
        <span>{label}</span>
        <Tooltip label={`Explain ${label}`}>{tooltip}</Tooltip>
      </div>
      <div className="mt-2 text-2xl font-semibold text-white">{value}</div>
      <div className="mt-1 text-sm leading-6 text-neutral-400">{subtext}</div>
    </div>
  );
}

function TableHead({ children }: { children: React.ReactNode }) {
  return <th className="px-4 py-3 font-medium">{children}</th>;
}

function TableCell({
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

function Pill({
  label,
  value,
  tooltip,
}: {
  label: string;
  value?: number;
  tooltip: React.ReactNode;
}) {
  return (
    <div className="inline-flex items-center rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5 text-sm text-neutral-200">
      <span className="text-neutral-400">{label}=</span>
      <span className="ml-1">{value != null ? value.toFixed(2) : "—"}</span>
      <Tooltip label={`Explain ${label}`}>{tooltip}</Tooltip>
    </div>
  );
}

export default function Page() {
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [availableFigures, setAvailableFigures] = useState<Record<string, boolean>>({});

  useEffect(() => {
    async function loadMetrics() {
      try {
        const res = await fetch("/metrics.json", { cache: "no-store" });
        const data = await res.json();
        setMetrics(data);
      } catch (e) {
        console.error("Failed to load metrics.json", e);
        setMetrics({});
      }
    }
    loadMetrics();
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
      setAvailableFigures(Object.fromEntries(entries));
    }
    checkFigures();
  }, []);

  const sessionRows = useMemo(() => (metrics ? extractSessionRows(metrics) : []), [metrics]);

  const rowsWithDelta = useMemo(
    () =>
      sessionRows.map((row) => ({
        ...row,
        deltaDense: relGainPct(row.session, row.dense),
      })),
    [sessionRows]
  );

  const means = useMemo(
    () => ({
      bm25: mean(sessionRows.map((r) => r.bm25)),
      dense: mean(sessionRows.map((r) => r.dense)),
      tool: mean(sessionRows.map((r) => r.tool)),
      session: mean(sessionRows.map((r) => r.session)),
      deltaDense: mean(rowsWithDelta.map((r) => r.deltaDense)),
    }),
    [sessionRows, rowsWithDelta]
  );

  const peakDeltaRow = useMemo(
    () => maxBy(rowsWithDelta, (r) => r.deltaDense),
    [rowsWithDelta]
  );

  const significance = useMemo(
    () => (metrics ? getSignificanceSample(metrics, sessionRows) : null),
    [metrics, sessionRows]
  );

  const figureInsights = useMemo(() => {
    if (!metrics) return {};

    const validRows = rowsWithDelta.filter(
      (r) => r.bm25 != null && r.dense != null && r.tool != null && r.session != null
    );

    const allWins =
      validRows.length > 0 &&
      validRows.every(
        (r) =>
          (r.session ?? -Infinity) >= (r.bm25 ?? -Infinity) &&
          (r.session ?? -Infinity) >= (r.dense ?? -Infinity) &&
          (r.session ?? -Infinity) >= (r.tool ?? -Infinity)
      );

    const hit5Rows = metrics.session_regime?.hit5_rows ?? [];
    const hit5Peak = maxBy(
      hit5Rows
        .map((row) => ({
          split: String(row.split ?? row.name ?? "Unknown split"),
          deltaDense: relGainPct(
            asNumber(row.sessionrerank_hit5 ?? row.sessionrerank ?? row.session_rerank),
            asNumber(row.dense_hit5 ?? row.dense)
          ),
        }))
        .filter((r) => r.deltaDense != null),
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
        allWins
          ? "SessionRerank is highest on every loaded session-regime split."
          : "SessionRerank is strongest on most loaded session-regime splits.",
        peakDeltaRow
          ? `Its biggest relative improvement over Dense appears on ${peakDeltaRow.split} at ${fmtPct(peakDeltaRow.deltaDense)}.`
          : null,
      ]
        .filter(Boolean)
        .join(" "),

      fig2: [
        "Hit@5 measures whether a correct API appears in the top 5 results.",
        hit5Peak
          ? `The biggest relative gain over Dense appears on ${hit5Peak.split} at ${fmtPct(hit5Peak.deltaDense)}.`
          : "Hit@5 insight will appear when hit@5 rows are available in metrics.json.",
      ].join(" "),

      fig3: [
        "NDCG@K shows ranking quality at multiple cutoff sizes.",
        g2Curve
          ? dominatesAllCutoffs
            ? "SessionRerank leads at every shown cutoff in the loaded curve data."
            : "The best method changes across cutoffs in the loaded curve data."
          : "Cutoff-specific insight will appear when curve data is present in metrics.json.",
      ].join(" "),

      fig4: [
        "Δ (delta) vs Dense means percentage improvement of SessionRerank over the Dense baseline.",
        means.deltaDense != null
          ? `The mean split-wise gain is ${fmtPct(means.deltaDense)}${peakDeltaRow ? `, with the largest gain on ${peakDeltaRow.split} at ${fmtPct(peakDeltaRow.deltaDense)}.` : "."}`
          : "Mean delta will appear when session-regime rows are loaded.",
      ].join(" "),

      fig5: [
        "The heatmap uses the same NDCG@5 values shown in the benchmark table.",
        peakDeltaRow
          ? `The strongest SessionRerank advantage should line up with ${peakDeltaRow.split}.`
          : null,
      ]
        .filter(Boolean)
        .join(" "),

      fig6: [
        "This figure tests the model without session history.",
        singleTurnDelta != null
          ? `SessionRerank changes by ${fmtPct(singleTurnDelta)} relative to Dense in the loaded no-history result.`
          : "When history is empty, SessionRerank should move closer to the Dense baseline.",
      ].join(" "),
    };
  }, [metrics, rowsWithDelta, means.deltaDense, peakDeltaRow]);

  const stageRows = useMemo(() => {
    const raw = metrics?.stage_benchmark?.rows ?? [];
    if (raw.length) {
      return raw.map((row) => ({
        split: String(row.split ?? row.name ?? "Unknown split"),
        stage: String(row.stage ?? "Stage 4 — current metrics"),
        value: asNumber(row.value ?? row.best_ndcg5 ?? row.ndcg5),
        winner: String(row.winner ?? "—"),
      }));
    }

    return sessionRows.map((row) => ({
      split: row.split,
      stage: "Stage 4 — current metrics",
      value: Math.max(
        row.bm25 ?? -Infinity,
        row.dense ?? -Infinity,
        row.tool ?? -Infinity,
        row.session ?? -Infinity
      ),
      winner: winnerName(row),
    }));
  }, [metrics, sessionRows]);

  const visibleFigures = FIGURES.filter((fig) => availableFigures[fig.id]);

  const encoder = metrics?.encoder ?? "—";
  const corpusSize = metrics?.corpus_size ?? null;
  const splitCount = sessionRows.length;

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
              History-aware tool re-ranking for LLM agents, with benchmark and demo results loaded directly from{" "}
              <code className="rounded bg-white/5 px-1.5 py-0.5 text-sm">metrics.json</code>.
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
              value={encoder}
              subtext={`Model used for the dense retriever. Corpus size: ${fmtInt(corpusSize)}`}
              tooltip={
                <>
                  This is the embedding model used to convert queries and APIs into vectors for dense retrieval. A larger or better encoder can improve baseline retrieval quality.
                </>
              }
            />

            <StatCard
              label="Mean NDCG@5 gain vs Dense"
              value={fmtPct(means.deltaDense)}
              subtext={`Average relative improvement across ${splitCount || "—"} ToolBench splits`}
              tooltip={
                <>
                  This tells you how much SessionRerank improves over Dense on average. NDCG@5 rewards putting the most relevant APIs near the top of the top-5 results.
                </>
              }
            />

            <StatCard
              label="Peak NDCG@5 gain"
              value={peakDeltaRow ? fmtPct(peakDeltaRow.deltaDense) : "—"}
              subtext={
                peakDeltaRow
                  ? `Largest improvement appears on ${peakDeltaRow.split}`
                  : "Largest split-wise relative improvement"
              }
              tooltip={
                <>
                  This is the single best relative improvement over Dense among all loaded splits. It helps show where session history helps the most.
                </>
              }
            />

            <StatCard
              label="Session significance sample"
              value={
                significance
                  ? `${significance.split} · ${significance.metric}`
                  : "—"
              }
              subtext={
                significance
                  ? `${fmtInt(significance.apisEvaluated)} APIs evaluated`
                  : "Not shown"
              }
              tooltip={
                <>
                  This card describes the sample used for a significance-style summary. If <code>significance_sample</code> exists in <code>metrics.json</code>, it is shown directly. Otherwise, it is derived from the loaded benchmark as the split with the largest relative NDCG@5 gain over Dense, plus the corpus size.
                </>
              }
            />
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 pb-8 md:px-8">
        <div className="rounded-3xl border border-white/10 bg-white/[0.03] p-6 md:p-8">
          <div className="flex items-center">
            <h2 className="text-2xl font-semibold text-white">Method</h2>
            <Tooltip label="Explain Method section">
              This section explains the scoring formula. The model starts from Dense similarity, then adds history-aware signals so APIs related to previous steps in the same session can be ranked higher.
            </Tooltip>
          </div>

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

          <div className="mt-5 flex flex-wrap gap-3">
            <Pill
              label="w_base"
              value={metrics?.weights?.w_base}
              tooltip={<>Weight on the original dense similarity score.</>}
            />
            <Pill
              label="w_cat"
              value={metrics?.weights?.w_cat}
              tooltip={<>Weight on category overlap between current candidate API and earlier session choices.</>}
            />
            <Pill
              label="w_sch"
              value={metrics?.weights?.w_sch}
              tooltip={<>Weight on schema-level similarity, such as input/output structure alignment.</>}
            />
            <Pill
              label="w_ept"
              value={metrics?.weights?.w_ept}
              tooltip={<>Weight on endpoint-level history affinity from previously used tools.</>}
            />
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 py-8 md:px-8">
        <div className="flex items-center">
          <h2 className="text-2xl font-semibold text-white">Results — NDCG@5 (session regime)</h2>
          <Tooltip label="Explain NDCG@5 table">
            NDCG@5 measures ranking quality in the top 5 results. Higher is better. The delta column shows how much SessionRerank improves over Dense in percentage terms.
          </Tooltip>
        </div>

        <p className="mt-3 text-neutral-300">
          Full evaluation on the ToolBench benchmark splits. Best per row appears in green. Δ (delta) vs Dense is computed dynamically from the loaded metrics.
        </p>

        <div className="mt-6 overflow-hidden rounded-3xl border border-white/10 bg-white/[0.03]">
          <div className="overflow-x-auto">
            <table className="min-w-full text-left text-sm">
              <thead className="bg-white/5 text-neutral-200">
                <tr>
                  <TableHead>Split</TableHead>
                  <TableHead>BM25</TableHead>
                  <TableHead>Dense</TableHead>
                  <TableHead>ToolRerank</TableHead>
                  <TableHead>SessionRerank</TableHead>
                  <TableHead>Δ (delta) vs Dense</TableHead>
                </tr>
              </thead>
              <tbody>
                {rowsWithDelta.map((row) => {
                  const best = Math.max(
                    row.bm25 ?? -Infinity,
                    row.dense ?? -Infinity,
                    row.tool ?? -Infinity,
                    row.session ?? -Infinity
                  );

                  return (
                    <tr key={row.split} className="border-t border-white/10">
                      <TableCell>{row.split}</TableCell>
                      <TableCell highlight={row.bm25 === best}>{fmtNum(row.bm25)}</TableCell>
                      <TableCell highlight={row.dense === best}>{fmtNum(row.dense)}</TableCell>
                      <TableCell highlight={row.tool === best}>{fmtNum(row.tool)}</TableCell>
                      <TableCell highlight={row.session === best}>{fmtNum(row.session)}</TableCell>
                      <TableCell className="text-emerald-300">{fmtPct(row.deltaDense)}</TableCell>
                    </tr>
                  );
                })}

                <tr className="border-t border-white/10 bg-white/5 font-semibold">
                  <TableCell>Mean</TableCell>
                  <TableCell>{fmtNum(means.bm25)}</TableCell>
                  <TableCell>{fmtNum(means.dense)}</TableCell>
                  <TableCell>{fmtNum(means.tool)}</TableCell>
                  <TableCell>{fmtNum(means.session)}</TableCell>
                  <TableCell className="text-emerald-300">{fmtPct(means.deltaDense)}</TableCell>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 py-8 md:px-8">
        <div className="flex items-center">
          <h2 className="text-2xl font-semibold text-white">Stage-by-Stage Benchmark</h2>
          <Tooltip label="Explain Stage-by-Stage Benchmark">
            This is a compact summary view. Each row shows the strongest score found for that split and the method that achieved it.
          </Tooltip>
        </div>

        <p className="mt-3 text-neutral-300">
          Each row shows the best NDCG@5 found in that stage row, plus the winning method name.
        </p>

        <div className="mt-6 overflow-hidden rounded-3xl border border-white/10 bg-white/[0.03]">
          <div className="overflow-x-auto">
            <table className="min-w-full text-left text-sm">
              <thead className="bg-white/5 text-neutral-200">
                <tr>
                  <TableHead>Split</TableHead>
                  <TableHead>{stageRows[0]?.stage ?? "Stage 4 — current metrics"}</TableHead>
                </tr>
              </thead>
              <tbody>
                {stageRows.map((row) => (
                  <tr key={row.split} className="border-t border-white/10">
                    <TableCell>{row.split}</TableCell>
                    <TableCell>
                      <span className="font-semibold text-emerald-300">{fmtNum(row.value)}</span>
                      <span className="ml-2 text-neutral-300">{row.winner}</span>
                    </TableCell>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 py-8 md:px-8">
        <div className="flex items-center">
          <h2 className="text-2xl font-semibold text-white">Publication-Ready Figures</h2>
          <Tooltip label="Explain Publication-Ready Figures">
            Each caption is generated from loaded metrics when possible. This helps keep the text aligned with the benchmark table instead of drifting away as values change.
          </Tooltip>
        </div>

        <p className="mt-3 text-neutral-300">
          Only figures that exist in the public folder are rendered, and the insight below each figure is generated dynamically from loaded metrics when the required fields exist.
        </p>

        <div className="mt-8 grid gap-6">
          {visibleFigures.map((fig) => (
            <figure
              key={fig.id}
              className="overflow-hidden rounded-3xl border border-white/10 bg-white/[0.03]"
            >
              <img src={fig.file} alt={fig.alt} className="w-full object-cover" />
              <figcaption className="border-t border-white/10 px-5 py-4 text-sm leading-7 text-neutral-300">
                <span className="font-medium text-white">{fig.title}</span>{" "}
                {(figureInsights as Record<string, string>)[fig.id] ?? ""}
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
          <div className="flex items-center">
            <h2 className="text-2xl font-semibold text-white">Live Demo — Ask the Agent</h2>
            <Tooltip label="Explain Live Demo">
              The demo lets you compare what the agent retrieves with and without session-aware reranking. Keep your current interactive UI here.
            </Tooltip>
          </div>

          <p className="mt-3 max-w-3xl text-neutral-300">
            Chat with the agent, or run rerank to compare baseline retrieval against session-aware reranking.
          </p>

          <div className="mt-6 rounded-2xl border border-white/10 bg-black/20 p-4 text-sm text-neutral-300">
            Keep your existing demo UI here. The main improvement in this page is that dashboard cards, table deltas, figure insights, and the significance sample are now data-driven instead of hard-coded.
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
