"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";

type AnyObj = Record<string, any>;

type MetricsData = {
  encoder?: string;
  corpus_size?: number;
  session_regime?: { rows?: AnyObj[] };
};

function asNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }
  return null;
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

export default function Page() {
  const [metrics, setMetrics] = useState<MetricsData | null>(null);

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

  const rows = useMemo(() => {
    const raw = metrics?.session_regime?.rows ?? [];
    return raw.map((row) => {
      const bm25 = asNumber(row.bm25_ndcg5 ?? row.bm25);
      const dense = asNumber(row.dense_ndcg5 ?? row.dense);
      const tool = asNumber(row.toolrerank_ndcg5 ?? row.toolrerank ?? row.tool_rerank);
      const session = asNumber(row.sessionrerank_ndcg5 ?? row.sessionrerank ?? row.session_rerank);
      return {
        split: String(row.split ?? row.name ?? "Unknown split"),
        bm25,
        dense,
        tool,
        session,
        deltaDense: relGainPct(session, dense),
      };
    });
  }, [metrics]);

  const meanDelta = useMemo(() => mean(rows.map((r) => r.deltaDense)), [rows]);
  const peakRow = useMemo(() => {
    return rows.reduce<(typeof rows)[number] | null>((best, row) => {
      if (row.deltaDense == null) return best;
      if (!best || (best.deltaDense ?? -Infinity) < row.deltaDense) return row;
      return best;
    }, null);
  }, [rows]);

  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-100">
      <section className="mx-auto max-w-6xl px-6 py-14 md:px-8">
        <div className="mb-4 text-xs font-medium uppercase tracking-[0.2em] text-emerald-400/80">
          ToolBench · Session-Aware Re-Ranking · Gemma 4 Good Hackathon Submission
        </div>

        <h1 className="text-4xl font-semibold tracking-tight text-white md:text-6xl">
          SessionRerank
        </h1>

        <p className="mt-4 max-w-3xl text-base leading-7 text-neutral-300 md:text-lg">
          History-aware tool re-ranking for LLM agents, with benchmark results loaded directly from{" "}
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
        </div>

        <div className="mt-8 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
            <div className="text-xs uppercase tracking-[0.16em] text-neutral-400">
              Encoder
              <span className="ml-2 cursor-help text-neutral-300" title="Embedding model used for dense retrieval.">?</span>
            </div>
            <div className="mt-2 text-2xl font-semibold text-white">{metrics?.encoder ?? "—"}</div>
            <div className="mt-1 text-sm text-neutral-400">Corpus size: {fmtInt(metrics?.corpus_size)}</div>
          </div>

          <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
            <div className="text-xs uppercase tracking-[0.16em] text-neutral-400">
              Mean NDCG@5 gain vs Dense
              <span className="ml-2 cursor-help text-neutral-300" title="Average percentage improvement of SessionRerank over Dense across all splits.">?</span>
            </div>
            <div className="mt-2 text-2xl font-semibold text-white">{fmtPct(meanDelta)}</div>
            <div className="mt-1 text-sm text-neutral-400">Average across {rows.length || "—"} splits</div>
          </div>

          <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
            <div className="text-xs uppercase tracking-[0.16em] text-neutral-400">
              Peak NDCG@5 gain
              <span className="ml-2 cursor-help text-neutral-300" title="Largest percentage improvement of SessionRerank over Dense on any single split.">?</span>
            </div>
            <div className="mt-2 text-2xl font-semibold text-white">{peakRow ? fmtPct(peakRow.deltaDense) : "—"}</div>
            <div className="mt-1 text-sm text-neutral-400">{peakRow ? peakRow.split : "Largest split-wise gain"}</div>
          </div>

          <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
            <div className="text-xs uppercase tracking-[0.16em] text-neutral-400">
              Session sample
              <span className="ml-2 cursor-help text-neutral-300" title="A simple descriptor of the split used to summarize session-aware improvement.">?</span>
            </div>
            <div className="mt-2 text-2xl font-semibold text-white">{peakRow ? `${peakRow.split} · NDCG@5` : "—"}</div>
            <div className="mt-1 text-sm text-neutral-400">{fmtInt(metrics?.corpus_size)} APIs evaluated</div>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 py-8 md:px-8">
        <h2 className="text-2xl font-semibold text-white">Results — NDCG@5</h2>
        <p className="mt-3 text-neutral-300">
          Δ (delta) vs Dense is computed as the percentage improvement of SessionRerank over Dense.
        </p>

        <div className="mt-6 overflow-hidden rounded-3xl border border-white/10 bg-white/[0.03]">
          <div className="overflow-x-auto">
            <table className="min-w-full text-left text-sm">
              <thead className="bg-white/5 text-neutral-200">
                <tr>
                  <th className="px-4 py-3 font-medium">Split</th>
                  <th className="px-4 py-3 font-medium">BM25</th>
                  <th className="px-4 py-3 font-medium">Dense</th>
                  <th className="px-4 py-3 font-medium">ToolRerank</th>
                  <th className="px-4 py-3 font-medium">SessionRerank</th>
                  <th className="px-4 py-3 font-medium">Δ (delta) vs Dense</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.split} className="border-t border-white/10">
                    <td className="px-4 py-3 text-neutral-200">{row.split}</td>
                    <td className="px-4 py-3 text-neutral-200">{fmtNum(row.bm25)}</td>
                    <td className="px-4 py-3 text-neutral-200">{fmtNum(row.dense)}</td>
                    <td className="px-4 py-3 text-neutral-200">{fmtNum(row.tool)}</td>
                    <td className="px-4 py-3 text-emerald-300">{fmtNum(row.session)}</td>
                    <td className="px-4 py-3 text-emerald-300">{fmtPct(row.deltaDense)}</td>
                  </tr>
                ))}
                <tr className="border-t border-white/10 bg-white/5 font-semibold">
                  <td className="px-4 py-3 text-neutral-200">Mean</td>
                  <td className="px-4 py-3 text-neutral-200">{fmtNum(mean(rows.map((r) => r.bm25)))}</td>
                  <td className="px-4 py-3 text-neutral-200">{fmtNum(mean(rows.map((r) => r.dense)))}</td>
                  <td className="px-4 py-3 text-neutral-200">{fmtNum(mean(rows.map((r) => r.tool)))}</td>
                  <td className="px-4 py-3 text-emerald-300">{fmtNum(mean(rows.map((r) => r.session)))}</td>
                  <td className="px-4 py-3 text-emerald-300">{fmtPct(meanDelta)}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </main>
  );
}
