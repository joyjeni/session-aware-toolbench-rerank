// Lightweight SessionRerank demo for the Vercel serverless runtime.
//
// Note on encoder: a heavyweight transformer is not appropriate inside a
// serverless function (cold-start, RAM). We therefore back the *base* score
// with a fast keyword-match score over the API description and metadata.
// The SessionRerank prior — category / schema / endpoint affinity — is the
// SAME formulation as the full evaluated implementation. This isolates the
// contribution of session-aware re-ranking, which is what the demo shows.

import fs from "fs";
import path from "path";

export interface API {
  uid: string;
  tool: string;
  api: string;
  category: string;
  description: string;
  schema: string;
  endpoint: string;
  emb: number[];
}

let _index: { apis: API[]; encoder: string } | null = null;
export function loadIndex() {
  if (_index) return _index;
  const p = path.join(process.cwd(), "public", "demo-index.json");
  _index = JSON.parse(fs.readFileSync(p, "utf-8"));
  return _index!;
}

function tokenize(s: string): string[] {
  return (s.toLowerCase().match(/[a-z0-9]+/g) || []).filter((t) => t.length > 1);
}

// Simple TF-IDF–lite query→doc match used as the *base* retriever in the demo.
export function baseScore(query: string, a: API): number {
  const q = new Set(tokenize(query));
  const doc = tokenize(
    `${a.category} ${a.tool} ${a.api} ${a.description} ${a.endpoint}`
  );
  if (doc.length === 0 || q.size === 0) return 0;
  let m = 0;
  for (const t of doc) if (q.has(t)) m++;
  // length-normalized count, slightly biased to short relevant fields
  return m / Math.sqrt(doc.length);
}

export interface SessionPrior {
  cat: Map<string, number>;
  sch: Map<string, number>;
  ept: Map<string, number>;
  n: number;
}

export function buildPrior(historyUids: string[], apis: API[]): SessionPrior {
  const map = new Map(apis.map((a) => [a.uid, a]));
  const decay = 0.85;
  const cat = new Map<string, number>();
  const sch = new Map<string, number>();
  const ept = new Map<string, number>();
  const hist = historyUids.map((u) => map.get(u)).filter(Boolean) as API[];
  for (let i = 0; i < hist.length; i++) {
    const w = Math.pow(decay, hist.length - 1 - i);
    const a = hist[i];
    cat.set(a.category, (cat.get(a.category) || 0) + w);
    sch.set(a.schema, (sch.get(a.schema) || 0) + w);
    ept.set(a.endpoint, (ept.get(a.endpoint) || 0) + w);
  }
  const norm = (m: Map<string, number>) => {
    const tot = Array.from(m.values()).reduce((s, v) => s + v, 0) || 1;
    m.forEach((v, k) => m.set(k, v / tot));
  };
  norm(cat); norm(sch); norm(ept);
  return { cat, sch, ept, n: hist.length };
}

// Same weights as fit on training trajectories
const W = { base: 1.4, cat: 1.17, sch: 0.31, ept: 0.1 };

export function rerank(
  query: string,
  apis: API[],
  prior?: SessionPrior
): { uid: string; score: number; baseScore: number }[] {
  const baseRaw = apis.map((a) => baseScore(query, a));
  // standardize base
  const mean = baseRaw.reduce((s, v) => s + v, 0) / baseRaw.length;
  const std = Math.sqrt(
    baseRaw.map((v) => (v - mean) ** 2).reduce((s, v) => s + v, 0) /
      baseRaw.length
  ) || 1e-6;
  const baseStd = baseRaw.map((v) => (v - mean) / std);

  const out = apis.map((a, i) => {
    let s = W.base * baseStd[i];
    if (prior && prior.n > 0) {
      const c = prior.cat.get(a.category) || 0;
      const sc = prior.sch.get(a.schema) || 0;
      const ep = prior.ept.get(a.endpoint) || 0;
      s +=
        W.cat * Math.log1p(c * 10) +
        W.sch * Math.log1p(sc * 10) +
        W.ept * Math.log1p(ep * 10);
    }
    return { uid: a.uid, score: s, baseScore: baseRaw[i] };
  });
  out.sort((a, b) => b.score - a.score);
  return out;
}
