import { NextResponse } from "next/server";
import { loadIndex, rerank, buildPrior, baseScore } from "@/lib/rerank";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: Request) {
  let body: any;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }
  const query: string = String(body?.query || "").trim();
  const history: string[] = Array.isArray(body?.history) ? body.history : [];
  const topK: number = Math.min(20, Math.max(1, body?.top_k || 8));

  if (!query) {
    return NextResponse.json({ error: "Missing query" }, { status: 400 });
  }

  const idx = loadIndex();

  // (1) baseline ranking — no session prior
  const baseline = rerank(query, idx.apis).slice(0, topK);

  // (2) session-aware ranking — with prior built from history
  const prior = buildPrior(history, idx.apis);
  const session = rerank(query, idx.apis, prior).slice(0, topK);

  // Build full result with metadata
  const byUid = new Map(idx.apis.map((a) => [a.uid, a]));
  const formatList = (list: { uid: string; score: number }[]) =>
    list.map((r, i) => {
      const a = byUid.get(r.uid)!;
      return {
        rank: i + 1,
        uid: a.uid,
        tool: a.tool,
        api: a.api,
        category: a.category,
        description: a.description,
        score: r.score,
      };
    });

  return NextResponse.json({
    query,
    history_size: history.length,
    encoder: idx.encoder,
    corpus_size: idx.apis.length,
    baseline: formatList(baseline),
    session_rerank: formatList(session),
    weights: { w_base: 1.4, w_cat: 1.17, w_sch: 0.31, w_ept: 0.1 },
  });
}

export async function GET() {
  const idx = loadIndex();
  // List a small sample so the UI can show "available APIs" to pick from
  return NextResponse.json({
    corpus_size: idx.apis.length,
    encoder: idx.encoder,
    sample: idx.apis.slice(0, 50).map((a) => ({
      uid: a.uid,
      tool: a.tool,
      api: a.api,
      category: a.category,
      description: a.description,
    })),
  });
}
