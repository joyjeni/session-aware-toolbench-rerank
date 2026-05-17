// Agent endpoint — turns a user question into a synthesized natural-language
// answer grounded in the SessionRerank top-k APIs.
//
// Path priority:
//   1. GROQ_API_KEY        → Groq Cloud free tier (llama-3.1-8b-instant)
//   2. OPENROUTER_API_KEY  → OpenRouter free models (meta-llama/llama-3.2-3b)
//   3. HF_TOKEN            → Hugging Face Inference router (zephyr-7b)
//   4. <no key set>        → deterministic offline fallback (template answer)
//
// The response always includes a `path` field so the UI can show which path
// served the answer.

import { NextResponse } from "next/server";
import { loadIndex, rerank, buildPrior } from "@/lib/rerank";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type RankedItem = {
  uid: string;
  tool: string;
  api: string;
  category: string;
  description: string;
  score: number;
  rank: number;
};

function buildSystemPrompt() {
  return [
    "You are SessionRerank+, a tool-routing assistant for LLM agents.",
    "You are given a user question and a ranked list of candidate APIs returned",
    "by the SessionRerank retriever. Pick the single best API and explain in",
    "2-3 short sentences why it answers the question. If the top APIs do not",
    "match the question, say so honestly. Keep the answer under 90 words.",
    "Reference APIs by name, e.g. 'ASIN Data / Category'.",
  ].join(" ");
}

type ChatTurn = { role: "user" | "assistant"; content: string };

function buildUserPrompt(query: string, top: RankedItem[], chat: ChatTurn[]) {
  const lines = top.slice(0, 5).map(
    (r, i) =>
      `${i + 1}. ${r.tool} / ${r.api}  —  category=${r.category}\n   ${r.description.slice(0, 240)}`
  );
  const transcript = chat
    .slice(-6)
    .map((t) => `${t.role === "user" ? "User" : "Assistant"}: ${t.content.slice(0, 400)}`)
    .join("\n");
  const ctx = transcript ? `Prior conversation (for context only — do not repeat verbatim):\n${transcript}\n\n` : "";
  return `${ctx}User question:\n${query}\n\nTop-ranked APIs from SessionRerank:\n${lines.join(
    "\n"
  )}\n\nWrite the assistant's answer now.`;
}

async function callGroq(query: string, top: RankedItem[], chat: ChatTurn[]): Promise<string> {
  const r = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "llama-3.1-8b-instant",
      temperature: 0.2,
      max_tokens: 220,
      messages: [
        { role: "system", content: buildSystemPrompt() },
        { role: "user", content: buildUserPrompt(query, top, chat) },
      ],
    }),
  });
  if (!r.ok) throw new Error(`Groq HTTP ${r.status}: ${await r.text()}`);
  const j = await r.json();
  return j.choices?.[0]?.message?.content?.trim() || "";
}

async function callOpenRouter(query: string, top: RankedItem[], chat: ChatTurn[]): Promise<string> {
  const r = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`,
      "Content-Type": "application/json",
      "HTTP-Referer": "https://github.com/joyjeni/session-aware-toolbench-rerank",
      "X-Title": "SessionRerank+",
    },
    body: JSON.stringify({
      model: "meta-llama/llama-3.2-3b-instruct:free",
      temperature: 0.2,
      max_tokens: 220,
      messages: [
        { role: "system", content: buildSystemPrompt() },
        { role: "user", content: buildUserPrompt(query, top, chat) },
      ],
    }),
  });
  if (!r.ok) throw new Error(`OpenRouter HTTP ${r.status}: ${await r.text()}`);
  const j = await r.json();
  return j.choices?.[0]?.message?.content?.trim() || "";
}

async function callHFInference(query: string, top: RankedItem[], chat: ChatTurn[]): Promise<string> {
  const prompt =
    `<|system|>\n${buildSystemPrompt()}\n` +
    `<|user|>\n${buildUserPrompt(query, top, chat)}\n<|assistant|>\n`;
  const r = await fetch(
    "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.HF_TOKEN}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        inputs: prompt,
        parameters: { max_new_tokens: 220, temperature: 0.2, return_full_text: false },
        options: { wait_for_model: true },
      }),
    }
  );
  if (!r.ok) throw new Error(`HF HTTP ${r.status}: ${await r.text()}`);
  const j = await r.json();
  const txt = Array.isArray(j) ? j[0]?.generated_text : j?.generated_text;
  return (txt || "").trim();
}

function offlineFallback(query: string, top: RankedItem[]): string {
  if (!top.length) {
    return `I could not find any matching tools for "${query}". Try rephrasing or broadening the query.`;
  }
  const t = top[0];
  const others = top.slice(1, 3).map((r) => `${r.tool} / ${r.api}`).join(", ");
  const tail = others ? ` Runner-ups: ${others}.` : "";
  return (
    `For "${query}", the best-matched tool is **${t.tool} / ${t.api}** ` +
    `(category: ${t.category}). ${t.description.slice(0, 200)}${
      t.description.length > 200 ? "…" : ""
    }${tail}\n\n` +
    `_This answer was generated by the offline template (no LLM key configured). ` +
    `Set GROQ_API_KEY, OPENROUTER_API_KEY, or HF_TOKEN to get a natural-language answer._`
  );
}

export async function POST(req: Request) {
  let body: any;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }
  const query: string = String(body?.query || "").trim();
  const history: string[] = Array.isArray(body?.history) ? body.history : [];
  const chatRaw: any[] = Array.isArray(body?.chat) ? body.chat : [];
  const chat: ChatTurn[] = chatRaw
    .filter((t) => t && (t.role === "user" || t.role === "assistant") && typeof t.content === "string")
    .map((t) => ({ role: t.role as "user" | "assistant", content: String(t.content) }));
  const topK = 5;

  if (!query) {
    return NextResponse.json({ error: "Missing query" }, { status: 400 });
  }

  const idx = loadIndex();
  const prior = buildPrior(history, idx.apis);
  const ranked = rerank(query, idx.apis, prior).slice(0, topK);
  const byUid = new Map(idx.apis.map((a) => [a.uid, a]));
  const top: RankedItem[] = ranked.map((r, i) => {
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

  let answer = "";
  let path: "groq" | "openrouter" | "hf" | "offline" = "offline";
  let pathError: string | null = null;

  const tryPaths: Array<["groq" | "openrouter" | "hf", () => Promise<string>]> = [];
  if (process.env.GROQ_API_KEY) tryPaths.push(["groq", () => callGroq(query, top, chat)]);
  if (process.env.OPENROUTER_API_KEY)
    tryPaths.push(["openrouter", () => callOpenRouter(query, top, chat)]);
  if (process.env.HF_TOKEN) tryPaths.push(["hf", () => callHFInference(query, top, chat)]);

  for (const [name, fn] of tryPaths) {
    try {
      const text = await fn();
      if (text) {
        answer = text;
        path = name;
        break;
      }
    } catch (e: any) {
      pathError = `${name}: ${String(e?.message || e).slice(0, 240)}`;
      // try next path
    }
  }

  if (!answer) {
    answer = offlineFallback(query, top);
    path = "offline";
  }

  return NextResponse.json({
    query,
    answer,
    path,
    path_error: pathError,
    top_apis: top,
    history_size: history.length,
  });
}

export async function GET() {
  // Health check + which paths are configured (no key values are returned).
  return NextResponse.json({
    ok: true,
    paths_available: {
      groq: Boolean(process.env.GROQ_API_KEY),
      openrouter: Boolean(process.env.OPENROUTER_API_KEY),
      hf: Boolean(process.env.HF_TOKEN),
    },
  });
}
