import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "SessionRerank — Session-Aware Tool Retrieval for ToolBench",
  description:
    "Lightweight session-aware re-ranking on top of ToolLLM's neural API retriever. " +
    "Improves NDCG@5 by +14.7% (mean) and Hit@5 by +8.5% on the full ToolBench benchmark.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
