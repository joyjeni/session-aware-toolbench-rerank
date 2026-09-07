#!/usr/bin/env python3
"""POST /rerank microservice for SessionRerank+ (no torch required)."""

from __future__ import annotations

import json
import math
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "agri_extension"))

from agri_extension.karnataka_apis import KARNATAKA_APIS  # noqa: E402
from sessionrerank.model.coactivation import CoactivationCache  # noqa: E402

CACHE = CoactivationCache()


def _tokens(text: str) -> set[str]:
    return {t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if len(t) > 1}


def rerank(body: dict) -> dict:
    query = str(body.get("query", ""))
    history = list(body.get("history_uids") or [])
    top_k = int(body.get("top_k") or 5)
    q = _tokens(query)
    scored = []
    for api in KARNATAKA_APIS:
        overlap = len(q & _tokens(api.doc()))
        dense = overlap / math.sqrt(max(len(q), 1) * max(len(_tokens(api.doc())), 1))
        cache = CACHE.score(history, api.uid)
        total = 1.0 * dense + 0.6 * cache
        scored.append(
            {
                "uid": api.uid,
                "name": api.name,
                "category": api.category,
                "score_total": total,
                "score_dense": dense,
                "score_cache": cache,
            }
        )
    scored.sort(key=lambda row: row["score_total"], reverse=True)
    if body.get("record_success") and scored:
        chosen = scored[0]["uid"]
        for uid in history:
            CACHE.update_edge(uid, chosen)
        CACHE.decay_step()
    return {"results": scored[:top_k], "formula": "dense + 0.6·coactivation", "service": "satr"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if urlparse(self.path).path in {"/", "/health", "/rerank"}:
            self._send(200, {"ok": True, "service": "satr", "repo": "session-aware-toolbench-rerank"})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/rerank":
            self._send(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        self._send(200, rerank(body))


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 43131
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"SATR microservice http://127.0.0.1:{port}/rerank", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
