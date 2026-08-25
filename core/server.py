#!/usr/bin/env python3
"""ulupi server — HTTP daemon over om-memory engine (OpenHuman-tier retrieval).

  GET  /ping          -> pong
  GET  /?q=...        -> recall() context block for the query
  GET  /search?q=..   -> JSON results
  POST /add?q=&a=     -> store memory (dupe-guarded)

Usage: python3 core/server.py   (port 8791)
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from core import engine

PORT = 8792


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _write(self, body, ctype="text/plain; charset=utf-8"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _qs(self):
        u = urlparse(self.path)
        if self.command == "POST":
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n).decode() if n else ""
            return {**parse_qs(u.query), **parse_qs(body)}
        return parse_qs(u.query)

    def do_GET(self):
        u = urlparse(self.path)
        q = self._qs().get("q", [""])[0].strip()
        if u.path == "/ping":
            self._write(b"pong")
        elif u.path == "/search" and q:
            mode = self._qs().get("mode", ["balanced"])[0]
            self._write(json.dumps(engine.search(q, mode=mode)).encode(), "application/json")
        elif q:
            self._write(engine.recall(q).encode())
        else:
            self._write(b"")

    def do_POST(self):
        qs = self._qs()
        if urlparse(self.path).path != "/add":
            self._write(b"unknown")
            return
        q = qs.get("q", [""])[0]
        a = qs.get("a", [""])[0]
        if not q or not a or a.startswith("/Users/") and a.count("/") > 3:
            self._write(b"skip")  # guard: shortcut handed us a file path, not an answer
            return
        fname = f"inbox/{q[:40].strip().replace('/', '-').lower() or 'note'}.md"
        r = engine.add(fname, f"{q}\n\n{a}")
        con = engine.db()
        thread = qs.get("thread", ["default"])[0]
        src = fname if r else None   # dupe -> no new note
        engine.log_message(con, thread, "user", q, source=src)
        engine.log_message(con, thread, "ulupi", a[:2000], source=src)
        con.close()
        self._write(b"ok" if r else b"dupe")


if __name__ == "__main__":
    con = engine.db(); engine.sync(con); con.close()
    # warm both brains: embed model + gemma4 fact extractor
    try:
        engine.embed("warm")
        engine.llm_extract_facts("warmup note")
        print("brains warmed", flush=True)
    except Exception as e:
        print("warmup degraded:", e, flush=True)
    print(f"ulupi up on :{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
