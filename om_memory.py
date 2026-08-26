#!/usr/bin/env python3
"""om-memory: OpenHuman-style personal memory retrieval. Free, local, stdlib-only.

Hybrid score = keyword (FTS5) + graph (shared-entity boost) + freshness.
Markdown files are the source of truth; SQLite is a rebuildable index.

Usage:
  python3 om_memory.py add <file.md> "note text"
  python3 om_memory.py search "query" [--mode balanced|lexical|graph_first|fresh]
  python3 om_memory.py rebuild
  python3 om_memory.py demo
"""

import json
import math
import re
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
STORE = ROOT / "data" / "memory"
DB = ROOT / "index.db"

WEIGHTS = {
    "balanced":   {"kw": 1.0, "graph": 1.0, "fresh": 0.5},
    "semantic":   {"kw": 0.5, "graph": 1.5, "fresh": 0.5},  # no vectors (free tier): graph stands in
    "lexical":    {"kw": 2.0, "graph": 0.3, "fresh": 0.2},
    "graph_first":{"kw": 0.3, "graph": 2.0, "fresh": 0.3},
    "fresh":      {"kw": 0.5, "graph": 0.5, "fresh": 2.0},
}

STOP = set("a an the is are was were be been of to in on at for with and or it this that".split())


def db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con


def init(con):
    con.executescript("""
    CREATE TABLE IF NOT EXISTS chunks(
      id INTEGER PRIMARY KEY, path TEXT, body TEXT,
      entities TEXT, mtime REAL);
    CREATE TABLE IF NOT EXISTS entities(name TEXT PRIMARY KEY);
    """)
    if not con.execute("SELECT 1 FROM sqlite_master WHERE name='chunks_fts'").fetchone():
        con.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(body, path, content='chunks', content_rowid='id')")
    con.commit()


TOKEN = re.compile(r"[A-Za-z]\w+")
WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")


def extract_entities(text):
    ents = set(WIKILINK.findall(text))
    # proper nouns not at sentence start, simple heuristic
    for m in re.finditer(r"(?<![.!?\n]\s)(?<!^)\b([A-Z][a-z]{2,})\b", text, re.M):
        ents.add(m.group(1))
    return {e.strip().lower() for e in ents if e.lower() not in STOP}


def chunk_text(text):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def index_file(con, path):
    rel = str(path.relative_to(STORE))
    text = path.read_text()
    mtime = path.stat().st_mtime
    for r in con.execute("SELECT id FROM chunks WHERE path=?", (rel,)).fetchall():
        con.execute("DELETE FROM chunks_fts WHERE rowid=?", (r["id"],))
    con.execute("DELETE FROM chunks WHERE path=?", (rel,))
    for para in chunk_text(text):
        ents = extract_entities(para)
        cur = con.execute(
            "INSERT INTO chunks(path,body,entities,mtime) VALUES(?,?,?,?)",
            (rel, para, json.dumps(sorted(ents)), mtime))
        con.execute("INSERT INTO chunks_fts(rowid,body,path) VALUES(?,?,?)",
                    (cur.lastrowid, para, rel))
        for e in ents:
            con.execute("INSERT OR IGNORE INTO entities VALUES(?)", (e,))
    con.commit()


def rebuild(con):
    init(con)
    con.execute("DELETE FROM chunks"); con.execute("DELETE FROM entities")
    for md in STORE.rglob("*.md"):
        index_file(con, md)


def kw_scores(con, query):
    toks = [t for t in TOKEN.findall(query.lower()) if t not in STOP]
    if not toks:
        return {}
    q = " OR ".join(f'"{t}"' for t in toks)
    try:
        rows = con.execute(
            "SELECT rowid, bm25(chunks_fts) AS r FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY r LIMIT 50",
            (q,)).fetchall()
    except Exception:
        return {}
    if not rows:
        return {}
    best = min(abs(r["r"]) for r in rows) or 1e-6
    return {r["rowid"]: max(0.0, 1 - abs(r["r"]) / (best * 4)) for r in rows}


def graph_scores(con, query_ents, query_toks=()):
    # query entities + any known entity mentioned verbatim in the query
    known = {r["name"] for r in con.execute("SELECT name FROM entities")}
    hits = set(query_ents) | ({t for t in query_toks if t in known})
    if not hits:
        return {}
    rows = con.execute("SELECT id, entities FROM chunks").fetchall()
    scores = {}
    for r in rows:
        ents = set(json.loads(r["entities"]))
        shared = ents & hits
        if shared:
            scores[r["id"]] = len(shared) / math.sqrt(len(ents) or 1)
    mx = max(scores.values(), default=1) or 1
    return {k: v / mx for k, v in scores.items()}


def fresh_scores(con):
    now = time.time()
    scores = {}
    for r in con.execute("SELECT id, mtime FROM chunks"):
        age_days = max(0, (now - r["mtime"]) / 86400)
        scores[r["id"]] = 1 / (1 + age_days / 30)  # half-life ~30d
    return scores


def search(query, mode="balanced", limit=8):
    con = db(); init(con)
    w = WEIGHTS.get(mode, WEIGHTS["balanced"])
    qtoks = {t for t in TOKEN.findall(query.lower()) if t not in STOP}
    qents = extract_entities(query)
    kw = kw_scores(con, query)
    gr = graph_scores(con, qents, qtoks)
    fr = fresh_scores(con)
    ids = set(kw) | set(gr)
    if not ids:  # pure freshness fallback
        ids = set(fr)
    out = []
    for cid in ids:
        s = w["kw"] * kw.get(cid, 0) + w["graph"] * gr.get(cid, 0) + w["fresh"] * fr.get(cid, 0)
        out.append((s, cid))
    out.sort(reverse=True)
    results = []
    for s, cid in out[:limit]:
        r = con.execute("SELECT path, body FROM chunks WHERE id=?", (cid,)).fetchone()
        results.append({"score": round(s, 3), "path": r["path"], "text": r["body"],
                        "signals": {"kw": round(kw.get(cid, 0), 2),
                                    "graph": round(gr.get(cid, 0), 2),
                                    "fresh": round(fr.get(cid, 0), 2)}})
    con.close()
    return results


def add(fname, text):
    p = STORE / fname
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(("\n\n" if p.exists() and p.stat().st_size else "") + text + "\n")
    con = db(); init(con); index_file(con, p); con.close()
    return str(p)


def seed():
    seed_files = {
        "people/niranjan.md": "Niranjan works at IONYIQ as an Associate Full Stack Developer.",
    }
    for name, txt in seed_files.items():
        p = STORE / name
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(txt)
    con = db(); rebuild(con); con.close()


def demo():
    seed()
    for mode in ("balanced", "graph_first"):
        print(f"\n== search 'where does niranjan work' [{mode}] ==")
        for r in search("where does niranjan work", mode=mode, limit=3):
            print(f"  {r['score']}  {r['path']}  {r['signals']}\n    {r['text'][:80]}")
    assert any("ionyiq" in r["path"] or "niranjan" in r["path"] for r in search("niranjan work")), "hybrid search failed"
    print("\nOK")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__); sys.exit(0)
    cmd = args[0]
    if cmd == "add":
        print(add(args[1], args[2]))
    elif cmd == "search":
        mode = "balanced"
        if "--mode" in args:
            mode = args[args.index("--mode") + 1]
        print(json.dumps(search(args[1], mode=mode), indent=2))
    elif cmd == "rebuild":
        con = db(); rebuild(con); con.close(); print("rebuilt")
    elif cmd == "demo":
        demo()
    else:
        print(__doc__)
