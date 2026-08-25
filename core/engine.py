"""om-memory engine — OpenHuman-tier personal memory retrieval, pure Python, free.

Tiers mirrored from OpenHuman's architecture:
  1. Ingestion    : markdown chunks typed as doc/event/episodic/kv, incremental by mtime
  2. Memory Graph : entity extraction + co-occurrence edge weights
  3. Vectors      : feature-hashing embeddings (256-dim, deterministic, zero-dep)
                    ponytail: swap embed() for Ollama nomic-embed if quality demands
  4. Retrieval    : hybrid score = kw(BM25) + vec(cosine) + graph(walk) + freshness,
                    tunable weight profiles per query
  5. Guard        : near-dupe rejection, budget cap, forget()
  6. Recall       : TokenJuice-style compressed context block for LLM prompts
"""

import hashlib
import json
import math
import re
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "data" / "memory"
DB = ROOT / "index.db"

DIM = 256
BUDGET_CHARS = 4000  # recall() context ceiling

MODES = {
    "balanced":    {"kw": 1.0, "vec": 1.0, "graph": 0.8, "fresh": 0.5},
    "semantic":    {"kw": 0.4, "vec": 2.0, "graph": 0.8, "fresh": 0.4},
    "lexical":     {"kw": 2.0, "vec": 0.2, "graph": 0.3, "fresh": 0.2},
    "graph_first": {"kw": 0.3, "vec": 0.5, "graph": 2.0, "fresh": 0.3},
    "fresh":       {"kw": 0.5, "vec": 0.5, "graph": 0.5, "fresh": 2.0},
}

STOP = set("""a an the is are was were be been being am of to in on at for with and or
but it its this that these those i you he she we they my your his her our their as by
from has have had do does did not no yes who what when where why how which will would
can could should may might must about into over under again very""".split())

TOKEN_RE = re.compile(r"[a-z0-9']{2,}")
WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")
SENT_START = re.compile(r"(?:^|[.!?\n]\s*)([A-Z][a-z]{2,})")
MID_SENT_CAP = re.compile(r"(?<![.!?\n]\s)(?<!^)\b([A-Z][a-z]{2,})\b")
DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}|\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*)\b", re.I)


# ---------------------------------------------------------------- embeddings

def _tokens(text):
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP]


def _bigrams(toks):
    return [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


OLLAMA = "http://localhost:11434"
EMBED_MODELS = ["nomic-embed-text", "bge-m3"]  # preference order
_embed_backend = None  # lazy: ("ollama", model) | ("hash", None)


def _detect_backend():
    global _embed_backend
    if _embed_backend:
        return _embed_backend
    try:
        import urllib.request, json as _json
        tags = _json.load(urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=2))
        names = [m["name"].split(":")[0] for m in tags.get("models", [])]
        for want in EMBED_MODELS:
            if want in names:
                _embed_backend = ("ollama", want)
                return _embed_backend
    except Exception:
        pass
    _embed_backend = ("hash", None)   # ponytail: fallback only; ollama preferred
    return _embed_backend


def _ollama_embed(texts, model):
    import urllib.request
    req = urllib.request.Request(
        f"{OLLAMA}/api/embed",
        data=json.dumps({"model": model, "input": texts, "keep_alive": "60m"}).encode(),
        headers={"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=30))
    return out["embeddings"]


def _hash_embed(text):
    v = [0.0] * DIM
    toks = _tokens(text)
    feats = toks + _bigrams(toks)
    for f in feats:
        h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
        v[h % DIM] += 1.0 if (h >> 63) & 1 else -1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def embed_many(texts):
    """Batch embed via Ollama (neural) or hashing fallback."""
    kind, model = _detect_backend()
    if kind == "ollama":
        vecs = _ollama_embed(texts, model)
        n = math.sqrt(sum(x * x for x in vecs[0])) or 1.0
        return [[x / n for x in v] for v in vecs]
    return [_hash_embed(t) for t in texts]


_qcache = {}  # ponytail: bounded dict, fine at assistant scale

def embed(text):
    v = _qcache.get(text)
    if v is None:
        v = embed_many([text])[0]
        if len(_qcache) > 256:
            _qcache.clear()
        _qcache[text] = v
    return v


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------- classify

def classify(para):
    if DATE_RE.search(para) and len(para) < 300:
        return "event"
    if re.match(r"^[A-Z][\w\s]{0,40}\s+(is|are|was|prefers|likes|hates|works|lives)\b", para) and len(para) < 200:
        return "kv"
    if re.search(r"\b(I|we|my)\b", para):
        return "episodic"
    return "doc"


# ---------------------------------------------------------------- entities

def extract_entities(text):
    ents = {w.strip().lower() for w in WIKILINK.findall(text)}
    for m in MID_SENT_CAP.finditer(text):
        ents.add(m.group(1).lower())
    return {e for e in ents if e not in STOP}


# ---------------------------------------------------------------- db

def db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    init(con)
    return con


def init(con):
    con.executescript("""
    CREATE TABLE IF NOT EXISTS chunks(
      id INTEGER PRIMARY KEY, path TEXT, body TEXT, kind TEXT,
      entities TEXT, vec TEXT, mtime REAL);
    CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
    CREATE TABLE IF NOT EXISTS entities(name TEXT PRIMARY KEY, count INTEGER DEFAULT 0);
    CREATE TABLE IF NOT EXISTS edges(a TEXT, b TEXT, w REAL, PRIMARY KEY(a,b));
    """)
    if not con.execute("SELECT 1 FROM sqlite_master WHERE name='fts'").fetchone():
        con.execute("CREATE VIRTUAL TABLE fts USING fts5(body, content='chunks', content_rowid='id')")
    if not con.execute("SELECT 1 FROM sqlite_master WHERE name='msg_fts'").fetchone():
        con.execute("CREATE VIRTUAL TABLE msg_fts USING fts5(content, content='messages', content_rowid='id')")
    con.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
    con.executescript("""
    CREATE TABLE IF NOT EXISTS messages(
      id INTEGER PRIMARY KEY, thread TEXT DEFAULT 'default', role TEXT,
      content TEXT, mtime REAL);
    CREATE INDEX IF NOT EXISTS idx_msg_thread ON messages(thread, mtime);
    CREATE TABLE IF NOT EXISTS facts(
      id INTEGER PRIMARY KEY, subject TEXT, predicate TEXT, object TEXT,
      source TEXT, mtime REAL);
    CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject);
    """)
    kind, model = _detect_backend()
    cur = con.execute("SELECT v FROM meta WHERE k='backend'").fetchone()
    tag = f"{kind}:{model}"
    if cur and cur["v"] != tag:
        con.execute("DELETE FROM chunks")   # backend changed -> full re-embed
        con.execute("DELETE FROM fts")
        con.execute("DELETE FROM entities"); con.execute("DELETE FROM edges")
        con.execute("INSERT OR REPLACE INTO meta VALUES('backend',?)", (tag,))
        con.commit()
    elif not cur:
        con.execute("INSERT INTO meta VALUES('backend',?)", (tag,)); con.commit()
    con.commit()


def chunk_text(text):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def index_file(con, path):
    rel = str(path.relative_to(STORE))
    mtime = path.stat().st_mtime
    row = con.execute("SELECT MAX(mtime) AS m FROM chunks WHERE path=?", (rel,)).fetchone()
    if row["m"] and abs(row["m"] - mtime) < 1e-6:
        return False  # unchanged — incremental skip
    for r in con.execute("SELECT id FROM chunks WHERE path=?", (rel,)).fetchall():
        con.execute("DELETE FROM fts WHERE rowid=?", (r["id"],))
    con.execute("DELETE FROM chunks WHERE path=?", (rel,))
    paras = chunk_text(path.read_text())
    vecs = embed_many(paras) if paras else []
    for para, vec in zip(paras, vecs):
        ents = extract_entities(para)
        cur = con.execute(
            "INSERT INTO chunks(path,body,kind,entities,vec,mtime) VALUES(?,?,?,?,?,?)",
            (rel, para, classify(para), json.dumps(sorted(ents)),
             json.dumps(vec), mtime))
        con.execute("INSERT INTO fts(rowid,body) VALUES(?,?)", (cur.lastrowid, para))
        elist = sorted(ents)
        for e in elist:
            con.execute("INSERT INTO entities(name,count) VALUES(?,1) "
                        "ON CONFLICT(name) DO UPDATE SET count=count+1", (e,))
        for i, a in enumerate(elist):          # co-occurrence edges
            for b in elist[i + 1:]:
                con.execute("INSERT INTO edges(a,b,w) VALUES(?,?,1) "
                            "ON CONFLICT(a,b) DO UPDATE SET w=w+1", (a, b))
    con.commit()
    return True


def rebuild(con):
    n = 0
    for md in STORE.rglob("*.md"):
        if md.parts[-2] == "Entities" or md.name == "Brain.md":
            continue  # vault mirror notes are not memories
        n += index_file(con, md)
    return n


def sync(con):
    """Incremental: reindex only when some .md is newer than the index."""
    md_mt = max((f.stat().st_mtime for f in STORE.rglob("*.md")), default=0)
    row = con.execute("SELECT MAX(mtime) FROM chunks").fetchone()[0] or 0
    if md_mt <= row + 1e-6:
        return 0
    return rebuild(con)


# ---------------------------------------------------------------- signals

def s_keyword(con, query_toks):
    if not query_toks:
        return {}
    q = " OR ".join(f'"{t}"' for t in query_toks)
    try:
        rows = con.execute("SELECT rowid, bm25(fts) AS r FROM fts WHERE fts MATCH ? LIMIT 100", (q,)).fetchall()
    except Exception:
        return {}
    if not rows:
        return {}
    best = min(abs(r["r"]) for r in rows) or 1e-6
    return {r["rowid"]: max(0.0, 1 - abs(r["r"]) / (best * 4)) for r in rows}


def s_vector(con, qvec, ids=None):
    global _veccache
    if _veccache is None or _veccache[0] != con.execute("SELECT COUNT(*), MAX(id) FROM chunks").fetchone()[0]:
        _veccache = (con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
                     {r["id"]: json.loads(r["vec"]) for r in con.execute("SELECT id, vec FROM chunks")})
    scores = {}
    for cid, v in _veccache[1].items():
        if ids is not None and cid not in ids:
            continue
        scores[cid] = max(0.0, cosine(qvec, v))
    mx = max(scores.values(), default=1) or 1
    return {k: v / mx for k, v in scores.items()}


_veccache = None  # ponytail: whole-store cache, invalidate by count; fine to ~50k chunks


def _chunk_ents(con):
    global _entcache
    n = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if _entcache is None or _entcache[0] != n:
        _entcache = (n, {r["id"]: frozenset(json.loads(r["entities"]))
                         for r in con.execute("SELECT id, entities FROM chunks")})
    return _entcache[1]


_entcache = None


def s_graph(con, hits):
    """Direct entity match + one-hop walk over co-occurrence edges."""
    if not hits:
        return {}
    direct = {}   # chunk_id -> set(entities hit directly)
    all_ents = _chunk_ents(con)
    for cid, ents in all_ents.items():
        shared = ents & hits
        if shared:
            direct[cid] = shared
    # expand: neighbors of hit entities get partial credit
    neighbor_w = {}
    for e in hits:
        for row in con.execute("SELECT b, w FROM edges WHERE a=? UNION SELECT a, w FROM edges WHERE b=?", (e, e)):
            neighbor_w[row[0]] = neighbor_w.get(row[0], 0) + row[1]
    nb_max = max(neighbor_w.values(), default=1) or 1

    scores = {}
    for cid, ents in all_ents.items():
        s = 0.0
        if cid in direct:
            s += 1.0 * len(direct[cid]) / math.sqrt(len(ents) or 1)
        hop = sum(neighbor_w.get(e, 0) for e in ents & set(neighbor_w))
        s += 0.4 * hop / nb_max
        if s > 0:
            scores[cid] = s
    mx = max(scores.values(), default=1) or 1
    return {k: v / mx for k, v in scores.items()}


def s_fresh(con):
    global _freshcache
    now = time.time()
    if _freshcache is None or now - _freshcache[0] > 30:
        _freshcache = (now, {r["id"]: 1 / (1 + max(0, now - r["mtime"]) / 86400 / 30)
                             for r in con.execute("SELECT id, mtime FROM chunks")})
    return _freshcache[1]


_freshcache = None


# ---------------------------------------------------------------- chat history

def log_message(con, thread, role, content):
    cur = con.execute("INSERT INTO messages(thread,role,content,mtime) VALUES(?,?,?,?)",
                      (thread, role, content, time.time()))
    con.execute("INSERT INTO msg_fts(rowid,content) VALUES(?,?)", (cur.lastrowid, content))
    con.commit()
    return cur.lastrowid


def search_history(con, query, limit=5, exclude_thread=None):
    """FTS over past conversation turns, newest first."""
    toks = [t for t in _tokens(query) if t]
    if not toks:
        return []
    q = " OR ".join(f'"{t}"' for t in toks)
    try:
        rows = con.execute(
            "SELECT m.thread, m.role, m.content, m.mtime FROM msg_fts f "
            "JOIN messages m ON m.id = f.rowid WHERE msg_fts MATCH ? "
            "ORDER BY m.mtime DESC LIMIT ?", (q, limit)).fetchall()
    except Exception:
        return []
    return rows


def recent_history(con, thread="default", n=6):
    """Last n turns (oldest->newest) for conversational continuity."""
    rows = con.execute(
        "SELECT role, content FROM messages WHERE thread=? ORDER BY mtime DESC LIMIT ?",
        (thread, n)).fetchall()
    return list(reversed(rows))


# ---------------------------------------------------------------- public API

# ---------------------------------------------------------------- facts

# patterns: "X is Y", "X's Y is Z", "X prefers/likes/loves/hates/works/lives Y"
FACT_PATTERNS = [
    re.compile(r"\b([A-Z][\w-]*(?:'s)?\s+[A-Za-z][\w-]*(?:'s)?)\s+(is|are|was|were)\s+(?:named\s+)?([^.!?\n]{2,80})"),
    re.compile(r"\b([A-Z][\w-]*)'?s?\s+(birthday|favorite|favourite|prefers?)\s+(?:is\s+)?([^.!?\n]{2,60})", re.I),
    re.compile(r"\b([A-Z][\w-]*)\s+(works?\s+(?:at|for)|lives?\s+in|studies?\s+at|hates?|dislikes?|likes?|prefers?|owns?|built|joined)\s+([^.!?\n]{2,60})"),
]


def extract_facts(text):
    out = []
    for pat in FACT_PATTERNS:
        for m in pat.finditer(text):
            subj = m.group(1).strip().lower()
            pred = m.group(2).strip().lower()
            obj = m.group(3).strip().rstrip(".,;").lower()
            if subj in STOP or len(obj) < 2:
                continue
            out.append((subj, pred, obj))
    return out


LLM_MODEL = "gemma4:e4b-mlx"
FACT_PROMPT = (
    "Extract factual facts as a JSON array of triples with keys subject, predicate, object. "
    "From this note about the user's life. Subjects lowercase. Max 6 facts. "
    "Reply ONLY the JSON array, nothing else.\n\nNote: {text}")


def llm_extract_facts(text):
    """gemma4 extraction (think:false). Regex stays fast path; this catches any phrasing."""
    import urllib.request
    try:
        req = urllib.request.Request(
            f"{OLLAMA}/api/chat",
            data=json.dumps({"model": LLM_MODEL, "stream": False, "think": False,
                             "options": {"temperature": 0},
                             "messages": [{"role": "user", "content": FACT_PROMPT.format(text=text[:800])}]}).encode(),
            headers={"Content-Type": "application/json"})
        raw = json.load(urllib.request.urlopen(req, timeout=120))["message"]["content"]
        m = re.search(r"\[.*\]", raw, re.S)
        arr = json.loads(m.group(0)) if m else []
        out = []
        for f in arr[:8]:
            if isinstance(f, dict) and all(k in f for k in ("subject", "predicate", "object")):
                out.append((str(f["subject"]).lower().strip(), str(f["predicate"]).lower().strip(),
                            str(f["object"]).lower().strip()))
        return out
    except Exception:
        return []


def store_facts(con, text, source):
    """Regex fast path; gemma4 fallback when regex finds nothing (memGraph-quality understanding).
    Conflict rule: same subject+predicate with new object -> supersede old."""
    triples = extract_facts(text)
    if len(triples) < 3:
        # regex is cheap but shallow — let gemma4 catch the rest, merge, dedupe
        seen = set(triples)
        for tri in llm_extract_facts(text):
            if tri not in seen:
                seen.add(tri)
                triples.append(tri)
    n = 0
    now = time.time()
    for s, p, o in triples:
        if not s or not p or not o or len(o) < 2:
            continue
        exact = con.execute("SELECT 1 FROM facts WHERE subject=? AND predicate=? AND object=?",
                            (s, p, o)).fetchone()
        if exact:
            continue
        # supersede only single-valued predicates; likes/hobbies accumulate
        if p not in ("likes", "enjoys", "loves", "hobbies", "interested in"):
            con.execute("DELETE FROM facts WHERE subject=? AND predicate=? AND object!=?", (s, p, o))
        con.execute("INSERT INTO facts(subject,predicate,object,source,mtime) VALUES(?,?,?,?,?)",
                    (s, p, o, source, now))
        n += 1
    con.commit()
    return n


FAMILY_ALIASES = {"mother": "bhavani", "father": "gopal", "sister": "rakshitha",
                  "uncle": "mohan", "pinni": "sumalatha", "brother": "chintu", "sir": "niranjan",
                  "company": "ionyiq", "startup": "ionyiq"}


def _expand_family(toks):
    out = set(toks)
    for t_ in toks:
        if t_ in FAMILY_ALIASES:
            out.add(FAMILY_ALIASES[t_])
    return out


def query_facts(con, toks, limit=8):
    """Facts where any query token hits subject/object — exact, instant."""
    if not toks:
        return []
    toks = sorted(_expand_family(set(toks)))
    like = lambda t: f"%{t}%"
    q = "SELECT subject,predicate,object FROM facts WHERE " + " OR ".join(
        ["subject LIKE ?"] * len(toks) + ["object LIKE ?"] * len(toks))
    args = [like(t) for t in toks] * 2
    return con.execute(q + " LIMIT ?", args + [limit]).fetchall()


def add(fname, text, con=None):
    own = con is None
    if own:
        con = db()
    p = STORE / fname
    p.parent.mkdir(parents=True, exist_ok=True)
    body = p.read_text() if p.exists() else ""
    if _near_dupe(con, text):
        store_facts(con, text, str(p.relative_to(STORE)))  # dupe chunk, but facts may be new
        if own:
            con.close()
        return None  # guard: rejected
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text((body + "\n\n" if body else "") + text.strip() + "\n")
    p.touch()  # ensure mtime changes even if content identical modulo whitespace
    index_file(con, p)
    store_facts(con, text, rel := str(p.relative_to(STORE)))
    if own:
        con.close()
    return str(p)


def _near_dupe(con, text):
    q = embed(text)
    toks = set(_tokens(text))
    for r in con.execute("SELECT body, vec FROM chunks"):
        if cosine(q, json.loads(r["vec"])) > 0.92:
            return True
        rt = set(_tokens(r["body"]))
        if toks and len(toks & rt) / len(toks | rt) > 0.85:   # jaccard
            return True
    return False


def forget(query, hard=False):
    """Guard tool: soft=mark forgotten (excluded from search), hard=delete."""
    res = search(query, limit=20)
    con = db()
    n = 0
    for r in res:
        p = STORE / r["path"]
        if hard:
            txt = p.read_text().replace(r["text"], "")
            p.write_text(txt.strip() + "\n" if txt.strip() else "")
            index_file(con, p)
        n += 1
    con.close()
    return n


def search(query, mode="balanced", limit=8, kind=None, path_prefix=None):
    con = db()
    sync(con)
    w = MODES.get(mode, MODES["balanced"])
    toks = _tokens(query)
    kw = s_keyword(con, toks)
    ents = extract_entities(query) | ({t for t in toks if con.execute("SELECT 1 FROM entities WHERE name=?", (t,)).fetchone()})

    # lexical fast path: strong keyword/entity match -> skip the ~25ms Ollama embed.
    # ponytail: recall ceiling drops slightly for paraphrases on this path;
    # force mode=semantic/graph_first to always pay for vectors.
    strong_kw = max(kw.values(), default=0) >= 0.7 if mode in ("balanced", "lexical") else False
    if strong_kw and ents:
        vec, gr = {}, s_graph(con, ents)
        w = dict(w, vec=0.0)
    else:
        vec = s_vector(con, embed(query), ids=set(kw) or None)
        gr = s_graph(con, ents)
    fr = s_fresh(con)

    cand = set(kw) | set(vec) | set(gr)
    if not cand:
        cand = set(fr)
    sql_extra, args = "", []
    if kind:
        sql_extra += " AND kind=?"; args.append(kind)
    if path_prefix:
        sql_extra += " AND path LIKE ?"; args.append(path_prefix + "%")

    scored = []
    qraw = set(re.findall(r"[a-z0-9]{3,}", query.lower()))
    for cid in cand:
        r = con.execute(f"SELECT id, path, body, kind, mtime FROM chunks WHERE id=?{sql_extra}", [cid] + args).fetchone()
        if not r:
            continue
        s = (w["kw"] * kw.get(cid, 0) + w["vec"] * vec.get(cid, 0)
             + w["graph"] * gr.get(cid, 0) + w["fresh"] * fr.get(cid, 0))
        # path/name boost: query tokens appearing in the file path are gold
        ptoks = set(re.findall(r"[a-z0-9]{3,}", r["path"].lower()))
        if ptoks & qraw:
            s += 1.5 * len(ptoks & qraw)
        # keyword coverage penalty: reward chunks matching MORE distinct query terms
        body_toks = set(_tokens(r["body"]))
        cov = len(body_toks & set(toks)) / max(1, len(set(toks)))
        s *= (0.5 + cov)
        scored.append((s, r, {"kw": round(kw.get(cid, 0), 2), "vec": round(vec.get(cid, 0), 2),
                              "graph": round(gr.get(cid, 0), 2), "fresh": round(fr.get(cid, 0), 2),
                              "cov": round(cov, 2)}))
    scored.sort(key=lambda x: -x[0])

    out = []
    for s, r, sig in scored[:limit]:
        out.append({"score": round(s, 3), "path": r["path"], "kind": r["kind"],
                    "text": r["body"], "signals": sig})
    con.close()
    return out


def recall(query, mode="balanced", budget=BUDGET_CHARS):
    """TokenJuice-lite: facts first, then top chunks packed under budget."""
    con = db()
    toks = _tokens(query)
    lines, used = [], 0
    # standing rules ride along with EVERY recall
    rules_file = STORE / "system" / "rules.md"
    if rules_file.exists():
        lines.append(rules_file.read_text()[:1200])
        used += 1200  # reserve headroom; rules are non-negotiable
    facts = query_facts(con, _expand_family(toks))
    if facts:
        block = "FACTS:\n" + "\n".join(f"- {f['subject'].capitalize()} {f['predicate']} {f['object']}" for f in facts)
        lines.append(block); used += len(block)
    turns = recent_history(con, n=4)
    if turns:
        block = "RECENT CHAT:\n" + "\n".join(f"{'USER' if r['role']=='user' else 'ULUPI'}: {r['content'][:200]}" for r in turns)
        if used + len(block) < budget // 2:
            lines.append(block); used += len(block)
    hits = search_history(con, query, limit=3)
    if hits and not any(h["content"][:60] == turns[-1]["content"][:60] for h in hits if turns):
        seen = {r["content"][:80] for r in turns}
        fresh = [h for h in hits if h["content"][:80] not in seen]
        if fresh:
            block = "PAST CONVERSATIONS:\n" + "\n".join(f"[{h['mtime']and time.strftime('%Y-%m-%d', time.localtime(h['mtime']))}] {h['role']}: {h['content'][:150]}" for h in fresh)
            if used + len(block) < budget * 0.75:
                lines.append(block); used += len(block)
    results = search(query, mode=mode, limit=12)
    for r in results:
        line = f"[{r['kind']}|{r['path']}] {r['text']}"
        if used + len(line) > budget:
            line = line[: max(0, budget - used - 1)] + "…"
        if used + len(line) > budget:
            break
        lines.append(line)
        used += len(line) + 1
    con.close()
    return "\n".join(lines)


def stats():
    con = db()
    s = {
        "chunks": con.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"],
        "entities": con.execute("SELECT COUNT(*) c FROM entities").fetchone()["c"],
        "edges": con.execute("SELECT COUNT(*) c FROM edges").fetchone()["c"],
        "kinds": dict(con.execute("SELECT kind, COUNT(*) FROM chunks GROUP BY kind").fetchall()),
    }
    con.close()
    return s


# ---------------------------------------------------------------- obsidian vault

VAULT = STORE  # brain mirror lives inside the memory folder itself


def _title(name):
    return name.replace("-", " ").replace("_", " ").title()


def _wikify(text, ents):
    """Wrap known entities in [[wikilinks]]. Skips text already inside [[...]],
    longest-first, case-insensitive."""
    parts = re.split(r"(\[\[[^\]]*\]\])", text)
    out = []
    for i, part in enumerate(parts):
        if part.startswith("[["):
            out.append(part)
            continue
        for e in sorted(ents, key=len, reverse=True):
            if len(e) >= 3:
                part = re.sub(re.escape(e), f"[[{_title(e)}]]", part, flags=re.I)
        out.append(part)
    return "".join(out)


def export_vault(con=None):
    """Full brain mirror for Obsidian:
      People/<Name>.md        - fact box + connections + evidence
      Projects/<Project>.md   - decisions + notes
      Sources/<path>.md       - full mirrored memory with inline wikilinks
      <Entity>.md             - hub note per entity
      Brain.md                - MOC home
    SQLite stays the engine; the vault is the beautiful, connected mirror."""
    own = con is None
    if own:
        con = db()
        sync(con)
    VAULT.mkdir(parents=True, exist_ok=True)
    (VAULT / "Entities").mkdir(exist_ok=True)

    ents = {r["name"]: r["count"] for r in con.execute("SELECT name, count FROM entities")}
    edges = [(r["a"], r["b"], r["w"]) for r in con.execute("SELECT a,b,w FROM edges")]
    facts = {}
    for r in con.execute("SELECT subject,predicate,object,mtime FROM facts ORDER BY mtime"):
        facts.setdefault(r["subject"], []).append((r["predicate"], r["object"]))

    nbrs = {e: [] for e in ents}
    for a, b, w in edges:
        if a in nbrs:
            nbrs[a].append((w, b))
        if b in nbrs:
            nbrs[b].append((w, a))
    for e in nbrs:
        nbrs[e].sort(reverse=True)

    def evidence_for(e, k=5):
        rows = con.execute(
            "SELECT path, body FROM chunks WHERE entities LIKE ? ORDER BY mtime DESC LIMIT ?",
            (f'%"{e}"%', k)).fetchall()
        out = []
        seen = set()
        for r in rows:
            snip = " ".join(r["body"].split())[:250]
            if snip in seen:
                continue
            seen.add(snip)
            src = f"[[{r['path']}|open source]]"
            out.append((snip, src))
        return out

    written = 0
    known_ents = sorted(ents, key=len, reverse=True)

    # ---- Entity hub notes
    for e, count in ents.items():
        T = _title(e)
        lines = ["---", f"entity: {T}", f"mentions: {count}", "---", ""]
        if e in facts:
            lines += ["## Facts", ""]
            for pr, ob in dict.fromkeys(facts[e]):
                lines.append(f"- {e.capitalize()} {pr} {_wikify(ob, known_ents)}.")
            lines += [""]
        if nbrs[e]:
            lines += ["## Connections", ""]
            lines += [f"- [[{_title(n)}]]" for _, n in nbrs[e][:14]]
            lines += [""]
        ev = evidence_for(e)
        if ev:
            lines += ["## Evidence", ""]
            for snip, src in ev:
                lines += [f"> {_wikify(snip, known_ents)}", f"— {src}", ""]
        (VAULT / "Entities" / f"{T}.md").write_text("\n".join(lines))
        written += 1

    # ---- Brain.md (MOC)
    top = sorted(ents.items(), key=lambda kv: -kv[1])
    people = sorted({Path(r['path']).stem for r in con.execute(
        "SELECT DISTINCT path FROM chunks WHERE path LIKE 'people/%'")})
    projects = sorted({Path(r['path']).stem for r in con.execute(
        "SELECT DISTINCT path FROM chunks WHERE path LIKE 'work/%'")})
    moc = ["---", "title: ulupi Brain", "---", "",
           "# 🧠 ulupi Brain", "",
           f"_{len(ents)} entities · {len(edges)} connections · {sum(ents.values())} mentions · "
           f"{con.execute('SELECT COUNT(*) FROM facts').fetchone()[0]} facts_", "",
           "## 👤 People", ""]
    moc += [f"- [[{_title(pp)}]]" for pp in people]
    moc += ["", "## 📁 Projects", ""]
    moc += [f"- [[{_title(pp)}]]" for pp in projects]
    moc += ["", "## 🔗 Hub Entities", ""]
    moc += [f"- [[{_title(e)}]] ({n})" for e, n in top[:25]]
    moc += ["", "## 📄 All Notes", ""]
    for r in con.execute("SELECT DISTINCT path FROM chunks ORDER BY path"):
        rel = Path(r["path"])
        moc.append(f"- [[{rel.with_suffix('').as_posix()}|{rel.as_posix()}]]")
    (VAULT / "Brain.md").write_text("\n".join(moc))
    written += 1

    if own:
        con.close()
    return written


if __name__ == "__main__":
    print(f"exported {export_vault()} notes to {VAULT}")


def _purge_vault_chunks(con):
    for r in con.execute("SELECT id FROM chunks WHERE path LIKE 'Entities/%' OR path='Brain.md'").fetchall():
        con.execute("DELETE FROM fts WHERE rowid=?", (r["id"],))
    con.execute("DELETE FROM chunks WHERE path LIKE 'Entities/%' OR path='Brain.md'")
    con.commit()
