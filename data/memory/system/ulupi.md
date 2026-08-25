Ulupi is sir Niranjan's personal AI assistant — and this memory store is its brain. Sir built it himself with opencode.

## What ulupi is
- A local-first personal memory + retrieval engine (OpenHuman-tier), free, zero cloud.
- Ulupi serves sir through an Apple Shortcut: ask → retrieve → ChatGPT answers with context → Q+A saved back to memory.

## How sir built it
- Started from memOTry (chilaka) — markdown + FTS only.
- Studied OpenHuman's open-source brain (Memory Tree, hybrid search) and rebuilt the ideas in pure Python.
- Created the om-memory folder on Desktop as the new engine; named the assistant "ulupi".
- Wired a shortcut cloned from chilaka's flow, port 8792 (8791 was squatted by memGraph).

## Architecture
- Store: markdown files in data/memory are source of truth; SQLite index.db is rebuildable.
- Retrieval: hybrid score = BM25 (FTS5) + neural vectors (Ollama nomic-embed-text) + entity graph walk + freshness + coverage + path boost.
- Facts: regex fast path → gemma4:e4b-mlx LLM extraction (think:false); S-P-O triples; likes are multi-valued; single-valued predicates supersede.
- Chat history: messages table + FTS5, layered into recall (FACTS → RECENT CHAT → PAST CONVERSATIONS → chunks).
- Recall(): TokenJuice-style context packer under char budget.
- Obsidian: data/memory IS the vault — Brain.md MOC + Entities/ hub notes with wikilinks.
- Guard: near-dupe rejection (cosine 0.92 + Jaccard 0.85), forget tool, budget caps.
- Family aliases: mother→bhavani, father→gopal, sister→rakshitha, uncle→mohan.

## Code map
- core/engine.py — the whole engine (embeddings, graph, facts, history, recall, vault export).
- core/server.py — HTTP daemon :8792 (/ping, /?q= recall, /search JSON, /add).
- core/ulupi.sh — shortcut entry; auto-starts daemon; persona + voice-free flow.
- om.py — CLI: add/search/recall/vault/demo/sync/stats.
- config/persona.md — super persona: answer-first, 3-4 lines max, seamless knowledge blending.
- shortcuts/ulupi.xml — the Apple Shortcut definition.

## Performance
- Warm retrieval ~18-21ms average; sync short-circuits when nothing changed.
- Both brains (nomic embed + gemma4) warmed at server start; keep_alive 60m.

## Repo
- github.com/ionyiqapps-creator/ulupi-memory
