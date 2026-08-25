# ulupi — om-memory

OpenHuman-tier personal memory + retrieval. Free, local, zero cloud.

## Architecture
- **Store**: markdown files (`data/memory/`) are source of truth; SQLite index is rebuildable
- **Retrieval**: hybrid score = BM25 (FTS5) + neural vectors (Ollama nomic-embed) + entity graph walk + freshness + coverage + path boost
- **Facts**: regex fast path → gemma4 LLM extraction fallback (`think:false`); subject-predicate-object triples with conflict supersede
- **Chat history**: threaded messages + FTS5, layered into every recall
- **Recall**: TokenJuice-style context packer — FACTS → RECENT CHAT → PAST CONVERSATIONS → chunks, under char budget
- **Guard**: near-dupe rejection (cosine 0.92 + Jaccard), forget tool, budget caps
- **Server**: HTTP daemon on :8792 (`/?q=` recall, `/search` JSON, `/add` dupe-guarded write)
- **Speed**: ~18ms warm average retrieval, all signals cached

## Usage
```bash
python3 om.py demo                  # seed + self-check
python3 om.py add people/arun.md "Arun prefers voice commands"
python3 om.py search "who owns chilaka" --mode graph_first
python3 om.py recall "context for my LLM prompt"
python3 core/server.py              # daemon on :8792
./core/ulupi.sh "question"          # shortcut entry (auto-starts daemon)
```

## Requirements
- Python 3.10+ (stdlib only)
- Ollama with `nomic-embed-text` (+ optional `gemma4:e4b-mlx` for fact extraction)

## Modes
`balanced` · `semantic` · `lexical` · `graph_first` · `fresh`
