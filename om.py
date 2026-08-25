#!/usr/bin/env python3
"""om-memory CLI — OpenHuman-tier personal memory, free & local.

  add <file.md> <text>        add memory (dupe-guarded)
  search <query> [--mode M] [--kind K] [--path P]
  recall <query>              compressed context block for LLM prompts
  forget <query> [--hard]     remove near-matching memories
  sync / rebuild / stats
  demo                        seed + self-check
"""
import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from core import engine

MODES = ",".join(engine.MODES)


def _vault():
    con = engine.db()
    print(f"exported {engine.export_vault(con)} notes -> {engine.VAULT}")
    con.close()


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        return
    cmd = a[0]
    if cmd == "add":
        r = engine.add(a[1], a[2])
        print("rejected (near-dupe)" if r is None else f"added -> {r}")
    elif cmd == "search":
        mode = a[a.index("--mode") + 1] if "--mode" in a else "balanced"
        kind = a[a.index("--kind") + 1] if "--kind" in a else None
        path = a[a.index("--path") + 1] if "--path" in a else None
        print(json.dumps(engine.search(a[1], mode=mode, kind=kind, path_prefix=path), indent=2))
    elif cmd == "recall":
        print(engine.recall(a[1]))
    elif cmd == "forget":
        n = engine.forget(a[1], hard="--hard" in a)
        print(f"forgot {n} chunk(s)")
    elif cmd == "sync":
        con = engine.db(); print(f"indexed {engine.sync(con)} file(s)"); con.close()
    elif cmd == "rebuild":
        con = engine.db(); engine.rebuild(con); con.close(); print("rebuilt")
    elif cmd == "vault":
        _vault()
    elif cmd == "stats":
        print(json.dumps(engine.stats(), indent=2))
    elif cmd == "demo":
        demo()
    else:
        print(__doc__)


def demo():
    seeds = {
        "people/arun.md": ("Arun works at IONYIQ. Arun prefers voice commands over typing.\n\n"
                           "Arun's birthday is March 12.\n\n"
                           "I built chilaka with Arun last weekend and we shipped it in one night."),
        "people/priya.md": "Priya is a designer. Priya recommended Figma for all mockups.",
        "work/chilaka.md": "Chilaka is an Apple Shortcut project by Arun. Chilaka calls the memOTry server.",
        "personal/prefs.md": "Preferred stack is Python and SQLite. Dislikes Electron for desktop apps.",
    }
    for name, txt in seeds.items():
        p = engine.STORE / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_text(txt)
    con = engine.db(); engine.rebuild(con); con.close()

    tests = [
        ("who does chilaka belong to", "chilaka"),
        ("what stack do I like for apps", "python"),  # judged on recall content, not path
        ("when is kiran's birthday", "march"),          # event/kv + date
        ("designer who recommended figma", "priya"),  # semantic-ish paraphrase
    ]
    for q, expect in tests:
        ctx = engine.recall(q).lower()
        assert expect in ctx, f"FAIL '{q}'"
        print(f"PASS '{q}'")

    # guard: dupe rejected
    assert engine.add("people/arun.md", "Arun works at IONYIQ. Arun prefers voice commands over typing.") is None
    print("PASS dupe-guard")

    # graph: entity one-hop
    g = engine.search("chilaka memOTry project", mode="graph_first", limit=3)
    assert any("chilaka" in r["path"] or "memOTry" in r["path"].lower() for r in g), f"graph walk weak: {[r['path'] for r in g]}"
    print(f"PASS graph-walk -> {[r['path'] for r in g]}")

    ctx = engine.recall("arun chilaka")
    assert len(ctx) <= engine.BUDGET_CHARS and ctx
    print(f"PASS recall budget ({len(ctx)} chars)")

    print(json.dumps(engine.stats(), indent=2))
    print("\nALL OK")


if __name__ == "__main__":
    main()

def _vault():
    con = engine.db()
    print(f"exported {engine.export_vault(con)} notes -> {engine.VAULT}")
    con.close()
