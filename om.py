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
    demo_file = engine.STORE / "inbox" / "_demo_dupe.md"
    if demo_file.exists():
        demo_file.unlink()
    con = engine.db(); engine.rebuild(con); con.close()

    tests = [
        ("who is chintu", "chintu"),
        ("where does niranjan work", "ionyiq"),
        ("who is the mother of niranjan", "bhavani"),
    ]
    for q, expect in tests:
        ctx = engine.recall(q).lower()
        assert expect in ctx, f"FAIL '{q}'"
        print(f"PASS '{q}'")

    # guard: dupe rejected
    p = engine.add("inbox/_demo_dupe.md", "Demo dupe guard note for testing.")
    assert p is not None and engine.add("inbox/_demo_dupe.md", "Demo dupe guard note for testing.") is None
    print("PASS dupe-guard")

    # graph: entity one-hop
    g = engine.search("ionyiq company work", mode="graph_first", limit=3)
    assert any("ionyiq" in r["path"] or "ups" in r["path"] for r in g), f"graph walk weak: {[r['path'] for r in g]}"
    print(f"PASS graph-walk -> {[r['path'] for r in g]}")

    ctx = engine.recall("niranjan family")
    assert len(ctx) <= engine.BUDGET_CHARS and ctx
    print(f"PASS recall budget ({len(ctx)} chars)")

    print(json.dumps(engine.stats(), indent=2))
    print("\nALL OK")


if __name__ == "__main__":
    main()
