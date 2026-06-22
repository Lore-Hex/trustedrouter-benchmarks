"""Grade phase-2 synth output on the decidable items.

Phase-2 output is [{id, synths:[{name, tool_calls}]}]; every item is decidable by
construction (oracle=100% on this subset), so each synth's score = how often it
recovers a correct tool call. Reports per-synth accuracy.

Run: PYTHONPATH=. .venv/bin/python scripts/phase2_grade.py <phase2-output-path>
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

from trbench.evals.bfcl.vendor.ast_checker import Language, ast_checker

GT = json.load(open("results/_wf_gt.json"))


def to_decode(tool_calls):
    out = []
    for tc in (tool_calls or []):
        name = tc.get("name")
        if not name:
            continue
        name = name.replace(".", "_")
        args = tc.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        out.append({name: args})
    return out


def grade(iid, tool_calls):
    g = GT.get(iid)
    if not g:
        return False
    dec = to_decode(tool_calls)
    if not dec:
        return False
    try:
        return bool(ast_checker(g["functions"], dec, g["ground_truth"], Language.PYTHON, g["cat"], "x")["valid"])
    except Exception:
        return False


def main():
    obj = json.loads(open(sys.argv[1]).read())
    items = obj["result"] if isinstance(obj, dict) and isinstance(obj.get("result"), list) else obj
    items = [it for it in items if it and it.get("id")]
    n = len(items)
    print(f"decidable items graded: {n} (oracle on this subset = 100% by construction)\n")
    correct = defaultdict(int)
    for it in items:
        for s in it.get("synths", []):
            if not s:
                continue
            correct[s.get("name", "?")] += grade(it["id"], s.get("tool_calls"))
    print("=== SYNTH variants on decidable items ===")
    for k, v in sorted(correct.items(), key=lambda kv: -kv[1]):
        print(f"  {k:18} {100*v/n:5.1f}   ({v}/{n})")


if __name__ == "__main__":
    main()
