"""Phase-1 -> decidable: grade the Haiku panel, keep only disagreement items.

Reads the phase-1 workflow output ([{id, panel}]), grades each stance with the
BFCL checker, classifies each item, and writes results/_wf_decidable.json =
[{id, request, functions, panel}] for the DECIDABLE items (some stance right, some
wrong) — the high-signal subset for the phase-2 synth sweep.

Run: PYTHONPATH=. .venv/bin/python scripts/phase1_decidable.py <phase1-output-path>
"""
from __future__ import annotations

import json
import sys

from trbench.evals.bfcl.vendor.ast_checker import Language, ast_checker

GT = json.load(open("results/_wf_gt.json"))
ITEMS = {it["id"]: it for it in json.load(open("results/_wf_items.json"))}


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
    raw = open(sys.argv[1]).read()
    obj = json.loads(raw)
    items = obj["result"] if isinstance(obj, dict) and isinstance(obj.get("result"), list) else obj
    items = [it for it in items if it and it.get("id")]
    print(f"phase-1 returned {len(items)} graded items")

    decidable, trivial, unwinnable = [], 0, 0
    for it in items:
        flags = [grade(it["id"], c.get("tool_calls")) for c in it.get("panel", [])]
        if not flags:
            continue
        if all(flags):
            trivial += 1
        elif not any(flags):
            unwinnable += 1
        else:
            src = ITEMS.get(it["id"], {})
            decidable.append({"id": it["id"], "request": src.get("request", ""),
                              "functions": src.get("functions", []), "panel": it["panel"]})
    open("results/_wf_decidable.json", "w").write(json.dumps(decidable))
    print(f"trivial(all-right)={trivial}  unwinnable(none-right)={unwinnable}  DECIDABLE={len(decidable)}")
    print(f"wrote results/_wf_decidable.json ({len(decidable)} items)")
    # phase-2 call budget check
    print(f"phase-2 calls = {len(decidable)} x (1 judge + 5 synth) = {len(decidable)*6}  (cap 1000)")


if __name__ == "__main__":
    main()
