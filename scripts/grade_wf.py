"""Grade the fusion-synth-mechanics workflow output with the BFCL checker.

Parses 'ITEM_JSON {...}' lines (logged by the workflow), converts each panel
candidate and each synth-variant tool call to BFCL decode format, and grades vs
the stashed ground truth. Reports per-synth-variant accuracy + panel oracle.

Run: PYTHONPATH=. .venv/bin/python scripts/grade_wf.py <path-with-ITEM_JSON-lines>
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict

from trbench.evals.bfcl.vendor.ast_checker import Language, ast_checker

GT = json.load(open("results/_wf_gt.json"))


def to_decode(tool_calls):
    """[{name, arguments(json-str)}] -> [{name: args_dict}] (BFCL decode format)."""
    out = []
    for tc in (tool_calls or []):
        name = tc.get("name")
        if not name:
            continue
        name = name.replace(".", "_")  # match BFCL checker's dot->underscore normalization
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
    path = sys.argv[1]
    raw = open(path).read()
    items = []
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and isinstance(obj.get("result"), list):
            items = obj["result"]
        elif isinstance(obj, list):
            items = obj
        elif isinstance(obj, dict) and isinstance(obj.get("logs"), list):
            for line in obj["logs"]:
                m = re.match(r"ITEM_JSON (\{.*\})\s*$", line)
                if m:
                    items.append(json.loads(m.group(1)))
    except Exception:
        for m in re.finditer(r"ITEM_JSON (\{.*?\})\s*$", raw, re.MULTILINE):
            try:
                items.append(json.loads(m.group(1)))
            except Exception:
                pass
    # drop failed (null) pipeline items, de-dup by id (keep last)
    items = [it for it in items if it and it.get("id")]
    by_id = {it["id"]: it for it in items}
    items = list(by_id.values())
    n = len(items)
    print(f"parsed {n} item results\n")
    if not n:
        print("NO ITEM_JSON found in", path); return

    stance_correct = defaultdict(int)
    synth_correct = defaultdict(int)
    synth_dec = defaultdict(int)
    oracle = 0
    decidable = []  # items where panel disagrees on correctness (some right, some wrong)
    for it in items:
        panel_flags = []
        for cand in it.get("panel", []):
            ok = grade(it["id"], cand.get("tool_calls"))
            stance_correct[cand.get("stance", "?")] += ok
            panel_flags.append(ok)
        oracle += any(panel_flags)
        is_dec = any(panel_flags) and not all(panel_flags)
        if is_dec:
            decidable.append(it["id"])
        for s in it.get("synths", []):
            ok = grade(it["id"], s.get("tool_calls"))
            synth_correct[s.get("name", "?")] += ok
            if is_dec:
                synth_dec[s.get("name", "?")] += ok

    print("=== PANEL (stance solos) ===")
    for k in sorted(stance_correct):
        print(f"  {k:14} {100*stance_correct[k]/n:5.1f}")
    print(f"  ORACLE (any stance) {100*oracle/n:5.1f}")
    nd = len(decidable)
    print(f"\n=== SYNTH variants — overall (n={n}) | DECIDABLE-only (n={nd}) ===")
    for k in sorted(synth_correct, key=lambda k: -synth_correct[k]):
        dec_s = f"{100*synth_dec[k]/nd:5.1f}" if nd else "  n/a"
        print(f"  {k:18} overall {100*synth_correct[k]/n:5.1f}   decidable {dec_s}")


if __name__ == "__main__":
    main()
