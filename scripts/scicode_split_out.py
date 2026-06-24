"""Split a scicode_fusion_mix Workflow output into a per-method {problem_id, code} list that
scicode_score.py can consume directly.

Run:  python scripts/scicode_split_out.py results/_full_b3_out.json fusion      results/_b3_fus.json
      python scripts/scicode_split_out.py results/_full_b3_out.json solo_sonnet results/_b3_son.json
"""
import json
import sys

OUT, KEY, DST = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(OUT))
res = d["result"] if isinstance(d, dict) else d
if isinstance(res, str):
    res = json.loads(res)
out = [{"problem_id": e["problem_id"], "code": e[KEY]} for e in res if e and KEY in e]
json.dump(out, open(DST, "w"))
print(f"wrote {DST}: {len(out)} problems (key={KEY}) pids={[e['problem_id'] for e in out]}")
