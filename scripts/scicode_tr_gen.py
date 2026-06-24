"""Clean-API SciCode generation via TrustedRouter (temperature 0) — to REPLICATE published
numbers that the CC-subagent backend can't match (subagents aren't temp-0 / carry an agentic
system prompt). Faithful to gencode.py; produces the same [{problem_id, code}] JSON that
scicode_score.py consumes.

Run:  python scripts/scicode_tr_gen.py test anthropic/claude-haiku-4.5 bg _TRBG
      python scripts/scicode_tr_gen.py test anthropic/claude-haiku-4.5 nobg _TR "21,22,23"
"""
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCI = Path(os.environ.get("SCICODE_HOME") or (Path(__file__).resolve().parent.parent / "scicode"))
sys.path.insert(0, str(SCI / "src"))
from scicode.parse.parse import (read_from_hf_dataset, extract_function_name,
                                 get_function_from_code)

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "anthropic/claude-haiku-4.5"
BG = len(sys.argv) > 3 and sys.argv[3] in ("bg", "1", "with_background")
SUF = sys.argv[4] if len(sys.argv) > 4 else "_TR"
PICK = sys.argv[5].split(",") if len(sys.argv) > 5 else None
CONC = int(sys.argv[6]) if len(sys.argv) > 6 else 12

TR_KEY = Path(os.path.expanduser("~/claude/.tr_key")).read_text().strip()
TR_URL = "https://api.trustedrouter.com/v1/chat/completions"
TEMPLATE = (SCI / ("eval/data/multistep_template.txt" if BG else
                   "eval/data/background_comment_template.txt")).read_text()
SPECIAL = {"13": 6, "62": 1, "76": 3}


def tr_call(prompt, model=MODEL, temperature=0, max_tokens=4096, _tries=5):
    body = json.dumps({"model": model, "temperature": temperature, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(TR_URL, data=body, headers={
        "Authorization": f"Bearer {TR_KEY}", "Content-Type": "application/json"})
    for t in range(_tries):
        try:
            with urllib.request.urlopen(req, timeout=240) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"] or ""
        except Exception as e:
            if t == _tries - 1:
                print(f"  tr_call failed: {str(e)[:120]}", file=sys.stderr)
                return ""
            time.sleep(2 * (t + 1))


def extract_python_script(response):  # verbatim from scicode.gen.models
    if '```' in response:
        s = response.split("```python")[1].split("```")[0] if '```python' in response else response.split('```')[1].split('```')[0]
    else:
        s = response
    return re.sub(r'^\s*(import .*|from .*\s+import\s+.*)', '', s, flags=re.MULTILINE)


def build_prompt(p, num, code):
    out = []
    for i in range(num - 1):
        out += [p["steps"][i]["desc"], code[i] or "", "------"]
    problem_steps_str = "\n\n".join(out[:-1])
    ns = p["steps"][num - 1]
    next_step_str = "\n\n".join([ns["desc"], ns["header"] + "\n\n" + ns["ret"]])
    return TEMPLATE.format(problem_steps_str=problem_steps_str, next_step_str=next_step_str,
                           dependencies=p["deps"])


data = read_from_hf_dataset(SPLIT)
problems = []
for prob in data:
    pid = prob["problem_id"]
    if PICK and pid not in PICK:
        continue
    steps = [{"desc": ss["step_description_prompt"] + ("\n" + ss["step_background"] if BG else ""),
              "header": ss["function_header"], "ret": ss["return_line"]} for ss in prob["sub_steps"]]
    skip = {}
    if pid in SPECIAL:
        n = SPECIAL[pid]
        gold = (SCI / f"eval/data/{pid}.{n}.txt").read_text()
        skip[str(n)] = get_function_from_code(gold, extract_function_name(prob["sub_steps"][n - 1]["function_header"]))
    problems.append({"problem_id": pid, "deps": prob["required_dependencies"], "steps": steps, "skip": skip})


def solve(p):
    code = [None] * len(p["steps"])
    for n in range(1, len(p["steps"]) + 1):
        if p["skip"].get(str(n)):
            code[n - 1] = p["skip"][str(n)]
            continue
        code[n - 1] = extract_python_script(tr_call(build_prompt(p, n, code)))
    return {"problem_id": p["problem_id"], "code": code}


t0 = time.time()
with ThreadPoolExecutor(max_workers=CONC) as ex:
    out = list(ex.map(solve, problems))
Path(f"results/_scicode{SUF}_out.json").write_text(json.dumps(out))
ncalls = sum(len(p["steps"]) - len(p["skip"]) for p in problems)
print(f"wrote results/_scicode{SUF}_out.json | {len(problems)} problems, {ncalls} TR calls "
      f"({MODEL}, temp0, {'with' if BG else 'without'}-bg) in {int(time.time()-t0)}s")
