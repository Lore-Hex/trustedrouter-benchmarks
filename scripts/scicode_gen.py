"""Generate a Workflow that runs the SciCode generation chain with Haiku subagents.

Faithful to SciCode's gencode.py (WITHOUT-background = the realistic leaderboard setting):
per step the prompt embeds the model's OWN prior-step code (a chain); the ```python``` block
is extracted and imports stripped; steps 13.6 / 62.1 / 76.3 are skipped using the gold .txt.
The workflow returns, per problem, the extracted script per step. Reconstruct + score with
the OFFICIAL test runner via scicode_score.py.

Run:  python scripts/scicode_gen.py validation _VAL haiku 12
      python scripts/scicode_gen.py test _TEST haiku 12
"""
import json
import os
import sys
from pathlib import Path

# Vendored SciCode lives at <repo>/scicode (override with SCICODE_HOME).
SCI = Path(os.environ.get("SCICODE_HOME") or (Path(__file__).resolve().parent.parent / "scicode"))
sys.path.insert(0, str(SCI / "src"))
from scicode.parse.parse import (read_from_hf_dataset, extract_function_name,
                                  get_function_from_code)

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "validation"
SUF = sys.argv[2] if len(sys.argv) > 2 else "_VAL"
MODEL = sys.argv[3] if len(sys.argv) > 3 else "haiku"
CONC = int(sys.argv[4]) if len(sys.argv) > 4 else 12
BG = len(sys.argv) > 5 and sys.argv[5] in ("bg", "1", "with_background")  # with-background setting

# with-background: the gold scientific background is GIVEN (multistep template + step_background
# appended to each description). without-background (default): model must recall the science.
TEMPLATE = (SCI / ("eval/data/multistep_template.txt" if BG else
                   "eval/data/background_comment_template.txt")).read_text()
SPECIAL = {"13": 6, "62": 1, "76": 3}  # prob_id -> 1-indexed step that is skipped (use gold)

data = read_from_hf_dataset(SPLIT)
problems = []
for prob in data:
    pid = prob["problem_id"]
    steps = []
    for ss in prob["sub_steps"]:
        desc = ss["step_description_prompt"] + ("\n" + ss["step_background"] if BG else "")
        steps.append({"desc": desc,
                      "header": ss["function_header"], "ret": ss["return_line"]})
    skip = {}
    if pid in SPECIAL:
        n = SPECIAL[pid]
        gold_txt = (SCI / f"eval/data/{pid}.{n}.txt").read_text()
        fname = extract_function_name(prob["sub_steps"][n - 1]["function_header"])
        skip[str(n)] = get_function_from_code(gold_txt, fname)
    problems.append({"problem_id": pid, "deps": prob["required_dependencies"],
                     "steps": steps, "skip": skip})

js = f"""export const meta = {{
  name: 'scicode-{SPLIT}-{MODEL}',
  description: 'SciCode {SPLIT} generation chain (without-background) on {MODEL} subagents',
  phases: [{{title:'Generate'}}],
}}
const PROBLEMS = {json.dumps(problems)}
const TEMPLATE = {json.dumps(TEMPLATE)}
function extractPython(resp){{
  resp = resp || ''
  let s
  if(resp.includes('```')){{
    s = resp.includes('```python') ? resp.split('```python')[1].split('```')[0] : resp.split('```')[1].split('```')[0]
  }} else {{ s = resp }}
  s = s.split('\\n').map(line => /^\\s*(import .*|from .*\\s+import\\s+.*)/.test(line) ? '' : line).join('\\n')
  return s
}}
function buildPrompt(p, num, code){{
  const out=[]
  for(let i=0;i<num-1;i++){{ out.push(p.steps[i].desc); out.push(code[i]||''); out.push('------') }}
  const problem_steps_str = out.slice(0,-1).join('\\n\\n')
  const ns = p.steps[num-1]
  const next_step_str = [ns.desc, ns.header + '\\n\\n' + ns.ret].join('\\n\\n')
  return TEMPLATE.split('{{problem_steps_str}}').join(problem_steps_str)
                 .split('{{next_step_str}}').join(next_step_str)
                 .split('{{dependencies}}').join(p.deps)
}}
async function solveProblem(p){{
  const code = new Array(p.steps.length).fill(null)
  for(let n=1;n<=p.steps.length;n++){{
    if(p.skip && p.skip[String(n)]){{ code[n-1] = p.skip[String(n)]; continue }}
    const prompt = buildPrompt(p, n, code)
    const resp = await agent(prompt, {{phase:'Generate', label:`${{p.problem_id}}.${{n}}`, model:'{MODEL}'}})
    code[n-1] = extractPython(resp || '')
  }}
  return {{problem_id:p.problem_id, code}}
}}
const jobs = PROBLEMS.map(p => () => solveProblem(p))
const out = []
for(let i=0;i<jobs.length;i+={CONC}){{ out.push(...(await parallel(jobs.slice(i,i+{CONC})))) }}
return out
"""
outp = Path(f"results/_scicode{SUF}.js")
outp.write_text(js)
nsteps = sum(len(p["steps"]) for p in problems)
nskip = sum(len(p["skip"]) for p in problems)
print(f"wrote {outp} | split={SPLIT} problems={len(problems)} steps={nsteps} "
      f"(skip {nskip} gold) gen-calls={nsteps-nskip} | model={MODEL} conc={CONC} | bytes={len(js)}")
