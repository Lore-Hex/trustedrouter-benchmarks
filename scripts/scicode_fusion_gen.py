"""SciCode: Sonnet SOLO vs Sonnet FUSION (5 panel -> judge -> evidence_decide synth) on a
small problem subset, to test whether per-step fusion beats solo on research code-gen.

For each problem it runs BOTH chains (each step's prompt uses ITS OWN prior generated code):
  solo:   1 Sonnet call per step
  fusion: 5 Sonnet stance-proposers -> Sonnet judge -> Sonnet synth (evidence_decide) per step
Returns {problem_id, solo:[code...], fusion:[code...]}; score each with scicode_score.py.

Run:  python scripts/scicode_fusion_gen.py "53,59,65" _SF sonnet
"""
import json
import os
import sys
from pathlib import Path

SCI = Path(os.environ.get("SCICODE_HOME") or (Path(__file__).resolve().parent.parent / "scicode"))
sys.path.insert(0, str(SCI / "src"))
from scicode.parse.parse import (read_from_hf_dataset, extract_function_name,
                                 get_function_from_code)

PICK = (sys.argv[1] if len(sys.argv) > 1 else "53,59,65").split(",")
SUF = sys.argv[2] if len(sys.argv) > 2 else "_SF"
MODEL = sys.argv[3] if len(sys.argv) > 3 else "sonnet"
CONC = int(sys.argv[4]) if len(sys.argv) > 4 else 3

TEMPLATE = (SCI / "eval/data/background_comment_template.txt").read_text()  # WITHOUT-background
SPECIAL = {"13": 6, "62": 1, "76": 3}

byid = {p["problem_id"]: p for p in read_from_hf_dataset("test")}
problems = []
for pid in PICK:
    prob = byid[pid]
    steps = [{"desc": ss["step_description_prompt"], "header": ss["function_header"],
              "ret": ss["return_line"]} for ss in prob["sub_steps"]]
    skip = {}
    if pid in SPECIAL:
        n = SPECIAL[pid]
        gold = (SCI / f"eval/data/{pid}.{n}.txt").read_text()
        skip[str(n)] = get_function_from_code(gold, extract_function_name(prob["sub_steps"][n - 1]["function_header"]))
    problems.append({"problem_id": pid, "deps": prob["required_dependencies"], "steps": steps, "skip": skip})

js = f"""export const meta = {{
  name: 'scicode-fusion-{MODEL}',
  description: 'SciCode {MODEL} solo vs fusion (5 panel->judge->synth) on {len(problems)} problems',
  phases: [{{title:'SoloVsFusion'}}],
}}
const PROBLEMS = {json.dumps(problems)}
const TEMPLATE = {json.dumps(TEMPLATE)}
const PANEL = [
  'STANCE: Derive the required math/algorithm carefully from first principles before coding; make sure the formula and constants are exactly right.',
  'STANCE: Write the simplest implementation that exactly satisfies the spec; avoid unnecessary complexity.',
  'STANCE: Be careful with array shapes/broadcasting, units, and numerical stability; handle edge cases.',
  'STANCE: Follow the function header, dependencies, and described inputs/outputs LITERALLY; match the expected return types and shapes exactly.',
  'STANCE: Consider an alternative correct formulation in case the obvious approach is subtly wrong; pick the one that truly matches the description.',
]
function extractPython(resp){{
  resp = resp || ''
  let s
  if(resp.includes('```')){{ s = resp.includes('```python') ? resp.split('```python')[1].split('```')[0] : resp.split('```')[1].split('```')[0] }}
  else {{ s = resp }}
  return s.split('\\n').map(line => /^\\s*(import .*|from .*\\s+import\\s+.*)/.test(line) ? '' : line).join('\\n')
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
async function fusionStep(prompt){{
  const cands = await parallel(PANEL.map((st,i)=>()=>agent(`${{st}}\\n\\n${{prompt}}`,{{phase:'SoloVsFusion',label:'panel'+i,model:'{MODEL}'}})))
  const candText = cands.map((c,i)=>`### Candidate ${{i+1}}\\n${{(c||'').slice(0,4000)}}`).join('\\n\\n')
  const judge = await agent(`You are reviewing candidate Python implementations for ONE step of a scientific-coding problem.\\n\\n${{prompt}}\\n\\nCandidate implementations:\\n${{candText}}\\n\\nCompare them: where do they agree, where do they differ (different formulas/approaches/shapes), which look buggy or mismatched to the spec, and which approach is most likely correct? Be concise.`,{{phase:'SoloVsFusion',label:'judge',model:'{MODEL}'}})
  const synth = await agent(`You are the fusion synthesizer. The candidate implementations and the reviewer analysis are EVIDENCE — use them, but INDEPENDENTLY write the correct implementation for THIS step from the problem description, function header, and dependencies. The correct code may match only one candidate, or none.\\n\\n${{prompt}}\\n\\nPanel candidates (evidence):\\n${{candText}}\\n\\nReviewer analysis:\\n${{judge}}\\n\\nNow output ONLY the complete, correct function for this step in a single \\`\\`\\`python\\`\\`\\` block — no prose, no example usage, no previous-step code.`,{{phase:'SoloVsFusion',label:'synth',model:'{MODEL}'}})
  return synth
}}
async function chain(p, fuse){{
  const code = new Array(p.steps.length).fill(null)
  for(let n=1;n<=p.steps.length;n++){{
    if(p.skip && p.skip[String(n)]){{ code[n-1]=p.skip[String(n)]; continue }}
    const prompt = buildPrompt(p, n, code)
    const resp = fuse ? await fusionStep(prompt) : await agent(prompt,{{phase:'SoloVsFusion',label:'solo',model:'{MODEL}'}})
    code[n-1] = extractPython(resp || '')
  }}
  return code
}}
async function solveProblem(p){{
  const [solo, fusion] = await parallel([()=>chain(p,false), ()=>chain(p,true)])
  return {{problem_id:p.problem_id, solo, fusion}}
}}
const jobs = PROBLEMS.map(p => () => solveProblem(p))
const out = []
for(let i=0;i<jobs.length;i+={CONC}){{ out.push(...(await parallel(jobs.slice(i,i+{CONC})))) }}
return out
"""
outp = Path(f"results/_scicode_fusion{SUF}.js")
outp.write_text(js)
ns = sum(len(p["steps"]) for p in problems)
print(f"wrote {outp} | problems={[p['problem_id'] for p in problems]} steps={ns} | "
      f"solo-calls={ns}, fusion-calls={ns*7} | model={MODEL} conc={CONC} | bytes={len(js)}")
