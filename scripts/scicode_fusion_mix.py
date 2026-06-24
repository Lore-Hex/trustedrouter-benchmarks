"""SciCode fusion with a Claude-TIER-DIVERSE panel (Sonnet + 2 Haiku + 1 Opus) -> Sonnet
synth, vs Sonnet solo (and Opus solo as the best-member reference). Subagents only (no TR $).

Tests whether tier diversity (different Claude tiers make different errors) lets a Sonnet-synth
fusion beat Sonnet solo on code-gen — where 5 same-Sonnet stances could not (they share Sonnet's
errors). Per problem runs N solo chains + 1 fusion chain; score each with scicode_score.py.

Run:  python scripts/scicode_fusion_mix.py "21,22,23,24,25,26,27,28" _MIX 2
"""
import json
import os
import sys
from pathlib import Path

SCI = Path(os.environ.get("SCICODE_HOME") or (Path(__file__).resolve().parent.parent / "scicode"))
sys.path.insert(0, str(SCI / "src"))
from scicode.parse.parse import (read_from_hf_dataset, extract_function_name,
                                 get_function_from_code)

PICK = (sys.argv[1] if len(sys.argv) > 1 else "21,22,23,24,25,26,27,28").split(",")
SUF = sys.argv[2] if len(sys.argv) > 2 else "_MIX"
CONC = int(sys.argv[3]) if len(sys.argv) > 3 else 2

TEMPLATE = (SCI / "eval/data/background_comment_template.txt").read_text()  # without-background
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
  name: 'scicode-fusion-mix',
  description: 'SciCode tier-diverse fusion (Sonnet+2Haiku+1Opus -> Sonnet synth) vs Sonnet/Opus solo',
  phases: [{{title:'MixFusion'}}],
}}
const PROBLEMS = {json.dumps(problems)}
const TEMPLATE = {json.dumps(TEMPLATE)}
// tier-diverse panel: different Claude tiers make DIFFERENT errors (real complementarity)
const PANEL = [
  {{model:'sonnet', stance:'Derive the math/algorithm carefully from first principles; get formulas and constants exactly right.'}},
  {{model:'haiku',  stance:'Write the simplest implementation that exactly satisfies the spec.'}},
  {{model:'haiku',  stance:'Be careful with array shapes/broadcasting, units, and numerical stability; handle edge cases.'}},
  {{model:'opus',   stance:'Consider the non-obvious/alternative correct formulation in case the obvious approach is subtly wrong.'}},
]
const SOLO_MODELS = ['sonnet','opus']   // baseline (under test) + best-member reference
const JUDGE='sonnet', SYNTH='sonnet'
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
  const cands = await parallel(PANEL.map((m,i)=>()=>agent(`${{m.stance}}\\n\\n${{prompt}}`,{{phase:'MixFusion',label:'panel:'+m.model+i,model:m.model}})))
  const candText = cands.map((c,i)=>`### Candidate ${{i+1}} (${{PANEL[i].model}})\\n${{(c||'').slice(0,4000)}}`).join('\\n\\n')
  const judge = await agent(`You are reviewing candidate Python implementations for ONE step of a scientific-coding problem.\\n\\n${{prompt}}\\n\\nCandidates:\\n${{candText}}\\n\\nCompare them: where do they agree/differ (formulas/approaches/shapes), which look buggy or mismatched to the spec, and which is most likely correct? Be concise.`,{{phase:'MixFusion',label:'judge',model:JUDGE}})
  const synth = await agent(`You are the fusion synthesizer. The candidate implementations and reviewer analysis are EVIDENCE — use them, but INDEPENDENTLY write the correct implementation for THIS step from the description, function header, and dependencies. The correct code may match only one candidate, or none. Trace the key formula/shape to check it.\\n\\n${{prompt}}\\n\\nCandidates (evidence):\\n${{candText}}\\n\\nReviewer:\\n${{judge}}\\n\\nOutput ONLY the complete correct function in one \\`\\`\\`python\\`\\`\\` block — no prose, no previous-step code.`,{{phase:'MixFusion',label:'synth',model:SYNTH}})
  return synth
}}
async function chain(p, mode){{  // mode = 'solo:<model>' or 'fusion'
  const code = new Array(p.steps.length).fill(null)
  for(let n=1;n<=p.steps.length;n++){{
    if(p.skip && p.skip[String(n)]){{ code[n-1]=p.skip[String(n)]; continue }}
    const prompt = buildPrompt(p, n, code)
    let resp
    if(mode==='fusion') resp = await fusionStep(prompt)
    else resp = await agent(prompt,{{phase:'MixFusion',label:mode,model:mode.split(':')[1]}})
    code[n-1] = extractPython(resp || '')
  }}
  return code
}}
async function solveProblem(p){{
  const modes = SOLO_MODELS.map(m=>'solo:'+m).concat(['fusion'])
  const res = await parallel(modes.map(mo=>()=>chain(p,mo)))
  const out = {{problem_id:p.problem_id}}
  SOLO_MODELS.forEach((m,i)=> out['solo_'+m]=res[i])
  out.fusion = res[res.length-1]
  return out
}}
const jobs = PROBLEMS.map(p => () => solveProblem(p))
const out = []
for(let i=0;i<jobs.length;i+={CONC}){{ out.push(...(await parallel(jobs.slice(i,i+{CONC})))) }}
return out
"""
outp = Path(f"results/_scicode_fusionmix{SUF}.js")
outp.write_text(js)
ns = sum(len(p["steps"]) for p in problems)
print(f"wrote {outp} | problems={[p['problem_id'] for p in problems]} steps={ns} | "
      f"per step: 2 solo + (4 panel+judge+synth=6) fusion = 8 calls | model mix sonnet/haiku/opus | bytes={len(js)}")
