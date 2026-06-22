"""Generate the phase-2 workflow: judge + 5 synth variants on the DECIDABLE items.

Reads results/_wf_decidable.json (items already carry their phase-1 panel) and
writes results/_fusion_phase2.js. All-Haiku. Run after phase1_decidable.py.

Run: PYTHONPATH=. .venv/bin/python scripts/gen_phase2.py
"""
import json

items = json.load(open("results/_wf_decidable.json"))

TC = """const TC = { type:'object', properties:{ tool_calls:{ type:'array', items:{ type:'object', properties:{ name:{type:'string'}, arguments:{type:'string', description:'JSON object string of the arguments'} }, required:['name','arguments'] } } }, required:['tool_calls'] }"""

SYNTH = """const SYNTH = {
  baseline: `TrustedRouter Fusion panel answers and judge analysis follow. Continue solving the original task using the panel answers as primary evidence and the judge analysis as guidance. If the next correct action is a tool call, emit the tool call directly.`,
  verify_each: `You are the fusion synthesizer. For EACH panel candidate, verify its function choice and every argument value against the request and schema (units, location format, required vs optional). Emit the tool call(s) that pass strictest. Prefer a panel candidate's exact call when it is correct rather than writing your own.`,
  evidence_decide: `You are the fusion synthesizer. Use the panel answers as evidence of the options, but independently determine the correct function and argument values from the request and schema. Emit that tool call. The correct answer may be one only a single panel member proposed.`,
  fresh_then_adopt: `You are the fusion synthesizer. First work out the correct tool call(s) from the request and schema yourself. Then compare to the panel candidates and adopt the candidate that matches your answer (it may catch a detail you missed). Emit the final tool call(s).`,
  verify_dean: `You are the fusion synthesizer. Independently derive the correct tool call(s) from the request and schema. Then verify each panel candidate's arguments against the request and adopt the most correct candidate, preferring a panel call over your own when it is right. Emit the final tool call(s).`,
}"""

js = f"""export const meta = {{
  name: 'fusion-phase2-synth',
  description: 'Phase 2: judge + 5 synth variants on decidable items (Haiku)',
  phases: [{{ title: 'Judge' }}, {{ title: 'Synth' }}],
}}
const ITEMS = {json.dumps(items)}
log(`judge+synth on ${{ITEMS.length}} decidable items`)
{TC}
{SYNTH}
const JUDGE = (it) =>
  `You are the TrustedRouter Fusion judge for a tool-calling task. Panel members each proposed tool calls.\\n\\n` +
  `Available functions:\\n${{JSON.stringify(it.functions)}}\\n\\nRequest:\\n${{it.request}}\\n\\n` +
  `Panel proposed tool calls:\\n${{JSON.stringify(it.panel)}}\\n\\n` +
  `Write a concise analysis: (1) where the candidates AGREE (same function, same argument values); (2) EXACTLY where they DIFFER - name the specific function or argument and state what the request/schema requires about it (units, location format, required vs optional). Do NOT pick a winner and do NOT write the final tool call.`
const SYNTHP = (it, judge, instr) =>
  `${{instr}}\\n\\nAvailable functions:\\n${{JSON.stringify(it.functions)}}\\n\\nRequest:\\n${{it.request}}\\n\\n` +
  `Panel candidates (proposed tool calls):\\n${{JSON.stringify(it.panel)}}\\n\\nJudge analysis:\\n${{judge}}\\n\\nOutput the final tool call(s). arguments must be a JSON object string.`
const out = await pipeline(ITEMS,
  it => agent(JUDGE(it), {{ phase: 'Judge', label: `judge:${{it.id}}`, model: 'haiku' }}).then(judge => ({{ it, judge }})),
  ({{ it, judge }}) => parallel(Object.entries(SYNTH).map(([name, instr]) => () =>
    agent(SYNTHP(it, judge, instr), {{ phase: 'Synth', label: `synth:${{name}}`, schema: TC, model: 'haiku' }})
      .then(o => ({{ name, tool_calls: (o && o.tool_calls) || [] }}))
  )).then(synths => ({{ id: it.id, synths }}))
)
return out
"""
open("results/_fusion_phase2.js", "w").write(js)
print("wrote results/_fusion_phase2.js for", len(items), "decidable items;", len(items)*6, "calls")
