"""Generic QA fusion-test harness: apply the decidable-focused panel->judge->synth
method to any text-answer benchmark (GSM8K, MMLU-Pro, ...). TR credits out, so the
panel is Haiku 4.5 with diverse stances; judge+synth Haiku too (constant strength
isolates the synth WORDING). Subagents answer in free text ending '#### <answer>'.

Phases (same as the BFCL study):
  build1 <bench> <n>  -> _qa_items.json, _qa_gt.json, _qa_phase1.js  (panel only)
  decidable <p1.out>  -> grade panel, write _qa_decidable.json + _qa_stats.json
  build2              -> _qa_phase2.js  (judge + 5 synths on decidable items)
  grade2 <p2.out>     -> synth scores on decidable + full-set fusion vs best-solo

Run: PYTHONPATH=. .venv/bin/python scripts/fusion_qa.py <cmd> [args]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

# ---------------- benchmark registry ----------------
def _gsm8k(n):
    from trbench.evals.gsm8k.loader import load
    out = []
    for it in load(limit=n):
        out.append({"id": it["id"], "prompt": it["question"], "target": it["target"]})
    return out

def _gsm8k_ok(text, target):
    from trbench.evals.gsm8k.score import extract_pred, normalize_num
    return extract_pred(text or "") == normalize_num(str(target))

def _mmlu(n):
    from trbench.evals.mmlu_pro.loader import load
    out = []
    for it in load(limit=n):
        letters = [chr(ord('A') + i) for i in range(len(it["options"]))]
        opts = "\n".join(f"{l}. {o}" for l, o in zip(letters, it["options"]))
        out.append({"id": it["id"], "prompt": f"{it['question']}\n\nOptions:\n{opts}", "target": it["answer"]})
    return out

def _mmlu_ok(text, target):
    import re
    if not text:
        return False
    tail = text.split("####")[-1]
    m = re.search(r"[A-J]", tail.upper())
    return bool(m) and m.group() == str(target).strip().upper()

BENCH = {
    "gsm8k": {"load": _gsm8k, "ok": _gsm8k_ok, "kind": "a number"},
    "mmlu_pro": {"load": _mmlu, "ok": _mmlu_ok, "kind": "the letter of the correct option"},
}

# ---------------- shared prompts (JS) ----------------
STANCES_JS = """const STANCES = [
  { k: 'careful',  p: 'You are a careful solver. Work the problem step by step and check each step.' },
  { k: 'quick',    p: 'You are a fast solver. Answer with your first instinct and minimal working.' },
  { k: 'alt',      p: 'You are a creative solver. Solve using a different approach than the most obvious one.' },
  { k: 'recheck',  p: 'You are a meticulous solver. Solve, then redo the key steps to confirm.' },
  { k: 'skeptic',  p: 'You are a skeptical solver. Solve, then actively look for a mistake before finalizing.' },
]"""
SYNTH_JS = """const SYNTH = {
  baseline: `Panel answers and a judge analysis follow. Use the panel answers as primary evidence and the judge analysis as guidance to give the final answer.`,
  evidence_decide: `Use the panel answers as evidence of the options, but INDEPENDENTLY re-derive the correct answer from the problem itself. The correct answer may be one only a single panel member gave.`,
  verify_each: `For EACH panel candidate, verify its reasoning and final answer step by step. Give the answer that checks out under scrutiny, not just the popular one.`,
  majority: `Consider the panel's final answers and give the answer that the most candidates agree on (self-consistency); break ties by which reasoning is soundest.`,
  fresh_then_adopt: `First solve the problem yourself. Then compare to the panel candidates and adopt the candidate matching your answer (it may catch a detail you missed).`,
}"""

def phase1_js(items, kind):
    return f"""export const meta = {{
  name: 'fusion-qa-phase1',
  description: 'QA fusion phase 1: Haiku panel only',
  phases: [{{ title: 'Panel' }}],
}}
const ITEMS = {json.dumps(items)}
const KIND = {json.dumps(kind)}
log(`panel on ${{ITEMS.length}} items`)
{STANCES_JS}
const PROMPT = (it, stance) => `${{stance}}\\n\\nProblem:\\n${{it.prompt}}\\n\\nSolve it, then end with a line exactly: #### <${{KIND}}>`
const out = await pipeline(ITEMS, it =>
  parallel(STANCES.map(s => () =>
    agent(PROMPT(it, s.p), {{ phase: 'Panel', label: `p:${{s.k}}`, model: 'haiku' }})
      .then(t => ({{ stance: s.k, text: t }}))
  )).then(panel => ({{ id: it.id, panel }}))
)
return out
"""

def phase2_js(items, kind):
    return f"""export const meta = {{
  name: 'fusion-qa-phase2',
  description: 'QA fusion phase 2: judge + 5 synths on decidable items',
  phases: [{{ title: 'Judge' }}, {{ title: 'Synth' }}],
}}
const ITEMS = {json.dumps(items)}
const KIND = {json.dumps(kind)}
log(`judge+synth on ${{ITEMS.length}} decidable items`)
{SYNTH_JS}
const PANEL = (it) => it.panel.map((c,i)=>`[${{i+1}}] (${{c.stance}}) ${{c.text}}`).join('\\n\\n')
const JUDGE = (it) => `You are a fusion judge. Several panel members solved a problem.\\n\\nProblem:\\n${{it.prompt}}\\n\\nPanel solutions:\\n${{PANEL(it)}}\\n\\nWrite a concise analysis: where the candidates AGREE on the final answer/steps, and EXACTLY where they DIFFER (which step or final value, and which looks right). Do NOT give the final answer.`
const SYNTHP = (it, judge, instr) => `${{instr}}\\n\\nProblem:\\n${{it.prompt}}\\n\\nPanel candidates:\\n${{PANEL(it)}}\\n\\nJudge analysis:\\n${{judge}}\\n\\nGive the final answer, ending with a line exactly: #### <${{KIND}}>`
const out = await pipeline(ITEMS,
  it => agent(JUDGE(it), {{ phase: 'Judge', label: `judge:${{it.id}}`, model: 'haiku' }}).then(judge => ({{ it, judge }})),
  ({{ it, judge }}) => parallel(Object.entries(SYNTH).map(([name, instr]) => () =>
    agent(SYNTHP(it, judge, instr), {{ phase: 'Synth', label: `s:${{name}}`, model: 'haiku' }})
      .then(t => ({{ name, text: t }}))
  )).then(synths => ({{ id: it.id, synths }}))
)
return out
"""

# ---------------- commands ----------------
def _load_items_for_grade(path):
    obj = json.loads(open(path).read())
    items = obj["result"] if isinstance(obj, dict) and isinstance(obj.get("result"), list) else obj
    return [it for it in items if it and it.get("id")]

def cmd_build1(bench, n):
    items = BENCH[bench]["load"](int(n))
    qa = [{"id": it["id"], "prompt": it["prompt"]} for it in items]
    gt = {it["id"]: it["target"] for it in items}
    open("results/_qa_items.json", "w").write(json.dumps(qa))
    open("results/_qa_gt.json", "w").write(json.dumps(gt))
    open("results/_qa_meta.json", "w").write(json.dumps({"bench": bench}))
    open("results/_qa_phase1.js", "w").write(phase1_js(qa, BENCH[bench]["kind"]))
    print(f"{bench}: {len(qa)} items -> results/_qa_phase1.js ({len(qa)*5} panel calls)")

def cmd_decidable(path):
    meta = json.load(open("results/_qa_meta.json")); ok = BENCH[meta["bench"]]["ok"]
    gt = json.load(open("results/_qa_gt.json"))
    items_src = {it["id"]: it for it in json.load(open("results/_qa_items.json"))}
    p1 = _load_items_for_grade(path)
    N = len(p1)
    stance = defaultdict(int); oracle = 0; trivial = 0; unwinnable = 0; decidable = []
    for it in p1:
        flags = [ok(c.get("text"), gt[it["id"]]) for c in it["panel"]]
        for c, f in zip(it["panel"], flags): stance[c["stance"]] += f
        if any(flags): oracle += 1
        if all(flags): trivial += 1
        elif not any(flags): unwinnable += 1
        else:
            decidable.append({"id": it["id"], "prompt": items_src[it["id"]]["prompt"], "panel": it["panel"]})
    open("results/_qa_decidable.json", "w").write(json.dumps(decidable))
    best_solo = max(stance.values()) if stance else 0
    json.dump({"N": N, "trivial": trivial, "best_solo": best_solo, "oracle": oracle,
               "stance": dict(stance)}, open("results/_qa_stats.json", "w"))
    print(f"N={N} trivial={trivial} unwinnable={unwinnable} DECIDABLE={len(decidable)}")
    for k in sorted(stance, key=lambda k: -stance[k]):
        print(f"  solo {k:9} {100*stance[k]/N:5.1f}")
    print(f"  BEST SOLO {100*best_solo/N:5.1f}  ORACLE {100*oracle/N:5.1f}")
    print(f"  phase-2 calls = {len(decidable)} x 6 = {len(decidable)*6}")

def cmd_build2():
    meta = json.load(open("results/_qa_meta.json"))
    items = json.load(open("results/_qa_decidable.json"))
    open("results/_qa_phase2.js", "w").write(phase2_js(items, BENCH[meta["bench"]]["kind"]))
    print(f"wrote results/_qa_phase2.js for {len(items)} decidable items ({len(items)*6} calls)")

def cmd_grade2(path):
    meta = json.load(open("results/_qa_meta.json")); ok = BENCH[meta["bench"]]["ok"]
    gt = json.load(open("results/_qa_gt.json"))
    stats = json.load(open("results/_qa_stats.json"))
    N, trivial, best_solo = stats["N"], stats["trivial"], stats["best_solo"]
    p2 = _load_items_for_grade(path)
    nd = len(p2)
    dec_correct = defaultdict(int)
    for it in p2:
        for s in it.get("synths", []):
            if s: dec_correct[s.get("name", "?")] += ok(s.get("text"), gt[it["id"]])
    print(f"=== {meta['bench']}: synth on {nd} decidable items (oracle 100% here) ===")
    for k in sorted(dec_correct, key=lambda k: -dec_correct[k]):
        print(f"  {k:16} decidable {100*dec_correct[k]/nd:5.1f}  ({dec_correct[k]}/{nd})")
    print(f"\n=== full-set (N={N}): fusion = trivial({trivial}) + decidable_correct, vs BEST SOLO {100*best_solo/N:.1f} ===")
    for k in sorted(dec_correct, key=lambda k: -dec_correct[k]):
        full = trivial + dec_correct[k]
        tag = ">> BEATS solo" if full > best_solo else ("= ties" if full == best_solo else "< below")
        print(f"  fusion[{k:16}] {100*full/N:5.1f}  {tag}")
    print(f"  ORACLE {100*stats['oracle']/N:.1f}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"build1": lambda: cmd_build1(sys.argv[2], sys.argv[3]),
     "decidable": lambda: cmd_decidable(sys.argv[2]),
     "build2": cmd_build2,
     "grade2": lambda: cmd_grade2(sys.argv[2])}[cmd]()
