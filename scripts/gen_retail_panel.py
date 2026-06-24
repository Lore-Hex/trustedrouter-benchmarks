"""Agentic fusion (panel -> synth-decides) on tau2 retail, subagent-driven.

The shippable design from the handoff: each step a diverse PANEL proposes candidate
next-moves, then the SYNTH decides the single next move ({reads | write | message}).
The synth -- an LLM -- classifies read/write and judges when to stop exploring; there is
NO code heuristic and NO write-verify stage. Reads fan out in parallel; a write is the
synth's single decision and is executed as-is. Grade with tau2's REAL evaluator
(scripts/tau2_grade.py). See docs/HANDOFF_fusion_agentic.md.

Run: python scripts/gen_retail_panel.py "0,1,2" "_PAN" sonnet 2
"""
import json
import sys

TAU = "/Users/jperla/claude/tau2-bench/data/tau2/domains/retail"
PICK = sys.argv[1].split(",") if len(sys.argv) > 1 else ["0", "1", "2"]
SUF = sys.argv[2] if len(sys.argv) > 2 else "_PAN"
MODEL = sys.argv[3] if len(sys.argv) > 3 else "sonnet"
CONC = int(sys.argv[4]) if len(sys.argv) > 4 else 2

db = json.load(open(f"{TAU}/db.json"))
tasks_all = {str(t["id"]): t for t in json.load(open(f"{TAU}/tasks.json"))}
policy = open(f"{TAU}/policy.md").read().strip()
guidelines = open(f"{TAU}/../../user_simulator/simulation_guidelines.md").read().strip()

AGENT_INSTRUCTION = ("You are a customer service agent that helps the user according to the <policy> below.\n"
    "Try to be helpful and always follow the policy.")
AGENT_SYS = f"<instructions>\n{AGENT_INSTRUCTION}\n</instructions>\n<policy>\n{policy}\n</policy>"

READS = ["find_user_id_by_name_zip", "find_user_id_by_email", "get_user_details", "get_order_details",
         "get_product_details", "get_item_details", "list_all_product_types", "calculate"]
WRITES = ["cancel_pending_order", "modify_pending_order_items", "modify_pending_order_address",
          "modify_pending_order_payment", "modify_user_address", "exchange_delivered_order_items",
          "return_delivered_order_items", "transfer_to_human_agents"]


def resolve_user(actions):
    for a in actions:
        if a["name"] == "find_user_id_by_name_zip":
            ar = a["arguments"]
            for uid, u in db["users"].items():
                if (u["name"]["first_name"].lower() == ar["first_name"].lower()
                        and u["name"]["last_name"].lower() == ar["last_name"].lower()
                        and u["address"]["zip"] == str(ar["zip"])):
                    return uid
        if a["name"] == "find_user_id_by_email":
            for uid, u in db["users"].items():
                if (u.get("email") or "").lower() == a["arguments"]["email"].lower():
                    return uid
    return None


sel_users, sel_orders, tasks = {}, {}, []
for tid in PICK:
    t = tasks_all[tid]
    uid = resolve_user(t["evaluation_criteria"]["actions"])
    if uid:
        sel_users[uid] = db["users"][uid]
        for oid in db["users"][uid].get("orders", []):
            if oid in db["orders"]:
                sel_orders[oid] = db["orders"][oid]
    ins = t["user_scenario"]["instructions"]
    scenario = "\n".join(filter(None, [ins.get("task_instructions", ""),
        f"Reason for contact: {ins.get('reason_for_call','')}", f"Known info: {ins.get('known_info','')}",
        f"Unknown info: {ins.get('unknown_info','')}"]))
    tasks.append({"id": tid, "scenario": scenario})

slice_db = {"products": db["products"], "users": sel_users, "orders": sel_orders}

TOOLS_DESC = """find_user_id_by_name_zip(first_name,last_name,zip); find_user_id_by_email(email)
get_user_details(user_id); get_order_details(order_id); get_product_details(product_id); get_item_details(item_id)
list_all_product_types(); calculate(expression)
cancel_pending_order(order_id, reason); modify_pending_order_items(order_id, item_ids, new_item_ids, payment_method_id)
modify_pending_order_address(order_id, address1,address2,city,state,country,zip); modify_pending_order_payment(order_id, payment_method_id)
modify_user_address(user_id, address1,address2,city,state,country,zip)
exchange_delivered_order_items(order_id, item_ids, new_item_ids, payment_method_id)
return_delivered_order_items(order_id, item_ids, payment_method_id); transfer_to_human_agents(summary)"""

DECIDE_SCHEMA = ("{type:'object',properties:{thought:{type:'string'},"
    "reads:{type:'array',items:{type:'object',properties:{tool_name:{type:'string'},arguments:{type:'object'}},required:['tool_name','arguments']}},"
    "write:{type:['object','null'],properties:{tool_name:{type:'string'},arguments:{type:'object'}}},"
    "message:{type:['string','null']}},required:['reads','write','message']}")

js = f"""export const meta = {{
  name: 'tau2-retail-panel-synth',
  description: 'Agentic fusion: diverse panel proposes -> synth decides next move, {MODEL}',
  phases: [{{title:'Episodes'}}],
}}
const TASKS = {json.dumps(tasks)}
const DB = {json.dumps(slice_db)}
const AGENT_SYS = {json.dumps(AGENT_SYS)}
const GUIDELINES = {json.dumps(guidelines)}
const TOOLS_DESC = {json.dumps(TOOLS_DESC)}
const READS = {json.dumps(READS)}
const WRITES = {json.dumps(WRITES)}
const DECIDE_SCHEMA = {DECIDE_SCHEMA}
const PANEL = [
  {{name:'thorough', stance:'STANCE: Be thorough. Before acting, gather ALL potentially relevant data first — look up the user, EVERY one of their orders, and every candidate product/variant. Prefer proposing several reads at once.'}},
  {{name:'direct', stance:'STANCE: Be direct and efficient. Identify the minimal information actually needed to fulfill the request, then propose the action. Avoid unnecessary lookups.'}},
  {{name:'skeptical', stance:'STANCE: Be skeptical and precise. Verify every id and precondition against the gathered data before any change. Watch for exact order-id formatting (e.g. a leading hash), the correct payment method, item availability, and pending-vs-delivered status. When unsure of an id, look it up from the user order list rather than guessing.'}},
  {{name:'policy', stance:'STANCE: Be policy-driven. Check the policy before any action: authenticate the user FIRST; only PENDING orders can be modified/cancelled; DELIVERED orders can be exchanged or returned (once each); refunds go only to the original payment method or a gift card. Propose the reads/actions that keep the request strictly within policy, and redirect anything it disallows.'}},
  {{name:'alternative', stance:'STANCE: Consider the non-obvious interpretation and edge cases. The user may have MULTIPLE matching orders, items may span different orders, a requested variant may be unavailable or only partially in stock, or an id may be formatted unexpectedly. Propose reads that surface these possibilities (list ALL the user orders, check EVERY candidate variant) rather than assuming the simplest case.'}},
]
function execTool(name,a){{
  try{{
    if(name==='find_user_id_by_name_zip'){{ for(const[id,u]of Object.entries(DB.users)){{ if(u.name.first_name.toLowerCase()===String(a.first_name||'').toLowerCase()&&u.name.last_name.toLowerCase()===String(a.last_name||'').toLowerCase()&&u.address.zip===String(a.zip)) return id }} return {{error:'User not found'}} }}
    if(name==='find_user_id_by_email'){{ for(const[id,u]of Object.entries(DB.users)){{ if((u.email||'').toLowerCase()===String(a.email||'').toLowerCase()) return id }} return {{error:'User not found'}} }}
    if(name==='get_user_details') return DB.users[a.user_id]||{{error:'User not found'}}
    if(name==='get_order_details') return DB.orders[a.order_id]||{{error:'Order not found'}}
    if(name==='get_product_details') return DB.products[a.product_id]||{{error:'Product not found'}}
    if(name==='get_item_details'){{ for(const p of Object.values(DB.products)){{ const v=(p.variants||{{}})[a.item_id]; if(v) return v }} return {{error:'Item not found'}} }}
    if(name==='list_all_product_types') return Object.fromEntries(Object.entries(DB.products).map(([id,p])=>[p.name,id]))
    if(name==='calculate') return {{result:'computed'}}
    return {{status:'success'}}
  }}catch(e){{return {{error:String(e)}}}}
}}
function convo(tr){{return tr.map(m=>`${{m.role.toUpperCase()}}: ${{typeof m.content==='string'?m.content:JSON.stringify(m.content)}}`).join('\\n')}}
async function userTurn(task,tr){{
  const sys=`${{GUIDELINES}}\\n\\n<scenario>\\n${{task.scenario}}\\n</scenario>`
  const t=await agent(`${{sys}}\\n\\nConversation so far:\\n${{convo(tr)||'(none yet — open the conversation)'}}\\n\\nWrite your next user message. If your request is fully handled reply ###STOP###; if transferred reply ###TRANSFER###.`,{{phase:'Episodes',label:'user',model:'{MODEL}'}})
  return (t||'').trim()
}}
const HEAD = `${{AGENT_SYS}}\\n\\nTools:\\n${{TOOLS_DESC}}\\nREAD tools (investigative, reversible, run in PARALLEL): ${{JSON.stringify(READS)}}\\nWRITE tools (mutating, irreversible): ${{JSON.stringify(WRITES)}}`
function panelPrompt(tr, stance){{
  return `${{HEAD}}\\n\\n${{stance}}\\n\\nConversation + data so far:\\n${{convo(tr)}}\\n\\nPropose your next move (use EXACTLY one): "reads" (one or more lookups to run in parallel), "write" (the single mutating call), or "message". Set the unused fields to [] or null.`
}}
function synthPrompt(tr, proposals){{
  return `${{HEAD}}\\n\\nConversation + data gathered so far:\\n${{convo(tr)}}\\n\\nOther assistants proposed these possible next actions (EVIDENCE — some may be wrong):\\n${{proposals}}\\n\\nTreat the proposals as EVIDENCE, not votes — independently re-derive what is correct from the policy, the conversation, and the data already read. The right move may be one only a single assistant proposed, or none of them.\\n\\nWork out the correct state-changing action and exactly which ids/values it depends on, then return the SINGLE next move:\\n- If any of those ids/values is NOT yet grounded in what you have read, return the "reads" that would ground it (read exactly what the action depends on, in parallel).\\n- Once every id/value of the correct action is exactly supported by the reads and the policy permits it, return that one "write" — do NOT re-ask for confirmation you already have, and do NOT keep investigating once the action is determined.\\n- Use "message" only for information that ONLY the user can supply and has not already given.`
}}
async function decide(tr){{
  const proposals=await parallel(PANEL.map(p=>()=>agent(panelPrompt(tr,p.stance),{{phase:'Episodes',label:'panel:'+p.name,schema:DECIDE_SCHEMA,model:'{MODEL}'}})))
  const summary=proposals.map((pr,i)=>`[${{PANEL[i].name}}] ${{JSON.stringify(pr||{{}})}}`).join('\\n')
  return await agent(synthPrompt(tr,summary),{{phase:'Episodes',label:'synth',schema:DECIDE_SCHEMA,model:'{MODEL}'}})
}}
async function episode(task){{
  const tr=[]; const trajectory=[]
  let u=await userTurn(task,tr); tr.push({{role:'user',content:u}})
  for(let step=0;step<14;step++){{
    const m=await decide(tr)
    if(m && Array.isArray(m.reads) && m.reads.length){{
      const res=await parallel(m.reads.map(r=>()=>execTool(r.tool_name,r.arguments||{{}})))
      m.reads.forEach((r,i)=>{{ trajectory.push({{tool_calls:[{{name:r.tool_name,arguments:r.arguments||{{}}}}]}}); tr.push({{role:'assistant',content:JSON.stringify(r)}}); tr.push({{role:'tool',content:JSON.stringify(res[i]).slice(0,1200)}}) }})
      continue
    }}
    if(m && m.write && m.write.tool_name){{
      const w=m.write
      trajectory.push({{tool_calls:[{{name:w.tool_name,arguments:w.arguments||{{}}}}]}})
      const res=execTool(w.tool_name,w.arguments||{{}})
      tr.push({{role:'assistant',content:JSON.stringify({{tool:w.tool_name,arguments:w.arguments}})}}); tr.push({{role:'tool',content:JSON.stringify(res).slice(0,1200)}})
      if(w.tool_name==='transfer_to_human_agents') break
      continue
    }}
    const msg=(m&&m.message)||''
    if(msg.trim()) trajectory.push({{content:msg}})
    tr.push({{role:'assistant',content:msg}})
    const ur=await userTurn(task,tr); if(ur.includes('###STOP###')||ur.includes('###TRANSFER###')) break; tr.push({{role:'user',content:ur}})
  }}
  return {{task_id:task.id, policy:'panel', trajectory}}
}}
const jobs=TASKS.map(t=>()=>episode(t))
const out=[]
for(let i=0;i<jobs.length;i+={CONC}){{ out.push(...(await parallel(jobs.slice(i,i+{CONC})))) }}
return out
"""
open(f"results/_retail_panel{SUF}.js", "w").write(js)
print(f"wrote results/_retail_panel{SUF}.js | tasks={len(tasks)} users={len(sel_users)} | model={MODEL} conc={CONC} | bytes={len(js)}")
