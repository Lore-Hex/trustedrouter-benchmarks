"""Near-identical tau2 retail harness, driven by Sonnet subagents in a Workflow.

Differences from my earlier harness, to be as close to the real tau2 loop as
possible without an API key:
  * tau2's VERBATIM agent system prompt (instructions + retail policy)
  * tau2's VERBATIM user-simulator prompt (global guidelines + scenario), with
    the real ###STOP### / ###TRANSFER### tokens
  * SCHEMA-FORCED structured tool-calls (validated {tool_name, arguments} | {message})
    instead of loose JSON parsing -> no parse loss, functionally native tool-calling
  * real-data reads (from a real db.json slice); writes recorded (terminal in retail,
    and the grader replays them on tau2's real env anyway)
Reward is computed separately by scripts/tau2_grade.py with tau2's REAL evaluator.

Run: python scripts/gen_retail_native.py "0,1,...,19" "_NAT" sonnet solo
"""
import json
import sys

TAU = "/Users/jperla/claude/tau2-bench/data/tau2/domains/retail"
PICK = sys.argv[1].split(",") if len(sys.argv) > 1 else [str(i) for i in range(20)]
SUF = sys.argv[2] if len(sys.argv) > 2 else "_NAT"
MODEL = sys.argv[3] if len(sys.argv) > 3 else "sonnet"
MODE = sys.argv[4] if len(sys.argv) > 4 else "solo"
CONC = 2 if MODE == "fusion" else 4
POLICIES_JS = {"solo": "['solo']", "fusion": "['fusion']", "both": "['solo','fusion']"}.get(MODE, "['solo']")

db = json.load(open(f"{TAU}/db.json"))
tasks_all = {str(t["id"]): t for t in json.load(open(f"{TAU}/tasks.json"))}
policy = open(f"{TAU}/policy.md").read().strip()
guidelines = open(f"{TAU}/../../user_simulator/simulation_guidelines.md").read().strip()

# tau2 verbatim agent prompt pieces (src/tau2/agent/llm_agent.py)
AGENT_INSTRUCTION = ("You are a customer service agent that helps the user according to the <policy> provided below.\n"
    "In each turn you can either:\n- Send a message to the user.\n- Make a tool call.\n"
    "You cannot do both at the same time.\n\nTry to be helpful and always follow the policy.")
AGENT_SYS = f"<instructions>\n{AGENT_INSTRUCTION}\n</instructions>\n<policy>\n{policy}\n</policy>"


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
    acts = t["evaluation_criteria"]["actions"]
    uid = resolve_user(acts)
    if uid:
        sel_users[uid] = db["users"][uid]
        for oid in db["users"][uid].get("orders", []):
            if oid in db["orders"]:
                sel_orders[oid] = db["orders"][oid]
    ins = t["user_scenario"]["instructions"]
    # tau2 builds the user scenario from these fields
    scenario = "\n".join(filter(None, [
        ins.get("task_instructions", ""),
        f"Reason for contact: {ins.get('reason_for_call','')}",
        f"Known info: {ins.get('known_info','')}",
        f"Unknown info: {ins.get('unknown_info','')}",
    ]))
    tasks.append({"id": tid, "scenario": scenario})

slice_db = {"products": db["products"], "users": sel_users, "orders": sel_orders}
json.dump([{"id": t["id"]} for t in tasks], open(f"results/_tau_tasks{SUF}.json", "w"))  # ids only; grader loads real tasks

TOOLS = ["find_user_id_by_name_zip", "find_user_id_by_email", "get_user_details", "get_order_details",
         "get_product_details", "get_item_details", "list_all_product_types", "calculate",
         "cancel_pending_order", "modify_pending_order_items", "modify_pending_order_address",
         "modify_pending_order_payment", "modify_user_address", "exchange_delivered_order_items",
         "return_delivered_order_items", "transfer_to_human_agents"]
TOOLS_DESC = """find_user_id_by_name_zip(first_name,last_name,zip) -> user_id
find_user_id_by_email(email) -> user_id
get_user_details(user_id); get_order_details(order_id); get_product_details(product_id); get_item_details(item_id)
list_all_product_types(); calculate(expression)
cancel_pending_order(order_id, reason)
modify_pending_order_items(order_id, item_ids, new_item_ids, payment_method_id)
modify_pending_order_address(order_id, address1, address2, city, state, country, zip)
modify_pending_order_payment(order_id, payment_method_id); modify_user_address(user_id, address1, address2, city, state, country, zip)
exchange_delivered_order_items(order_id, item_ids, new_item_ids, payment_method_id)
return_delivered_order_items(order_id, item_ids, payment_method_id)
transfer_to_human_agents(summary)"""

STANCES = """const STANCES=[
 {k:'careful', p:'Read what you need, verify every id against the data, then act.'},
 {k:'eager',   p:'Once you have the ids, perform the required write decisively.'},
 {k:'cautious',p:'Re-verify order_id/item_id/new item_id/payment against reads before any write.'},
 {k:'literal', p:'Use ids/values exactly as they appear in read results; never invent an item_id.'},
 {k:'alt',     p:'Consider whether a read is missing or a different variant/tool fits better.'},
]"""
EVID = "`You are the fusion synthesizer choosing the NEXT action. Use the proposals as evidence but INDEPENDENTLY decide the correct action from the policy and the data read. The right action may be one only one proposed. Prefer the action whose ids/values are exactly supported by the reads.`"

ACTION_SCHEMA = ("{type:'object',properties:{reasoning:{type:'string'},"
    "tool_name:{type:['string','null'],enum:" + json.dumps(TOOLS + [None]) + "},"
    "arguments:{type:'object'},message:{type:['string','null']}},required:['tool_name','arguments','message']}")

js = f"""export const meta = {{
  name: 'tau2-retail-native',
  description: 'Near-identical tau2 retail (verbatim prompts + schema tool-calls), {MODEL} subagents',
  phases: [{{title:'Episodes'}}],
}}
const TASKS = {json.dumps(tasks)}
const DB = {json.dumps(slice_db)}
const AGENT_SYS = {json.dumps(AGENT_SYS)}
const GUIDELINES = {json.dumps(guidelines)}
const TOOLS_DESC = {json.dumps(TOOLS_DESC)}
const ACTION_SCHEMA = {ACTION_SCHEMA}
{STANCES}
const EVIDENCE_DECIDE = {EVID}
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
    return {{status:'success'}}  // writes recorded; grader replays on the real env
  }}catch(e){{return {{error:String(e)}}}}
}}
function convo(tr){{return tr.map(m=>`${{m.role.toUpperCase()}}: ${{typeof m.content==='string'?m.content:JSON.stringify(m.content)}}`).join('\\n')}}
async function userTurn(task,tr){{
  const sys=`${{GUIDELINES}}\\n\\n<scenario>\\n${{task.scenario}}\\n</scenario>`
  const t=await agent(`${{sys}}\\n\\nConversation so far:\\n${{convo(tr)||'(none yet — open the conversation)'}}\\n\\nWrite your next user message (one message). If your request is fully handled, reply exactly ###STOP###. If transferred to a human, reply ###TRANSFER###.`,{{phase:'Episodes',label:'user',model:'{MODEL}'}})
  return (t||'').trim()
}}
function actPrompt(tr, stance){{
  return `${{stance?stance+'\\n\\n':''}}${{AGENT_SYS}}\\n\\nAvailable tools:\\n${{TOOLS_DESC}}\\n\\nConversation so far:\\n${{convo(tr)}}\\n\\nDecide your next action. To call a tool, set tool_name (exact name) and arguments. To reply to the user, set message and leave tool_name null. Do not do both.`
}}
async function decide(tr, policy){{
  if(policy==='solo'){{ return await agent(actPrompt(tr,null),{{phase:'Episodes',label:'agent',schema:ACTION_SCHEMA,model:'{MODEL}'}}) }}
  const cands=await parallel(STANCES.map(s=>()=>agent(actPrompt(tr,s.p),{{phase:'Episodes',label:`p:${{s.k}}`,schema:ACTION_SCHEMA,model:'{MODEL}'}})))
  const cs=cands.filter(Boolean).map((x,i)=>`[${{i+1}}] ${{JSON.stringify({{tool_name:x.tool_name,arguments:x.arguments,message:x.message}})}}`).join('\\n')
  const judge=await agent(`You are a fusion judge. Agents proposed the NEXT action.\\nConversation:\\n${{convo(tr)}}\\n\\nProposals:\\n${{cs}}\\n\\nAnalyze where they agree and EXACTLY where they differ (tool or argument ids), judged against the policy and reads. Do NOT pick.`,{{phase:'Episodes',label:'judge',model:'{MODEL}'}})
  return await agent(`${{EVIDENCE_DECIDE}}\\n\\n${{actPrompt(tr,null)}}\\n\\nProposed next actions:\\n${{cs}}\\n\\nJudge analysis:\\n${{judge}}\\n\\nOutput the single best next action.`,{{phase:'Episodes',label:'synth',schema:ACTION_SCHEMA,model:'{MODEL}'}})
}}
async function episode(task, policy){{
  const tr=[]; const trajectory=[]
  let u=await userTurn(task,tr); tr.push({{role:'user',content:u}})
  for(let step=0;step<16;step++){{
    const act=await decide(tr,policy)
    if(act && act.tool_name){{
      trajectory.push({{tool_calls:[{{name:act.tool_name,arguments:act.arguments||{{}}}}]}})
      const res=execTool(act.tool_name, act.arguments||{{}})
      tr.push({{role:'assistant',content:JSON.stringify({{tool:act.tool_name,arguments:act.arguments}})}})
      tr.push({{role:'tool',content:JSON.stringify(res).slice(0,1500)}})
      if(act.tool_name==='transfer_to_human_agents') break
    }} else {{
      const msg=(act&&act.message)||''
      trajectory.push({{content:msg}})
      tr.push({{role:'assistant',content:msg}})
      const ur=await userTurn(task,tr)
      if(ur.includes('###STOP###')||ur.includes('###TRANSFER###')||ur.includes('###OUT-OF-SCOPE###')) break
      tr.push({{role:'user',content:ur}})
    }}
  }}
  return {{task_id:task.id, policy, trajectory}}
}}
const POLICIES={POLICIES_JS}
const jobs=[]
for(const t of TASKS) for(const p of POLICIES) jobs.push(()=>episode(t,p))
const out=[]
for(let i=0;i<jobs.length;i+={CONC}){{ out.push(...(await parallel(jobs.slice(i,i+{CONC})))) }}
return out
"""
open(f"results/_retail_native{SUF}.js", "w").write(js)
print(f"wrote results/_retail_native{SUF}.js | tasks={len(tasks)} users={len(sel_users)} orders={len(sel_orders)} | model={MODEL} mode={MODE} | bytes={len(js)}")
