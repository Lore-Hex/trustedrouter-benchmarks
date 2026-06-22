"""Generate the retail agentic-fusion workflow (tau2 retail domain, Haiku, no TR).

Reads return REAL data from a per-task slice of the actual retail db.json (so the
agent gets real user/order/product ids); writes are recorded + ack'd. Success is
graded (separately) on whether the agent's tool calls match the task's
evaluation_criteria.actions. Solo vs fusion (per-step panel->judge->evidence_decide).
"""
import json
import sys

TAU = "/Users/jperla/claude/tau2-bench/data/tau2/domains/retail"
PICK = sys.argv[1].split(",") if len(sys.argv) > 1 else ["0", "1", "7", "9", "11", "13"]
SUF = sys.argv[2] if len(sys.argv) > 2 else ""
MODEL = sys.argv[3] if len(sys.argv) > 3 else "haiku"
SOLO = len(sys.argv) > 4 and sys.argv[4] == "solo"
CONC = 4
POLICIES_JS = "['solo']" if SOLO else "['solo','fusion']"

db = json.load(open(f"{TAU}/db.json"))
tasks_all = json.load(open(f"{TAU}/tasks.json"))
policy = open(f"{TAU}/policy.md").read().strip()
by_id = {t["id"]: t for t in tasks_all}


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


sel_users, sel_orders = {}, {}
tasks = []
for tid in PICK:
    t = by_id[tid]
    acts = t["evaluation_criteria"]["actions"]
    uid = resolve_user(acts)
    if uid:
        sel_users[uid] = db["users"][uid]
        for oid in db["users"][uid].get("orders", []):
            if oid in db["orders"]:
                sel_orders[oid] = db["orders"][oid]
    ins = t["user_scenario"]["instructions"]
    ui = (f"{ins.get('reason_for_call','')}\nWhat you know: {ins.get('known_info','')}\n"
          f"What you do NOT know: {ins.get('unknown_info','')}\nYour style: {ins.get('task_instructions','')}")
    tasks.append({"id": tid, "user_instructions": ui,
                  "expected_actions": [{"name": a["name"], "arguments": a.get("arguments", {})} for a in acts]})

slice_db = {"products": db["products"], "users": sel_users, "orders": sel_orders}
json.dump(tasks, open(f"results/_tau_tasks{SUF}.json", "w"))
print(f"tasks={len(tasks)} users={len(sel_users)} orders={len(sel_orders)} products={len(db['products'])}")
print("slice bytes:", len(json.dumps(slice_db)))

TOOLS_DESC = """READ (return real data):
- find_user_id_by_name_zip(first_name, last_name, zip) -> user_id
- find_user_id_by_email(email) -> user_id
- get_user_details(user_id) -> name, address, email, payment_methods, orders
- get_order_details(order_id) -> status, items (each has item_id, product_id, name, options, price), payment
- get_product_details(product_id) -> product with all variants (each variant has item_id, options, price, available)
- get_item_details(item_id) -> a single variant
- list_all_product_types() -> {name: product_id}
- calculate(expression) -> arithmetic
WRITE (perform the change):
- cancel_pending_order(order_id, reason)
- modify_pending_order_items(order_id, item_ids, new_item_ids, payment_method_id)
- modify_pending_order_address(order_id, address1, address2, city, state, country, zip)
- modify_pending_order_payment(order_id, payment_method_id)
- modify_user_address(user_id, address1, address2, city, state, country, zip)
- exchange_delivered_order_items(order_id, item_ids, new_item_ids, payment_method_id)
- return_delivered_order_items(order_id, item_ids, payment_method_id)
- transfer_to_human_agents(summary)"""

STANCES = """const STANCES=[
 {k:'careful', p:'Follow the policy exactly. Read what you need (user, order, product variants) before any write, and choose the minimal correct next action.'},
 {k:'eager',   p:'Move the task forward decisively: once you have the needed ids, perform the write.'},
 {k:'cautious',p:'Before any write, re-verify every id (order_id, item_id, new item_id, payment_method_id) against the data you have read.'},
 {k:'literal', p:'Use ids and option values exactly as they appear in the read results; never invent an item_id.'},
 {k:'alt',     p:'Consider whether a read is still missing, or whether a different variant/tool better matches the user request.'},
]"""
EVID = "`You are the fusion synthesizer choosing the agent's NEXT action. Use the proposals as evidence, but INDEPENDENTLY decide the correct next action from the policy, the conversation, and the data already read. The right action may be one only a single agent proposed. Prefer the action whose tool and argument ids/values are exactly supported by the read results.`"

js = f"""export const meta = {{
  name: 'agentic-fusion-retail',
  description: 'Agentic fusion vs solo on tau2 RETAIL domain (Haiku, real-data reads)',
  phases: [{{title:'Episodes'}}],
}}
const TASKS = {json.dumps(tasks)}
const DB = {json.dumps(slice_db)}
const POLICY = {json.dumps(policy)}
const TOOLS_DESC = {json.dumps(TOOLS_DESC)}
{STANCES}
const EVIDENCE_DECIDE = {EVID}
const AGENTSYS = `You are a retail customer-service agent. Policy:\\n${{POLICY}}\\n\\nTools:\\n${{TOOLS_DESC}}\\n\\nEach turn respond with ONLY JSON: {{"tool":"<name>","arguments":{{...}}}} to call a tool, or {{"message":"<text>"}} to reply to the user. A {{"message"}} performs NO action; the ONLY way to look up data or make a change is the {{"tool"}} JSON. Read what you need, then perform the required write with ids taken EXACTLY from the read results. Do NOT use any external tools of your own — your entire response must be one JSON object.`
function execTool(name,a){{
  try{{
    if(name==='find_user_id_by_name_zip'){{ for(const[id,u]of Object.entries(DB.users)){{ if(u.name.first_name.toLowerCase()===String(a.first_name||'').toLowerCase()&&u.name.last_name.toLowerCase()===String(a.last_name||'').toLowerCase()&&u.address.zip===String(a.zip)) return id }} return {{error:'User not found'}} }}
    if(name==='find_user_id_by_email'){{ for(const[id,u]of Object.entries(DB.users)){{ if((u.email||'').toLowerCase()===String(a.email||'').toLowerCase()) return id }} return {{error:'User not found'}} }}
    if(name==='get_user_details') return DB.users[a.user_id]||{{error:'User not found'}}
    if(name==='get_order_details') return DB.orders[a.order_id]||{{error:'Order not found'}}
    if(name==='get_product_details') return DB.products[a.product_id]||{{error:'Product not found'}}
    if(name==='get_item_details'){{ for(const p of Object.values(DB.products)){{ const v=(p.variants||{{}})[a.item_id]; if(v) return v }} return {{error:'Item not found'}} }}
    if(name==='list_all_product_types') return Object.fromEntries(Object.entries(DB.products).map(([id,p])=>[p.name,id]))
    if(name==='calculate') return {{result:'see figures in the data'}}
    return {{status:'success', note:`${{name}} performed`, arguments:a}}
  }}catch(e){{return {{error:String(e)}}}}
}}
function parseAction(text){{
  if(!text) return {{type:'message',text:''}}
  const s=text.replace(/```(json)?/gi,''); const m=s.match(/\\{{[\\s\\S]*\\}}/); if(!m) return {{type:'message',text:s}}
  try{{const o=JSON.parse(m[0]); if(o.tool) return {{type:'tool',name:o.tool,arguments:o.arguments||{{}}}}; if(o.message!==undefined) return {{type:'message',text:String(o.message)}}; return {{type:'message',text:s}} }}catch(e){{return {{type:'message',text:s}}}}
}}
function convo(tr){{return tr.map(m=>`${{m.role.toUpperCase()}}: ${{typeof m.content==='string'?m.content:JSON.stringify(m.content)}}`).join('\\n')}}
async function userTurn(task,tr){{
  const t=await agent(`You are the USER (a retail customer). Your situation: ${{task.user_instructions}}\\nBe concise. Provide details you KNOW when asked (name, zip); say you don't know the ones you don't. Do NOT do the agent's job.\\nConversation so far:\\n${{convo(tr)||'(none yet)'}}\\n\\nWrite your next message. Reply ###DONE### ONLY after the agent has ACTUALLY completed your request (a write tool succeeded), not when it merely promises or asks.`,{{phase:'Episodes',label:'user',model:'haiku'}})
  return (t||'').trim()
}}
async function decide(task,tr,policy){{
  const c=convo(tr)
  if(policy==='solo'){{ const t=await agent(`${{AGENTSYS}}\\n\\nConversation so far:\\n${{c}}\\n\\nYour next action (JSON only):`,{{phase:'Episodes',label:'solo',model:'haiku'}}); return parseAction(t) }}
  const cands=await parallel(STANCES.map(s=>()=>agent(`${{s.p}}\\n\\n${{AGENTSYS}}\\n\\nConversation so far:\\n${{c}}\\n\\nYour next action (JSON only):`,{{phase:'Episodes',label:`p:${{s.k}}`,model:'haiku'}}).then(parseAction)))
  const cs=cands.map((x,i)=>`[${{i+1}}] ${{JSON.stringify(x)}}`).join('\\n')
  const judge=await agent(`You are a fusion judge. Agents proposed the NEXT action.\\nConversation:\\n${{c}}\\n\\nProposals:\\n${{cs}}\\n\\nAnalyze where they AGREE and EXACTLY where they DIFFER (tool choice or argument ids/values, judged against the policy + the data already read). Do NOT pick.`,{{phase:'Episodes',label:'judge',model:'haiku'}})
  const st=await agent(`${{EVIDENCE_DECIDE}}\\n\\n${{AGENTSYS}}\\n\\nConversation so far:\\n${{c}}\\n\\nProposed next actions:\\n${{cs}}\\n\\nJudge analysis:\\n${{judge}}\\n\\nOutput the single best next action as JSON only.`,{{phase:'Episodes',label:'synth',model:'haiku'}})
  return parseAction(st)
}}
async function episode(task,policy){{
  const tr=[]; const actions=[]
  let u=await userTurn(task,tr); tr.push({{role:'user',content:u}})
  for(let step=0;step<10;step++){{
    const act=await decide(task,tr,policy)
    if(act.type==='tool'){{
      actions.push({{name:act.name,arguments:act.arguments}})
      const res=execTool(act.name,act.arguments)
      tr.push({{role:'assistant',content:JSON.stringify({{tool:act.name,arguments:act.arguments}})}})
      tr.push({{role:'tool',content:JSON.stringify(res).slice(0,1500)}})
      if(act.name==='transfer_to_human_agents') break
    }} else {{
      tr.push({{role:'assistant',content:act.text}})
      const ur=await userTurn(task,tr); if(ur.includes('###DONE###')) break; tr.push({{role:'user',content:ur}})
    }}
  }}
  return {{task_id:task.id,policy,actions}}
}}
const POLICIES = {POLICIES_JS}
const CONC = {CONC}
const jobs=[]
for(const t of TASKS) for(const p of POLICIES) jobs.push(()=>episode(t,p))
const out=[]
for(let i=0;i<jobs.length;i+=CONC){{ out.push(...(await parallel(jobs.slice(i,i+CONC)))) }}
return out
"""
js = js.replace("model:'haiku'", f"model:'{MODEL}'")
open(f"results/_retail_fusion{SUF}.js", "w").write(js)
print("wrote results/_retail_fusion{SUF}.js; episodes:", len(tasks)*2, "| script bytes:", len(js))
