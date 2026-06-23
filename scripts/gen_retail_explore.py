"""Agentic fusion v3: read/write-asymmetric exploration (tau2 retail, subagents).

Per step the agent emits {reads:[...], write:?, message:?}:
  * reads  -> run ALL in parallel (investigative, reversible) and keep exploring
  * write  -> fusion CONCENTRATED here: a small panel independently re-derives the
              correct mutating call from the data read + policy; commit the
              self-consistent majority (else fall back to the agent's proposal)
  * message-> reply to the user (then a user turn)
Reads are cheap/parallel (explore broadly, reach more state automatically); the
expensive deliberation is spent only at the few irreversible commits. Grade with
tau2's REAL evaluator (scripts/tau2_grade.py). See docs/agentic_fusion_explore_plan.md.

Run: python scripts/gen_retail_explore.py "0,1,2" "_EXP" sonnet
"""
import json
import sys

TAU = "/Users/jperla/claude/tau2-bench/data/tau2/domains/retail"
PICK = sys.argv[1].split(",") if len(sys.argv) > 1 else ["0", "1", "2"]
SUF = sys.argv[2] if len(sys.argv) > 2 else "_EXP"
MODEL = sys.argv[3] if len(sys.argv) > 3 else "sonnet"
CONC = int(sys.argv[4]) if len(sys.argv) > 4 else 3
NVERIFY = 3

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
WRITE_SCHEMA = "{type:'object',properties:{tool_name:{type:'string'},arguments:{type:'object'}},required:['tool_name','arguments']}"

js = f"""export const meta = {{
  name: 'tau2-retail-explore',
  description: 'Read/write-asymmetric agentic fusion (read fan-out + write-fusion), {MODEL}',
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
const WRITE_SCHEMA = {WRITE_SCHEMA}
const NVERIFY = {NVERIFY}
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
function ckey(w){{ return w?JSON.stringify([w.tool_name, JSON.stringify(w.arguments||{{}}, Object.keys(w.arguments||{{}}).sort())]):'' }}
async function userTurn(task,tr){{
  const sys=`${{GUIDELINES}}\\n\\n<scenario>\\n${{task.scenario}}\\n</scenario>`
  const t=await agent(`${{sys}}\\n\\nConversation so far:\\n${{convo(tr)||'(none yet — open the conversation)'}}\\n\\nWrite your next user message. If your request is fully handled reply ###STOP###; if transferred reply ###TRANSFER###.`,{{phase:'Episodes',label:'user',model:'{MODEL}'}})
  return (t||'').trim()
}}
function decidePrompt(tr){{
  return `${{AGENT_SYS}}\\n\\nTools:\\n${{TOOLS_DESC}}\\n\\nREAD tools (investigative, reversible, run in PARALLEL): ${{JSON.stringify(READS)}}\\nWRITE tools (mutating, irreversible — commit only when sure): ${{JSON.stringify(WRITES)}}\\n\\nConversation + data so far:\\n${{convo(tr)}}\\n\\nChoose your next move (use EXACTLY one):\\n- "reads": a list of READ tool calls to run in parallel — request EVERYTHING you still need at once (all the user's orders, every candidate product variant, etc.).\\n- "write": the single mutating tool call, only once the data fully supports it (ids/payment taken from the read results).\\n- "message": text to send the user.\\nSet the unused fields to [] or null.`
}}
function verifyPrompt(tr, proposed){{
  return `${{AGENT_SYS}}\\n\\nTools:\\n${{TOOLS_DESC}}\\n\\nConversation + data read so far:\\n${{convo(tr)}}\\n\\nA mutating action is about to be committed:\\n${{JSON.stringify(proposed)}}\\n\\nIndependently determine the EXACTLY correct mutating tool call for this task using ONLY the data read and the policy. Verify the tool, order_id, item_ids, new_item_ids and payment_method_id against the read results. Output the correct {{tool_name, arguments}}.`
}}
async function commitWrite(tr, proposed){{
  // fusion concentrated at the irreversible action: NVERIFY independent re-derivations + self-consistency
  const panel=await parallel(Array.from({{length:NVERIFY}},()=>()=>agent(verifyPrompt(tr,proposed),{{phase:'Episodes',label:'verify',schema:WRITE_SCHEMA,model:'{MODEL}'}})))
  const votes={{}}; const byKey={{}}
  for(const w of [proposed,...panel].filter(Boolean)){{ const k=ckey(w); votes[k]=(votes[k]||0)+1; byKey[k]=w }}
  let best=ckey(proposed); for(const k of Object.keys(votes)) if(votes[k]>(votes[best]||0)) best=k
  return byKey[best]||proposed
}}
async function episode(task){{
  const tr=[]; const trajectory=[]
  let u=await userTurn(task,tr); tr.push({{role:'user',content:u}})
  for(let step=0;step<14;step++){{
    const m=await agent(decidePrompt(tr),{{phase:'Episodes',label:'decide',schema:DECIDE_SCHEMA,model:'{MODEL}'}})
    if(m && Array.isArray(m.reads) && m.reads.length){{
      const res=await parallel(m.reads.map(r=>()=>execTool(r.tool_name,r.arguments||{{}})))
      m.reads.forEach((r,i)=>{{ trajectory.push({{tool_calls:[{{name:r.tool_name,arguments:r.arguments||{{}}}}]}}); tr.push({{role:'assistant',content:JSON.stringify(r)}}); tr.push({{role:'tool',content:JSON.stringify(res[i]).slice(0,1200)}}) }})
      continue
    }}
    if(m && m.write && m.write.tool_name){{
      const w=await commitWrite(tr,m.write)
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
  return {{task_id:task.id, policy:'explore', trajectory}}
}}
const jobs=TASKS.map(t=>()=>episode(t))
const out=[]
for(let i=0;i<jobs.length;i+={CONC}){{ out.push(...(await parallel(jobs.slice(i,i+{CONC})))) }}
return out
"""
open(f"results/_retail_explore{SUF}.js", "w").write(js)
print(f"wrote results/_retail_explore{SUF}.js | tasks={len(tasks)} users={len(sel_users)} | model={MODEL} | bytes={len(js)}")
