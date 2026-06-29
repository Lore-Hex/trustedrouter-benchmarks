import os, sys, subprocess, time
REPO="/Users/jperla/claude/trustedrouter-benchmarks"
os.chdir(f"{REPO}/runs/sonnet2")
if os.fork() > 0: os._exit(0)
os.setsid()
if os.fork() > 0: os._exit(0)
log = open("_sonnet.log","a"); os.dup2(log.fileno(),1); os.dup2(log.fileno(),2)
open("_sonnet.pid","w").write(str(os.getpid()))
# 1) wait for Haiku re-run to finish (avoid 2 harbor procs on same dataset)
rerun_done = f"{REPO}/runs/haiku21/_rerun_done.txt"
print(f"[{time.strftime('%H:%M:%S')}] waiting for Haiku re-run to finish ({rerun_done})...", flush=True)
waited=0
while not os.path.exists(rerun_done):
    time.sleep(20); waited+=20
    if waited % 300 == 0: print(f"[{time.strftime('%H:%M:%S')}] still waiting ({waited//60}m)...", flush=True)
print(f"[{time.strftime('%H:%M:%S')}] Haiku re-run done; starting Sonnet.", flush=True)
time.sleep(10)
env=dict(os.environ); env["PYTHONPATH"]=f"{REPO}/scripts"; env["TB_CLAUDE_TIMEOUT"]="600"
tasks=["fix-code-vulnerability","cancel-async-tasks"]
cmd=["caffeinate","-dimsu","harbor","run","-d","terminal-bench/terminal-bench-2-1",
     "--agent-import-path","harbor_haiku_agent:HaikuTerminus2","-m","claude-sonnet-4-6",
     "-n","1","--agent-timeout-multiplier","3","--agent-setup-timeout-multiplier","10"]
for t in tasks: cmd += ["-i", f"terminal-bench/{t}"]
print(f"[{time.strftime('%H:%M:%S')}] SONNET run: {tasks}", flush=True)
rc=subprocess.call(cmd, env=env)
open("_sonnet_done.txt","w").write(f"rc={rc} at {time.strftime('%H:%M:%S')}\n")
print(f"[{time.strftime('%H:%M:%S')}] Sonnet run finished rc={rc}", flush=True)
