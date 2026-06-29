import os, subprocess, time
REPO="/Users/jperla/claude/trustedrouter-benchmarks"
os.chdir(f"{REPO}/runs/sonnet2b")
if os.fork() > 0: os._exit(0)
os.setsid()
if os.fork() > 0: os._exit(0)
log=open("_sonnet2.log","a"); os.dup2(log.fileno(),1); os.dup2(log.fileno(),2)
open("_sonnet2.pid","w").write(str(os.getpid()))
wait_on=f"{REPO}/runs/sonnet2/_sonnet_done.txt"
print(f"[{time.strftime('%H:%M:%S')}] waiting for Sonnet batch-1 to finish ({wait_on})...", flush=True)
w=0
while not os.path.exists(wait_on):
    time.sleep(20); w+=20
    if w%300==0: print(f"[{time.strftime('%H:%M:%S')}] still waiting ({w//60}m)...", flush=True)
print(f"[{time.strftime('%H:%M:%S')}] batch-1 done; starting Sonnet batch-2.", flush=True)
time.sleep(10)
env=dict(os.environ); env["PYTHONPATH"]=f"{REPO}/scripts"; env["TB_CLAUDE_TIMEOUT"]="600"
tasks=["password-recovery","feal-linear-cryptanalysis"]
cmd=["caffeinate","-dimsu","harbor","run","-d","terminal-bench/terminal-bench-2-1",
     "--agent-import-path","harbor_haiku_agent:HaikuTerminus2","-m","claude-sonnet-4-6",
     "-n","1","--agent-timeout-multiplier","3","--agent-setup-timeout-multiplier","10"]
for t in tasks: cmd+=["-i",f"terminal-bench/{t}"]
print(f"[{time.strftime('%H:%M:%S')}] SONNET batch-2: {tasks}", flush=True)
rc=subprocess.call(cmd, env=env)
open("_sonnet2_done.txt","w").write(f"rc={rc} at {time.strftime('%H:%M:%S')}\n")
print(f"[{time.strftime('%H:%M:%S')}] batch-2 finished rc={rc}", flush=True)
