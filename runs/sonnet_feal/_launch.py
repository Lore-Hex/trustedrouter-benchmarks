import os, subprocess, time
REPO="/Users/jperla/claude/trustedrouter-benchmarks"
os.chdir(f"{REPO}/runs/sonnet_feal")
if os.fork()>0: os._exit(0)
os.setsid()
if os.fork()>0: os._exit(0)
log=open("_log.txt","a"); os.dup2(log.fileno(),1); os.dup2(log.fileno(),2)
open("_pid","w").write(str(os.getpid()))
env=dict(os.environ); env["PYTHONPATH"]=f"{REPO}/scripts"; env["TB_CLAUDE_TIMEOUT"]="1800"  # Sonnet is slow
cmd=["caffeinate","-dimsu","harbor","run","-d","terminal-bench/terminal-bench-2-1",
     "--agent-import-path","harbor_haiku_agent:HaikuTerminus2","-m","claude-sonnet-4-6",
     "-n","1","--agent-timeout-multiplier","4","--agent-setup-timeout-multiplier","10",
     "-i","terminal-bench/feal-linear-cryptanalysis"]
print(f"[{time.strftime('%H:%M:%S')}] SONNET feal-linear re-run (1800s call timeout)", flush=True)
rc=subprocess.call(cmd, env=env)
open("_done.txt","w").write(f"rc={rc} at {time.strftime('%H:%M:%S')}\n")
print(f"[{time.strftime('%H:%M:%S')}] done rc={rc}", flush=True)
