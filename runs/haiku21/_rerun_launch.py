import os, sys, subprocess, time
os.chdir("/Users/jperla/claude/trustedrouter-benchmarks/runs/haiku21")
# double-fork daemonize
if os.fork() > 0: os._exit(0)
os.setsid()
if os.fork() > 0: os._exit(0)
log = open("_rerun.log", "a")
os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
open("_rerun.pid","w").write(str(os.getpid()))
env = dict(os.environ)
env["PYTHONPATH"] = "/Users/jperla/claude/trustedrouter-benchmarks/scripts"
env["TB_CLAUDE_TIMEOUT"] = "600"
tasks = ["cancel-async-tasks","sparql-university","gpt2-codegolf","fix-ocaml-gc","extract-moves-from-video"]
cmd = ["caffeinate","-dimsu","harbor","run","-d","terminal-bench/terminal-bench-2-1",
       "--agent-import-path","harbor_haiku_agent:HaikuTerminus2","-m","claude-haiku-4-5",
       "-n","1","--agent-timeout-multiplier","3","--agent-setup-timeout-multiplier","10"]
for t in tasks: cmd += ["-i", f"terminal-bench/{t}"]
print(f"[{time.strftime('%H:%M:%S')}] launching rerun: {' '.join(tasks)}", flush=True)
rc = subprocess.call(cmd, env=env)
open("_rerun_done.txt","w").write(f"rc={rc} at {time.strftime('%H:%M:%S')}\n")
print(f"[{time.strftime('%H:%M:%S')}] rerun finished rc={rc}", flush=True)
