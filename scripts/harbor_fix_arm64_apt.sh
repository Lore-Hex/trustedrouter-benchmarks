#!/usr/bin/env bash
# Fix for Harbor / Terminal-Bench 2 tasks that error with `AgentSetupTimeoutError` on Apple Silicon.
#
# ROOT CAUSE: a few TB2 tasks build their environment `FROM ubuntu:24.04` + apt (e.g. gpt2-codegolf:
#   RUN apt-get update && apt-get install -y curl gcc). On arm64 Macs the Ubuntu *ports* mirror
#   (ports.ubuntu.com) intermittently fails to fetch some `.deb` archives, and apt's default retry
#   count is too low, so the step fails. This bites in TWO places:
#     1. the task's environment Docker BUILD (the apt-get install above), and
#     2. Harbor's terminus-2 RUNTIME install of tmux + asciinema at tmux-session start
#        (TmuxSession._attempt_tmux_installation) — bare ubuntu images ship neither.
#   Harbor retries/hangs until the agent-setup timeout (default 360s x --agent-setup-timeout-multiplier)
#   and the trial errors with AgentSetupTimeoutError. (The other ~28 hard tasks ship prebuilt images
#   that already include their tools, so they never apt-install and never hit this.)
#
# FIX: bake a higher apt retry count into the LOCAL `ubuntu:24.04` base image. Because the affected
#   tasks build `FROM ubuntu:24.04`, both the build apt AND the runtime tmux/asciinema install inherit
#   the retry config and become reliable. `apt-get install -o Acquire::Retries=10/15` was verified to
#   succeed where the default failed 6/6 times.
#
# Usage:  bash scripts/harbor_fix_arm64_apt.sh
#         # then re-run the affected task; Harbor rebuilds its env FROM this base (cache invalidated):
#         #   PYTHONPATH=scripts harbor run -d terminal-bench/terminal-bench-2 \
#         #     --agent-import-path harbor_agy_agent:AgyTerminus2 -m gemini-3.1-pro \
#         #     -i terminal-bench/gpt2-codegolf -n 1
set -euo pipefail

docker build -t ubuntu:24.04 - <<'EOF'
FROM ubuntu:24.04
RUN printf 'Acquire::Retries "15";\nAcquire::http::Timeout "60";\nAcquire::https::Timeout "60";\n' \
    > /etc/apt/apt.conf.d/80-retries
EOF

echo "Local ubuntu:24.04 now carries apt retries (Acquire::Retries 15)."
echo "Re-run the affected TB2 task; Harbor will rebuild its environment FROM this base and setup will pass."
