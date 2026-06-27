#!/usr/bin/env bash
# Make Terminal-Bench 2.x task images reliable on Apple Silicon (arm64 + Rosetta amd64 emulation).
#
# ROOT CAUSE (two separate false-failure modes, both from the SAME flaky mirror):
#   TB2 tasks run as amd64 under Rosetta. The task images (alexgshaw/<task>:<tag>) are bare
#   ubuntu/debian and ship neither tmux/asciinema nor curl/uv. At runtime:
#     1. terminus-2 installs tmux+asciinema via apt  -> AGENT SETUP HANGS when the amd64
#        ubuntu/debian mirror intermittently stalls under Rosetta -> AgentSetupTimeoutError.
#     2. the verifier's /tests/test.sh does `apt-get install -y curl` then `curl | sh` to install
#        uv, then `uvx pytest ...` -> when apt fails, curl/uvx are "command not found" and the
#        test CRASHES BEFORE TESTING -> reward 0 = a FALSE NEGATIVE (the solution was never run).
#   Both are invisible on the reference Linux/amd64 CI (fast, reliable mirror); they only bite here.
#
# FIX: bake the deps into the task image so BOTH the agent's tmux check and the verifier's curl/uv
#   are satisfied WITHOUT touching the network at run time. terminus-2 skips its apt install when
#   `tmux -V` and `asciinema --version` already succeed (tmux_session.py:_attempt_tmux_installation),
#   and test.sh's curl/uv steps become no-ops because the binaries already exist.
#
# Usage:  bash scripts/harbor_patch_task_images.sh <task1> <task2> ...
#         (pass the bare task names; the script resolves the docker_image from the dataset task.toml,
#          or falls back to alexgshaw/<task>:20251031)
#
# Idempotent: re-running verifies and skips images already patched.
set -uo pipefail

TASKS_DIR="${TB_TASKS_DIR:-/tmp/tb21/tasks}"   # where the dataset tasks live (for task.toml lookup)
UV_VER="${TB_UV_VERSION:-0.9.5}"

resolve_image() {
  local t="$1" img=""
  if [ -f "$TASKS_DIR/$t/task.toml" ]; then
    img=$(grep -E '^\s*docker_image' "$TASKS_DIR/$t/task.toml" | head -1 | sed -E 's/.*"(.*)".*/\1/')
  fi
  echo "${img:-alexgshaw/$t:20251031}"
}

check() {  # returns "OK" if tmux+asciinema+curl+uv all present
  docker run --rm --platform linux/amd64 "$1" sh -c \
    'tmux -V >/dev/null 2>&1 && asciinema --version >/dev/null 2>&1 && command -v curl >/dev/null 2>&1 && test -f /root/.local/bin/uvx && echo OK' 2>/dev/null
}

patch() {  # build the patched image, tagging over the original
  local img="$1"
  docker build --platform linux/amd64 -t "$img" - <<EOF >/dev/null 2>&1
FROM $img
RUN apt-get update -o Acquire::Retries=20 \
 && apt-get install -y --no-install-recommends tmux asciinema curl ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && (curl -LsSf https://astral.sh/uv/${UV_VER}/install.sh | sh)
EOF
}

[ $# -ge 1 ] || { echo "usage: $0 <task> [task ...]"; exit 2; }
for t in "$@"; do
  img=$(resolve_image "$t")
  if [ "$(check "$img")" = "OK" ]; then echo "  $t ($img) -> already OK"; continue; fi
  ok=""
  for a in 1 2 3 4; do
    echo "  patching $t ($img) attempt $a ..."
    timeout 240 bash -c "$(declare -f patch); patch '$img'"
    if [ "$(check "$img")" = "OK" ]; then echo "  $t -> OK"; ok=1; break; fi
    sleep 3
  done
  [ -z "$ok" ] && echo "  $t -> STILL FAILING (retry later when mirror/buildkit is less contended)"
done
