#!/usr/bin/env bash
# Make locally migrated Harbor / Terminal-Bench task Docker builds less flaky.
#
# This patches only the temporary migrated dataset used for local Harbor runs. It
# inserts apt retry/timeout config before Dockerfile apt installs, then can
# prebuild the images so random mirror failures do not become model failures.
#
# Usage:
#   scripts/harbor_prepare_local_docker.sh /tmp/tb21-fast-hard-harbor-... task1 task2
#   scripts/harbor_prepare_local_docker.sh --prebuild /tmp/tb21-fast-hard-harbor-... task1 task2
#
# The patch is intentionally narrow: it changes package-manager reliability, not
# task files, tests, prompts, or scoring.
set -euo pipefail

prebuild=0
if [ "${1:-}" = "--prebuild" ]; then
  prebuild=1
  shift
fi

dataset="${1:-}"
if [ -z "$dataset" ] || [ ! -d "$dataset" ]; then
  echo "usage: $0 [--prebuild] <harbor-dataset-dir> <task> [task ...]" >&2
  exit 2
fi
shift

if [ "$#" -eq 0 ]; then
  set -- $(find "$dataset" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort)
fi

patch_dockerfile() {
  local dockerfile="$1"
  local marker="# trustedrouter-benchmarks: apt retry config for local Harbor runs"

  if grep -qF "$marker" "$dockerfile"; then
    return 0
  fi

  python3 - "$dockerfile" "$marker" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
marker = sys.argv[2]
lines = path.read_text().splitlines(keepends=True)

for idx, line in enumerate(lines):
    if line.lstrip().upper().startswith("FROM "):
        insert = [
            "\n",
            f"{marker}\n",
            "RUN if command -v apt-get >/dev/null 2>&1 || command -v apt >/dev/null 2>&1; then \\\n",
            "      mkdir -p /etc/apt/apt.conf.d && \\\n",
            "      printf 'Acquire::Retries \"20\";\\nAcquire::http::Timeout \"90\";\\nAcquire::https::Timeout \"90\";\\n' > /etc/apt/apt.conf.d/80-local-retries; \\\n",
            "    fi\n",
        ]
        lines[idx + 1:idx + 1] = insert
        path.write_text("".join(lines))
        break
else:
    raise SystemExit(f"no FROM line found in {path}")
PY
}

for task in "$@"; do
  dockerfile="$dataset/$task/environment/Dockerfile"
  if [ ! -f "$dockerfile" ]; then
    echo "  $task -> missing environment/Dockerfile" >&2
    continue
  fi

  if grep -Eq 'apt(-get)?[[:space:]]+(update|install)|apt[[:space:]]+install' "$dockerfile"; then
    patch_dockerfile "$dockerfile"
    echo "  $task -> apt retries patched"
  else
    echo "  $task -> no apt install found"
  fi

  if [ "$prebuild" -eq 1 ]; then
    tag="tb-local-${task}:prepared"
    ok=0
    for attempt in 1 2 3; do
      echo "  $task -> docker build attempt $attempt"
      if docker build -t "$tag" "$dataset/$task/environment"; then
        ok=1
        break
      fi
      sleep 5
    done
    if [ "$ok" -ne 1 ]; then
      echo "  $task -> docker build still failing after retries" >&2
      exit 1
    fi
  fi
done
