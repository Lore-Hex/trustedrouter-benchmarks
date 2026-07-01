#!/usr/bin/env bash
# Stabilize verifier bootstrap in a temporary Harbor-migrated Terminal-Bench dataset.
#
# This is for local benchmarking only. It changes dependency setup and unbounded
# verifier subprocess execution, not the pass/fail assertions themselves.
#
# Usage:
#   scripts/harbor_patch_verifier_bootstrap.sh /tmp/tb21-fast-hard-harbor-... task1 task2
set -euo pipefail

dataset="${1:-}"
if [ -z "$dataset" ] || [ ! -d "$dataset" ]; then
  echo "usage: $0 <harbor-dataset-dir> <task> [task ...]" >&2
  exit 2
fi
shift

if [ "$#" -eq 0 ]; then
  set -- $(find "$dataset" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort)
fi

docker_marker="# trustedrouter-benchmarks: verifier bootstrap tools for local Harbor runs"
test_marker="# trustedrouter-benchmarks: bounded verifier bootstrap for local Harbor runs"
subprocess_marker="# trustedrouter-benchmarks: bounded verifier subprocess"
uv_version="${TB_UV_VERSION:-0.7.13}"

patch_dockerfile() {
  local dockerfile="$1"
  local tmp

  if grep -qF "$docker_marker" "$dockerfile"; then
    return 0
  fi

  tmp="$(mktemp)"
  awk -v marker="$docker_marker" -v uv_version="$uv_version" '
    BEGIN { inserted = 0 }
    /^[[:space:]]*FROM[[:space:]]+/ && inserted == 0 {
      print
      print ""
      print marker
      print "RUN if command -v apt-get >/dev/null 2>&1; then \\"
      print "      mkdir -p /etc/apt/apt.conf.d && \\"
      print "      printf '\''Acquire::Retries \"20\";\\nAcquire::http::Timeout \"30\";\\nAcquire::https::Timeout \"30\";\\nAcquire::ForceIPv4 \"true\";\\n'\'' > /etc/apt/apt.conf.d/80-local-verifier-retries && \\"
      print "      apt-get update && \\"
      print "      apt-get install -y --no-install-recommends curl ca-certificates && \\"
      print "      rm -rf /var/lib/apt/lists/*; \\"
      print "    fi"
      print "RUN if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/" uv_version "/install.sh | sh; fi"
      print "ENV PATH=\"/root/.local/bin:${PATH}\""
      inserted = 1
      next
    }
    { print }
  ' "$dockerfile" > "$tmp"
  mv "$tmp" "$dockerfile"
}

patch_test_sh() {
  local test_sh="$1"

  if grep -qF "$test_marker" "$test_sh"; then
    perl -0pi -e '
      s#export PATH="/\.local/bin:"#export PATH="\$HOME/.local/bin:\$PATH"#g;
      s#\[ -f "/\.local/bin/env" \]#[ -f "\$HOME/.local/bin/env" ]#g;
      s#source "/\.local/bin/env"#source "\$HOME/.local/bin/env"#g;
    ' "$test_sh"
    return 0
  fi

  perl -0pi -e '
    s/# Install curl\napt-get update\napt-get install -y curl/'"$test_marker"'\n# Install curl only if the image does not already provide it.\nexport PATH="\$HOME\/.local\/bin:\$PATH"\nif ! command -v curl >\/dev\/null 2>\&1; then\n    mkdir -p \/etc\/apt\/apt.conf.d\n    printf '\''Acquire::Retries "5";\\nAcquire::http::Timeout "20";\\nAcquire::https::Timeout "20";\\nAcquire::ForceIPv4 "true";\\n'\'' > \/etc\/apt\/apt.conf.d\/80-local-verifier-retries\n    apt-get update\n    apt-get install -y curl\nfi/s;
    s/# Install uv\ncurl -LsSf https:\/\/astral\.sh\/uv\/0\.7\.13\/install\.sh \| sh\n\n?source \$HOME\/\.local\/bin\/env/# Install uv only if the image does not already provide it.\nif ! command -v uv >\/dev\/null 2>\&1; then\n    curl --connect-timeout 20 --max-time 120 -LsSf https:\/\/astral.sh\/uv\/0.7.13\/install.sh | sh\nfi\nif [ -f "\$HOME\/.local\/bin\/env" ]; then\n    source "\$HOME\/.local\/bin\/env"\nfi\nexport PATH="\$HOME\/.local\/bin:\$PATH"/s;
  ' "$test_sh"
}

patch_unbounded_subprocesses() {
  local test_py="$1"

  if ! grep -q 'os\.popen("python3 /app/steal.py")\.read()' "$test_py"; then
    return 0
  fi
  if grep -qF "$subprocess_marker" "$test_py"; then
    return 0
  fi

  perl -0pi -e '
    s/import os\n/import os\nimport subprocess\n/ if $_ !~ /import subprocess\n/;
    s/([ \t]*)os\.popen\("python3 \/app\/steal\.py"\)\.read\(\)/$1'"$subprocess_marker"'\n$1subprocess.run(\n$1    ["python3", "\/app\/steal.py"],\n$1    stdout=subprocess.PIPE,\n$1    stderr=subprocess.STDOUT,\n$1    text=True,\n$1    timeout=int(os.environ.get("TB_VERIFIER_SUBPROCESS_TIMEOUT_SEC", "150")),\n$1    check=False,\n$1)/;
  ' "$test_py"
}

for task in "$@"; do
  task_dir="$dataset/$task"
  if [ ! -d "$task_dir" ]; then
    echo "  $task -> missing task directory" >&2
    continue
  fi

  dockerfile="$task_dir/environment/Dockerfile"
  test_sh="$task_dir/tests/test.sh"

  if [ -f "$dockerfile" ]; then
    patch_dockerfile "$dockerfile"
    echo "  $task -> Dockerfile verifier bootstrap patched"
  fi

  if [ -f "$test_sh" ]; then
    patch_test_sh "$test_sh"
    echo "  $task -> tests/test.sh verifier bootstrap patched"
  fi

  while IFS= read -r -d "" test_py; do
    patch_unbounded_subprocesses "$test_py"
  done < <(find "$task_dir/tests" -type f -name '*.py' -print0 2>/dev/null)
done
