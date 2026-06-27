#!/usr/bin/env bash
set -uo pipefail

cd /Users/jperla/claude/trustedrouter-benchmarks || exit 1

RUN_ROOT="${RUN_ROOT:-runs/harbor-v21/synth-hard-v21-agent3x-tmux-20260625}"
JOB_SUFFIX="${JOB_SUFFIX:-20260625a}"
mkdir -p "$RUN_ROOT"

tasks=(
  bn-fit-modify
  cancel-async-tasks
  circuit-fibsqrt
  configure-git-webserver
  dna-assembly
  extract-moves-from-video
  feal-differential-cryptanalysis
  feal-linear-cryptanalysis
  fix-code-vulnerability
  fix-ocaml-gc
  gpt2-codegolf
  install-windows-3.11
  llm-inference-batching-scheduler
  make-doom-for-mips
  make-mips-interpreter
  mcmc-sampling-stan
  model-extraction-relu-logits
  password-recovery
  path-tracing-reverse
  polyglot-rust-c
  protein-assembly
  regex-chess
  sam-cell-seg
  sparql-university
  torch-pipeline-parallelism
  torch-tensor-parallelism
  train-fasttext
  video-processing
  write-compressor
  path-tracing
)

if [ ! -f "$RUN_ROOT/runner.tsv" ]; then
  printf "started_at\ttask\tjob\tstatus\tfinished_at\n" > "$RUN_ROOT/runner.tsv"
fi

for task in "${tasks[@]}"; do
  if awk -F '\t' -v task="$task" 'NR > 1 && $2 == task { found = 1 } END { exit found ? 0 : 1 }' "$RUN_ROOT/runner.tsv"; then
    printf "[%s] SKIP %s already recorded\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$task"
    continue
  fi

  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  safe_task="${task//./-}"
  job="synth-v21h3x-${safe_task}-${JOB_SUFFIX}"

  printf "[%s] START %s job=%s\n" "$started" "$task" "$job"
  TB_TR_CONFIRM=1 PYTHONPATH=scripts /opt/homebrew/bin/uvx --from harbor --with trusted-router-py harbor run \
    --path /tmp/tb21/tasks \
    --agent-import-path harbor_tr_agent:TRHarborTerminus \
    -m trustedrouter/synth \
    --n-concurrent 1 \
    --jobs-dir runs/harbor-v21 \
    --job-name "$job" \
    --agent-kwarg max_tokens=65536 \
    --agent-kwarg max_empty_retries=2 \
    --agent-kwarg record_terminal_session=false \
    --agent-timeout-multiplier 3 \
    --agent-setup-timeout-multiplier 3 \
    -i "$task" \
    --yes
  status=$?
  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  printf "%s\t%s\t%s\t%s\t%s\n" "$started" "$task" "$job" "$status" "$finished" >> "$RUN_ROOT/runner.tsv"
  printf "[%s] END %s status=%s\n" "$finished" "$task" "$status"
done
