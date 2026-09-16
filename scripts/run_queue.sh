#!/usr/bin/env bash
# Run several experiment configs back-to-back for one model, waiting for any sweep
# already running for that model to finish first. Every cell is resumable, so the
# queue can be killed and restarted at any point.
#
#   scripts/run_queue.sh <python-bin> <model> <config.yaml> [config.yaml ...]
set -uo pipefail
PY="${1:?python bin}"; MODEL="${2:?model}"; shift 2
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/results/queue-$MODEL-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$ROOT/results"

# wait for ANY in-flight evaluation on this machine -- two policies on one GPU contend
# for memory and skew every latency number we record
while pgrep -f "vlaquantbench.cli run --model " >/dev/null; do sleep 60; done

echo "== queue $MODEL | $(date -Is) | configs: $*" | tee -a "$LOG"
for cfg in "$@"; do
  echo "==== config $cfg | $(date -Is)" | tee -a "$LOG"
  "$PY" -m vlaquantbench.cli matrix "$cfg" --print --only-model "$MODEL" 2>/dev/null | while read -r cmd; do
    args="${cmd#*vqb run }"
    echo "-- $(date +%H:%M:%S) $args" | tee -a "$LOG"
    eval "$PY -m vlaquantbench.cli run $args" >> "$LOG" 2>&1 || echo "   !! FAILED (rc=$?)" | tee -a "$LOG"
  done
done
echo "== queue done $(date -Is)" | tee -a "$LOG"
