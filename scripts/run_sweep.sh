#!/usr/bin/env bash
# Run a subset of an experiment matrix with an explicit interpreter, sequentially,
# logging each cell. Runs are resumable, so re-invoking fills only what is missing.
#
#   scripts/run_sweep.sh <config.yaml> <python-bin> <model> [extra vqb args...]
set -uo pipefail
CFG="${1:?config}"; PY="${2:?python bin}"; MODEL="${3:?model}"; shift 3
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT/results/sweep-$MODEL-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$ROOT/results"
echo "== sweep $MODEL | $(date -Is) | $* " | tee -a "$LOG"
"$PY" -m vlaquantbench.cli matrix "$CFG" --print --only-model "$MODEL" 2>/dev/null | while read -r cmd; do
  # strip the `conda run -n <env> vqb ` prefix that `matrix --print` emits and use $PY directly
  args="${cmd#*vqb run }"
  full="$PY -m vlaquantbench.cli run $args $*"
  echo "-- $(date +%H:%M:%S) $full" | tee -a "$LOG"
  eval "$full" >> "$LOG" 2>&1 || echo "   !! FAILED (rc=$?)" | tee -a "$LOG"
done
echo "== done $(date -Is)" | tee -a "$LOG"
