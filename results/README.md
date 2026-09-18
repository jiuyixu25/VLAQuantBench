# results/

One JSONL file per (benchmark, suite, model, method, preset, scope) cell, written by `vqb run`:

```
results/<benchmark>/<suite>/<model>/<method>-<PRESET>-<scope>[-lmhead].jsonl
```

* line 1: a `RunHeader` (`kind: header`) — model, checkpoint, preset, scope, method, baseline dtype,
  per-component quantization report (layers / params quantized), seed, git commit, host, GPU, timestamp.
* following lines: one `EpisodeRecord` per episode — task id/name, episode index, seed, `success`,
  `steps`, wall time, per-benchmark scalar (`subtasks_completed` for CALVIN, `progress_score` for VLABench),
  policy latency (mean / p95 seconds per action-prediction call after warm-up) and peak VRAM.

Runs are resumable: re-running the same cell appends only the missing episodes. A header mismatch
(different preset/scope/checkpoint for the same path) is refused.

`vqb summarize results/ [--ci] [--format md|latex|csv|json]` aggregates everything; LIBERO additionally
gets a `libero_avg4` row (mean of the four suite success rates).

## Other directories under `results/`

* `latency/<model>/` — real-kernel profiling runs (20 LIBERO episodes each, `--real-kernel`);
  `session1..3/` are three further independent sessions of the same five settings. These are
  timing measurements, not accuracy cells, and are excluded from `summary.csv` and from the
  accuracy audit.
* `fidelity/<model>/` — the paired-observation action-fidelity test (`scripts/action_fidelity.py`):
  `record.npz` holds every observation and action of one held-out full-precision episode;
  `<method>-<preset>-<scope>.json` holds the replayed deviation and its definition, and the
  matching `.actions.npy` the replayed action sequence.
* `../diagnostics/` — on-policy activation statistics per layer; `v2/` is the recollection on
  held-out init states with full provenance (checkpoint revision, seed, init states, MuJoCo,
  PyTorch, code revision, GPU) recorded in each file.
