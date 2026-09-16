# Implementation status

What is implemented, and what has actually been executed end-to-end. "Verified"
means a closed-loop rollout was run on the hardware named in the last column and
produced a sane result — not that the full protocol was swept.

Hardware used for verification: **RTX 4090 (24 GB)**, driver 590.48, Linux.

## Model adapters

| model | adapter | benchmarks | verified | notes |
|---|---|---|---|---|
| π0.5 | `pi05` | LIBERO | ✅ closed-loop (baseline / W4 / W4A4, e2e + component scopes) | `lerobot/pi05_libero_finetuned_v044`, bf16 |
| π0 | `pi0` | LIBERO | ⚠️ same code path as π0.5, not yet run | `lerobot/pi0_libero_finetuned_v044` |
| OpenVLA-OFT | `openvla_oft` | LIBERO | ✅ closed-loop (baseline / W4 e2e / LLM-only RTN, NF4, LLM.int8(), AWQ) | wraps the official eval helpers |
| OpenVLA | `openvla` | LIBERO | ✅ closed-loop (baseline / W4 LLM-only) | autoregressive; forced FA2/eager, never the fork's SDPA |
| X-VLA | `xvla` | LIBERO, SIMPLER, CALVIN, VLABench | ✅ LIBERO, CALVIN, VLABench closed-loop (baseline / W4 e2e); SIMPLER ⚠️ not run | fp32; per-benchmark client conventions reproduced |
| CogACT | `cogact` | SIMPLER | ⚠️ implemented, not run | wraps the official `CogACTInference`; needs a 48 GB card in fp32 or `--dtype bf16` |
| InternVLA-M1 | `internvla_m1` | LIBERO, SIMPLER | ⚠️ implemented, not run | needs flash-attn + the InternVLA-M1 checkout |

## Benchmark runners

| benchmark | runner | protocol reproduced | verified |
|---|---|---|---|
| LIBERO | `libero` | official init states, 50 eps/task, 10 wait steps, per-model horizons, 180° rotation, delta or absolute control | ✅ |
| SIMPLER | `simpler` | VM / VA sweeps expanded verbatim from `scripts/rt1_*.sh` (244 VM / 92 VA configs), official episode enumeration and variant-weighted metric | ⚠️ task table generated and parsed; rollouts not run (needs a SimplerEnv install) |
| CALVIN | `calvin` | LH-MTLC, 1000 chains × 5 sub-tasks, official `get_sequences` + initial conditions (vendored, no `pyhash`), oracle from `new_playtable_tasks.yaml` | ✅ |
| VLABench | `vlabench` | 5 tracks × 10 tasks × 50 pre-generated episodes, official IK/action path, success / progress / intention scores | ✅ |

## Quantization

| piece | status |
|---|---|
| RTN presets W2/W3/W4/W8/W4A4/W4A8/W8A8 (+ `W3G64SYM`-style custom specs) | ✅ unit-tested (levels, ragged tails, dtype/shape preservation, per-token scales) |
| in-place weight quantization + activation pre-hooks, no module swaps | ✅ unit-tested |
| component scopes `e2e` / `ve` / `mp` / `llm` / `ah` / combinations, disjointness via exclusion patterns | ✅ unit-tested |
| NF4, LLM.int8() (bitsandbytes real kernels) | ✅ run on OpenVLA-OFT's LLaMA backbone |
| AWQ (official `llm-awq` search + pseudo-quant, W3/W4) | ✅ run; search results cached under `~/.cache/vlaquantbench/awq` |
| SmoothQuant (official smoothing + fake W8A8, or `torch._int_mm` real INT8) | ⚠️ implemented; the INT8 linear itself is GPU-tested, the full pipeline is not |
| RTN packed into the AutoAWQ INT4 GEMM kernel (same-kernel comparison) | ⚠️ implemented; needs `autoawq-kernels` built for the target GPU |

## Cross-environment serving

`vqb serve` + `--model remote` lets the policy and the simulator live in
different conda environments. Verified: X-VLA (Python 3.10, transformers 4.51)
serving CALVIN (Python 3.8, PyBullet stack) and VLABench (Python 3.10,
dm_control) — this is the only practical way to run CALVIN, whose
`calvin_models` stack cannot coexist with a modern torch.

## Known upstream issues worked around

* **LIBERO** `get_task_init_states` uses `torch.load` without `weights_only=False`
  and fails on torch ≥ 2.6 → the runner loads the `.pruned_init` files itself.
* **LIBERO** prompts on `stdin` at import when `~/.libero/config.yaml` is absent →
  the setup scripts write it.
* **calvin_env** ends `PlayTableSimEnv.__init__` with
  `get_git_commit_hash(Path(calvin_env.__file__))`, which is `None` for a
  namespace-package install → the runner sets `__file__` first.
* **calvin_models** depends on `pyhash`, which no longer builds; the two
  functions actually needed are vendored with a pure-Python FNV-1-32 over
  UTF-16-LE (verified against pyhash's own test vectors).
* **VLABench** needs `from VLABench.tasks/robots import *` for registration, and
  its `rrt-algorithms` dependency ships `rrt_algorithms/rrt/` without an
  `__init__.py`, so `find_packages()` drops it → the setup script adds it.
* **VLABench** `max_substeps` differs per model (official script 1, X-VLA's own
  client 10) and materially changes the score → adapters declare it.
* **llm-awq** `run_awq` already applies its scales/clips in place
  (`pre_quant.py:215,234`); the official `entry.py` dumps the results and
  `exit(0)`s, and `apply_awq` is only ever used on a *freshly loaded* model
  (`--load_awq`). Calling both on one instance double-applies the
  reparameterisation. Measured on OpenVLA-OFT's LLaMA backbone (mean absolute
  action deviation from bf16 over a fixed observation):

  | setting | mean abs. action deviation |
  |---|---|
  | RTN W4 (this repo's protocol quantizer) | 0.0144 |
  | llm-awq's own pseudo-quant, no search | 0.0160 |
  | AWQ W4, scales only | 0.0057 |
  | AWQ W4, scales + clipping (correct) | 0.0065 |
  | AWQ W3, scales + clipping (correct) | 0.0169 |
  | **AWQ W4 with the double-application bug** | **0.0897** |

  i.e. correct AWQ is ~2× more faithful than RTN at W4 and W3-AWQ ≈ W4-RTN,
  while the bug made it 6× *worse* than RTN — the kind of error that silently
  turns into a "method X does not transfer to VLAs" conclusion.
* **llm-awq** `move_embed` assumes `model.model.rotary_emb` (transformers ≥ 4.45)
  while OpenVLA pins 4.40.1 → patched.
* **MuJoCo/EGL contexts leak across tasks in one process.** Running a LIBERO
  sweep for X-VLA in the `xvla-stable` environment builds the first task's
  environment and completes its episodes normally, then hangs *permanently* on
  the second task's environment construction: 100% CPU, 0% GPU, file
  descriptors stuck on robosuite's Panda meshes, no progress for 70+ minutes.
  The runner's env cache (close the old env, build the new one) triggers it.
  Workaround: `scripts/run_xvla.sh` runs **one task per process** (`--tasks N`);
  because every cell is resumable, the ten invocations fill in their slices of
  the same JSONL. The same sweep does not hang when the simulator lives in a
  different environment from the model (the `vqb serve` path), so it is
  specific to that environment's robosuite/EGL combination.
  *Operational lesson:* a liveness check on the process is not enough - watch
  the age of the newest episode record, since a hung process stays "running".
* **bitsandbytes ≥ 0.44** pulls a newer torch and silently breaks the pinned
  OpenVLA-OFT stack → the setup script installs `0.43.3` with `--no-deps`.

## Not implemented

* The real-robot (Franka) experiment — deliberately out of scope for this repo.
* Quantization-aware fine-tuning; only PTQ.
* SIMPLER WidowX/Bridge adapters for models other than X-VLA and CogACT.

## Reproducing a full experiment

```bash
# expand a matrix to see what it will run (and in which environment)
vqb matrix configs/experiments/exp1_end2end.yaml --print

# run it; `--execute` dispatches each cell to the conda env named in configs/models/<model>.yaml
vqb matrix configs/experiments/exp1_end2end.yaml --execute --keep-going

# runs are resumable: re-running the same command only fills in missing episodes
vqb summarize results/ --ci
vqb summarize results/ --format latex --rows benchmark,suite,model,scope > tables.tex
```

Rough cost on a single RTX 4090, measured from the smoke runs:

| model / setting | per episode | one LIBERO suite (10x50) |
|---|---|---|
| π0.5 (bf16, 50-step chunks) | ~6-12 s | ~1.5 h |
| OpenVLA-OFT (bf16, 8-step chunks) | ~25-40 s | ~5 h |
| OpenVLA (bf16, 1 action/step) | ~60-90 s | ~12 h |
| X-VLA (fp32, 30-step chunks, 800-step horizon) | ~20-60 s | ~6 h |

An AWQ search costs ~8 min per (checkpoint, bit-width) and is cached, so the
four LIBERO suites of one model share a single search.
