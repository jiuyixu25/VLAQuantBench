# VLAQuantBench

**Closed-loop evaluation of post-training quantization for vision–language–action models.**

Quantization decisions for VLA policies are usually made with proxy signals — action
error against the full-precision policy, or calibration-time activation statistics.
VLAQuantBench measures the thing that actually matters instead: **task success when the
quantized policy is rolled out in the simulator under the benchmark's own protocol.**

Every number in the paper is reproducible from this repository. The per-episode records
of all 370 evaluation cells (96,468 rollouts) are included, and every table is regenerated
from those records by a script — nothing is transcribed by hand.

```bash
vqb run --model pi05 --benchmark libero --suite libero_spatial --preset W4A4 --scope ah
vqb summarize results/ --ci
python scripts/verify_cells.py            # independent audit of every shipped cell
```

## What it measures

| | |
|---|---|
| **Models** | OpenVLA-OFT, π0, π0.5, X-VLA (plus adapters for OpenVLA, CogACT, InternVLA-M1) |
| **Benchmarks** | LIBERO (4 suites), SIMPLER (visual matching + variant aggregation), CALVIN ABC→D, VLABench (5 tracks) |
| **Formats** | W2/W3/W4/W8 weight-only; W4A4, W4A6, W4A8, W8A8 with activations; per-group / per-channel, symmetric / asymmetric |
| **Scopes** | end-to-end · one component (VE / MP / LLM / AH) · a layer group · a single named layer |
| **Methods** | RTN anchor, plus AWQ, NF4, LLM.int8(), SmoothQuant on the LLM backbone, in both fake-quant and real-kernel paths |
| **Scale** | 370 cells, 96,468 closed-loop episodes, 200 episodes per cell with 95% Wilson intervals (X-VLA 190; five VLABench W4A4 tracks at a documented reduced budget) |

## Three findings the code reproduces

1. **Sensitivity is not additive.** On π0.5's action head at W4A4, attention alone costs
   1.0 points and the adaRMS modulators alone 5.5, but the two together cost 97.5.
   Quantizing 41 *more* layers on top of a collapsed 126-layer subset recovers 63.5 points.
2. **Collapse is localized, and architectural role does not predict where.** One
   28,672-parameter projection carries OpenVLA-OFT's entire activation-quantization
   failure, while the same-role projection in π0.5 is free to quantize.
3. **Proxies fail measurably.** Two OFT layer groups with kurtosis 834 and 835 differ by
   74 success points; the statistic-to-damage rank correlation inverts on X-VLA; and AWQ
   is 2.2× closer to the full-precision actions than RTN with no gain in task success.

## Install

The four VLAs and four simulators have mutually incompatible dependency stacks, so the
core package is installed **into each model's own environment** rather than the other way
round. `scripts/env/` pins the upstream commit for each stack.

```bash
git clone https://github.com/OWNER/VLAQuantBench && cd VLAQuantBench
bash scripts/env/setup_pi05.sh        # creates the env, clones upstream, installs this package
pip install -e .                      # or install into an environment you already have
```

Pin `mujoco==3.3.2` for OpenVLA-OFT and the π models. Versions ≥3.4.0 change box–box
collision resolution so that one LIBERO-Spatial task's stored initial state no longer settles
into its intended configuration — the failure is invisible in aggregate and costs 3–9 suite
points. The X-VLA environment pins MuJoCo 3.1.6, which predates the change; the version used
is recorded in every `diagnostics/v2/*.json`. See [`docs/protocol.md`](docs/protocol.md).

## Reproducing a table

```bash
python scripts/verify_cells.py                 # recompute every cell from its records;
                                               # rejects any cell whose recorded quantized-layer
                                               # count disagrees with its declared scope
python scripts/ablation_tables.py              # within-component decomposition
python scripts/precision_budget.py             # per-component activation-bit budgets
python scripts/diag_vs_damage.py               # activation statistics vs. measured damage
python scripts/profile_kernels.py              # real-kernel latency and memory
python scripts/action_fidelity.py record ...   # paired-observation action fidelity: record one
python scripts/action_fidelity.py replay ...   #   full-precision episode, replay it quantized
```

Activation calibration (`vqb run --act-calib ...`) collects its statistics on init states
starting at 20 by default (`--act-calib-from`), disjoint from the evaluation states 0–19; the
cells collected in August 2026 used `--act-calib-from 0`, and the three-set replication in
`results/libero/libero_spatial/pi05/calib_set*` shows the choice does not change the outcome.
Every run header records the calibration components, episode range, seed, smoothing strength,
clip quantile, cache tag, and code revision, and the statistics cache is keyed on all of them.

## Results layout

```
results/<benchmark>/<suite>/<model>/<method>-<PRESET>-<scope>.jsonl
```

Line 1 is a run header: model, checkpoint, preset, scope, method, the per-component
quantization report (layers and parameters actually quantized), seed, git commit, host,
GPU. Every following line is one episode: task, episode index, seed, success, steps, wall
time, per-benchmark scalar (CALVIN subtasks, VLABench progress), policy and inference
latency, peak VRAM. Runs are resumable and a header mismatch is refused.
`results/summary.csv` carries one row per cell with its Wilson interval. Real-kernel profiling
runs (`results/latency/`), the paired-observation fidelity test (`results/fidelity/`), and the
activation diagnostics with full provenance (`diagnostics/v2/`) are described in
[`results/README.md`](results/README.md).

## Evaluation-stack pitfalls

Three defects we hit while building this each moved measured success by more than a
typical method delta, and each is invisible without per-task auditing:

* **Simulator physics version** — see the MuJoCo pin above.
* **Action-chunk execution length** — a released π-family config executes 50-step chunks
  open-loop where the protocol replans every 10; the policy looks healthy and loses 10–25
  points.
* **Incomplete activation hooks** — hooks that silently miss the action head report
  weight-only behaviour for it. Because OFT's activation failure lives in one layer, a
  2–36% outcome becomes a healthy-looking 94–96%.

`scripts/verify_cells.py` asserts hook coverage per component and rejects any cell whose
recorded layer count mismatches its scope. It caught a real silent failure during this
study: a pattern-mismatch ablation that had quantized nothing and re-measured the baseline.

## Adding your own

* A model: [`docs/adding_a_model.md`](docs/adding_a_model.md) — implement `load`,
  `component_map`, `reset`, `act`.
* A benchmark: [`docs/adding_a_benchmark.md`](docs/adding_a_benchmark.md) — implement
  `tasks`, `run_episode`, and the official metric.
* A quantizer: apply it in place, then run any scope; the harness records what was
  actually quantized.

## Citation

See [`CITATION.cff`](CITATION.cff). Released under the [MIT License](LICENSE).
