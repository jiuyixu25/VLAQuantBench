# VLAQuantBench

**Closed-loop evaluation of post-training quantization for vision–language–action models.**

Quantization decisions for VLA policies are usually made with proxy signals — action
error against the full-precision policy, or calibration-time activation statistics.
VLAQuantBench measures the thing that actually matters instead: **task success when the
quantized policy is rolled out in the simulator under the benchmark's own protocol.**

Every number in the paper is reproducible from this repository. The per-episode records
of all 370 evaluation cells (96,468 rollouts) are included. The paper analyzes the
manifest-defined subset of 363 runs (85,374 simulation episodes): the seven SIMPLER
variant-aggregation runs are shipped but excluded because their per-variant coverage is
unequal. Runs include baselines and evaluation-seed repetitions, so 363 is not a count of
distinct precision assignments. Every table is regenerated from the records by a script —
nothing is transcribed by hand.

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
| **Scale** | 370 shipped cells / 96,468 closed-loop episodes; the paper's manifest subset is 363 runs / 85,374 simulation episodes. 200 episodes per LIBERO cell with 95% Wilson intervals (X-VLA 190); budgets and native metrics differ per benchmark (five VLABench W4A4 tracks at a documented reduced budget) |

## Main findings (all regenerated from the shipped records)

1. **More quantized layers can improve task success.** Expanding a π0.5 W4A4 action-head
   scope from 126 to 167 layers raises LIBERO-Spatial success from 7.0% to 70.5%
   (baseline 99.0%). The recovery repeats on LIBERO-Object at three evaluation seeds
   (1.0/0.0/2.0% → 60.0/63.0/61.0%).
2. **Isolated success losses can miss joint failure.** On the same action head, attention
   alone costs 1.0 percentage points and the adaRMS modulators alone 5.5, but their
   108-layer union costs 97.5. This is non-additivity in task success; the underlying
   numerical mechanism remains open.
3. **Local activation statistics can miss large sensitivity differences.** OpenVLA-OFT's
   28,672-parameter output projection and its residual-layer group have activation
   kurtosis 834.3 and 835.3, yet W4A8 success is 18.0% and 99.5%, an 81.5-point gap.
   On one held-out OFT trajectory, AWQ has 1.44× lower action MAE than RTN at W4
   (1.15× at W3), while near-ceiling task scores do not resolve whether that fidelity
   gain changes success.

## Install

The four VLAs and four simulators have mutually incompatible dependency stacks, so the
core package is installed **into each model's own environment** rather than the other way
round. `scripts/env/` pins the upstream commit for each stack.

```bash
git clone https://github.com/jiuyixu25/VLAQuantBench && cd VLAQuantBench
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
