# VLAQuantBench evaluation protocol

This document is the normative description of what a VLAQuantBench number
means. Anything not covered here is an implementation detail; anything that
contradicts this document is a bug.

## 1. What is measured

Every setting is evaluated **under the official closed-loop protocol of the
benchmark** and reported with that benchmark's native metric:

| benchmark | protocol | metric | episodes |
|---|---|---|---|
| LIBERO | `libero_spatial` / `object` / `goal` / `10`, official initial states, 10 settle steps | success rate (%) | 10 tasks x 50 |
| SIMPLER (Google Robot) | Visual Matching (VM) and Variant Aggregation (VA), sweeps expanded verbatim from `scripts/rt1_*.sh` | success rate (%), equal weight per variant then per family | official env/variant lists (244 VM / 92 VA configs) |
| CALVIN ABC->D | LH-MTLC, 1000 chains of 5 sub-tasks | average successful sequence length (0-5) | 1000 |
| VLABench | official VLA evaluation tracks | progress score (%) (+ success, intention score) | 10 tasks x 50 per track |

### 1.1 Model-dependent protocol constants

Some constants are set by the *model's* official evaluation client rather than by
the benchmark, and they change the score materially. Adapters declare them, the
runner honours them, and the value used is written into every episode record:

| constant | declared as | observed values |
|---|---|---|
| LIBERO episode horizon | `libero_max_steps` | 220/280/300/520 (OpenVLA, OpenVLA-OFT, InternVLA-M1) · 280/280/300/520 (LeRobot pi0/pi0.5) · 800/900 (X-VLA) |
| LIBERO control mode | `libero_action_mode` | `delta` · `absolute` (X-VLA sets `controller.use_delta=False`) |
| LIBERO render size | `libero_camera_size` | 256 · 360 (LeRobot) |
| SIMPLER control mode | `simpler_control_mode` | the official delta-pose mode · X-VLA's absolute base-pose mode (needs the `255isWhite/SimplerEnv` fork) |
| SIMPLER step budget | `simpler_max_steps_factor` | 1 (official) · 2 (X-VLA's client) |
| CALVIN sub-task budget | `calvin_ep_len` | 360 (CALVIN's protocol) · 720 (X-VLA's client) |
| VLABench physics substeps | `vlabench_max_substeps` | 1 (`scripts/evaluate_policy.py`) · 10 (X-VLA's client) |

Override any of them from the command line with `--bench-kwargs` (e.g.
`--bench-kwargs honor_adapter_protocol=false max_substeps=1`) when the goal is a
strictly like-for-like comparison across models rather than each model's own
official setting.

### 1.2 What "correct" is measured against

The reference implementation is **the upstream code** - each model's official
inference client and each benchmark's official evaluation script - not any
published results table. A disagreement with a published number is expected
whenever the protocol constants above differ, and is not by itself evidence of a
bug here; a disagreement with the upstream *code* is.

Action error against the full-precision policy is **not** a metric of this
benchmark: the full-precision trajectory is not the unique task-optimal one.

Per-episode outcomes are always persisted (`results/**.jsonl`) so that
**95% Wilson intervals** (success rates) and **bootstrap intervals**
(CALVIN / VLABench) can be reported; `vqb summarize --ci` prints them.

## 2. Component decomposition

Every VLM-based VLA is split into four disjoint sets of module roots:

| key | component | typical modules |
|---|---|---|
| `ve` | vision encoder(s) | SigLIP / DINOv2 / Qwen2.5-VL ViT / DaViT towers |
| `mp` | multimodal projector | MLP / merger mapping vision features into the LM token space |
| `llm` | language / LLM backbone | LLaMA-2-7B, Gemma-2B, Qwen2.5-3B, Florence language model ... |
| `ah` | action head | L1/MLP heads, diffusion / flow action experts (pi0 expert, CogACT DiT, X-VLA action transformer) incl. their input/output/time projections and any proprio projector feeding them |

Components absent from an architecture (OpenVLA has no `ah`, X-VLA has no
`mp`) are simply skipped. Adapters document their mapping in
`ComponentMap.notes`; `vqb inspect` prints it together with parameter counts.

## 3. Quantization boundary

The boundary is chosen to coincide with what real integer kernels execute, so
that fake-quant accuracy numbers and real-kernel deployment numbers describe
the same computation.

**Quantized**: every `torch.nn.Linear` inside a selected component — attention
and MLP projections of transformers, the action head's projections (input,
output, timestep-embedding MLPs, adaLN modulation, proprio projectors), the
multimodal projector.

**Kept at baseline precision**:

* token / patch / position embeddings (`nn.Embedding`, the patch-embedding
  `Conv2d`; `--include-conv` opts the latter in for ablations);
* all normalisation layers and all biases;
* the LLM vocabulary head `lm_head`, following LLM PTQ deployment practice
  (bitsandbytes, AWQ and GPTQ skip it by default). For OpenVLA, whose
  discrete action tokens are decoded through `lm_head`, this keeps the
  experiment comparable with the real-kernel runs; `--quantize-lm-head`
  enables it for ablation;
* attention score/probability matmuls (QK^T, PV), softmax, residual stream,
  KV cache — i.e. anything that is not a GEMM operand. A `W_xA_y` setting
  therefore means: weights of every quantized Linear at `x` bits, the *input
  activation* of every quantized Linear at `y` bits.

## 4. Quantizer (round-to-nearest)

| preset | weights | activations |
|---|---|---|
| `W2` / `W3` / `W4` | per-group (g = 128 along the input dim), **asymmetric**, zero-point | none |
| `W8` | per-channel, symmetric | none |
| `W4A4` / `W4A8` | as `W4` | per-token, symmetric, dynamic absmax |
| `W8A8` | as `W8` | per-token, symmetric, dynamic absmax |

* Symmetric grid: `q ∈ [-2^(b-1), 2^(b-1)-1]`, `s = max|x| / (2^(b-1)-1)`.
* Asymmetric grid: `q ∈ [0, 2^b-1]`, `s = (max-min)/(2^b-1)`, `z = round(-min/s)`;
  the range is widened to include 0 so that `z` is on the grid.
* If `in_features` is not divisible by the group size the last group is
  shorter (no zero padding).
* Activation scales are computed from the instantaneous tensor at every
  forward pass; **no calibration data** is used anywhere in the RTN protocol.
* Arithmetic in float32, results cast back to the baseline dtype. Weights are
  rewritten in place; activation quantization is a forward pre-hook. Module
  types never change, so model code that inspects `weight.dtype` etc. is
  unaffected.
* Custom settings use the grammar `W<bits>[G<group>][A<bits>][SYM|ASYM]`.

## 5. Scopes

* `e2e` — every component present.
* `ve` / `mp` / `llm` / `ah` — exactly one component; everything else at
  baseline precision (component-isolated sensitivity).
* `a+b` — any combination.

## 6. Baseline precision

The officially released inference precision of each checkpoint: BF16 unless
the release runs in FP32 (X-VLA, CogACT). It is recorded in every run header.
All model-specific inference hyper-parameters (image preprocessing, prompt,
action chunk length, number of denoising steps, control mode) are those of the
official evaluation code and are never changed between settings.

## 7. LLM PTQ methods (Experiment 3)

Applied to the LLM backbone only (`--scope llm`), using each method's official
implementation and its default calibration data:

| method | accuracy run | deployment run (real kernel) | calibration |
|---|---|---|---|
| RTN | in-repo fake quant | weights packed into the AWQ GEMM kernel (W4) | none |
| AWQ | official `llm-awq` search + pseudo-quant (W3, W4) | AutoAWQ GEMM (W4) | pile-val, 128 x 512 tokens |
| NF4 | bitsandbytes (W4) | bitsandbytes | none |
| LLM.int8() | bitsandbytes (W8) | bitsandbytes | none |
| SmoothQuant | official `smoothquant` smoothing (alpha 0.5) + fake W8A8 | smoothing + `torch._int_mm` INT8 GEMM (W8A8) | pile-val, 512 x 512 tokens |

Latency is the wall-clock of one action-prediction call (observation in,
action chunk out) with CUDA synchronisation, averaged over rollout steps after
warm-up; memory is the CUDA peak allocation during closed-loop rollouts at
batch size 1.

## 8. Determinism

Episode initial states come from the benchmarks' official files; environment
seeds are fixed per task; sampling-based action heads (flow matching /
diffusion) are seeded per episode by the adapters. The run header records the
git commit, hostname, GPU and every option so that a JSONL file is
self-describing.
