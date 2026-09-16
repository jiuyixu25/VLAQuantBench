# Measured findings (2026-08-22 → 08-24)

150 cells, 29,775 closed-loop rollouts on LIBERO, 200 episodes per cell (X-VLA 190; one
truncated cell is marked where it is used), official initial states, every number with a
95 % Wilson interval. Two GPUs: RTX 4090 and RTX 5080.

Raw per-episode records are in `results/`. **Every number below was re-derived from those
files by `scripts/verify_cells.py`, not read from a run log** — the log's own inline check
had a hole (`grep -c` prints `0` *and* exits non-zero, so `n=$(grep -c ... || echo 0)`
yields `"0
0"` and the following integer test errors instead of failing, which can log an
empty cell as DONE). The verifier refuses any RTN cell whose header records zero quantized
layers; that check caught one cell whose `--only` pattern matched nothing and had silently
re-measured the baseline (see §4b).

Regenerate: `scripts/verify_cells.py` (all cells + intervals), `scripts/ablation_tables.py`
(LaTeX), `scripts/diag_vs_damage.py` (diagnostics vs damage), `scripts/final_report.py`.

Seed check (π0.5, W3, action head, 3 seeds): 88.0 / 87.5 / 88.5 %. Sampling noise at n=200
is well under a point. The two headline cells were additionally re-run at an independent
seed (§8).

## 1. There is no universal bottleneck component

Weight-only, 2-bit, one component at a time (delta vs each model's own baseline):

| model | ve | llm | ah |
|---|---|---|---|
| X-VLA (96.8) | −0.5 | +0.0 | −3.7 |
| pi0.5 (87.5) | −5.0 | **−87.5** | −33.0 |
| pi0 (61.5) | +1.5 | −15.5 | **−46.5** |
| OpenVLA-OFT (95.0) | −20.5 | −82.0 | **−95.0** |

Four architectures, four patterns: X-VLA has no 2-bit bottleneck at all, pi0.5's is the
LLM, pi0's and OFT's are the action head. The paper's "the action head is universally the
most sensitive component" holds for OFT and pi0 and is *reversed* for pi0.5.

pi0 vs pi0.5 is the controlled pair: same SigLIP + Gemma-2B backbone, same flow-matching
recipe, differing mainly in the action expert's normalisation (adaRMS or not) — and their
bottlenecks are opposite. Architecture, not scale, decides: X-VLA is the smallest model
(0.9B vs OFT's 7.5B) and the most robust.

## 2. Weight robustness does not imply activation robustness

X-VLA is immune to 2-bit *weights* everywhere (worst −3.7) but its action head is
annihilated by 4-bit *activations* (−96.8). Two of four models have different bottlenecks
in the two regimes, so component sensitivity has to be reported per regime.

## 3. 4-bit activations are dangerous everywhere; 8-bit is safe *almost* everywhere

W4A4, per component: OFT collapses in all three (−95.0); pi0.5 loses 38–88 points; X-VLA
loses its action head only; pi0 is nearly unaffected (worst −17.5). The spread is the
finding — "W4A4 universally collapses" is true end-to-end but hides that pi0 barely
notices it.

W4A8 is within ±3.5 for eleven of twelve cells. The exception matters: **OpenVLA-OFT's
action head drops to 20.5 % (−74.5) at 8-bit activations.** That head is a 4-layer
MLPResNet, while every other model's head has 131–167 linear layers. A shallow regression
head has no depth over which to average quantization noise, and it is the same head that
uniquely zeroes out at 2-bit weights. Depth/redundancy of the action head, not bit-width
alone, decides whether activation quantization is survivable.

## 4. Component sensitivity is not additive, in both directions

pi0.5, W4A4 applied to subsets of the *same* component (the action head, 167 quantizable
layers = 18 blocks x (4 attn + 3 MLP + 2 adaRMS) + 5 plumbing layers). Baseline 87.5 %.

| quantized subset | layers | success | Δ |
|---|---|---|---|
| attention only | 72 | 84.5 [79,89] | −3.0 |
| adaRMS only | 36 | 68.5 [62,75] | −19.0 |
| **attention + adaRMS** | **108** | **15.0 [11,21]** | **−72.5** |
| `gate_proj` | 18 | 81.0 [75,86] | −6.5 |
| `up_proj` | 18 | 85.0 [79,89] | −2.5 |
| `down_proj` | 18 | 24.5 [19,31] | −63.0 |
| `action_out_proj` | 1 | 87.5 [82,91] | 0.0 |
| MLP | 54 | 27.5 [22,34] | −60.0 |
| MLP + adaRMS | 90 | 21.5 [16,28] | −66.0 |
| `down_proj` + attention | 90 | 18.0 [13,24] | −69.5 |
| `down_proj` + attn + adaRMS | 126 | 12.0 [8,17] | −75.5 |
| all but adaRMS | 131 | 32.0 [26,39] | −55.5 |
| all but `action_out_proj` | 166 | 56.0 [49,63] | −31.5 |
| **everything** | **167** | **49.5 [43,56]** | **−38.0** |

Three independent violations, and they do not share a sign:

* **Synergy.** Attention alone costs 3.0 and adaRMS alone costs 19.0; together they cost
  72.5, with no MLP quantized at all — 50 points beyond the sum of the parts.
* **Protection by co-quantization.** `down_proj`+attn+adaRMS (126 layers) scores 12.0.
  Adding `gate_proj` and `up_proj`, worth 6.5 and 2.5 on their own, brings the superset of
  167 layers to 49.5. Quantizing *more* recovers 37.5 points. Neither endpoint is at the
  floor, so this is not saturation.
* **Zero alone, negative in company.** `action_out_proj` costs exactly 0.0 on its own, but
  removing it from the full set moves 49.5 to 56.0 — a marginal contribution of −6.5.

Any mixed-precision search that scores layers or components independently and sums is
measuring the wrong quantity on this model.

Caveat, and it is a large one: on pi0 — same PaliGemma + Gemma-expert architecture, differing
mainly in adaRMS conditioning — none of this reproduces. Every pi0 subset lands between
+0.5 and −6.0. Four attributions were proposed and falsified by the data: adaRMS as the
sensitive part, MLP dominance as architectural, MLP<->adaRMS as the compensating pair, and
`down_proj` dominance as a property of the layer's fan-in.

## 4b. The layer that matters is not the layer with the parameters

OpenVLA-OFT's action head is four linears. W4A8, baseline 95.0 %:

| quantized | params | success | Δ |
|---|---|---|---|
| `fc1` | 117,440,512 | 96.0 [92,98] | +1.0 |
| both residual blocks | 33,554,432 | 93.5 [89,96] | −1.5 |
| **`fc2`** | **28,672** | **19.5 [15,26]** | **−75.5** |
| `fc1` + `fc2` | 117,469,184 | 19.0 [14,25] | −76.0 |
| all four | 151,023,616 | 20.5 [15,27] | −74.5 |

99.98 % of the head's parameters quantize to W4A8 for free; the remaining 0.02 % costs 75
points. And it is the activations, not the weights: W4 weight-only over the whole head
scores 94.0, W8A8 scores 36.0, W4A8 scores 20.5, W4A4 scores 0.0.

X-VLA's action head is 98 layers (24 blocks x qkv/proj/fc1/fc2, plus two input
projections). W4A4, baseline 96.8 %:

| quantized | layers | success | Δ |
|---|---|---|---|
| `mlp.fc2` | 24 | 94.2 [90,97] | −2.6 |
| `mlp.fc1` | 24 | 66.3 [59,73] | −30.5 |
| MLP (`fc1`+`fc2`) | 48 | 24.2 [19,31] | −72.6 |
| **attention** (`qkv`+`proj`) | 48 | **13.2 [9,19]** | **−83.6** |
| everything | 98 | 0.0 [0,2] | −96.8 |

Within the MLP the parts sum to −33.1 and cost −72.6 together: a 39.5-point super-additive
gap on a third architecture. And the ordering is the reverse of π0.5's — here the
*up*-projection `fc1` carries the MLP sensitivity and the down-projection `fc2` is nearly
free, where on π0.5 `down_proj` cost 63.0 and `up_proj` 2.5.

Two explanations were proposed and both were refuted. "The final action projection is the
bottleneck" fails because pi0.5's `action_out_proj` (32,768 params, same role, harsher
setting) costs 0.0. "Flow matching averages the error away" fails because pi0.5's immunity
survives at `num_inference_steps=1`, where there is nothing to average — measured against
matched unquantized controls at 10, 2 and 1 steps (87.5 / 89.0 / 89.0).

`scripts/act_diagnostics.py` measures per-layer activation statistics on real on-policy
activations, before any rollout: per-token absmax dispersion, kurtosis of |x|, the activation
grid's own relative error, and the relative output error of the quantized layer.
`scripts/diag_vs_damage.py` pairs each ablation cell with the layers that cell recorded
quantizing (read from its `--only` patterns, never re-derived by hand) and rank-correlates.

| | cells | kurtosis | out_rel_err | act_rel_err |
|---|---|---|---|---|
| within π0.5 | 17 | +0.527 | +0.493 | +0.480 |
| within OpenVLA-OFT | 4 | +0.200 | +0.400 | +0.200 |
| within π0 | 6 | +0.143 | +0.429 | +0.257 |
| within X-VLA | 4 | **−0.800** | −0.400 | −0.800 |
| pooled | 31 | +0.544 | +0.456 | +0.327 |

It does not work. Two counterexamples are enough to close the question:

* Inside OpenVLA-OFT's action head, `fc2` has activation kurtosis 834.3 and costs 75.5
  points; the two residual blocks have kurtosis 835.3 and cost 1.5. Same statistic to
  within 0.1 %, 74 points apart in closed-loop success.
* On X-VLA the ranking inverts: `mlp.fc2` has the highest kurtosis in the head (355) and
  costs 2.6 points, while attention has kurtosis 140 and costs 83.7.

Calibration-time activation statistics — the quantities standard PTQ methods are built to
optimise — do not identify which layer will break a closed-loop policy. This is the
strongest available argument that closed-loop measurement cannot be replaced by a proxy.

## 5. Off-the-shelf LLM PTQ brings no consistent gain over RTN

OpenVLA-OFT, LLM backbone only, same protocol and episode budget:

| method | W3 | W4 | W8 |
|---|---|---|---|
| RTN | **98.0** [95,99] | 92.5 [88,95] | — |
| AWQ | 95.0 [91,97] | 94.5 [90,97] | — |
| NF4 | — | 95.0 [91,97] | — |
| LLM.int8() | — | — | 95.5 [92,98] |

No method wins at both bit-widths; all intervals overlap. This confirms the paper's claim.

The mechanism is visible in a paired measurement: with the AWQ integration fixed, AWQ W4 is
**2.2× more faithful than RTN in action space** (mean |Δaction| 0.0065 vs 0.0144) yet gives
no task-level gain. Task success is a thresholded metric with a dead zone; fidelity
improvements inside the recovery basin are invisible to it. That is direct evidence for the
paper's Sec. 5.3 argument, previously only asserted.

## 6. Horizon changes magnitude, not ordering

pi0.5 and pi0 on libero_spatial (220–280 steps) and libero_10 (520 steps): each model's
component ordering is identical on both suites, but damage grows with horizon
(pi0.5 W2·ah −33.0 → −45.0; W4A4·ve −60.5 → −78.0; pi0 W3·ah −15.0 → −20.0). Quantization
tolerance is therefore only meaningful relative to a stated episode budget — and the
per-model official budgets differ by up to 4× (220 vs 800 steps).

## 7. Quantization does not slow policies down; it makes them fail

Median steps-to-success is flat across every surviving cell (pi0.5 baseline 109, quantized
105–110; OFT baseline 101, quantized 94–111). Only already-severely-degraded cells show a
rise (pi0.5 W2·ah 117, W4A4·ah 120). The hypothesis that efficiency degrades before
capability — that success rate is a lagging indicator — was tested and **rejected**.

## 8. Replication

The two numbers a paper would lean on hardest were re-run at an independent seed, same
200-episode budget, π0.5 W4A4 on the action head:

| cell | seed 0 | seed 1 |
|---|---|---|
| attention only (72 layers) | 84.5 | 88.0 |
| adaRMS only (36 layers) | 68.5 | 67.5 |
| **attention + adaRMS (108 layers)** | **15.0** | **11.0** |
| `down_proj` only (18 layers) | 24.5 | 23.0 |

The synergy — two groups costing 3.0 and 19.0 alone, 72.5 together — reproduces.

## What this means for the paper's claims

| paper's claim | verdict |
|---|---|
| "first benchmark evaluating VLA quantization under closed-loop success" | **false** — SQAP-VLA, QVLA and QuantVLA all report LIBERO success rates. Reposition as a systematic comparison, not a first. |
| "the action head is universally the most sensitive component" | **false as written** — true for OFT and π0, reversed for π0.5 (its LLM collapses first), and X-VLA has no weight-side bottleneck at all |
| "W4A4 universally collapses" | **needs qualifying** — true end-to-end, but π0 is nearly unaffected component-wise |
| "W4–W8 weight-only is largely safe" | **needs qualifying** — π0's action head loses 8.5 points at W4 |
| "W4A8 is safe" | **false for OpenVLA-OFT** — its 4-layer head drops to 20.5 %, and the drop is one 28,672-parameter layer |
| "off-the-shelf LLM PTQ brings no consistent gain over RTN" | **confirmed** — four bit-widths, with intervals, plus the paired evidence that AWQ is 2.2× more faithful in action space and still gains nothing at task level |
| "activation outliers arise in vision tokens rather than LLM states" | **X-VLA-specific** — π0.5 is the opposite |
| per-task tolerance margin / precision budgets (contribution 5) | **still unimplemented in the paper**; this repo now has the data (A4/A6/A8 per component per model) to write it |
| π0 LIBERO baseline 73.7 % | our LeRobot π0 gives 61.5 % — state the checkpoint; not comparable to openpi's ~94 % |
| Sec. 5.3 softmax/argmax dead-zone argument | applies only to the original autoregressive OpenVLA, and Table 4's own numbers run the other way — but the AWQ fidelity-vs-success result supports the underlying point |
| Sec. 3 quantization format description | inconsistent with Table 7 / Appendix A (signed z=0 + per-channel vs per-group asymmetric + per-token) |
| no rollout counts, seeds or variance reported | every cell here is 200 episodes (X-VLA 190) with Wilson intervals; seed variance measured at 88.0 / 87.5 / 88.5 |

### What is new here that the paper does not have

1. **Within-component non-additivity, measured on a lattice of 14 subsets** (§4), replicated
   at a second seed (§8) and reproduced on a third architecture (§4b). This is a concrete
   problem for sensitivity-guided mixed precision, which is the dominant practical recipe.
2. **A single 28,672-parameter layer that costs 75 points** while the other 151 M parameters
   of the same head cost nothing (§4b) — parameter count and layer role do not predict risk.
3. **A negative result with teeth** (§4b): calibration-time activation statistics, including
   the outlier metrics PTQ methods are built around, fail to rank layers by closed-loop
   damage, and invert entirely on X-VLA.
4. **Contribution 5, actually measured**: per-component activation-bit budgets from the
   A4 / A6 / A8 sweep.
