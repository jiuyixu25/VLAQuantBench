# Paper ↔ repository map

Every table and figure of the paper (version of 21 September 2026) is computed from the shipped
episode records. This page lists, for each one, the result files it uses and the script that
regenerates the numbers. Paths are relative to `results/`; `<S>` is a LIBERO suite
(`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`), `<M>` a model directory
(`pi05`, `pi0`, `openvla_oft`, `xvla`). `summary.csv` (from `scripts/make_summary.py`) holds one
row per cell with its Wilson interval and is the quickest way to look any cell up.

The paper analyzes 409 runs / 94,574 episodes: every shipped cell except the seven SIMPLER
variant-aggregation runs (`simpler/google_robot_va/xvla/*`, unequal per-variant coverage) and
`libero/libero_10/openvla_oft/ablate-W3-e2e-no_any_fc2.jsonl` (an exclusion pattern that also
matched 54 vision/projector layers; superseded by `ablate-W3-e2e-no_ah_fc2`). The 20 replay
reference trajectories are reported separately and are not counted as evaluation episodes.

## Main text

| Paper | Result files | Regenerate with |
|---|---|---|
| Figure 1 — uncalibrated W4A4 action-head interventions in π0.5 | `libero/libero_spatial/pi05/{rtn-BASELINE-e2e, ablate-W4A4-ah-onlyself_attn, ablate-W4A4-ah-onlyadarms, ablate-W4A4-ah-attn_adarms, ablate-W4A4-ah-down_attn_adarms, rtn-W4A4-ah}.jsonl` and their `seed1-`/`seed2-` repeats | `vqb summarize results/libero/libero_spatial --ci` (the plotting script lives with the paper source) |
| Figure 2 — same-observation, same-noise replay across formats | `replay/pi05_spatial/{W4A4,W4,W4A8}/analysis.json` | `scripts/replay_pi05_subsets.py record / replay / analyze`, then `scripts/replay_table.py` |
| Figure 3 — evaluation axes | schematic | — |
| Table 1 — end-to-end RTN on LIBERO | `libero/<S>/<M>/rtn-{BASELINE,W3,W4,W8,W4A4,W4A8,W8A8}-e2e.jsonl` | `vqb summarize results/libero --ci` |
| Table 2 — X-VLA beyond LIBERO | `simpler/google_robot_vm/xvla/*.jsonl`, `calvin/ABC_D/xvla/*.jsonl`, `vlabench/track_*/xvla/*.jsonl` | `vqb summarize results/simpler results/calvin results/vlabench --ci` |
| Table 3 — component-only W4A4 | `libero/libero_spatial/<M>/rtn-W4A4-{ve,llm,ah}.jsonl` | `vqb summarize` |
| Table 4 — π0.5 W4A4 scope controls (RTN vs. AH-calibrated, recovery split, replay columns) | `libero/libero_spatial/pi05/{ablate-W4A4-ah-*, calib_ah-W4A4-ah, calib_ah-W4A4-ah-*, ablate-W4A4-ah-s126_gateup, ablate-W4A4-ah-s126_others, rtn-W4A4-ah}.jsonl`; `replay/pi05_spatial/W4A4/analysis.json` | `scripts/replay_table.py` (replay columns); `vqb summarize` (success) |
| Table 5 — layer-group interventions on Spatial | OFT: `libero/libero_spatial/openvla_oft/{ablate-W4A8-ah-only_fc1, ablate-W4A8-ah-only_blocks, ablate-W4A8-ah-only_fc2, rtn-W4A8-ah}.jsonl`; X-VLA: `libero/libero_spatial/xvla/{ablate-W4A4-ah-only_qkv, only_proj, only_fc1, only_fc2, rtn-W4A4-ah}.jsonl`; π0.5: `ablate-W4A4-ah-only_action_out.jsonl` | `scripts/ablation_tables.py` |
| Table 6 — OFT-Long W3 localization | `libero/libero_10/openvla_oft/{rtn-BASELINE-e2e, rtn-W3-e2e, rtn-W3-llm, rtn-W3-ve, rtn-W3-ve_mp_llm, rtn-W3-ah, ablate-W3-ah-only_fc1, ablate-W3-ah-only_blocks, ablate-W3-ah-only_fc2, seed1-ablate-W3-ah-only_fc2, rtn-W3G64-ah, rtn-W3G32-ah, ablate-W3-e2e-no_ah_fc2, seed1-ablate-W3-e2e-no_ah_fc2, seed2-ablate-W3-e2e-no_ah_fc2}.jsonl` | `vqb summarize results/libero/libero_10 --ci` |
| Table 7 — OFT protected-output comparisons at 8-bit activations | `libero/<S>/openvla_oft/{rtn-BASELINE-e2e, rtn-W4A8-e2e, rtn-W8A8-e2e, ablate-W4A8-e2e-no_ah_fc2, ablate-W8A8-e2e-no_ah_fc2}.jsonl` | `vqb summarize` |
| Table 8 — weight-grid controls for π0.5 | `libero/libero_spatial/pi05/rtn-W{3,4}G{0,64,128}{SYM,ASYM}-e2e.jsonl` (group-128 asymmetric = `rtn-W3-e2e`, `rtn-W4-e2e`) | `vqb summarize` |
| Table 9 — W4A4 smoothing-and-clipping controls, three models | `libero/libero_spatial/<M>/{rtn-W4A4-{ve,llm,ah,e2e}, calib_ve-W4A4-ve, calib_llm-W4A4-llm, calib_ah-W4A4-ah, calib_all-W4A4-e2e, calib_all_a75-W4A4-e2e}.jsonl` (π0 also `calib_all_a25-W4A4-e2e`) | `scripts/calib_scope_table.py` |
| Table 10 — real-kernel measurements | `latency/openvla_oft/{baseline, awq-real, nf4-real, int8-real, smoothquant-real}.jsonl` | `scripts/profile_kernels.py` |
| Table 11 — Franka Research 3 study | not part of this repository (physical study) | — |

## Appendix

| Paper | Result files | Regenerate with |
|---|---|---|
| Table 12 — RTN presets | `vlaquantbench/quant/presets.py` | — |
| Table 13 — selected layer counts / parameters per component | run headers (`quant_report`) | `vqb summarize --format json` |
| Table 14 — checkpoint identifiers and diagnostic revisions | run headers; `diagnostics/v2/*.json` (`provenance`) | — |
| Tables 15–16 — π0.5 calibration scope/strength and calibration-set stability | `libero/libero_spatial/pi05/{calib_ah-W4A4-ah, calib_llm-W4A4-llm, calib_ah-W4A4-e2e, calib_llm-W4A4-e2e, calib_all-W4A4-e2e, calib_all_a75-W4A4-e2e, calib_set{0,1,2}_ah-W4A4-ah, calib_set{0,1,2}_all_a75-W4A4-e2e, seed1-calib_ah-W4A4-ah}.jsonl` | `vqb summarize` |
| Table 17 — π0 calibration controls | `libero/libero_spatial/pi0/calib_*.jsonl` | `scripts/calib_scope_table.py` |
| Table 18 — evaluation-seed replications | every `seed1-*` / `seed2-*` file | `vqb summarize` |
| Table 19 — task-clustered uncertainty | the cells named in `scripts/clustered_table.py` | `scripts/clustered_table.py` (`scripts/clustered_ci.py` for one contrast) |
| Table 20 — Spearman correlations of layer diagnostics vs. damage | `../diagnostics/` (original) and `../diagnostics/v2/` (recollection with provenance); action-head ablation cells | `scripts/act_diagnostics.py`, `scripts/diag_vs_damage.py` |
| Table 21 — repeated profiling sessions | `latency/openvla_oft/session{1,2,3}/*.jsonl` | `scripts/profile_kernels.py` |
| Table 22 — LLM-only PTQ methods on OFT | `libero/libero_spatial/openvla_oft/{rtn-W{3,4,8}-llm, rtn-W8A8-llm, awq-W{3,4}-llm, nf4-W4-llm, int8-W8-llm, smoothquant-W8A8-llm}.jsonl` | `vqb summarize` |
| Tables 23–24 — paired-observation action fidelity | `fidelity/openvla_oft/{record.npz, baseline, rtn-W4-llm, awq-W4-llm, rtn-W3-llm, awq-W3-llm}.json` | `scripts/action_fidelity.py record / replay` |
| Tables 25–26 — LoRA hyperparameters (physical study) | not part of this repository | — |
| Tables 27–28 — component weight floors and activation budgets | `libero/libero_spatial/<M>/rtn-W{2,3,4}-{ve,llm,ah}.jsonl`, `rtn-W4A{4,5,6,8}-{ve,llm,ah}.jsonl` (A5 only where measured) | `scripts/analyze_floors.py`, `scripts/precision_budget.py` |
| Table 29 — π0.5 action-head subsets (14 rows) | `libero/libero_spatial/pi05/ablate-W4A4-ah-*.jsonl`, `rtn-W4A4-ah.jsonl` | `scripts/ablation_tables.py` |
| Table 30 — six settings on Object at three seeds | `libero/libero_object/pi05/{,seed1-,seed2-}{rtn-BASELINE-e2e, ablate-W4A4-ah-onlyself_attn, ablate-W4A4-ah-onlyadarms, ablate-W4A4-ah-attn_adarms, ablate-W4A4-ah-down_attn_adarms, rtn-W4A4-ah}.jsonl` | `vqb summarize results/libero/libero_object --ci` |
| Table 31 — supporting scope, format and flow-step controls | `libero/libero_spatial/pi05/{ablate-W4A4-ah-attn_inputadarms, attn_postadarms, {q,k,v,o}proj_adarms, ablate-W4A6-ah-attn_adarms, ablate-W4A8-ah-attn_adarms, steps{1,2}-rtn-BASELINE-e2e, steps{1,2}-ablate-W4A4-ah-only_action_out}.jsonl`; `libero/libero_10/pi05/ablate-W4A4-ah-attn_adarms.jsonl`; `libero/libero_spatial/xvla/ablate-W4A4-ah-onlymlp.jsonl` | `vqb summarize` |
| Table 32 — six settings at W4 / W4A8 / W4A4 | `libero/libero_spatial/pi05/ablate-{W4,W4A8,W4A4}-ah-*.jsonl`, `rtn-{W4,W4A8,W4A4}-ah.jsonl` | `vqb summarize` |
| Tables 33–35 — replay composition, formats, signed deviations | `replay/pi05_spatial/{W4A4,W4,W4A8}/analysis.json` | `scripts/replay_pi05_subsets.py analyze`, `scripts/replay_table.py` |

## Provenance

* Each result file's header records model, checkpoint, preset, scope, method, the per-component
  quantized layer and parameter counts, the `--only` / `--exclude` patterns, calibration settings
  (`quant_report.act_calib`), seed, code revision, host and GPU.
* `scripts/verify_cells.py results/` recomputes every cell from its episode lines and rejects any
  cell whose recorded layer count disagrees with its declared scope.
* The paper's frozen manifest (file paths, episode counts and SHA-256 hashes) is distributed with
  the paper source; the shipped files are byte-identical to the ones it hashes.
