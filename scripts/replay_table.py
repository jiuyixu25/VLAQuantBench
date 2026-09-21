#!/usr/bin/env python
"""Turn a replay analysis (scripts/replay_pi05_subsets.py analyze) plus the matching closed-loop cells
into a LaTeX table and a Markdown summary.

    python scripts/replay_table.py --analysis results/replay/pi05_spatial/W4A4/analysis.json \
        --cells results/libero/libero_spatial/pi05 --format W4A4 --tex out.tex --md out.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROWS = [  # (config key, label, closed-loop cell file stem per format)
    ("attention", "attention", "ablate-{F}-ah-onlyself_attn"),
    ("adarms", "adaRMS", "ablate-{F}-ah-onlyadarms"),
    ("union", "attention + adaRMS", "ablate-{F}-ah-attn_adarms"),
    ("s126", "+ down\\_proj (126)", "ablate-{F}-ah-down_attn_adarms"),
    ("s167", "full action head (167)", "rtn-{F}-ah"),
]


def sr_of(path: Path):
    if not path.exists():
        return None
    n = ok = 0
    for line in path.open():
        if '"kind": "episode"' in line:
            d = json.loads(line); n += 1; ok += bool(d.get("success"))
    return 100.0 * ok / n if n else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True); ap.add_argument("--cells", required=True)
    ap.add_argument("--format", default="W4A4"); ap.add_argument("--tex", required=True); ap.add_argument("--md", required=True)
    a = ap.parse_args()
    d = json.load(open(a.analysis))
    cells = Path(a.cells)
    base_sr = sr_of(cells / "rtn-BASELINE-e2e.jsonl")
    # scale of the full-precision executed actions, for a relative reading of the deviations
    if "fp_mean_abs_action" in d:
        scale = float(d["fp_mean_abs_action"])
    else:  # older analysis files: recompute from the baseline replay arrays
        z = np.load(Path(a.analysis).parent / "baseline.npz", allow_pickle=False)
        scale = float(np.abs(np.concatenate([z[k][:, :7] for k in z.files if k.endswith("_actions")])).mean())
    lines_tex, lines_md = [], []
    lines_md.append(f"| subset | #L | SR % (closed loop) | executed-action MAE | relative | chunk MAE | divergence at last flow step | gripper bias |")
    lines_md.append("|---|---|---|---|---|---|---|---|")
    for key, label, stem in ROWS:
        v = d["configs"][key]
        sr = sr_of(cells / (stem.format(F=a.format) + ".jsonl"))
        srs = "--" if sr is None else f"{sr:.1f}"
        mae, ch, div = v["executed_action_mae"], v["chunk_mae_mean"], v["flow_step_divergence"][-1]
        grip = v["executed_action_signed_mean_per_dim"][6]
        lines_tex.append(f"{label} & {v['quantized_layers']} & {srs} & {mae:.3f} & {ch:.3f} & {div:.3f} \\\\")
        lines_md.append(f"| {label.replace(chr(92), '')} | {v['quantized_layers']} | {srs} | {mae:.4f} | {mae/scale:.0%} | {ch:.3f} | {div:.3f} | {grip:+.3f} |")
    add, rec = d.get("additivity", {}), d.get("recovery", {})
    n_rec = len(d["records"])
    tex = "\n".join([
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\begin{tabular}{lrrrrr}", "\\toprule",
        "Quantized subset (of the AH) & \\#L & SR \\% & $|\\Delta a|$ exec. & $|\\Delta a|$ chunk & div.\\ step 10 \\\\",
        "\\midrule", *lines_tex, "\\bottomrule", "\\end{tabular}",
        f"\\caption{{Same-observation replay of the $\\pi_{{0.5}}$ action-head subsets at {a.format} on LIBERO-Spatial. "
        f"Each quantized policy replays the {n_rec} held-out full-precision trajectories (10 tasks, initial states 20--21; "
        f"baseline {base_sr:.1f}\\%) with identical observations and identical flow-matching noise; the full-precision replay "
        f"reproduces the recorded actions exactly (floor {d['repeatability_floor']:.5f}). $|\\Delta a|$ exec.\\ is the mean absolute "
        f"deviation of the executed 7-D action in environment units (full-precision mean $|a|$ = {scale:.3f}); $|\\Delta a|$ chunk is the "
        f"same over the full 50-step normalized chunk; div.\\ step 10 is the mean absolute deviation of the denoising state at the "
        f"last of 10 flow steps. SR is the closed-loop success of the matching cell (200 episodes). "
        f"Over {add.get('n_chunks', 0)} chunks, the union's error norm is {add['norm_union_over_sum_of_norms']['median']:.1f}$\\times$ the sum of the two "
        f"isolated error norms (IQR {add['norm_union_over_sum_of_norms']['iqr'][0]:.1f}--{add['norm_union_over_sum_of_norms']['iqr'][1]:.1f}), and the "
        f"167-layer head has lower chunk error than the 126-layer subset on {rec['frac_chunks_167_lower']:.1%} of chunks.}}",
        f"\\label{{tab:replay_{a.format.lower()}}}", "\\end{table}", ""])
    Path(a.tex).write_text(tex)
    md = "\n".join([f"### {a.format}: {n_rec} trajectories, floor {d['repeatability_floor']}, FP mean |a| = {scale:.3f}", *lines_md, "",
                    f"additivity (n={add.get('n_chunks')}): ||e_union|| / (||e_attn|| + ||e_adaRMS||) median {add['norm_union_over_sum_of_norms']['median']:.2f} "
                    f"(IQR {add['norm_union_over_sum_of_norms']['iqr'][0]:.2f}-{add['norm_union_over_sum_of_norms']['iqr'][1]:.2f}); "
                    f"residual ||e_union - (e_attn + e_adaRMS)|| / ||e_union|| median {add['residual_over_union']['median']:.2f}; "
                    f"cosine {add['cosine_union_vs_sum']['median']:.2f}",
                    f"recovery: chunk MAE 126 = {rec['chunk_mae_126']:.3f} vs 167 = {rec['chunk_mae_167']:.3f}; 167 lower on {rec['frac_chunks_167_lower']:.1%} of {rec['n_chunks']} chunks", ""])
    Path(a.md).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
