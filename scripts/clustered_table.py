#!/usr/bin/env python
"""Task-clustered uncertainty for the paper's headline contrasts (appendix table).

For each contrast (cell A minus cell B, same suite and model), reports the point difference, the
episode-level normal-approximation interval, and a task-paired cluster bootstrap interval that
resamples the 10 LIBERO tasks with replacement (and episodes within task) in both cells jointly.
Missing cells are skipped, so the table can be regenerated as cells arrive.

    python scripts/clustered_table.py --tex tabs/clustered_ci.tex --md /dev/stdout
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clustered_ci import cluster_boot, load  # noqa: E402

R = Path("results/libero")
# (label, suite, model, cell A, cell B)
CONTRASTS = [
    ("$\\pi_{0.5}$ Spatial W4A4: attention vs. baseline", "libero_spatial", "pi05", "ablate-W4A4-ah-onlyself_attn", "rtn-BASELINE-e2e"),
    ("$\\pi_{0.5}$ Spatial W4A4: adaRMS vs. baseline", "libero_spatial", "pi05", "ablate-W4A4-ah-onlyadarms", "rtn-BASELINE-e2e"),
    ("$\\pi_{0.5}$ Spatial W4A4: attention+adaRMS vs. baseline", "libero_spatial", "pi05", "ablate-W4A4-ah-attn_adarms", "rtn-BASELINE-e2e"),
    ("$\\pi_{0.5}$ Spatial W4A4: full head (167) vs. 126-layer subset", "libero_spatial", "pi05", "rtn-W4A4-ah", "ablate-W4A4-ah-down_attn_adarms"),
    ("$\\pi_{0.5}$ Spatial W4A4: 126 + five conditioning layers (131) vs. 126", "libero_spatial", "pi05", "ablate-W4A4-ah-s126_others", "ablate-W4A4-ah-down_attn_adarms"),
    ("$\\pi_{0.5}$ Spatial W4A4: 126 + gate/up (162) vs. 126", "libero_spatial", "pi05", "ablate-W4A4-ah-s126_gateup", "ablate-W4A4-ah-down_attn_adarms"),
    ("$\\pi_{0.5}$ Spatial W4A4: full head vs. baseline", "libero_spatial", "pi05", "rtn-W4A4-ah", "rtn-BASELINE-e2e"),
    ("$\\pi_{0.5}$ Spatial W4A4, AH-calibrated: attention+adaRMS vs. baseline", "libero_spatial", "pi05", "calib_ah-W4A4-ah-attn_adarms", "rtn-BASELINE-e2e"),
    ("$\\pi_{0.5}$ Spatial W4A4, AH-calibrated: 126-layer subset vs. baseline", "libero_spatial", "pi05", "calib_ah-W4A4-ah-down_attn_adarms", "rtn-BASELINE-e2e"),
    ("$\\pi_{0.5}$ Spatial W4A4, calibrated end-to-end ($\\alpha$=0.75) vs. baseline", "libero_spatial", "pi05", "calib_all_a75-W4A4-e2e", "rtn-BASELINE-e2e"),
    ("$\\pi_{0.5}$ Object W4A4: full head vs. 126-layer subset", "libero_object", "pi05", "rtn-W4A4-ah", "ablate-W4A4-ah-down_attn_adarms"),
    ("X-VLA Spatial W4A4: MLP fc1+fc2 vs. baseline", "libero_spatial", "xvla", "ablate-W4A4-ah-onlymlp", "rtn-BASELINE-e2e"),
    ("OFT Spatial W4A8: fc2 only vs. residual blocks only", "libero_spatial", "openvla_oft", "ablate-W4A8-ah-only_fc2", "ablate-W4A8-ah-only_blocks"),
    ("OFT Spatial W4A8: end-to-end vs. baseline", "libero_spatial", "openvla_oft", "rtn-W4A8-e2e", "rtn-BASELINE-e2e"),
    ("OFT Spatial W4A8: all but fc2 vs. baseline", "libero_spatial", "openvla_oft", "ablate-W4A8-e2e-no_ah_fc2", "rtn-BASELINE-e2e"),
    ("OFT Object W4A8: all but fc2 vs. baseline", "libero_object", "openvla_oft", "ablate-W4A8-e2e-no_ah_fc2", "rtn-BASELINE-e2e"),
    ("OFT Goal W4A8: all but fc2 vs. baseline", "libero_goal", "openvla_oft", "ablate-W4A8-e2e-no_ah_fc2", "rtn-BASELINE-e2e"),
    ("OFT Long W4A8: all but fc2 vs. baseline", "libero_10", "openvla_oft", "ablate-W4A8-e2e-no_ah_fc2", "rtn-BASELINE-e2e"),
    ("OFT Long W3: end-to-end vs. baseline", "libero_10", "openvla_oft", "rtn-W3-e2e", "rtn-BASELINE-e2e"),
    ("OFT Long W3: fc2 only vs. baseline", "libero_10", "openvla_oft", "ablate-W3-ah-only_fc2", "rtn-BASELINE-e2e"),
    ("OFT Long W3: all but fc2 (441 layers) vs. baseline", "libero_10", "openvla_oft", "ablate-W3-e2e-no_ah_fc2", "rtn-BASELINE-e2e"),
    ("OFT Goal W4: end-to-end vs. baseline", "libero_goal", "openvla_oft", "rtn-W4-e2e", "rtn-BASELINE-e2e"),
    ("$\\pi_0$ Spatial W4: end-to-end vs. baseline", "libero_spatial", "pi0", "rtn-W4-e2e", "rtn-BASELINE-e2e"),
    ("$\\pi_0$ Spatial W4A4: end-to-end vs. baseline", "libero_spatial", "pi0", "rtn-W4A4-e2e", "rtn-BASELINE-e2e"),
    ("$\\pi_0$ Spatial W4A4, calibrated end-to-end ($\\alpha$=0.5) vs. uncalibrated", "libero_spatial", "pi0", "calib_all-W4A4-e2e", "rtn-W4A4-e2e"),
    ("$\\pi_0$ Spatial W4A4, AH-calibrated AH-only vs. uncalibrated AH-only", "libero_spatial", "pi0", "calib_ah-W4A4-ah", "rtn-W4A4-ah"),
    ("$\\pi_0$ Spatial W4A4, LLM-calibrated LLM-only vs. uncalibrated LLM-only", "libero_spatial", "pi0", "calib_llm-W4A4-llm", "rtn-W4A4-llm"),
    ("$\\pi_{0.5}$ Spatial W4A4, 126 + five conditioning layers vs. full head", "libero_spatial", "pi05", "ablate-W4A4-ah-s126_others", "rtn-W4A4-ah"),
    ("OFT Spatial W4A4, LLM-calibrated LLM-only vs. uncalibrated LLM-only", "libero_spatial", "openvla_oft", "calib_llm-W4A4-llm", "rtn-W4A4-llm"),
    ("OFT Spatial W4A4, AH-calibrated AH-only vs. uncalibrated AH-only", "libero_spatial", "openvla_oft", "calib_ah-W4A4-ah", "rtn-W4A4-ah"),
]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tex", required=True); ap.add_argument("--md", required=True)
    ap.add_argument("--B", type=int, default=5000); a = ap.parse_args()
    rng = np.random.default_rng(0)
    rows_tex, rows_md = [], ["| contrast | A | B | diff (pp) | episode-level 95% | task-paired bootstrap 95% |", "|---|---|---|---|---|---|"]
    for label, suite, model, ca, cb in CONTRASTS:
        pa, pb = R / suite / model / f"{ca}.jsonl", R / suite / model / f"{cb}.jsonl"
        if not (pa.exists() and pb.exists()):
            print(f"skip {label}: missing", file=sys.stderr); continue
        A, Bc = load(pa), load(pb)
        common = sorted(set(A) & set(Bc)); A = {t: A[t] for t in common}; Bc = {t: Bc[t] for t in common}
        na, nb = sum(len(v) for v in A.values()), sum(len(v) for v in Bc.values())
        sa, sb = sum(int(v.sum()) for v in A.values()) / na, sum(int(v.sum()) for v in Bc.values()) / nb
        diff = 100 * (sa - sb)
        se = 100 * math.sqrt(sa * (1 - sa) / na + sb * (1 - sb) / nb)
        boot = cluster_boot(A, a.B, rng, ref=Bc)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows_tex.append(f"{label} & {100*sa:.1f} & {100*sb:.1f} & {diff:+.1f} & [{diff-1.96*se:+.1f}, {diff+1.96*se:+.1f}] & [{lo:+.1f}, {hi:+.1f}] \\\\")
        rows_md.append(f"| {label.replace('$','').replace(chr(92),'')} | {100*sa:.1f} | {100*sb:.1f} | {diff:+.1f} | [{diff-1.96*se:+.1f}, {diff+1.96*se:+.1f}] | [{lo:+.1f}, {hi:+.1f}] |")
    tex = "\n".join(["\\begin{table}[t]", "\\centering", "\\scriptsize", "\\begin{tabular}{lrrrrr}", "\\toprule",
                     "Contrast & A & B & $\\Delta$ (pp) & episode-level 95\\% & task-paired bootstrap 95\\% \\\\", "\\midrule", *rows_tex,
                     "\\bottomrule", "\\end{tabular}",
                     "\\caption{Task-clustered uncertainty for the headline contrasts. A and B are success rates (\\%) of the two cells; $\\Delta$ = A $-$ B. "
                     "The episode-level interval is the normal approximation treating the 200 episodes of each cell as independent. The task-paired interval is a "
                     "cluster bootstrap (5{,}000 draws) that resamples the ten LIBERO tasks with replacement, and episodes within each task, jointly in both cells; "
                     "it accounts for task-level clustering and for the shared initial states of paired cells. Contrasts whose task-paired interval excludes zero are "
                     "robust to which tasks were sampled; the $\\pi_0$ W4 contrast is not.}", "\\label{tab:clustered}", "\\end{table}", ""])
    Path(a.tex).write_text(tex); Path(a.md).write_text("\n".join(rows_md) + "\n"); print("\n".join(rows_md))


if __name__ == "__main__":
    main()
