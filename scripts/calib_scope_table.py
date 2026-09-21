#!/usr/bin/env python
"""Three-model calibration-by-scope table (LIBERO-Spatial, W4A4): RTN versus two-episode SmoothQuant-style
calibration of the quantized scope (alpha 0.5 unless noted; init states 20-21 for the new cells, 0-1 for
the original pi0.5 cells of Table 7, whose disjoint-set replications are in Table 8).

    python scripts/calib_scope_table.py --tex tabs/calibration_three_models.tex --md /dev/stdout
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

R = Path("results/libero/libero_spatial")
MODELS = [("$\\pi_{0.5}$", "pi05"), ("$\\pi_0$", "pi0"), ("OpenVLA-OFT", "openvla_oft")]
ROWS = [("VE only", "rtn-W4A4-ve", "calib_ve-W4A4-ve"), ("LLM only", "rtn-W4A4-llm", "calib_llm-W4A4-llm"),
        ("AH only", "rtn-W4A4-ah", "calib_ah-W4A4-ah"), ("end-to-end, $\\alpha$=0.5", "rtn-W4A4-e2e", "calib_all-W4A4-e2e"),
        ("end-to-end, $\\alpha$=0.75", "rtn-W4A4-e2e", "calib_all_a75-W4A4-e2e")]


def sr(model, cell):
    p = R / model / f"{cell}.jsonl"
    if not p.exists():
        return None
    n = ok = 0
    for line in p.open():
        if '"kind": "episode"' in line:
            d = json.loads(line); n += 1; ok += bool(d.get("success"))
    return 100 * ok / n if n else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tex", required=True); ap.add_argument("--md", required=True); a = ap.parse_args()
    f = lambda x: "--" if x is None else f"{x:.1f}"
    base = {m: sr(m, "rtn-BASELINE-e2e") for _, m in MODELS}
    tex = ["\\begin{table}[t]", "\\centering", "\\small", "\\begin{tabular}{l" + "rr" * len(MODELS) + "}", "\\toprule",
           "& " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{lab} (baseline {base[m]:.1f})}}" for lab, m in MODELS) + " \\\\",
           "Quantized scope & " + " & ".join("RTN & calibrated" for _ in MODELS) + " \\\\", "\\midrule"]
    md = ["| scope | " + " | ".join(f"{lab.replace('$','').replace(chr(92),'')} RTN | calib" for lab, m in MODELS) + " |", "|---|" + "---|---|" * len(MODELS)]
    for lab, u, c in ROWS:
        vals = []
        for _, m in MODELS:
            vals += [f(sr(m, u)), f(sr(m, c))]
        tex.append(f"{lab} & " + " & ".join(vals) + " \\\\"); md.append(f"| {lab.replace('$','').replace(chr(92),'')} | " + " | ".join(vals) + " |")
    tex += ["\\bottomrule", "\\end{tabular}",
            "\\caption{The same two-episode calibration recipe (SmoothQuant-style channel smoothing folded into the quantized weights, 99.9th-percentile clipping, statistics from two unquantized episodes of task 0) applied to three models at W4A4 on LIBERO-Spatial (SR \\%, 200 episodes per cell). "
            "Calibration acts on the quantized scope only; the projector is never calibrated. The outcome is model- and scope-dependent: on $\\pi_{0.5}$ it removes the action-head interactions and, at $\\alpha$=0.75, recovers end-to-end W4A4; on $\\pi_0$ it lowers success at every scope; on OpenVLA-OFT it partially recovers the language backbone and action head in isolation but not the vision encoder or the full policy. "
            "Calibration is therefore part of the precision assignment and must be validated in closed loop with it.}", "\\label{tab:calibration_three_models}", "\\end{table}", ""]
    Path(a.tex).write_text("\n".join(tex)); Path(a.md).write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
