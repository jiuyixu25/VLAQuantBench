#!/usr/bin/env python
"""Consolidate every result cell into the tables the paper needs."""
from __future__ import annotations
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlaquantbench.results import read_run
from vlaquantbench.stats import wilson_ci

R = "results/libero"
MODELS = [("xvla", "X-VLA"), ("pi05", "pi0.5"), ("pi0", "pi0"), ("openvla_oft", "OpenVLA-OFT")]
SCOPES = ["ve", "llm", "ah"]

def cell(model, suite, name):
    f = f"{R}/{suite}/{model}/{name}.jsonl"
    if not os.path.exists(f):
        return None
    _, eps = read_run(f)
    if not eps:
        return None
    ci = wilson_ci(sum(e.success for e in eps), len(eps))
    return 100 * ci.point, 100 * ci.low, 100 * ci.high, len(eps)

def table(suite, presets, title):
    print(f"\n### {title}  ({suite})\n")
    hdr = "| model | baseline | " + " | ".join(f"{p} {s}" for p in presets for s in SCOPES) + " |"
    print(hdr); print("|" + "---|" * (2 + len(presets) * 3))
    for m, label in MODELS:
        b = cell(m, suite, "rtn-BASELINE-e2e")
        if not b:
            continue
        row = [label, f"{b[0]:.1f}"]
        for p in presets:
            for s in SCOPES:
                c = cell(m, suite, f"rtn-{p}-{s}")
                row.append("—" if not c else f"{c[0]:.1f} ({c[0]-b[0]:+.1f})")
        print("| " + " | ".join(row) + " |")

table("libero_spatial", ["W2", "W3", "W4"], "Weight-only quantization, one component at a time")
table("libero_spatial", ["W4A4", "W4A8"], "Activation quantization, one component at a time")
table("libero_10", ["W2", "W3", "W4"], "Weight-only, long horizon")
table("libero_10", ["W4A4", "W4A8"], "Activation, long horizon")

print("\n### LLM-only: off-the-shelf PTQ vs RTN (OpenVLA-OFT)\n")
print("| method | W3 | W4 | W8 |"); print("|---|---|---|---|")
for meth in ["rtn", "awq", "nf4", "int8"]:
    row = [meth]
    for p in ["W3", "W4", "W8"]:
        c = cell("openvla_oft", "libero_spatial", f"{meth}-{p}-llm")
        row.append("—" if not c else f"{c[0]:.1f} [{c[1]:.0f},{c[2]:.0f}]")
    print("| " + " | ".join(row) + " |")

print("\n### Layer-group ablation inside the action head (W4A4)\n")
print("| model | baseline | attention only | all | MLP only | adaRMS only | non-adaRMS |")
print("|---|---|---|---|---|---|---|")
for m, label in [("pi05", "pi0.5"), ("pi0", "pi0")]:
    b = cell(m, "libero_spatial", "rtn-BASELINE-e2e")
    row = [label, f"{b[0]:.1f}"]
    for name in ["ablate-W4A4-ah-onlyself_attn", "rtn-W4A4-ah", "ablate-W4A4-ah-onlymlp",
                 "ablate-W4A4-ah-onlyadarms", "ablate-W4A4-ah-noadarms"]:
        c = cell(m, "libero_spatial", name)
        row.append("—" if not c else f"{c[0]:.1f} ({c[0]-b[0]:+.1f})")
    print("| " + " | ".join(row) + " |")

print("\n### Seed variance (pi0.5, W3, action head)\n")
for seed, name in [(0, "rtn-W3-ah"), (1, "rtn-W3-ah-seed1"), (2, "rtn-W3-ah-seed2")]:
    c = cell("pi05", "libero_spatial", name)
    if c:
        print(f"  seed {seed}: {c[0]:.1f}% [{c[1]:.0f},{c[2]:.0f}]  n={c[3]}")
