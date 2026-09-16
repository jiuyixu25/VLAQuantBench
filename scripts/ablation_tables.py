#!/usr/bin/env python
"""LaTeX tables for the within-component ablations.

Three tables the paper does not currently have:

  T1  action-head layer decomposition, per model -- shows that the layer that
      matters is not the layer with the parameters;
  T2  pi0.5 subset lattice -- ten subsets of one component, demonstrating that
      component sensitivity is not additive in either direction;
  T3  weight-vs-activation isolation -- W4 weight-only / W4A8 / W8A8 / W4A4.

    python scripts/ablation_tables.py results/ --out tables/ablations.tex
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlaquantbench.stats import wilson_ci  # noqa: E402

PRETTY = {"pi05": r"$\pi_{0.5}$", "pi0": r"$\pi_0$",
          "openvla_oft": "OpenVLA-OFT", "xvla": "X-VLA", "openvla": "OpenVLA"}


def load(root: Path, suite: str) -> dict[tuple[str, str], dict]:
    cells = {}
    for path in root.rglob("*.jsonl"):
        header, n, ok = None, 0, 0
        with path.open() as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("kind") == "episode":
                    n += 1
                    ok += bool(rec.get("success"))
                elif header is None:
                    header = rec
        if not header or header.get("suite") != suite or n == 0:
            continue
        vals = [v for v in (header.get("quant_report") or {}).values() if isinstance(v, dict)]
        cells[(header.get("model"), path.stem)] = dict(
            n=n, ok=ok, ci=wilson_ci(ok, n),
            layers=sum(v.get("layers", 0) for v in vals),
            params=sum(v.get("params", 0) for v in vals),
        )
    return cells


def fmt(cell, base=None) -> str:
    if cell is None:
        return r"\multicolumn{1}{c}{--}"
    ci = cell["ci"]
    s = f"{100*ci.point:.1f}\\,\\tiny[{100*ci.low:.0f},{100*ci.high:.0f}]"
    if base is not None:
        d = 100 * (ci.point - base)
        s += f" & {d:+.1f}"
    return s


def table1(cells) -> str:
    """Action-head layer decomposition."""
    specs = [
        ("openvla_oft", "W4A8", "rtn-BASELINE-e2e", [
            ("\\texttt{fc1}", "ablate-W4A8-ah-only_fc1"),
            ("both residual blocks", "ablate-W4A8-ah-only_blocks"),
            ("\\textbf{\\texttt{fc2}} (outputs the action)", "ablate-W4A8-ah-only_fc2"),
            ("\\texttt{fc1}+\\texttt{fc2}", "ablate-W4A8-ah-fc1_fc2"),
            ("all four layers", "rtn-W4A8-ah"),
        ]),
        ("pi05", "W4A4", "rtn-BASELINE-e2e", [
            ("\\texttt{gate\\_proj}", "ablate-W4A4-ah-only_gate"),
            ("\\texttt{up\\_proj}", "ablate-W4A4-ah-only_up"),
            ("\\texttt{down\\_proj}", "ablate-W4A4-ah-only_down"),
            ("\\textbf{\\texttt{action\\_out\\_proj}} (outputs the action)",
             "ablate-W4A4-ah-only_action_out"),
            ("all 167 layers", "rtn-W4A4-ah"),
        ]),
        ("xvla", "W4A4", "rtn-BASELINE-e2e", [
            ("attention (\\texttt{qkv}+\\texttt{proj})", "ablate-W4A4-ah-only_attn"),
            ("\\texttt{mlp.fc1}", "ablate-W4A4-ah-only_fc1"),
            ("\\texttt{mlp.fc2}", "ablate-W4A4-ah-only_fc2"),
            ("MLP (\\texttt{fc1}+\\texttt{fc2})", "ablate-W4A4-ah-onlymlp"),
            ("all 98 layers", "rtn-W4A4-ah"),
        ]),
    ]
    out = [r"\begin{tabular}{llrrrr}", r"\toprule",
           r"model & quantized layers & \#layers & params & success (\%) & $\Delta$ \\",
           r"\midrule"]
    for model, preset, base_tag, rows in specs:
        b = cells.get((model, base_tag))
        if not b:
            continue
        out.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{PRETTY.get(model, model)}}}, "
                   f"{preset} on the action head, baseline "
                   f"{100*b['ci'].point:.1f}\\%}} \\\\")
        for label, tag in rows:
            c = cells.get((model, tag))
            if not c:
                out.append(f"  & {label} & \\multicolumn{{4}}{{c}}{{not run}} \\\\")
                continue
            out.append(f"  & {label} & {c['layers']} & {c['params']:,} & "
                       f"{fmt(c, b['ci'].point)} \\\\")
        out.append(r"\midrule")
    out[-1] = r"\bottomrule"
    out.append(r"\end{tabular}")
    return "\n".join(out)


def table2(cells) -> str:
    """pi0.5 subset lattice, ordered by layer count."""
    rows = [
        ("attention", "ablate-W4A4-ah-onlyself_attn"),
        ("adaRMS", "ablate-W4A4-ah-onlyadarms"),
        ("attention + adaRMS", "ablate-W4A4-ah-attn_adarms"),
        ("\\texttt{gate\\_proj}", "ablate-W4A4-ah-only_gate"),
        ("\\texttt{up\\_proj}", "ablate-W4A4-ah-only_up"),
        ("\\texttt{down\\_proj}", "ablate-W4A4-ah-only_down"),
        ("\\texttt{down\\_proj} + \\texttt{up\\_proj}", "ablate-W4A4-ah-down_up"),
        ("MLP", "ablate-W4A4-ah-onlymlp"),
        ("MLP + adaRMS", "ablate-W4A4-ah-mlp_plus_adarms"),
        ("\\texttt{down\\_proj} + attention", "ablate-W4A4-ah-down_attn"),
        ("\\texttt{down\\_proj} + attn + adaRMS", "ablate-W4A4-ah-down_attn_adarms"),
        ("everything except adaRMS", "ablate-W4A4-ah-noadarms"),
        ("everything except \\texttt{action\\_out\\_proj}", "ablate-W4A4-ah-no_action_out"),
        ("\\textbf{everything}", "rtn-W4A4-ah"),
    ]
    b = cells.get(("pi05", "rtn-BASELINE-e2e"))
    out = [r"\begin{tabular}{lrrr}", r"\toprule",
           r"quantized subset of the action head & \#layers & success (\%) & $\Delta$ \\",
           r"\midrule"]
    for label, tag in rows:
        c = cells.get(("pi05", tag))
        if not c:
            continue
        out.append(f"{label} & {c['layers']} & {fmt(c, b['ci'].point if b else None)} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def table3(cells) -> str:
    """Weight-only vs activation quantization, per model, on the action head."""
    presets = [("W4 (weight-only)", "rtn-W4-ah"), ("W8A8", "rtn-W8A8-ah"),
               ("W4A8", "rtn-W4A8-ah"), ("W4A6", "rtn-W4A6-ah"), ("W4A4", "rtn-W4A4-ah")]
    models = ["openvla_oft", "pi05", "pi0", "xvla"]
    out = [r"\begin{tabular}{l" + "r" * (len(presets) + 1) + "}", r"\toprule",
           "model & baseline & " + " & ".join(p for p, _ in presets) + r" \\", r"\midrule"]
    for m in models:
        b = cells.get((m, "rtn-BASELINE-e2e"))
        if not b:
            continue
        cs = [f"{100*b['ci'].point:.1f}"]
        for _, tag in presets:
            c = cells.get((m, tag))
            cs.append(f"{100*c['ci'].point:.1f}" if c else "--")
        out.append(f"{PRETTY.get(m, m)} & " + " & ".join(cs) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="results")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cells = load(Path(args.root), args.suite)
    text = "\n\n".join([
        "% T1: action-head layer decomposition",
        table1(cells),
        "% T2: pi0.5 subset lattice",
        table2(cells),
        "% T3: weight-only vs activation quantization",
        table3(cells),
    ])
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
