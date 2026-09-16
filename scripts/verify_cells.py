#!/usr/bin/env python
"""Independently re-verify every result cell, ignoring what the queue logs claim.

The queue scripts validate each cell inline, but that check has a hole: GNU
`grep -c` prints "0" *and* exits non-zero when nothing matches, so the idiom
`n=$(grep -c ... || echo 0)` yields "0\n0" for an empty file and the following
integer test errors out instead of failing. An empty cell can therefore be
logged as DONE. This script is the ground truth: it reads the files.

    python scripts/verify_cells.py results/ --min-episodes 180
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlaquantbench.stats import wilson_ci  # noqa: E402


def read(path: Path):
    header, eps = None, []
    with path.open() as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") == "episode":
                eps.append(rec)
            elif header is None:
                header = rec
    return header or {}, eps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="results")
    ap.add_argument("--min-episodes", type=int, default=180)
    args = ap.parse_args()

    rows, bad = [], []
    for path in sorted(Path(args.root).rglob("*.jsonl")):
        header, eps = read(path)
        n = len(eps)
        report = header.get("quant_report") or {}
        vals = [v for v in report.values() if isinstance(v, dict)]
        layers = sum(v.get("layers", 0) for v in vals)
        params = sum(v.get("params", 0) for v in vals)
        preset = (header.get("preset") or "?").upper()
        ok = sum(1 for e in eps if e.get("success"))
        opts = header.get("extra") or {}
        only = opts.get("only") or []
        excl = opts.get("exclude") or []
        steps = ((opts.get("adapter_options") or {}).get("num_inference_steps"))
        if n == 0:
            bad.append((path, "ZERO EPISODES"))
            continue
        if n < args.min_episodes:
            bad.append((path, f"only {n} episodes"))
        method = header.get("method") or "rtn"
        # AWQ/NF4/int8/SmoothQuant write a flat quant_report ({method, w_bits, ...})
        # rather than {component: {layers, params}}, so the layer check applies to RTN only.
        if method == "rtn" and preset != "BASELINE" and layers == 0:
            bad.append((path, "RTN cell quantized ZERO layers"))
        if method != "rtn" and not report:
            bad.append((path, f"method {method!r} recorded no quantization at all"))
        ci = wilson_ci(ok, n)
        rows.append(dict(
            model=header.get("model", "?"), suite=header.get("suite", "?"),
            preset=preset, scope=header.get("scope", "?"),
            tag=path.stem, n=n, ok=ok, pt=ci.point, lo=ci.low, hi=ci.high,
            layers=layers, params=params,
            only=" ".join(only), exclude=" ".join(excl), steps=steps,
        ))

    w = max((len(r["tag"]) for r in rows), default=10)
    print(f"{'model':12s} {'suite':15s} {'tag':{w}s} {'n':>4s} {'succ':>7s} "
          f"{'95% CI':>13s} {'layers':>7s} {'params':>13s} steps")
    for r in sorted(rows, key=lambda r: (r["model"], r["suite"], r["tag"])):
        print(f"{r['model']:12s} {r['suite']:15s} {r['tag']:{w}s} {r['n']:4d} "
              f"{100*r['pt']:6.1f}% [{100*r['lo']:4.1f},{100*r['hi']:5.1f}] "
              f"{r['layers']:7d} {r['params']:13,d} {r['steps'] if r['steps'] else ''}")

    print()
    if bad:
        print(f"!! {len(bad)} cell(s) FAILED verification -- do not use:")
        for p, why in bad:
            print(f"   {p}: {why}")
    else:
        print(f"all {len(rows)} cells verified: >= {args.min_episodes} episodes, "
              f"and every non-baseline cell quantized at least one layer")


if __name__ == "__main__":
    main()
