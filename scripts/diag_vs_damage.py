#!/usr/bin/env python
"""Does a calibration-time statistic predict closed-loop damage?

Pairs every single-group ablation cell with the activation statistics of the
layers that cell actually quantized, and reports the rank correlation between
each statistic and the measured drop in success rate.

The layer set is not re-derived here: it is read from each result file's
recorded `--only` patterns, so the layers scored are by construction the layers
that were quantized. (An earlier hand-written pattern list is exactly how a
cell that quantized nothing got mistaken for a cell that showed no effect.)

    python scripts/diag_vs_damage.py --diagnostics diagnostics --results results
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlaquantbench.stats import wilson_ci  # noqa: E402

STATS = ["act_kurtosis", "out_rel_err", "act_rel_err", "act_dispersion"]


def read_cell(path: Path):
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
    return header, n, ok


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    x, y = ranks(a), ranks(b)
    mx, my = st.mean(x), st.mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = (sum((xi - mx) ** 2 for xi in x) * sum((yi - my) ** 2 for yi in y)) ** 0.5
    return num / den if den else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostics", default="diagnostics")
    ap.add_argument("--results", default="results")
    ap.add_argument("--suite", default="libero_spatial")
    args = ap.parse_args()

    diags = {}
    for p in Path(args.diagnostics).glob("*.json"):
        d = json.loads(p.read_text())
        diags[(d["model"], d["scope"], d["preset"])] = d["layers"]
    if not diags:
        raise SystemExit(f"no diagnostics found in {args.diagnostics}/")

    baselines = {}
    cells = []
    for path in Path(args.results).rglob("*.jsonl"):
        header, n, ok = read_cell(path)
        if not header or header.get("suite") != args.suite or n == 0:
            continue
        model, preset = header.get("model"), (header.get("preset") or "").upper()
        if preset == "BASELINE" and header.get("scope") == "e2e" and path.stem == "rtn-BASELINE-e2e":
            baselines[model] = wilson_ci(ok, n).point
        only = (header.get("extra") or {}).get("only") or []
        # cells run with a non-default inference config have their own control
        # and must not be differenced against the default baseline
        if path.stem.startswith("steps"):
            continue
        if only and header.get("method", "rtn") in (None, "rtn"):
            cells.append((model, preset, header.get("scope"), path.stem, only, n, ok))

    rows = []
    for model, preset, scope, tag, only, n, ok in cells:
        layers = diags.get((model, scope, preset))
        if layers is None or model not in baselines:
            continue
        sel = [v for k, v in layers.items() if any(fnmatch.fnmatchcase(k, p) for p in only)]
        if not sel:
            print(f"  (skipping {model}/{tag}: its --only patterns match no probed layer)")
            continue
        damage = 100 * (baselines[model] - wilson_ci(ok, n).point)
        rows.append(dict(model=model, tag=tag, preset=preset, n_layers=len(sel),
                         damage=damage, **{s: st.mean(x[s] for x in sel) for s in STATS}))

    if not rows:
        raise SystemExit("no ablation cell could be paired with a diagnostic")

    rows.sort(key=lambda r: -r["damage"])
    w = max(len(r["tag"]) for r in rows)
    print(f"\n{'model':12s} {'cell':{w}s} {'lyr':>4s} {'damage':>7s} " +
          " ".join(f"{s:>14s}" for s in STATS))
    for r in rows:
        print(f"{r['model']:12s} {r['tag']:{w}s} {r['n_layers']:4d} {r['damage']:7.1f} " +
              " ".join(f"{r[s]:14.4f}" for s in STATS))

    def report(label, sub):
        if len(sub) < 4:
            print(f"  {label:28s} (only {len(sub)} cells -- not reported)")
            return
        d = [r["damage"] for r in sub]
        cs = "  ".join(f"{s.split(chr(95))[-1]}={spearman(d, [r[s] for r in sub]):+.3f}" for s in STATS)
        print(f"  {label:28s} n={len(sub):2d}  {cs}")

    print("\nrank correlation of each statistic with measured damage:")
    print("  (kurtosis / rel_err = out_rel_err / err = act_rel_err / dispersion)")
    for m in sorted({r["model"] for r in rows}):
        report(f"within {m}", [r for r in rows if r["model"] == m])
    report("pooled across models", rows)
    print("\nA statistic that ranks layers correctly *within* a model but not across\n"
          "models is a per-model triage tool, not a transferable law.")


if __name__ == "__main__":
    main()
