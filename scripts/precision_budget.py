#!/usr/bin/env python
"""Per-component activation-bit budget -- the table the paper promises and never runs.

For each (model, component) it reports the closed-loop success rate at every measured
activation width with 4-bit weights, and the *budget*: the lowest activation width whose
interval still reaches within `--tolerance` of the model's baseline.

    python scripts/precision_budget.py results/ --suite libero_spatial
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlaquantbench.stats import wilson_ci  # noqa: E402

WIDTHS = ["W4A4", "W4A6", "W4A8", "W4"]          # harshest first; W4 = no activation quant
LABEL = {"W4A4": "A4", "W4A6": "A6", "W4A8": "A8", "W4": "none"}
COMPONENTS = ["ve", "llm", "ah"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="results")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tolerance", type=float, default=5.0, help="allowed absolute drop, points")
    args = ap.parse_args()

    cells, base = {}, {}
    for path in Path(args.root).rglob("*.jsonl"):
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
        if not header or header.get("suite") != args.suite or n == 0:
            continue
        if (header.get("extra") or {}).get("only") or (header.get("method") or "rtn") != "rtn":
            continue
        m, pre, sc = header.get("model"), (header.get("preset") or "").upper(), header.get("scope")
        if pre == "BASELINE" and sc == "e2e":
            base[m] = (100 * wilson_ci(ok, n).point, n)
        else:
            cells[(m, sc, pre)] = (100 * wilson_ci(ok, n).point, n)

    print(f"# Activation-bit budget, {args.suite}, 4-bit weights throughout")
    print(f"# budget = lowest activation width within {args.tolerance:.0f} points of baseline\n")
    print(f"| model | baseline | component | " + " | ".join(LABEL[w] for w in WIDTHS) + " | budget |")
    print("|" + "---|" * (len(WIDTHS) + 4))
    for m in sorted(base):
        b, _ = base[m]
        for c in COMPONENTS:
            row, budget = [], "—"
            for w in WIDTHS:                       # harshest first, so the first pass wins
                cell = cells.get((m, c, w))
                if cell is None:
                    row.append("—")
                    continue
                pt, n = cell
                row.append(f"{pt:.1f}" + (f"<sub>n={n}</sub>" if n < 190 else ""))
                if budget == "—" and pt >= b - args.tolerance:
                    budget = LABEL[w]
            print(f"| {m} | {b:.1f} | {c} | " + " | ".join(row) + f" | **{budget}** |")
    print("\n`none` = the component tolerates 4-bit weights but no activation quantization at "
          "any width measured. A budget is a floor, not a guarantee: read it with the full row, "
          "since the columns are not always monotone.")


if __name__ == "__main__":
    main()
