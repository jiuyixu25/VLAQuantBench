#!/usr/bin/env python
"""Turn the component sweep into the two tables the paper needs.

1. **Bit-width floor table** - per (model, component) the lowest weight-only
   bit-width whose closed-loop success stays within ``--tolerance`` of the
   model's baseline. This is the table the paper reports (its Table 3) without
   having run the bit-widths it names.

2. **Horizon sensitivity** - what each cell's success rate *would have been*
   under a shorter episode budget. Successful episodes record the step at which
   the task was solved, so any horizon shorter than the one actually used can be
   evaluated post hoc, at zero GPU cost. This quantifies how much of a
   quantization conclusion is an artefact of the episode budget - the per-model
   protocol constants differ by up to 4x (220 vs 800 steps), so a claim like
   "W3 is safe" is only meaningful relative to a stated horizon.

    python scripts/analyze_floors.py results/ --benchmark libero --suite libero_spatial
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlaquantbench.results import iter_runs  # noqa: E402
from vlaquantbench.stats import wilson_ci  # noqa: E402

BIT_ORDER = ["W2", "W3", "W4", "W8"]
COMPONENTS = ["ve", "mp", "llm", "ah"]


def load(root: Path, benchmark: str, suite: str | None):
    """(model, method, scope, preset) -> list of episode records."""
    cells: dict[tuple, list] = {}
    for _, h, eps in iter_runs(root):
        if h.benchmark != benchmark or (suite and h.suite != suite):
            continue
        cells[(h.model, h.method, h.scope, h.preset.upper())] = eps
    return cells


def rate(eps) -> tuple[float, float, float, int]:
    ci = wilson_ci(sum(1 for e in eps if e.success), len(eps))
    return ci.point, ci.low, ci.high, ci.n


def floor_table(cells, tolerance: float) -> str:
    models = sorted({k[0] for k in cells})
    out = ["", f"## Weight-only bit-width floor (success within {tolerance:.0%} of baseline)", ""]
    out.append("| model | baseline | " + " | ".join(f"{c} floor" for c in COMPONENTS) + " |")
    out.append("|" + "---|" * (len(COMPONENTS) + 2))
    for m in models:
        base = cells.get((m, "rtn", "e2e", "BASELINE"))
        if not base:
            continue
        b, *_ = rate(base)
        row = [m, f"{100 * b:.1f}"]
        for c in COMPONENTS:
            floor = "—"
            for bit in BIT_ORDER:  # lowest first
                eps = cells.get((m, "rtn", c, bit))
                if not eps:
                    continue
                p, *_ = rate(eps)
                if p >= b - tolerance:
                    floor = bit
                    break
            row.append(floor)
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    out.append("Floor = the *lowest* bit-width that holds; `—` means no swept bit-width held "
               "(or the cell was not run). Read with the full grid below - a floor hides "
               "non-monotonicity.")
    return "\n".join(out)


def grid_table(cells) -> str:
    models = sorted({k[0] for k in cells})
    presets = [p for p in ["BASELINE", "W2", "W3", "W4", "W8", "W4A4", "W4A8", "W8A8"]
               if any(k[3] == p for k in cells)]
    out = ["", "## Full component grid (success rate %, 95% Wilson interval)", ""]
    out.append("| model | scope | " + " | ".join(presets) + " |")
    out.append("|" + "---|" * (len(presets) + 2))
    for m in models:
        for scope in ["e2e", *COMPONENTS]:
            if not any(k[0] == m and k[2] == scope for k in cells):
                continue
            row = [m, scope]
            for p in presets:
                eps = cells.get((m, "rtn", scope, p))
                if not eps:
                    row.append("—")
                    continue
                pt, lo, hi, n = rate(eps)
                row.append(f"{100 * pt:.1f} [{100 * lo:.0f},{100 * hi:.0f}] n={n}")
            out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def methods_table(cells) -> str:
    """RTN vs the off-the-shelf LLM PTQ methods, same scope and episode budget."""
    rows = [(k, v) for k, v in cells.items() if k[2] == "llm"]
    if not rows:
        return ""
    methods = sorted({k[1] for k, _ in rows})
    presets = [p for p in BIT_ORDER if any(k[3] == p for k, _ in rows)]
    out = ["", "## LLM-only: off-the-shelf PTQ vs RTN (same protocol, same episodes)", ""]
    out.append("| model | method | " + " | ".join(presets) + " |")
    out.append("|" + "---|" * (len(presets) + 2))
    for m in sorted({k[0] for k, _ in rows}):
        for meth in methods:
            if not any(k[0] == m and k[1] == meth for k, _ in rows):
                continue
            row = [m, meth]
            for p in presets:
                eps = cells.get((m, meth, "llm", p))
                if not eps:
                    row.append("—")
                    continue
                pt, lo, hi, n = rate(eps)
                row.append(f"{100 * pt:.1f} [{100 * lo:.0f},{100 * hi:.0f}]")
            out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def horizon_table(cells, horizons: list[int]) -> str:
    """Success rate recomputed under shorter episode budgets (free, post hoc)."""
    out = ["", "## Horizon sensitivity (success rate % if the budget had been H steps)", ""]
    out.append("| model | scope | preset | actual | " + " | ".join(f"H={h}" for h in horizons) + " |")
    out.append("|" + "---|" * (len(horizons) + 4))
    interesting = [k for k in sorted(cells) if k[3] in ("BASELINE", "W3", "W4")]
    for k in interesting:
        eps = cells[k]
        if not eps:
            continue
        actual_max = max((e.extra or {}).get("max_steps", 0) for e in eps) or max(e.steps for e in eps)
        pt, *_ = rate(eps)
        row = [k[0], k[2], k[3], f"{100 * pt:.1f} (H={actual_max})"]
        for h in horizons:
            # an episode counts as solved under budget h iff it succeeded at or before step h
            n_ok = sum(1 for e in eps if e.success and e.steps <= h)
            row.append(f"{100 * n_ok / len(eps):.1f}")
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    out.append("Only *shorter* budgets are sound: an episode that failed within H steps might "
               "still have succeeded later, so longer budgets cannot be extrapolated.")
    return "\n".join(out)


def steps_shift(cells) -> str:
    """Does quantization make successful rollouts slower, even when they still succeed?"""
    out = ["", "## Time-to-success of the episodes that did succeed (median steps)", ""]
    models = sorted({k[0] for k in cells})
    presets = [p for p in ["BASELINE", "W2", "W3", "W4", "W4A8"] if any(k[3] == p for k in cells)]
    out.append("| model | scope | " + " | ".join(presets) + " |")
    out.append("|" + "---|" * (len(presets) + 2))
    for m in models:
        for scope in ["e2e", *COMPONENTS]:
            if not any(k[0] == m and k[2] == scope for k in cells):
                continue
            row = [m, scope]
            for p in presets:
                eps = cells.get((m, "rtn", scope, p))
                ok = [e.steps for e in (eps or []) if e.success]
                row.append(f"{int(np.median(ok))}" if ok else "—")
            out.append("| " + " | ".join(row) + " |")
    out.append("")
    out.append("A rise here with a flat success rate is the mechanism behind horizon sensitivity: "
               "the policy still solves the task but needs more steps, so a shorter budget would "
               "score it as a failure.")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="results")
    ap.add_argument("--benchmark", default="libero")
    ap.add_argument("--suite", default=None)
    ap.add_argument("--tolerance", type=float, default=0.10, help="allowed absolute drop vs baseline")
    ap.add_argument("--horizons", default="100,150,220,300")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cells = load(Path(args.root), args.benchmark, args.suite)
    if not cells:
        raise SystemExit(f"no {args.benchmark} runs found under {args.root}")
    horizons = [int(h) for h in args.horizons.split(",")]
    n_eps = sum(len(v) for v in cells.values())
    text = "\n".join([
        f"# {args.benchmark} / {args.suite or 'all suites'} — {len(cells)} cells, {n_eps} episodes",
        floor_table(cells, args.tolerance),
        grid_table(cells),
        methods_table(cells),
        steps_shift(cells),
        horizon_table(cells, horizons),
    ])
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
