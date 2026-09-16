"""Aggregate JSONL runs into the paper's tables (Markdown / LaTeX / CSV)."""

from __future__ import annotations

import csv
import io
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .results import EpisodeRecord, RunHeader, iter_runs
from .stats import Interval, bootstrap_ci, wilson_ci

__all__ = ["Cell", "collect", "pivot", "render_markdown", "render_latex", "render_csv", "PRESET_ORDER"]

PRESET_ORDER = ["BASELINE", "W2", "W3", "W4", "W8", "W4A4", "W4A8", "W8A8"]
LIBERO_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

# metric per benchmark: (field on EpisodeRecord, display scale, aggregate kind)
METRIC = {
    "libero": ("success", 100.0, "binomial", "SR (%)"),
    "simpler": ("success", 100.0, "binomial", "SR (%)"),
    "calvin": ("subtasks_completed", 1.0, "mean", "Avg. Len."),
    "vlabench": ("progress_score", 100.0, "mean", "Avg. PS"),
}


@dataclass
class Cell:
    interval: Interval
    scale: float
    n_runs: int = 1

    def fmt(self, ci: bool = False, digits: int = 1) -> str:
        if ci:
            return self.interval.fmt(self.scale, digits)
        return f"{self.interval.point * self.scale:.{digits}f}"


def _metric(header: RunHeader, eps: list[EpisodeRecord]) -> tuple[Interval, float] | None:
    spec = METRIC.get(header.benchmark)
    if spec is None or not eps:
        return None
    field, scale, kind, _ = spec
    vals = [getattr(e, field) for e in eps]
    if kind == "binomial":
        return wilson_ci(int(sum(bool(v) for v in vals)), len(vals)), scale
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return None
    return bootstrap_ci(vals), scale


def _simpler_official(eps: list[EpisodeRecord]) -> Interval | None:
    """SIMPLER's official aggregation: equal-weight mean over *variants* within a
    family, then over families; camera variants are excluded from the VA score
    (``tools/calc_metrics_evaluation_videos.py``). Straight episode averaging
    would over-weight the variants that ship more episodes.

    The point estimate is exactly the official one. The interval is only an
    approximation: it is the Wilson interval of an equivalent binomial with the
    same rate over the scored episodes, which ignores the between-variant
    variance the weighting introduces. Use the per-variant rows (``--rows
    benchmark,suite,model,scope`` on the un-aggregated suite) for an exact
    uncertainty statement."""
    per_variant: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for e in eps:
        if not e.extra.get("scored", True):
            continue
        per_variant[(e.extra.get("family", "?"), e.extra.get("variant", "?"))].append(bool(e.success))
    if not per_variant:
        return None
    per_family: dict[str, list[float]] = defaultdict(list)
    for (family, _), successes in per_variant.items():
        per_family[family].append(float(np.mean(successes)))
    families = [float(np.mean(v)) for v in per_family.values()]
    n = sum(len(v) for v in per_variant.values())
    point = float(np.mean(families))
    lo, hi = wilson_ci(int(round(point * n)), n).low, wilson_ci(int(round(point * n)), n).high
    return Interval(point, lo, hi, n)


def collect(root: str | Path) -> dict[tuple, Cell]:
    """Key: (benchmark, suite, model, method, scope, preset, lm_head) -> Cell."""
    cells: dict[tuple, Cell] = {}
    for _, header, eps in iter_runs(root):
        m = _metric(header, eps)
        if m is None:
            continue
        key = (header.benchmark, header.suite, header.model, header.method, header.scope, header.preset.upper(), header.quantize_lm_head)
        cells[key] = Cell(m[0], m[1])
        if header.benchmark == "simpler":
            official = _simpler_official(eps)
            if official is not None:
                cells[(header.benchmark, header.suite + "_official", *key[2:])] = Cell(official, 100.0)
    # LIBERO: add the 4-suite average (mean of suite-level rates, as reported in the paper)
    groups: dict[tuple, dict[str, Cell]] = defaultdict(dict)
    for key, cell in cells.items():
        if key[0] == "libero" and key[1] in LIBERO_SUITES:
            groups[(key[0], key[2], key[3], key[4], key[5], key[6])][key[1]] = cell
    for (bench, model, method, scope, preset, lm), per_suite in groups.items():
        if len(per_suite) == len(LIBERO_SUITES):
            pts = [c.interval.point for c in per_suite.values()]
            lows = [c.interval.low for c in per_suite.values()]
            highs = [c.interval.high for c in per_suite.values()]
            n = sum(c.interval.n for c in per_suite.values())
            cells[(bench, "libero_avg4", model, method, scope, preset, lm)] = Cell(
                Interval(float(np.mean(pts)), float(np.mean(lows)), float(np.mean(highs)), n), 100.0, len(per_suite)
            )
    return cells


def pivot(
    cells: dict[tuple, Cell],
    *,
    rows: Iterable[str] = ("benchmark", "suite", "model", "method", "scope"),
    presets: list[str] | None = None,
) -> tuple[list[str], list[str], dict[tuple, dict[str, Cell]]]:
    """Return (row_fields, preset_columns, table) with table[row_key][preset] = Cell."""
    fields = list(rows)
    idx = {"benchmark": 0, "suite": 1, "model": 2, "method": 3, "scope": 4, "preset": 5, "lm_head": 6}
    table: dict[tuple, dict[str, Cell]] = defaultdict(dict)
    seen_presets: set[str] = set()
    for key, cell in cells.items():
        row_key = tuple(key[idx[f]] for f in fields)
        table[row_key][key[5]] = cell
        seen_presets.add(key[5])
    cols = [p for p in (presets or PRESET_ORDER) if p in seen_presets]
    cols += sorted(seen_presets - set(cols))
    return fields, cols, dict(sorted(table.items()))


def render_markdown(fields: list[str], cols: list[str], table: dict[tuple, dict[str, Cell]], *, ci: bool = False) -> str:
    out = io.StringIO()
    out.write("| " + " | ".join(fields + cols) + " |\n")
    out.write("|" + "---|" * (len(fields) + len(cols)) + "\n")
    for row_key, per in table.items():
        vals = [per[c].fmt(ci=ci) if c in per else "—" for c in cols]
        out.write("| " + " | ".join([str(x) for x in row_key] + vals) + " |\n")
    return out.getvalue()


def render_latex(fields: list[str], cols: list[str], table: dict[tuple, dict[str, Cell]], *, ci: bool = False, caption: str = "") -> str:
    def esc(s: Any) -> str:
        return str(s).replace("_", r"\_")

    out = io.StringIO()
    out.write("\\begin{table}[t]\n\\centering\n\\small\n")
    out.write("\\begin{tabular}{" + "l" * len(fields) + "c" * len(cols) + "}\n\\toprule\n")
    out.write(" & ".join([esc(f) for f in fields] + [esc(c) for c in cols]) + " \\\\\n\\midrule\n")
    for row_key, per in table.items():
        vals = [per[c].fmt(ci=ci) if c in per else "--" for c in cols]
        out.write(" & ".join([esc(x) for x in row_key] + vals) + " \\\\\n")
    out.write("\\bottomrule\n\\end{tabular}\n")
    if caption:
        out.write(f"\\caption{{{caption}}}\n")
    out.write("\\end{table}\n")
    return out.getvalue()


def render_csv(fields: list[str], cols: list[str], table: dict[tuple, dict[str, Cell]]) -> str:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(fields + [f"{c}" for c in cols] + [f"{c}_low" for c in cols] + [f"{c}_high" for c in cols] + [f"{c}_n" for c in cols])
    for row_key, per in table.items():
        pts = [f"{per[c].interval.point * per[c].scale:.3f}" if c in per else "" for c in cols]
        lows = [f"{per[c].interval.low * per[c].scale:.3f}" if c in per else "" for c in cols]
        highs = [f"{per[c].interval.high * per[c].scale:.3f}" if c in per else "" for c in cols]
        ns = [str(per[c].interval.n) if c in per else "" for c in cols]
        w.writerow(list(row_key) + pts + lows + highs + ns)
    return out.getvalue()


def to_json(cells: dict[tuple, Cell]) -> str:
    payload = [
        {
            "benchmark": k[0], "suite": k[1], "model": k[2], "method": k[3], "scope": k[4], "preset": k[5],
            "quantize_lm_head": k[6], **{kk: vv for kk, vv in c.interval.as_dict().items()}, "scale": c.scale,
        }
        for k, c in cells.items()
    ]
    return json.dumps(payload, indent=1)
