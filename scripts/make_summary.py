#!/usr/bin/env python
"""Regenerate results/summary.csv from the shipped records (one row per accuracy cell).

Excludes profiling runs (results/latency) and paired-observation fidelity files (results/fidelity).
Wilson 95% intervals on the pooled episodes; CALVIN adds mean completed subtasks, VLABench mean progress.

    python scripts/make_summary.py --root results --out results/summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def wilson(ok, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = ok / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="results"); ap.add_argument("--out", default="results/summary.csv")
    a = ap.parse_args()
    rows = []
    for p in sorted(Path(a.root).rglob("*.jsonl")):
        parts = p.relative_to(a.root).parts
        if parts[0] in ("latency", "fidelity", "replay"):
            continue
        header = None; n = ok = 0; subtasks = []; progress = []
        for line in p.open():
            line = line.strip().strip("\x00")
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("kind") == "header":
                header = d; continue
            if d.get("kind") != "episode":
                continue
            n += 1; ok += bool(d.get("success"))
            if d.get("subtasks_completed") is not None:
                subtasks.append(d["subtasks_completed"])
            if d.get("progress_score") is not None:
                progress.append(d["progress_score"])
        if header is None or n == 0:
            continue
        lo, hi = wilson(ok, n)
        rows.append({
            "benchmark": header["benchmark"], "suite": header["suite"], "model": header["model"], "cell": p.stem,
            "method": header["method"], "preset": header["preset"], "scope": header["scope"], "seed": header["seed"],
            "checkpoint": header["checkpoint"], "episodes": n, "successes": ok, "success_rate": round(100 * ok / n, 2),
            "ci_low": round(lo, 2), "ci_high": round(hi, 2),
            "mean_subtasks": round(sum(subtasks) / len(subtasks), 3) if subtasks else "",
            "mean_progress": round(100 * sum(progress) / len(progress), 2) if progress else "",
        })
    fields = ["benchmark", "suite", "model", "cell", "method", "preset", "scope", "seed", "checkpoint", "episodes", "successes",
              "success_rate", "ci_low", "ci_high", "mean_subtasks", "mean_progress"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} cells, {sum(r['episodes'] for r in rows)} episodes -> {a.out}")


if __name__ == "__main__":
    main()
