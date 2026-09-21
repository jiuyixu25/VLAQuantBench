#!/usr/bin/env python
"""Task-clustered uncertainty for closed-loop cells.

Each LIBERO cell is 10 tasks x 20 episodes. The episode-level Wilson interval treats the 200
episodes as i.i.d.; this script adds (a) a cluster bootstrap that resamples *tasks* with
replacement (and, within each task, episodes), and (b) for a pair of cells, a task-paired
bootstrap of the success difference (the same tasks are resampled in both cells). Reads the
JSONL records directly.

    python scripts/clustered_ci.py --cell results/libero/libero_goal/openvla_oft/rtn-W4-e2e.jsonl
    python scripts/clustered_ci.py --cell A.jsonl --ref B.jsonl     # paired difference A - B
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path):
    per = defaultdict(list)
    for line in open(path):
        if '"kind": "episode"' in line:
            d = json.loads(line)
            per[int(d["task_id"])].append(int(bool(d.get("success"))))
    return {t: np.array(v) for t, v in sorted(per.items())}


def wilson(ok, n, z=1.96):
    p = ok / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def cluster_boot(cell, B, rng, ref=None):
    tasks = sorted(cell)
    T = len(tasks)
    out = []
    for _ in range(B):
        pick = rng.integers(0, T, T)
        vals = []
        for i in pick:
            t = tasks[i]
            a = cell[t]; a = a[rng.integers(0, len(a), len(a))]
            if ref is None:
                vals.append(a.mean())
            else:
                b = ref[t]; b = b[rng.integers(0, len(b), len(b))]
                vals.append(a.mean() - b.mean())
        out.append(100 * np.mean(vals))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True); ap.add_argument("--ref", default=None)
    ap.add_argument("--B", type=int, default=5000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    cell = load(a.cell)
    n = sum(len(v) for v in cell.values()); ok = sum(int(v.sum()) for v in cell.values())
    lo, hi = wilson(ok, n)
    boot = cluster_boot(cell, a.B, rng)
    per_task = " ".join(f"{100*v.mean():.0f}" for v in cell.values())
    print(f"{Path(a.cell).name}: SR {100*ok/n:.1f}  Wilson [{lo:.1f},{hi:.1f}]  task-cluster bootstrap [{np.percentile(boot,2.5):.1f},{np.percentile(boot,97.5):.1f}]  per-task: {per_task}")
    if a.ref:
        ref = load(a.ref)
        common = sorted(set(cell) & set(ref))
        cell = {t: cell[t] for t in common}; ref = {t: ref[t] for t in common}
        d = cluster_boot(cell, a.B, rng, ref=ref)
        n2 = sum(len(v) for v in ref.values()); ok2 = sum(int(v.sum()) for v in ref.values())
        diff = 100 * (ok / n - ok2 / n2)
        print(f"  vs {Path(a.ref).name}: diff {diff:+.1f} pp  task-paired bootstrap [{np.percentile(d,2.5):+.1f},{np.percentile(d,97.5):+.1f}]  "
              f"P(diff<0)={np.mean(d<0):.3f}")


if __name__ == "__main__":
    main()
