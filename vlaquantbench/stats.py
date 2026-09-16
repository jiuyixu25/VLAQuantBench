"""Uncertainty estimates for closed-loop metrics.

* success rates   -> Wilson score interval (binomial)
* CALVIN Avg. Len / VLABench progress score -> percentile bootstrap
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = ["Interval", "wilson_ci", "bootstrap_ci", "success_summary", "mean_summary"]


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float
    n: int

    def fmt(self, scale: float = 1.0, digits: int = 1) -> str:
        return f"{self.point * scale:.{digits}f} [{self.low * scale:.{digits}f}, {self.high * scale:.{digits}f}]"

    def as_dict(self) -> dict[str, float | int]:
        return {"point": self.point, "low": self.low, "high": self.high, "n": self.n}


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Interval:
    """Wilson score interval for ``k`` successes out of ``n`` trials (95% by default)."""
    if n <= 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    low, high = max(0.0, centre - half), min(1.0, centre + half)
    if low < 1e-12:
        low = 0.0
    if high > 1 - 1e-12:
        high = 1.0
    return Interval(p, low, high, n)


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap CI of the mean."""
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(float(arr.mean()), float(lo), float(hi), n)


def success_summary(successes: Sequence[bool | int]) -> Interval:
    k = int(sum(1 for s in successes if s))
    return wilson_ci(k, len(successes))


def mean_summary(values: Sequence[float], **kw) -> Interval:
    return bootstrap_ci(values, **kw)
