"""Latency / memory measurement helpers for the deployment experiment.

Latency is measured around *one action-prediction call* (whatever the policy
does to turn an observation into an action, including action-chunk decoding)
with explicit CUDA synchronisation; the first ``warmup`` calls are discarded.
Memory is the CUDA peak allocation measured by the caching allocator.
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

__all__ = ["LatencyMeter", "reset_peak_memory", "peak_memory_gb", "module_memory_gb"]


@dataclass
class LatencyMeter:
    warmup: int = 5
    samples: list[float] = field(default_factory=list)
    _seen: int = 0

    @contextmanager
    def measure(self):
        sync = torch.cuda.is_available()
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if sync:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            self._seen += 1
            if self._seen > self.warmup:
                self.samples.append(dt)

    @property
    def n(self) -> int:
        return len(self.samples)

    def mean(self) -> float | None:
        return statistics.fmean(self.samples) if self.samples else None

    def median(self) -> float | None:
        return statistics.median(self.samples) if self.samples else None

    def p95(self) -> float | None:
        if not self.samples:
            return None
        s = sorted(self.samples)
        return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]

    def summary(self) -> dict[str, float | int | None]:
        return {"n": self.n, "mean_s": self.mean(), "median_s": self.median(), "p95_s": self.p95()}


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1024**3


def module_memory_gb(module: torch.nn.Module) -> float:
    """Bytes held by parameters + buffers of ``module`` (as stored, e.g. packed int4)."""
    total = 0
    for p in module.parameters():
        total += p.numel() * p.element_size()
    for b in module.buffers():
        total += b.numel() * b.element_size()
    return total / 1024**3
