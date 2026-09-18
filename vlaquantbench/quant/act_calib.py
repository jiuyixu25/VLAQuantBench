"""SmoothQuant-style activation calibration for closed-loop VLA policies.

The main sweep deliberately uses calibration-free per-token absmax activation
quantization. This module is the controlled *counter*-condition: a lightweight,
rollout-based calibration that answers whether the W4A4 collapse is intrinsic
to VLA activations or an artifact of naive dynamic scaling — and, applied
component-wise, *which* component's activations are the ones that need it.

Procedure (no retraining, no hyperparameter search):

1. Run the un-quantized policy through a few closed-loop episodes and record,
   at the input of every target ``nn.Linear``, per-channel absolute maxima and
   a high quantile of per-channel magnitude.
2. From those statistics derive, per layer,
   * a per-channel smoothing vector ``s_j = max(|x_j|)^α / max(|W_:j|)^(1-α)``
     (SmoothQuant, α=0.5), and
   * a per-channel clipping threshold ``c_j`` = the recorded quantile of the
     *smoothed* activation, which absorbs the rare residual spikes.
3. At quantization time the smoothing is folded into the weights
   (``W ← W · diag(s)`` before weight RTN), and the activation hook computes
   ``Q_a(clamp(x / s, ±c))`` instead of ``Q_a(x)``. Since
   ``(W diag(s)) (diag(s)^{-1} x) = W x``, the transform is exact at full
   precision; only the quantization grids see the redistributed ranges.

Statistics are collected once per (model, checkpoint, suite, component set)
and cached on disk, so the six cells of the calibration experiment share one
collection run.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import torch
from torch import nn

log = logging.getLogger(__name__)

__all__ = ["LayerCalib", "collect_stats", "build_calibration", "cache_path"]

_EPS = 1e-5


@dataclass
class LayerCalib:
    """Per-layer calibration: smoothing vector and post-smoothing clip, both (in_features,)."""

    smooth: torch.Tensor
    clip: torch.Tensor

    def to(self, device, dtype=torch.float32) -> "LayerCalib":
        return LayerCalib(self.smooth.to(device, dtype), self.clip.to(device, dtype))


class _Observer:
    """Accumulates per-channel |x| statistics for one linear layer."""

    def __init__(self) -> None:
        self.absmax: torch.Tensor | None = None   # (C,) running max over all tokens
        self.q_hi: torch.Tensor | None = None     # (C,) running max of per-call p99.9
        self.calls = 0

    def __call__(self, module: nn.Module, args: tuple):
        if not args or not isinstance(args[0], torch.Tensor):
            return None
        x = args[0]
        if not x.is_floating_point() or x.shape[-1] != module.weight.shape[1]:
            return None
        with torch.no_grad():
            xa = x.reshape(-1, x.shape[-1]).abs().float()
            if xa.numel() == 0:
                return None
            amax = xa.amax(dim=0)
            # per-call high quantiles per channel; running max across calls
            if xa.shape[0] > 1:
                # the q tensor must live on the input's device (torch < 2.3 raises otherwise), and
                # torch.quantile refuses inputs above 2**24 elements, so very long token batches are
                # subsampled on a fixed stride -- the statistic is a per-channel high quantile, which
                # a uniform subsample estimates without bias.
                xq = xa
                limit = 2 ** 24
                if xq.numel() > limit:
                    stride = (xq.numel() + limit - 1) // limit
                    xq = xq[::stride]
                qs = torch.quantile(xq, torch.tensor([0.99, 0.999], device=xq.device, dtype=xq.dtype), dim=0)
                q99, q999 = qs[0], qs[1]
            else:
                q99 = q999 = amax
            self.absmax = amax if self.absmax is None else torch.maximum(self.absmax, amax)
            self.q_hi = q999 if self.q_hi is None else torch.maximum(self.q_hi, q999)
            self.q_99 = q99 if getattr(self, "q_99", None) is None else torch.maximum(self.q_99, q99)
            self.calls += 1
        return None


@torch.no_grad()
def collect_stats(
    layers: Mapping[str, nn.Linear],
    run_episodes,
) -> dict[str, dict[str, torch.Tensor]]:
    """Observe ``layers`` while ``run_episodes()`` drives the un-quantized policy.

    Returns ``{name: {"absmax": (C,), "q_hi": (C,)}}`` on CPU. Layers that were
    never called (0 observations) are dropped with a warning — calibrating them
    would install garbage.
    """
    observers: dict[str, _Observer] = {}
    handles = []
    for name, mod in layers.items():
        obs = _Observer()
        observers[name] = obs
        handles.append(mod.register_forward_pre_hook(obs))
    try:
        run_episodes()
    finally:
        for h in handles:
            h.remove()
    out: dict[str, dict[str, torch.Tensor]] = {}
    for name, obs in observers.items():
        if obs.calls == 0 or obs.absmax is None:
            log.warning("act-calib: layer %s was never called during calibration; skipping", name)
            continue
        out[name] = {"absmax": obs.absmax.cpu(), "q_hi": obs.q_hi.cpu(),
                     "q_99": getattr(obs, "q_99", obs.q_hi).cpu(), "calls": torch.tensor(obs.calls)}
    log.info("act-calib: collected statistics for %d/%d layers", len(out), len(layers))
    return out


@torch.no_grad()
def build_calibration(
    layers: Mapping[str, nn.Linear],
    stats: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    alpha: float = 0.5,
    clip_quantile: float = 0.999,
) -> dict[str, LayerCalib]:
    """Turn raw statistics into per-layer smoothing + clip vectors."""
    calib: dict[str, LayerCalib] = {}
    for name, mod in layers.items():
        st = stats.get(name)
        if st is None:
            continue
        absmax = st["absmax"].float().clamp_min(_EPS)
        w_absmax = mod.weight.detach().abs().amax(dim=0).float().cpu().clamp_min(_EPS)
        smooth = (absmax.pow(alpha) / w_absmax.pow(1.0 - alpha)).clamp_min(_EPS)
        q = st.get("q_99") if clip_quantile <= 0.99 else st["q_hi"]
        if q is None:
            q = st["q_hi"]
        clip = (q.float() / smooth).clamp_min(_EPS)
        calib[name] = LayerCalib(smooth=smooth, clip=clip)
    return calib


def cache_path(model: str, checkpoint: str, suite: str, components: Iterable[str], episodes: int, *,
               episodes_from: int = 0, tasks: int = 1, seed: int = 0, commit: str | None = None) -> Path:
    """Cache file for collected statistics. The key covers every input that changes them:
    which init states were rolled out (``episodes_from``, ``episodes``, ``tasks``), the RNG seed
    of the un-quantized policy, and the code revision -- so a cache entry can never be reused
    across a change that would have produced different statistics."""
    key = (f"{model}|{checkpoint}|{suite}|{'+'.join(sorted(components))}|{episodes}"
           f"|from={episodes_from}|tasks={tasks}|seed={seed}|commit={commit or 'unknown'}")
    tag = hashlib.sha1(key.encode()).hexdigest()[:12]
    root = Path.home() / ".cache" / "vlaquantbench" / "actcalib"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{tag}.pt"
