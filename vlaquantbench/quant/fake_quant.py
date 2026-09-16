"""Round-to-nearest (RTN) fake quantization primitives.

Everything here is *simulated* quantization: tensors are mapped to a
``b``-bit integer grid and immediately dequantized back to floating point, so
the result has at most ``2**b`` distinct values per scale group but keeps the
original dtype. This is the standard way to study the *accuracy* effect of
PTQ independently of kernel availability (see ``vlaquantbench.methods`` for
real-kernel execution).

Conventions (identical to the paper's Appendix A):

* symmetric grid:  ``q_min = -2**(b-1)``, ``q_max = 2**(b-1) - 1``,
  ``s = max|x| / q_max``, zero-point ``z = 0``.
* asymmetric grid: ``[0, 2**b - 1]`` with ``s = (max - min) / (2**b - 1)`` and
  ``z = round(-min / s)``, where the range is widened to include 0 so that the
  zero-point is an integer on the grid.
* weights are grouped along the *input* dimension (``dim=1`` of a
  ``[out, in]`` matrix). ``per_channel`` means one group per output channel,
  ``per_group`` means contiguous groups of ``group_size`` input features. If
  ``in_features`` is not divisible by ``group_size`` the *last group is
  shorter*; we never zero-pad, so padding can not pollute a group's range.
* activations are quantized per token (one scale per row of the
  ``[..., in]`` activation matrix), symmetric, computed on the fly from the
  instantaneous tensor (``calibration-free dynamic quantization``).

All arithmetic is carried out in float32 and the result is cast back to the
input dtype.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

__all__ = [
    "WeightQuantSpec",
    "ActQuantSpec",
    "fake_quant_symmetric",
    "fake_quant_asymmetric",
    "fake_quant_weight",
    "fake_quant_act_per_token",
    "fake_quant_act_per_tensor",
    "weight_quant_error",
]

_EPS = 1e-8


@dataclass(frozen=True)
class WeightQuantSpec:
    """How to quantize a weight matrix."""

    bits: int
    granularity: Literal["per_channel", "per_group", "per_tensor"] = "per_channel"
    group_size: int | None = None
    symmetric: bool = True

    def __post_init__(self) -> None:
        if not (1 <= self.bits <= 16):
            raise ValueError(f"weight bits must be in [1, 16], got {self.bits}")
        if self.granularity == "per_group":
            if not self.group_size or self.group_size <= 0:
                raise ValueError("per_group weight quantization requires a positive group_size")
        elif self.group_size is not None:
            raise ValueError(f"group_size only applies to per_group granularity, got {self.granularity}")

    def short(self) -> str:
        g = f"g{self.group_size}" if self.granularity == "per_group" else (
            "pc" if self.granularity == "per_channel" else "pt"
        )
        return f"w{self.bits}{g}{'sym' if self.symmetric else 'asym'}"


@dataclass(frozen=True)
class ActQuantSpec:
    """How to quantize the *input* activation of a linear layer."""

    bits: int
    granularity: Literal["per_token", "per_tensor"] = "per_token"
    symmetric: bool = True

    def __post_init__(self) -> None:
        if not (1 <= self.bits <= 16):
            raise ValueError(f"activation bits must be in [1, 16], got {self.bits}")
        if not self.symmetric:
            raise NotImplementedError("asymmetric activation quantization is not part of the protocol")

    def short(self) -> str:
        return f"a{self.bits}{'pt' if self.granularity == 'per_token' else 'pT'}"


# --------------------------------------------------------------------------- #
# Grid primitives. ``x`` is float32 and the reduction happens over ``dim=-1``.
# --------------------------------------------------------------------------- #
def fake_quant_symmetric(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric RTN over the last dimension of a float32 tensor."""
    q_max = 2 ** (bits - 1) - 1
    q_min = -(2 ** (bits - 1))
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_(min=_EPS) / q_max
    return torch.round(x / scale).clamp_(q_min, q_max).mul_(scale)


def fake_quant_asymmetric(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Asymmetric (zero-point) RTN over the last dimension of a float32 tensor.

    The integer zero-point must lie on the grid, so the covered range always
    includes 0 (``min <= 0 <= max``), as in Jacob et al. (2018) and the GPTQ /
    AutoGPTQ quantizers. For weight groups this is immaterial (a group of 128
    weights practically always spans both signs) but it keeps degenerate
    all-positive groups well defined.
    """
    levels = 2**bits - 1
    x_max = x.amax(dim=-1, keepdim=True).clamp_(min=0.0)
    x_min = x.amin(dim=-1, keepdim=True).clamp_(max=0.0)
    scale = ((x_max - x_min) / levels).clamp_(min=_EPS)
    zero = torch.round(-x_min / scale).clamp_(0, levels)
    q = torch.round(x / scale).add_(zero).clamp_(0, levels)
    return q.sub_(zero).mul_(scale)


def _quant_rows(x: torch.Tensor, bits: int, symmetric: bool) -> torch.Tensor:
    return fake_quant_symmetric(x, bits) if symmetric else fake_quant_asymmetric(x, bits)


# --------------------------------------------------------------------------- #
# Weights
# --------------------------------------------------------------------------- #
@torch.no_grad()
def fake_quant_weight(weight: torch.Tensor, spec: WeightQuantSpec) -> torch.Tensor:
    """Return an RTN fake-quantized copy of ``weight`` (same shape and dtype).

    ``weight`` may be 2-D ``[out, in]`` (``nn.Linear``) or N-D with the output
    channel first (``nn.Conv2d``: ``[out, in, kh, kw]``); N-D weights are
    flattened to ``[out, -1]`` for grouping and reshaped back.
    """
    if spec.bits >= 16:
        return weight.clone()
    orig_shape, orig_dtype = weight.shape, weight.dtype
    w = weight.detach().reshape(orig_shape[0], -1).to(torch.float32)
    out_f, in_f = w.shape

    if spec.granularity == "per_tensor":
        deq = _quant_rows(w.reshape(1, -1), spec.bits, spec.symmetric).reshape(out_f, in_f)
    elif spec.granularity == "per_channel":
        deq = _quant_rows(w, spec.bits, spec.symmetric)
    elif spec.granularity == "per_group":
        g = int(spec.group_size)  # type: ignore[arg-type]
        n_full = in_f // g
        deq = torch.empty_like(w)
        if n_full:
            head = w[:, : n_full * g].reshape(out_f, n_full, g)
            deq[:, : n_full * g] = _quant_rows(head, spec.bits, spec.symmetric).reshape(out_f, n_full * g)
        if in_f % g:  # ragged tail: its own (shorter) group, no zero padding
            deq[:, n_full * g :] = _quant_rows(w[:, n_full * g :], spec.bits, spec.symmetric)
    else:  # pragma: no cover - guarded by dataclass validation
        raise ValueError(f"unknown granularity {spec.granularity}")
    return deq.reshape(orig_shape).to(orig_dtype)


@torch.no_grad()
def weight_quant_error(weight: torch.Tensor, spec: WeightQuantSpec) -> dict[str, float]:
    """Relative reconstruction error statistics (for reports / sanity checks)."""
    w = weight.detach().to(torch.float32)
    w_hat = fake_quant_weight(w, spec)
    err = (w - w_hat)
    denom = w.pow(2).sum().clamp_min(_EPS)
    return {
        "rel_mse": float(err.pow(2).sum() / denom),
        "max_abs_err": float(err.abs().max()),
        "n_unique_per_row_mean": float(
            torch.tensor([w_hat[i].unique().numel() for i in range(min(8, w_hat.shape[0]))], dtype=torch.float32).mean()
        ),
    }


# --------------------------------------------------------------------------- #
# Activations (dynamic, calibration-free)
# --------------------------------------------------------------------------- #
def fake_quant_act_per_token(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-token symmetric RTN of an activation ``[..., in]`` (scale per row)."""
    if bits >= 16:
        return x
    return fake_quant_symmetric(x.to(torch.float32), bits).to(x.dtype)


def fake_quant_act_per_tensor(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-tensor symmetric RTN of an activation (one scale for the whole tensor)."""
    if bits >= 16:
        return x
    x32 = x.to(torch.float32)
    return fake_quant_symmetric(x32.reshape(1, -1), bits).reshape(x.shape).to(x.dtype)


def fake_quant_act(x: torch.Tensor, spec: ActQuantSpec) -> torch.Tensor:
    if spec.granularity == "per_token":
        return fake_quant_act_per_token(x, spec.bits)
    return fake_quant_act_per_tensor(x, spec.bits)
