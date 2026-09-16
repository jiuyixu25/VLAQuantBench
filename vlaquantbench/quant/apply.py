"""Apply a :class:`QuantSpec` to a module tree without changing module types.

Design
------
* **Weights** are fake-quantized *in place* (``linear.weight.data.copy_(...)``).
  No module is swapped, so code that inspects ``isinstance(m, nn.Linear)``,
  ``m.weight.dtype`` / ``m.weight.shape`` keeps working and there is zero
  runtime overhead for weight-only settings.
* **Activations** are quantized with a *forward pre-hook* on the same module,
  which rewrites the first positional input (the GEMM operand). Hooks can be
  removed again with :func:`remove_activation_hooks`.
* Only ``nn.Linear`` layers are quantized by default, matching standard LLM
  PTQ practice; ``nn.Conv2d`` (patch embeddings) can be opted in with
  ``include_conv=True`` for ablations. Embeddings, norms, biases and anything
  not matched are left at baseline precision.
* A module reached through several roots (e.g. tied/shared layers) is
  quantized at most once; the caller is responsible for not listing the
  LM head inside an ``llm`` root unless that is intended (see
  :data:`DEFAULT_EXCLUDE_PATTERNS`).
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
from torch import nn

from .fake_quant import ActQuantSpec, fake_quant_act, fake_quant_weight, weight_quant_error
from .presets import QuantSpec

__all__ = [
    "DEFAULT_EXCLUDE_PATTERNS",
    "LayerReport",
    "QuantReport",
    "iter_quantizable",
    "quantize_modules",
    "remove_activation_hooks",
    "count_linear_params",
]

log = logging.getLogger(__name__)

#: Module-name patterns that are never quantized unless explicitly allowed.
#: ``lm_head`` follows LLM PTQ deployment practice (bitsandbytes / AWQ / GPTQ
#: keep it in full precision by default); the rest are embeddings that are
#: not ``nn.Linear`` anyway but are listed for clarity.
DEFAULT_EXCLUDE_PATTERNS: tuple[str, ...] = (
    "*lm_head*",
    "*embed_tokens*",
    "*patch_embed*",
    "*embeddings*",
)

_ACT_HOOK_ATTR = "_vqb_act_hook_handle"
_QUANT_MARK_ATTR = "_vqb_quantized_with"


@dataclass
class LayerReport:
    name: str
    kind: str
    n_params: int
    shape: tuple[int, ...]
    weight_rel_mse: float | None = None
    act_bits: int | None = None
    act_calibrated: bool = False


@dataclass
class QuantReport:
    spec: str
    layers: list[LayerReport] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def n_params(self) -> int:
        return sum(l.n_params for l in self.layers)

    def mean_rel_mse(self) -> float | None:
        vals = [l.weight_rel_mse for l in self.layers if l.weight_rel_mse is not None]
        return sum(vals) / len(vals) if vals else None

    def summary(self) -> str:
        mse = self.mean_rel_mse()
        mse_s = f", mean weight rel-MSE {mse:.3e}" if mse is not None else ""
        return (
            f"[{self.spec}] quantized {self.n_layers} layers / {self.n_params / 1e6:.1f}M params"
            f"{mse_s}; skipped {len(self.skipped)}"
        )


def _matches_any(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def iter_quantizable(
    roots: Iterable[tuple[str, nn.Module]],
    *,
    include_conv: bool = False,
    exclude_patterns: Sequence[str] = DEFAULT_EXCLUDE_PATTERNS,
    allow_patterns: Sequence[str] = (),
    only_patterns: Sequence[str] | None = None,
):
    """Yield ``(qualified_name, module)`` for every quantizable leaf under ``roots``.

    ``roots`` is an iterable of ``(prefix, module)``; the prefix is only used
    to build readable names. Shared modules are yielded once.

    * ``exclude_patterns`` - names matching these are skipped ...
    * ``allow_patterns``   - ... unless they also match one of these;
    * ``only_patterns``    - if given, a whitelist: only matching names are yielded.
    """
    seen: set[int] = set()
    for prefix, root in roots:
        for name, mod in root.named_modules():
            qual = f"{prefix}.{name}" if (prefix and name) else (prefix or name)
            is_linear = type(mod) is nn.Linear or (isinstance(mod, nn.Linear) and _plain_linear(mod))
            is_conv = include_conv and isinstance(mod, nn.Conv2d)
            if not (is_linear or is_conv):
                continue
            if id(mod) in seen:
                continue
            if _matches_any(qual, exclude_patterns) and not _matches_any(qual, allow_patterns):
                continue
            if only_patterns is not None and not _matches_any(qual, only_patterns):
                continue
            seen.add(id(mod))
            yield qual, mod


def _plain_linear(mod: nn.Module) -> bool:
    """Subclasses of nn.Linear that still own a dense float ``weight`` (e.g. LoRA-merged)."""
    w = getattr(mod, "weight", None)
    return isinstance(w, torch.Tensor) and w.dtype.is_floating_point and w.dim() == 2


def _make_act_hook(spec: ActQuantSpec):
    def hook(module: nn.Module, args: tuple):
        if not args:
            return None
        x = args[0]
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            return None
        return (fake_quant_act(x, spec), *args[1:])

    return hook


def _make_calibrated_act_hook(spec: ActQuantSpec, smooth: torch.Tensor, clip: torch.Tensor):
    """Smooth, clip, then quantize. The matching ``W · diag(smooth)`` fold happens
    at weight-quantization time, so the product is exact at full precision."""

    def hook(module: nn.Module, args: tuple):
        if not args:
            return None
        x = args[0]
        if not isinstance(x, torch.Tensor) or not x.is_floating_point() or x.shape[-1] != smooth.shape[0]:
            return None
        s = smooth.to(x.device, x.dtype)
        c = clip.to(x.device, x.dtype)
        xs = torch.clamp(x / s, -c, c)
        return (fake_quant_act(xs, spec), *args[1:])

    return hook


@torch.no_grad()
def quantize_modules(
    roots: Iterable[tuple[str, nn.Module]],
    spec: QuantSpec,
    *,
    include_conv: bool = False,
    exclude_patterns: Sequence[str] = DEFAULT_EXCLUDE_PATTERNS,
    allow_patterns: Sequence[str] = (),
    only_patterns: Sequence[str] | None = None,
    compute_error: bool = False,
    error_max_layers: int = 64,
    act_calib: dict | None = None,
) -> QuantReport:
    """Fake-quantize every quantizable layer under ``roots`` according to ``spec``.

    Returns a :class:`QuantReport`. Calling this twice on the same layer with a
    weight spec raises, because RTN of an already-rounded tensor would silently
    produce a different grid.
    """
    report = QuantReport(spec=spec.name)
    if spec.is_baseline:
        return report
    roots = list(roots)
    n_err = 0
    for qual, mod in iter_quantizable(
        roots,
        include_conv=include_conv,
        exclude_patterns=exclude_patterns,
        allow_patterns=allow_patterns,
        only_patterns=only_patterns,
    ):
        weight = mod.weight
        entry = LayerReport(name=qual, kind=type(mod).__name__, n_params=weight.numel(), shape=tuple(weight.shape))
        calib = act_calib.get(qual) if act_calib else None
        if calib is not None and spec.act is None:
            raise ValueError(f"{qual}: activation calibration given but the spec has no activation quantization")
        if calib is not None:
            # fold the smoothing into the weight BEFORE weight RTN so the
            # quantizer sees the redistributed per-channel ranges
            s = calib.smooth.to(weight.device, weight.dtype)
            weight.data.mul_(s)  # broadcasts over the input dimension
        if spec.weight is not None:
            if getattr(mod, _QUANT_MARK_ATTR, None) is not None:
                raise RuntimeError(f"{qual} was already quantized with {getattr(mod, _QUANT_MARK_ATTR)}")
            if compute_error and n_err < error_max_layers:
                entry.weight_rel_mse = weight_quant_error(weight, spec.weight)["rel_mse"]
                n_err += 1
            weight.data.copy_(fake_quant_weight(weight.data, spec.weight))
            setattr(mod, _QUANT_MARK_ATTR, spec.name)
        if spec.act is not None:
            if isinstance(mod, nn.Conv2d):
                raise NotImplementedError("activation quantization of Conv2d inputs is not part of the protocol")
            old = getattr(mod, _ACT_HOOK_ATTR, None)
            if old is not None:
                old.remove()
            if calib is not None:
                handle = mod.register_forward_pre_hook(
                    _make_calibrated_act_hook(spec.act, calib.smooth, calib.clip)
                )
                entry.act_calibrated = True
            else:
                handle = mod.register_forward_pre_hook(_make_act_hook(spec.act))
            setattr(mod, _ACT_HOOK_ATTR, handle)
            entry.act_bits = spec.act.bits
        report.layers.append(entry)
    log.info(report.summary())
    return report


def remove_activation_hooks(roots: Iterable[tuple[str, nn.Module]]) -> int:
    """Remove activation-quantization hooks installed by :func:`quantize_modules`."""
    n = 0
    for _, root in roots:
        for mod in root.modules():
            handle = getattr(mod, _ACT_HOOK_ATTR, None)
            if handle is not None:
                handle.remove()
                delattr(mod, _ACT_HOOK_ATTR)
                n += 1
    return n


def count_linear_params(root: nn.Module, *, include_conv: bool = False) -> tuple[int, int]:
    """``(n_layers, n_params)`` of quantizable layers under ``root`` (ignoring exclusions)."""
    n_l = n_p = 0
    for _, mod in iter_quantizable([("", root)], include_conv=include_conv, exclude_patterns=()):
        n_l += 1
        n_p += mod.weight.numel()
    return n_l, n_p
