"""SmoothQuant (Xiao et al., 2023) on a VLA's LLM backbone.

Pipeline (official ``mit-han-lab/smoothquant``):

1. ``get_act_scales`` - per-input-channel activation absmax collected with
   forward hooks over the default calibration corpus (Pile validation, 512
   documents truncated to 512 tokens). Since the collector only needs hooks on
   ``nn.Linear`` modules, ``collect_act_scales_from_texts`` below reproduces it
   without the ``datasets``/``zstandard`` round-trip when a text list is given.
2. ``smooth_lm(model, scales, alpha)`` - migrates activation outliers into the
   weights by dividing the preceding RMSNorm and multiplying ``q/k/v`` and
   ``gate/up``. This is an exact reparameterisation, so it stays valid for the
   VLA's multimodal inputs even though the scales were measured on text.
   ``alpha=0.85`` is the value the official repo uses for LLaMA-2-7B.
3. Execution:
   * ``real_kernel=False`` - fake W8A8 (per-channel symmetric weights,
     per-token dynamic activations) via the benchmark's own quantizer, so the
     numbers sit on the same protocol as every other accuracy result;
   * ``real_kernel=True`` - :class:`vlaquantbench.methods.int8_gemm.W8A8Linear`,
     a genuine INT8 tensor-core GEMM through ``torch._int_mm`` (the official
     ``torch-int`` CUTLASS kernels only implement OPT, not LLaMA).
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import torch
from torch import nn

from ..quant.apply import quantize_modules
from ..quant.presets import QuantSpec, parse_spec

log = logging.getLogger(__name__)

__all__ = ["apply_smoothquant", "collect_act_scales_from_texts", "real_w8a8"]

DEFAULT_ALPHA = 0.85  # official value for LLaMA-2-7B (examples/ppl_eval.sh)


@torch.no_grad()
def collect_act_scales_from_texts(lm: nn.Module, tokenizer, texts: Sequence[str], seq_len: int = 512) -> dict[str, torch.Tensor]:
    """Per-input-channel absmax over ``texts`` - the computation of ``get_act_scales``."""
    scales: dict[str, torch.Tensor] = {}
    device = next(lm.parameters()).device

    def hook(name):
        def fn(module, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            x = x.detach().reshape(-1, x.shape[-1]).abs().amax(dim=0).float().cpu()
            scales[name] = torch.maximum(scales[name], x) if name in scales else x

        return fn

    handles = [m.register_forward_hook(hook(n)) for n, m in lm.named_modules() if isinstance(m, nn.Linear)]
    try:
        for text in texts:
            ids = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True).input_ids.to(device)
            lm(ids)
    finally:
        for h in handles:
            h.remove()
    return scales


def _act_scales(lm: nn.Module, tokenizer, calib_path: str | None, n_samples: int, seq_len: int) -> dict[str, torch.Tensor]:
    from .calib import pile_val_path, pile_val_texts

    try:
        from smoothquant.calibration import get_act_scales  # type: ignore

        return get_act_scales(lm, tokenizer, pile_val_path(calib_path), num_samples=n_samples, seq_len=seq_len)
    except ImportError:
        log.warning("`smoothquant` package not importable; collecting activation scales with the in-repo equivalent")
        return collect_act_scales_from_texts(lm, tokenizer, pile_val_texts(n_samples, calib_path), seq_len=seq_len)


def real_w8a8(lm: nn.Module, *, smooth: bool = True) -> int:
    """Swap every ``nn.Linear`` of ``lm`` for a real INT8 GEMM layer."""
    from .int8_gemm import int_mm_supported, replace_linear_with_int8

    if not int_mm_supported():
        raise RuntimeError("torch._int_mm is unavailable on this device; W8A8 real-kernel runs need a CUDA GPU (SM75+)")
    n = replace_linear_with_int8(lm)
    log.info("replaced %d layers with torch._int_mm W8A8 linears (smoothed=%s)", n, smooth)
    return n


def apply_smoothquant(lm: nn.Module, tokenizer, *, calib: str | None = None, calib_path: str | None = None,
                      real_kernel: bool = False, spec: QuantSpec | None = None, alpha: float = DEFAULT_ALPHA,
                      n_samples: int = 512, seq_len: int = 512) -> dict[str, Any]:
    """Smooth ``lm`` and quantize it to W8A8 (fake or real kernel)."""
    if tokenizer is None:
        raise ValueError("SmoothQuant needs the backbone's tokenizer (adapter.llm_tokenizer())")
    calib_path = calib_path or calib
    scales = _act_scales(lm, tokenizer, calib_path, n_samples, seq_len)
    try:
        from smoothquant.smooth import smooth_lm  # type: ignore

        smooth_lm(lm, scales, alpha)
        smoothed = "smoothquant.smooth.smooth_lm"
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pip install git+https://github.com/mit-han-lab/smoothquant (the smoothing step is method-specific "
            "and is intentionally not re-implemented here)"
        ) from e

    info: dict[str, Any] = {"alpha": alpha, "calib_samples": n_samples, "act_scales": len(scales), "smoothed_by": smoothed}
    if real_kernel:
        info["layers"] = real_w8a8(lm, smooth=True)
        info["kernel"] = "torch._int_mm"
    else:
        report = quantize_modules([("", lm)], spec or parse_spec("W8A8"))
        info["layers"] = report.n_layers
        info["kernel"] = "fake"
    return info
