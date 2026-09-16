"""Off-the-shelf LLM PTQ methods applied to a VLA's LLM backbone (Experiment 3).

All methods operate on the HF-style causal LM returned by
``adapter.llm_causal_lm()`` (e.g. the ``LlamaForCausalLM`` inside OpenVLA) and
use the *official* implementation of each method:

======  ======================================  ==========================================
method  accuracy run (fake quant)               deployment run (``--real-kernel``)
======  ======================================  ==========================================
rtn     in-repo RTN (protocol quantizer)        RTN weights packed into the AutoAWQ GEMM kernel (W4)
awq     llm-awq search + pseudo quant (W3/W4)   AutoAWQ GEMM kernel (W4)
nf4     bitsandbytes NF4 (real kernel, same)    bitsandbytes NF4
int8    bitsandbytes LLM.int8() (real, same)    bitsandbytes LLM.int8()
smooth  smoothquant smoothing + fake W8A8       smoothing + ``torch._int_mm`` INT8 GEMM (W8A8)
======  ======================================  ==========================================
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from ..components import quantize_scope
from ..quant.presets import QuantSpec

log = logging.getLogger(__name__)

__all__ = ["apply_method", "METHODS"]

METHODS = ("rtn", "awq", "nf4", "int8", "smoothquant")


def _cache_tag(adapter) -> str:
    """Identity of the quantized backbone, so cached AWQ search results are never
    reused across different models or checkpoints."""
    import hashlib

    raw = f"{adapter.name}|{adapter.checkpoint}"
    return f"{adapter.name}-{hashlib.sha1(raw.encode()).hexdigest()[:10]}"


def _llm(adapter):
    lm = adapter.llm_causal_lm()
    if lm is None:
        raise RuntimeError(f"adapter {adapter.name} does not expose an HF causal LM; LLM PTQ methods need one")
    return lm


def apply_method(adapter, method: str, spec: QuantSpec, *, calib: str | None = None, real_kernel: bool = False) -> dict[str, Any]:
    """Quantize the LLM backbone of ``adapter`` with ``method`` at the bit-width of ``spec``."""
    method = method.lower().replace("-", "").replace("_", "")
    if method in ("smooth", "sq", "smoothquant"):
        method = "smoothquant"
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    w_bits = spec.weight.bits if spec.weight else 16
    a_bits = spec.act.bits if spec.act else 16
    info: dict[str, Any] = {"method": method, "w_bits": w_bits, "a_bits": a_bits, "real_kernel": real_kernel}

    if method == "rtn":
        if not real_kernel:
            reports = quantize_scope(adapter.component_map(), spec, scope="llm")
            info["layers"] = reports["llm"].n_layers
            return info
        if a_bits < 16:
            from .smoothquant import real_w8a8

            if (w_bits, a_bits) != (8, 8):
                raise ValueError("real-kernel RTN with activations is only available as W8A8")
            info["layers"] = real_w8a8(_llm(adapter), smooth=False)
            return info
        from .awq import pack_rtn_into_awq_kernel

        info["layers"] = pack_rtn_into_awq_kernel(_llm(adapter), spec)
        return info

    if method == "awq":
        from .awq import apply_awq_fake, apply_awq_real

        fn = apply_awq_real if real_kernel else apply_awq_fake
        group = spec.weight.group_size if (spec.weight and spec.weight.group_size) else 128
        info.update(fn(_llm(adapter), adapter.llm_tokenizer(), w_bits=w_bits, calib_path=calib, group_size=group,
                       cache_tag=_cache_tag(adapter)))
        return info

    if method in ("nf4", "int8"):
        from .bnb import quantize_with_bnb

        expected = 4 if method == "nf4" else 8
        if w_bits != expected:
            raise ValueError(f"{method} is a {expected}-bit method; got preset {spec.name}")
        info["layers"] = quantize_with_bnb(_llm(adapter), method, compute_dtype=adapter.dtype)
        return info

    if method == "smoothquant":
        from .smoothquant import apply_smoothquant

        if (w_bits, a_bits) != (8, 8):
            raise ValueError(f"SmoothQuant is evaluated at W8A8; got preset {spec.name}")
        info.update(apply_smoothquant(_llm(adapter), adapter.llm_tokenizer(), calib=calib, real_kernel=real_kernel, spec=spec))
        return info
    raise AssertionError("unreachable")
