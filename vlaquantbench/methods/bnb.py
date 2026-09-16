"""bitsandbytes NF4 (QLoRA, Dettmers et al. 2023) and LLM.int8() (Dettmers et
al. 2022) applied to an already-loaded HF causal LM, layer by layer.

Both are *real* kernels: NF4 stores 4-bit NormalFloat weights and dequantizes
blocks on the fly inside the matmul; LLM.int8() runs INT8 GEMM with
mixed-precision outlier decomposition (threshold 6.0, the official default).
There is no separate fake-quant path - the accuracy of the deployed kernel is
what is measured.
"""

from __future__ import annotations

import gc
import logging

import torch
from torch import nn

log = logging.getLogger(__name__)

__all__ = ["quantize_with_bnb"]


def _iter_linears(root: nn.Module, skip: tuple[str, ...], prefix: str = ""):
    for name, child in list(root.named_children()):
        qual = f"{prefix}.{name}" if prefix else name
        if type(child) is nn.Linear:
            if any(s in qual for s in skip):
                continue
            yield root, name, qual, child
        else:
            yield from _iter_linears(child, skip, qual)


@torch.no_grad()
def quantize_with_bnb(
    root: nn.Module,
    mode: str,
    *,
    compute_dtype: torch.dtype = torch.bfloat16,
    skip: tuple[str, ...] = ("lm_head",),
    int8_threshold: float = 6.0,
    double_quant: bool = False,
) -> int:
    """Replace every ``nn.Linear`` under ``root`` with a bitsandbytes layer.

    ``mode`` is ``"nf4"`` or ``"int8"``. Returns the number of layers replaced.
    """
    try:
        import bitsandbytes as bnb  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install bitsandbytes>=0.43 (inside the model's env) to use nf4/int8") from e

    mode = mode.lower()
    if mode not in ("nf4", "int8"):
        raise ValueError(f"bnb mode must be nf4 or int8, got {mode}")
    n = 0
    for parent, name, qual, lin in _iter_linears(root, skip):
        device = lin.weight.device
        has_bias = lin.bias is not None
        if mode == "nf4":
            new = bnb.nn.Linear4bit(
                lin.in_features, lin.out_features, bias=has_bias, compute_dtype=compute_dtype,
                compress_statistics=double_quant, quant_type="nf4",
            )
            new.weight = bnb.nn.Params4bit(
                lin.weight.data.to(compute_dtype).cpu(), requires_grad=False, quant_type="nf4",
                compress_statistics=double_quant,
            )
        else:
            new = bnb.nn.Linear8bitLt(
                lin.in_features, lin.out_features, bias=has_bias, has_fp16_weights=False, threshold=int8_threshold,
            )
            new.weight = bnb.nn.Int8Params(lin.weight.data.to(torch.float16).cpu(), requires_grad=False, has_fp16_weights=False)
        if has_bias:
            new.bias = nn.Parameter(lin.bias.data.to(compute_dtype if mode == "nf4" else torch.float16).clone(), requires_grad=False)
        new = new.to(device)  # <- quantization happens when the bnb params are moved to CUDA
        setattr(parent, name, new)
        del lin
        n += 1
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("bitsandbytes %s: replaced %d Linear layers", mode, n)
    return n
