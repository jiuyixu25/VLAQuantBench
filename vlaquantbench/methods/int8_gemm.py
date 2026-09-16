"""Real INT8 (W8A8) linear layer on top of ``torch._int_mm``.

``torch._int_mm`` dispatches to cuBLASLt's INT8 tensor-core GEMM, so this is a
genuine integer kernel with no custom CUDA extension to build. Quantization:

* weights: per-output-channel symmetric INT8 (``s_w`` shape ``[1, out]``),
  optionally *after* SmoothQuant smoothing has been folded into the weights
  and the preceding LayerNorm by the official ``smoothquant`` package;
* activations: per-token symmetric INT8 computed dynamically in the forward
  pass (the same scheme the fake-quant protocol uses).

The dynamic activation quantization is done with plain torch ops (absmax,
round, cast) and is therefore *not fused* into the GEMM; production kernels
fuse it, so the latency reported here is a slightly conservative estimate.

cuBLASLt requires ``M > 16`` (and ``K``, ``N`` multiples of 8); inputs with
fewer rows are zero-padded to 32 rows.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["W8A8Linear", "replace_linear_with_int8", "int_mm_supported"]

_MIN_ROWS = 32


def int_mm_supported(device: torch.device | str = "cuda") -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        a = torch.randint(-128, 127, (_MIN_ROWS, 64), dtype=torch.int8, device=device)
        b = torch.randint(-128, 127, (64, 64), dtype=torch.int8, device=device)
        torch._int_mm(a, b)
        return True
    except Exception:  # pragma: no cover
        return False


class W8A8Linear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        w = linear.weight.detach().to(torch.float32)
        out_f, in_f = w.shape
        if in_f % 8 or out_f % 8:
            raise ValueError(f"W8A8Linear needs in/out features divisible by 8, got {in_f}x{out_f}")
        s_w = w.abs().amax(dim=1, keepdim=True).clamp_(min=1e-8) / 127.0
        w_int8 = torch.round(w / s_w).clamp_(-128, 127).to(torch.int8)
        self.in_features, self.out_features = in_f, out_f
        self.register_buffer("weight_int8_t", w_int8.t().contiguous())  # [in, out]
        self.register_buffer("w_scale", s_w.t().contiguous())  # [1, out]
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().clone())
        else:
            self.bias = None
        self.out_dtype = linear.weight.dtype

    @property
    def weight(self) -> torch.Tensor:  # some model code reads .weight.dtype / .device
        return self.w_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).to(torch.float32)
        s_x = x2.abs().amax(dim=1, keepdim=True).clamp_(min=1e-8) / 127.0
        x_int8 = torch.round(x2 / s_x).clamp_(-128, 127).to(torch.int8)
        m = x_int8.shape[0]
        if m < _MIN_ROWS or m % 8:
            rows = max(_MIN_ROWS, (m + 7) // 8 * 8)
            pad = torch.zeros(rows - m, x_int8.shape[1], dtype=torch.int8, device=x_int8.device)
            x_int8 = torch.cat([x_int8, pad], dim=0)
        y = torch._int_mm(x_int8, self.weight_int8_t)[:m].to(torch.float32)
        y = y * s_x * self.w_scale
        if self.bias is not None:
            y = y + self.bias.to(torch.float32)
        return y.to(self.out_dtype).reshape(*shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, kernel=torch._int_mm"


@torch.no_grad()
def replace_linear_with_int8(root: nn.Module, *, skip: tuple[str, ...] = ("lm_head",), prefix: str = "") -> int:
    """Swap every ``nn.Linear`` under ``root`` for :class:`W8A8Linear` (in place)."""
    n = 0
    for name, child in list(root.named_children()):
        qual = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and type(child) is nn.Linear:
            if any(s in qual for s in skip):
                continue
            setattr(root, name, W8A8Linear(child).to(child.weight.device))
            n += 1
        else:
            n += replace_linear_with_int8(child, skip=skip, prefix=qual)
    return n
