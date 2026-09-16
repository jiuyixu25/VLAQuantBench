"""AWQ (Lin et al., 2024) applied to a VLA's LLM backbone, plus RTN packing into
the same kernel so that accuracy and latency can be compared kernel-for-kernel.

Two paths:

``apply_awq_fake``
    The official ``llm-awq`` search (``run_awq`` -> ``apply_awq`` -> layer-wise
    pseudo quantization) with its default calibration set (Pile validation,
    128 documents of at most 512 tokens re-chunked into 512-token blocks).
    Works for W3 and W4; the result is fake-quantized weights in the model's
    dtype, i.e. directly comparable with the benchmark's RTN protocol.

``apply_awq_real`` / ``pack_rtn_into_awq_kernel``
    W4 only. The AWQ search (or plain RTN) produces per-group asymmetric
    scales/zero-points which are packed into AutoAWQ's ``WQLinear_GEMM``
    (4-bit weights, INT4 GEMM kernel). Running RTN through the *same* kernel
    isolates the effect of the algorithm from the effect of the kernel.

Notes / gotchas handled here:

* ``import awq.quantize.pre_quant`` pulls in the compiled ``awq_inference_engine``
  extension even for the pseudo-quant path; a stub module is injected when the
  extension is absent.
* ``transformers >= 4.54`` returns a bare tensor from ``LlamaDecoderLayer.forward``
  while ``pre_quant`` indexes ``[0]``; the layer output is normalised.
* The AWQ search runs its own text-only calibration forward with ``input_ids``,
  which is fine for a VLA backbone: the resulting scales are folded into the
  RMSNorm weights and into ``v_proj``/``down_proj``, an exact reparameterisation
  for *any* input, including vision tokens.
* ``llm-awq`` moves layers to CPU as it goes; the model is moved back afterwards.
* **``run_awq`` already applies its scales and clips to the model in place**
  (``pre_quant.py`` lines 215/234); the official ``entry.py`` dumps the results
  and exits, and ``apply_awq`` is only used on a *freshly loaded* model
  (``--load_awq``). Calling both on one instance double-applies the
  reparameterisation and destroys the model, so this module never does.
  Search results are cached under ``$VQB_CALIB_DIR/awq`` keyed by
  ``(model, checkpoint, bits, group size)`` so repeated runs of the same
  backbone reuse one ~8-minute search and different checkpoints never share one.
  Caching is off when no ``cache_tag`` is given.
"""

from __future__ import annotations

import logging
import os
import sys
import types
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..quant.fake_quant import WeightQuantSpec
from ..quant.presets import QuantSpec

log = logging.getLogger(__name__)

__all__ = ["apply_awq_fake", "apply_awq_real", "pack_rtn_into_awq_kernel", "rtn_scales_zeros"]


def _default_cache() -> str:
    return str(Path(os.environ.get("VQB_CALIB_DIR", Path.home() / ".cache" / "vlaquantbench")) / "awq")


def _stub_awq_kernel() -> None:
    """Allow importing ``awq.quantize.*`` without the compiled inference engine."""
    for name in ("awq_inference_engine", "awq_v2_ext"):
        if name not in sys.modules:
            try:
                __import__(name)
            except ImportError:
                sys.modules[name] = types.ModuleType(name)


def _patch_move_embed() -> None:
    """``llm-awq``'s ``move_embed`` assumes ``model.model.rotary_emb`` (transformers >= 4.45).

    OpenVLA / OpenVLA-OFT pin transformers 4.40.1, where the rotary embedding lives
    inside each ``LlamaAttention``. Move whatever is actually there.
    """
    from awq.quantize import pre_quant  # type: ignore

    if getattr(pre_quant, "_vqb_move_embed_patched", False):
        return
    orig = pre_quant.move_embed

    def move_embed(model, device):
        try:
            return orig(model, device)
        except AttributeError:
            inner = getattr(model, "model", model)
            for attr in ("embed_tokens", "rotary_emb", "wte", "wpe"):
                mod = getattr(inner, attr, None)
                if isinstance(mod, nn.Module):
                    mod.to(device)

    pre_quant.move_embed = move_embed
    pre_quant._vqb_move_embed_patched = True


def _patch_decoder_layer_output() -> None:
    """transformers >= 4.54 returns a tensor (not a tuple) from decoder layers."""
    try:
        from awq.quantize import pre_quant  # type: ignore
    except ImportError:  # pragma: no cover
        return
    import inspect

    src = inspect.getsource(pre_quant.run_awq)
    if "isinstance(" in src or "[0]" not in src:
        return
    orig = pre_quant.run_awq

    def run_awq(model, enc, **kw):  # pragma: no cover - exercised only on new transformers
        import transformers

        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        if (major, minor) < (4, 54):
            return orig(model, enc, **kw)
        layers = pre_quant.get_blocks(model)
        patched = []
        for layer in layers:
            fwd = layer.forward

            def wrapper(*a, __fwd=fwd, **k):
                out = __fwd(*a, **k)
                return out if isinstance(out, tuple) else (out,)

            layer.forward = wrapper
            patched.append((layer, fwd))
        try:
            return orig(model, enc, **kw)
        finally:
            for layer, fwd in patched:
                layer.forward = fwd

    pre_quant.run_awq = run_awq


def _q_config(w_bits: int, group_size: int) -> dict[str, Any]:
    return {"zero_point": True, "q_group_size": int(group_size)}


def _cache_path(cache_dir: str | None, tag: str) -> "Path | None":
    if cache_dir is None:
        return None
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"awq-search-{tag}.pt"


def _search_or_load(lm: nn.Module, tokenizer, *, w_bits: int, group_size: int, n_samples: int, seq_len: int,
                    cache_dir: str | None, tag: str) -> tuple[dict[str, Any], bool]:
    """Return ``(awq_results, applied_in_place)``.

    A fresh search applies its scales/clips to ``lm`` as it goes; a cached
    result must be applied explicitly with ``apply_awq``.
    """
    from awq.quantize.pre_quant import apply_awq, run_awq  # type: ignore

    path = _cache_path(cache_dir, tag)
    if path is not None and path.exists():
        log.info("loading cached AWQ search results from %s", path)
        results = torch.load(path, map_location="cpu")
        apply_awq(lm, results)
        return results, True
    log.info("running AWQ search (w%d, g%d, %d calib samples)", w_bits, group_size, n_samples)
    results = run_awq(lm, tokenizer, w_bit=w_bits, q_config=_q_config(w_bits, group_size),
                      n_samples=n_samples, seqlen=seq_len)  # applies scales/clips in place
    if path is not None:
        torch.save(results, path)
        log.info("cached AWQ search results at %s", path)
    return results, True


def _prepare(lm: nn.Module, tokenizer) -> tuple[torch.device, Any]:
    _stub_awq_kernel()
    _patch_decoder_layer_output()
    _patch_move_embed()
    if tokenizer is None:
        raise ValueError("AWQ needs the backbone's tokenizer (adapter.llm_tokenizer())")
    use_cache = getattr(lm.config, "use_cache", None)
    lm.config.use_cache = False
    return next(lm.parameters()).device, use_cache


def apply_awq_fake(lm: nn.Module, tokenizer, *, w_bits: int, group_size: int = 128,
                   calib_path: str | None = None, n_samples: int = 128, seq_len: int = 512,
                   cache_dir: str | None = None, cache_tag: str | None = None) -> dict[str, Any]:
    """Official AWQ search + pseudo quantization of ``lm`` (in place)."""
    device, use_cache = _prepare(lm, tokenizer)
    from awq.quantize.quantizer import pseudo_quantize_model_weight  # type: ignore

    results, _ = _search_or_load(lm, tokenizer, w_bits=w_bits, group_size=group_size, n_samples=n_samples,
                                 seq_len=seq_len, cache_dir=(cache_dir or _default_cache()) if cache_tag else None,
                                 tag=f"{cache_tag}-w{w_bits}g{group_size}")
    pseudo_quantize_model_weight(lm, w_bit=w_bits, q_config=_q_config(w_bits, group_size))
    lm.to(device)  # llm-awq leaves the blocks on CPU
    if use_cache is not None:
        lm.config.use_cache = use_cache
    n_scale, n_clip = len(results.get("scale", [])), len(results.get("clip", []))
    log.info("AWQ applied: %d scale groups, %d clipped layers", n_scale, n_clip)
    return {"awq_scale_groups": n_scale, "awq_clipped": n_clip, "group_size": group_size, "calib_samples": n_samples}


# --------------------------------------------------------------------------- #
# real kernel
# --------------------------------------------------------------------------- #
@torch.no_grad()
def rtn_scales_zeros(weight: torch.Tensor, bits: int, group_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric per-group RTN of ``[out, in]`` -> ``(dequantized, scales, zeros)``.

    ``scales`` / ``zeros`` have shape ``[out, n_groups]`` and use the AWQ
    convention ``q = round(w / s) + z``, ``w_hat = (q - z) * s``.
    """
    out_f, in_f = weight.shape
    if in_f % group_size:
        raise ValueError(f"real-kernel packing needs in_features ({in_f}) divisible by the group size ({group_size})")
    w = weight.detach().to(torch.float32).reshape(out_f, in_f // group_size, group_size)
    levels = 2**bits - 1
    w_max = w.amax(dim=-1, keepdim=True).clamp_(min=0.0)
    w_min = w.amin(dim=-1, keepdim=True).clamp_(max=0.0)
    scales = ((w_max - w_min) / levels).clamp_(min=1e-8)
    zeros = torch.round(-w_min / scales).clamp_(0, levels)
    q = torch.round(w / scales).add_(zeros).clamp_(0, levels)
    deq = (q - zeros).mul_(scales).reshape(out_f, in_f)
    return deq.to(weight.dtype), scales.squeeze(-1), zeros.squeeze(-1)


_WQLINEAR_GEMM = None


def _import_wqlinear_gemm():
    """Import AutoAWQ's ``WQLinear_GEMM`` without executing any AutoAWQ ``__init__``.

    Two independent reasons the naive import fails here:

    * ``third_party/llm-awq`` (the search) and ``third_party/AutoAWQ`` (the INT4
      kernel wrapper) both ship a top-level package named ``awq`` and shadow
      each other in one process;
    * AutoAWQ's ``awq/__init__.py`` imports its full model zoo, which requires
      a newer ``transformers`` than the pinned OpenVLA fork (4.40.1) provides.

    Only ``awq/modules/linear/gemm.py`` and three torch-only leaf modules under
    ``awq/utils`` are actually needed, so synthetic parent packages (with real
    ``__path__`` but no executed ``__init__``) are registered for the import,
    and the previous ``awq*`` bindings are restored afterwards so llm-awq keeps
    working in the same process.
    """
    global _WQLINEAR_GEMM
    if _WQLINEAR_GEMM is not None:  # the function recurses per module-tree level
        return _WQLINEAR_GEMM
    import importlib
    import sys
    import types
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "third_party" / "AutoAWQ"
    saved = {k: v for k, v in sys.modules.items() if k == "awq" or k.startswith("awq.")}
    for k in saved:
        del sys.modules[k]
    try:
        # parents whose real __init__ must NOT run (top-level pulls the model
        # zoo; modules.linear pulls exllama/marlin). Their subdirs still need
        # to be importable, so each synthetic package carries the real path.
        for name, sub in (("awq", ""), ("awq.modules", "modules"),
                          ("awq.modules.linear", "modules/linear")):
            m = types.ModuleType(name)
            m.__path__ = [str(root / "awq" / sub)]
            m.__package__ = name
            sys.modules[name] = m
        mod = importlib.import_module("awq.modules.linear.gemm")
        _WQLINEAR_GEMM = mod.WQLinear_GEMM
        return _WQLINEAR_GEMM
    finally:
        for k in [k for k in sys.modules if k == "awq" or k.startswith("awq.")]:
            del sys.modules[k]
        sys.modules.update(saved)


@torch.no_grad()
def _swap_to_wqlinear(root: nn.Module, w_bits: int, group_size: int, *, skip=("lm_head",), prefix: str = "") -> int:
    WQLinear_GEMM = _import_wqlinear_gemm()

    n = 0
    for name, child in list(root.named_children()):
        qual = f"{prefix}.{name}" if prefix else name
        if type(child) is nn.Linear:
            if any(s in qual for s in skip):
                continue
            deq, scales, zeros = rtn_scales_zeros(child.weight.data, w_bits, group_size)
            child.weight.data.copy_(deq)  # from_linear re-derives the integers from the dequantized weight
            device = child.weight.device
            q = WQLinear_GEMM.from_linear(
                child.half() if child.weight.dtype != torch.float16 else child,
                w_bit=w_bits, group_size=group_size,
                scales=scales.t().contiguous().half(), zeros=zeros.t().contiguous().half(),
            )
            setattr(root, name, q.to(device))
            n += 1
        else:
            n += _swap_to_wqlinear(child, w_bits, group_size, skip=skip, prefix=qual)
    return n


def pack_rtn_into_awq_kernel(lm: nn.Module, spec: QuantSpec) -> int:
    """Quantize ``lm`` with plain RTN and execute it with the AutoAWQ INT4 GEMM kernel."""
    w: WeightQuantSpec | None = spec.weight
    if w is None or w.bits != 4:
        raise ValueError(f"the AWQ kernel is 4-bit only; got preset {spec.name}")
    group_size = w.group_size or 128
    n = _swap_to_wqlinear(lm, 4, group_size)
    log.info("packed %d RTN layers into WQLinear_GEMM (w4, g%d)", n, group_size)
    return n


def apply_awq_real(lm: nn.Module, tokenizer, *, w_bits: int, group_size: int = 128,
                   calib_path: str | None = None, n_samples: int = 128, seq_len: int = 512,
                   cache_dir: str | None = None, cache_tag: str | None = None) -> dict[str, Any]:
    """AWQ search, then execution with the AutoAWQ INT4 GEMM kernel (W4 only)."""
    if w_bits != 4:
        raise ValueError(f"the AWQ inference kernel supports 4-bit only; got w{w_bits}")
    device, use_cache = _prepare(lm, tokenizer)
    _search_or_load(lm, tokenizer, w_bits=w_bits, group_size=group_size, n_samples=n_samples, seq_len=seq_len,
                    cache_dir=(cache_dir or _default_cache()) if cache_tag else None,
                    tag=f"{cache_tag}-w{w_bits}g{group_size}")
    lm.to(device)
    if use_cache is not None:
        lm.config.use_cache = use_cache
    n = _swap_to_wqlinear(lm, w_bits, group_size)
    log.info("AWQ search applied and %d layers packed into WQLinear_GEMM", n)
    return {"layers": n, "group_size": group_size, "calib_samples": n_samples, "kernel": "WQLinear_GEMM"}
