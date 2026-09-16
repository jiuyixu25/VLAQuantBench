#!/usr/bin/env python
"""Kernel-level latency / memory micro-benchmark for the LLM backbone (complements Table 5).

The closed-loop numbers in ``results/*.jsonl`` already contain per-step policy
latency and peak VRAM. This script isolates the *LLM backbone* and times two
synthetic workloads that bracket how VLAs use it:

* ``prefill``: one forward over ``--tokens`` tokens (OpenVLA-OFT style parallel decoding);
* ``decode``:  prefill + ``--decode-steps`` single-token autoregressive steps with KV cache
               (OpenVLA style action-token decoding).

Example (inside the OpenVLA env, RTX 4090):

    python scripts/profile_kernels.py --model openvla_oft --checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
        --methods baseline rtn_awqkernel awq nf4 int8 smoothquant --tokens 300
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlaquantbench.methods import apply_method  # noqa: E402
from vlaquantbench.profiling import module_memory_gb  # noqa: E402
from vlaquantbench.quant.presets import parse_spec  # noqa: E402
from vlaquantbench.registry import get_adapter_cls  # noqa: E402

METHOD_PRESET = {
    "baseline": ("rtn", "BASELINE", False),
    "rtn_fake_w4": ("rtn", "W4", False),
    "rtn_awqkernel": ("rtn", "W4", True),
    "awq": ("awq", "W4", True),
    "nf4": ("nf4", "W4", True),
    "int8": ("int8", "W8", True),
    "smoothquant": ("smoothquant", "W8A8", True),
    "rtn_int8gemm": ("rtn", "W8A8", True),
}


@torch.no_grad()
def time_fn(fn, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    return {"mean_ms": 1e3 * statistics.fmean(samples), "median_ms": 1e3 * statistics.median(samples),
            "p95_ms": 1e3 * sorted(samples)[int(0.95 * (len(samples) - 1))]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--methods", nargs="+", default=list(METHOD_PRESET))
    ap.add_argument("--tokens", type=int, default=300, help="prefill length (vision + text tokens)")
    ap.add_argument("--decode-steps", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = {}
    for name in args.methods:
        method, preset, real = METHOD_PRESET[name]
        adapter = get_adapter_cls(args.model)(args.checkpoint, device="cuda", dtype=args.dtype)
        adapter.load()
        lm = adapter.llm_causal_lm()
        if lm is None:
            raise SystemExit(f"model {args.model} does not expose an HF causal LM backbone")
        torch.cuda.reset_peak_memory_stats()
        if preset != "BASELINE":
            apply_method(adapter, method, parse_spec(preset), real_kernel=real)
        hidden = lm.config.hidden_size
        # the *activation* dtype is the model's baseline dtype even when the weights are
        # packed (bitsandbytes / AWQ layers keep uint8/int32 parameters)
        x = torch.randn(1, args.tokens, hidden, device="cuda", dtype=adapter.dtype)

        def prefill():
            lm(inputs_embeds=x, use_cache=False)

        def decode():
            out = lm(inputs_embeds=x, use_cache=True)
            past = out.past_key_values
            tok = out.logits[:, -1:].argmax(-1)
            for _ in range(args.decode_steps):
                out = lm(input_ids=tok, past_key_values=past, use_cache=True)
                past = out.past_key_values
                tok = out.logits[:, -1:].argmax(-1)

        r = {
            "prefill": time_fn(prefill, args.warmup, args.iters),
            "decode": time_fn(decode, args.warmup, max(5, args.iters // 5)),
            "llm_weight_gb": module_memory_gb(lm),
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }
        results[name] = r
        print(f"{name:16s} prefill {r['prefill']['mean_ms']:8.1f} ms | decode {r['decode']['mean_ms']:8.1f} ms | "
              f"weights {r['llm_weight_gb']:.2f} GB | peak {r['peak_vram_gb']:.2f} GB", flush=True)
        adapter.unload()
        del lm, adapter
        torch.cuda.empty_cache()
    if args.out:
        Path(args.out).write_text(json.dumps({"args": vars(args), "gpu": torch.cuda.get_device_name(0), "results": results}, indent=2))


if __name__ == "__main__":
    main()
