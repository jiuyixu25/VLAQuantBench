#!/usr/bin/env python
"""Can a calibration-time statistic predict which layer will break the policy?

Every post-hoc explanation we tried for *why* one layer is fragile and another
is not (parameter count, position in the head, "it emits the action", flow-
matching step averaging) was contradicted by the closed-loop data. The useful
question is the forward one: measured on real on-policy activations and before
any rollout, does a cheap per-layer statistic rank the layers the way the
closed-loop success rate does?

For every quantizable layer in a scope this records, over the activations the
policy actually sees:

  act_dispersion   per-token absmax, p99 / median. The classic outlier metric:
                   per-token scaling is set by the largest entry in a token, so
                   a heavy tail wastes the grid on a few coordinates.
  act_kurtosis     kurtosis of |x|, a scale-free version of the same idea.
  act_rel_err      ||Q_a(x) - x|| / ||x||, the error the activation grid alone
                   introduces.
  out_rel_err      ||W_q Q_a(x) - W x|| / ||W x||, the error that actually
                   reaches the next layer -- the quantity the closed-loop
                   result should track, if anything does.

    python scripts/act_diagnostics.py --model pi05 --scope ah --preset W4A4 \
        --tasks 0 1 --episodes 2 --out diagnostics/pi05-ah-W4A4.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlaquantbench.cli import build_adapter, resolve_checkpoint  # noqa: E402
from vlaquantbench.components import ComponentMap  # noqa: E402
from vlaquantbench.quant.apply import iter_quantizable  # noqa: E402
from vlaquantbench.quant.fake_quant import fake_quant_act, fake_quant_weight  # noqa: E402
from vlaquantbench.quant.presets import parse_spec  # noqa: E402
from vlaquantbench.registry import get_runner_cls  # noqa: E402


class LayerProbe:
    """Accumulates activation statistics for one linear layer."""

    def __init__(self, name: str, mod: torch.nn.Module, spec) -> None:
        self.name, self.mod, self.spec = name, mod, spec
        self.sums: dict[str, float] = defaultdict(float)
        self.n = 0
        self.wq = None  # quantized weight, computed once

    def __call__(self, mod, inputs):
        x = inputs[0]
        if not torch.is_tensor(x) or x.dim() < 2:
            return
        with torch.no_grad():
            xf = x.reshape(-1, x.shape[-1]).float()
            if xf.shape[0] == 0:
                return
            absmax = xf.abs().amax(dim=-1)                      # per token
            med = absmax.median().clamp_min(1e-12)
            p99 = absmax.quantile(0.99)
            xa = xf.abs()
            mu, sd = xa.mean(), xa.std().clamp_min(1e-12)
            kurt = (((xa - mu) / sd) ** 4).mean()

            xq = fake_quant_act(xf, self.spec.act) if self.spec.act else xf
            act_err = (xq - xf).norm() / xf.norm().clamp_min(1e-12)

            if self.wq is None:
                w = self.mod.weight.detach().float()
                self.wq = fake_quant_weight(w, self.spec.weight) if self.spec.weight else w
                self.w = w
            ref = xf @ self.w.T
            got = xq @ self.wq.T
            out_err = (got - ref).norm() / ref.norm().clamp_min(1e-12)

            for k, v in (("act_dispersion", p99 / med), ("act_kurtosis", kurt),
                         ("act_rel_err", act_err), ("out_rel_err", out_err)):
                self.sums[k] += float(v)
            self.n += 1

    def result(self) -> dict:
        out = {k: v / max(self.n, 1) for k, v in self.sums.items()}
        out["calls"] = self.n
        out["in_features"] = int(self.mod.weight.shape[1])
        out["out_features"] = int(self.mod.weight.shape[0])
        out["params"] = int(self.mod.weight.numel())
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmark", default="libero")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--scope", default="ah")
    ap.add_argument("--preset", default="W4A4")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--model-kwargs", nargs="*", default=None)
    ap.add_argument("--tasks", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--episodes-from", type=int, default=20,
                    help="first init-state index (default 20: disjoint from the evaluation states 0-19)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    args.checkpoint = resolve_checkpoint(args.model, args.checkpoint, args.benchmark, args.suite)
    spec = parse_spec(args.preset.upper())
    adapter = build_adapter(args)
    runner = get_runner_cls(args.benchmark)(args.suite, episodes_per_task=args.episodes, seed=0)
    if hasattr(runner, "configure_for"):
        runner.configure_for(adapter)

    cmap: ComponentMap = adapter.component_map()
    layers = cmap.effective_layers(args.scope)
    probes, handles = [], []
    for name, mod in iter_quantizable(cmap.roots(args.scope)):
        if id(mod) not in layers:
            continue
        p = LayerProbe(name, mod, spec)
        probes.append(p)
        handles.append(mod.register_forward_pre_hook(p))
    if not probes:
        raise SystemExit(f"no quantizable layers in scope {args.scope!r} -- check the scope name")
    print(f"probing {len(probes)} layers in scope {args.scope!r} with {spec.describe()}")

    all_tasks = runner.tasks()
    picked = [all_tasks[i] for i in args.tasks if i < len(all_tasks)]
    runner.run(adapter, writer=None, tasks=picked,
               episodes=range(args.episodes_from, args.episodes_from + args.episodes))
    for h in handles:
        h.remove()

    rows = {p.name: p.result() for p in probes if p.n}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import datetime as _dt
    import platform
    import torch
    from vlaquantbench.results import git_commit, gpu_name
    try:
        import mujoco
        mujoco_version = mujoco.__version__
    except Exception:  # pragma: no cover
        mujoco_version = None
    try:  # resolve the exact checkpoint revision from the local HF cache when the id is a repo id
        from huggingface_hub import snapshot_download
        snap = snapshot_download(args.checkpoint, local_files_only=True)
        checkpoint_revision = Path(snap).name
    except Exception:
        checkpoint_revision = None
    out.write_text(json.dumps(
        {"model": args.model, "suite": args.suite, "scope": args.scope,
         "preset": args.preset.upper(), "spec": spec.describe(),
         "provenance": {
             "checkpoint": args.checkpoint, "checkpoint_revision": checkpoint_revision,
             "seed": args.seed, "tasks": [t.task_id for t in picked],
             "episodes_from": args.episodes_from, "episodes": args.episodes,
             "vqb_commit": git_commit(), "mujoco": mujoco_version, "torch": torch.__version__,
             "python": platform.python_version(), "gpu": gpu_name(),
             "created": _dt.datetime.now().isoformat(timespec="seconds")},
         "layers": rows}, indent=1))

    rank = sorted(rows.items(), key=lambda kv: -kv[1]["out_rel_err"])
    print(f"\n{'layer':60s} {'out_rel_err':>11s} {'act_rel_err':>11s} "
          f"{'dispersion':>10s} {'kurtosis':>9s}")
    for name, r in rank[:25]:
        print(f"{name:60s} {r['out_rel_err']:11.4f} {r['act_rel_err']:11.4f} "
              f"{r['act_dispersion']:10.2f} {r['act_kurtosis']:9.1f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
