#!/usr/bin/env python
"""Action-space fidelity of a quantized policy, measured on paired observations.

Two modes, run as separate processes so each quantization is applied to freshly loaded
weights:

  record   roll out the full-precision policy on one held-out episode (init state >= 20,
           disjoint from the evaluation states 0-19) and save every observation the policy
           saw together with the action it returned;
  replay   load the model, apply a quantization exactly as ``vqb run`` would, feed the
           recorded observations in order, and save the returned actions.

Output of ``replay`` is a JSON summary with the mean absolute deviation from the recorded
full-precision actions (over all steps and all 7 action dimensions, in the environment's
unnormalized action units), per-dimension means, the number of steps, and full provenance
(checkpoint, method, preset, scope, commit). A BASELINE replay is the self-check: its
deviation is the replay's own noise floor.

    python scripts/action_fidelity.py record --model openvla_oft --suite libero_spatial \
        --out results/fidelity/openvla_oft/record.npz
    python scripts/action_fidelity.py replay --model openvla_oft --suite libero_spatial \
        --method awq --preset W4 --scope llm --record results/fidelity/openvla_oft/record.npz \
        --out results/fidelity/openvla_oft/awq-W4-llm.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlaquantbench.benchmarks import base as _b  # noqa: E402
from vlaquantbench.cli import apply_quantization, build_adapter, build_parser, resolve_checkpoint  # noqa: E402
from vlaquantbench.registry import get_runner_cls  # noqa: E402
from vlaquantbench.results import git_commit, gpu_name  # noqa: E402


def _run_args(argv: list[str]) -> argparse.Namespace:
    """Parse a ``vqb run`` command line so quantization is applied by the same code path."""
    return build_parser().parse_args(["run", *argv, "--out", "/dev/null"])


def _copy(obj):
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if isinstance(obj, dict):
        return {k: _copy(v) for k, v in obj.items()}
    return obj


def record(a: argparse.Namespace) -> None:
    args = _run_args(["--model", a.model, "--benchmark", a.benchmark, "--suite", a.suite, "--preset", "BASELINE",
                      "--scope", "e2e", "--seed", str(a.seed)] + (["--checkpoint", a.checkpoint] if a.checkpoint else []))
    args.checkpoint = resolve_checkpoint(args.model, args.checkpoint, args.benchmark, args.suite)
    runner = get_runner_cls(args.benchmark)(args.suite, episodes_per_task=50, seed=args.seed)
    adapter = build_adapter(args)
    if hasattr(runner, "configure_for"):
        runner.configure_for(adapter)
    task = runner.tasks()[a.task]
    obs_log, act_log, orig = [], [], adapter.act

    def recorder(obs, *rest, **kw):
        obs_log.append({"images": _copy(obs.images), "instruction": obs.instruction,
                        "state": None if obs.state is None else np.asarray(obs.state).copy(), "step": obs.step})
        out = orig(obs, *rest, **kw)
        act_log.append(np.asarray(out, dtype=np.float64).reshape(-1).copy())
        return out

    adapter.act = recorder
    try:
        summary = runner.run(adapter, None, tasks=[task], episodes=range(a.episode, a.episode + 1), log_every=1000)
    finally:
        adapter.act = orig
    cams = sorted(obs_log[0]["images"])
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, actions=np.stack(act_log), states=np.stack([o["state"] for o in obs_log]) if obs_log[0]["state"] is not None else np.zeros(0),
        steps=np.array([o["step"] for o in obs_log]), instructions=np.array([o["instruction"] for o in obs_log]),
        cameras=np.array(cams), **{f"img_{c}": np.stack([o["images"][c] for o in obs_log]) for c in cams},
        meta=json.dumps({"model": a.model, "benchmark": a.benchmark, "suite": a.suite, "task_id": task.task_id,
                         "task_name": task.task_name, "episode": a.episode, "seed": a.seed,
                         "checkpoint": adapter.checkpoint, "success": bool(round(summary["success_rate"]["point"])),
                         "n_steps": len(obs_log), "vqb_commit": git_commit(), "gpu": gpu_name(),
                         "created": _dt.datetime.now().isoformat(timespec="seconds")}))
    print(f"recorded {len(obs_log)} steps of task {task.task_id} episode {a.episode} -> {out}")


def replay(a: argparse.Namespace) -> None:
    rec = np.load(a.record, allow_pickle=False)
    meta = json.loads(str(rec["meta"]))
    argv = ["--model", a.model, "--benchmark", meta["benchmark"], "--suite", meta["suite"], "--preset", a.preset,
            "--scope", a.scope, "--seed", str(meta["seed"])]
    if a.method:
        argv += ["--method", a.method]
    if a.checkpoint:
        argv += ["--checkpoint", a.checkpoint]
    for flag, vals in (("--only", a.only), ("--exclude", a.exclude)):
        if vals:
            argv += [flag, *vals]
    args = _run_args(argv)
    args.checkpoint = resolve_checkpoint(args.model, args.checkpoint, args.benchmark, args.suite)
    runner = get_runner_cls(args.benchmark)(args.suite, episodes_per_task=50, seed=args.seed)
    adapter = build_adapter(args)
    if hasattr(runner, "configure_for"):
        runner.configure_for(adapter)
    report = apply_quantization(adapter, args)
    task = runner.tasks()[meta["task_id"]]
    cams = [str(c) for c in rec["cameras"]]
    n = int(meta["n_steps"])
    ref = rec["actions"]
    adapter.reset(task)
    acts = []
    for i in range(n):
        obs = _b.Observation(images={c: rec[f"img_{c}"][i] for c in cams}, instruction=str(rec["instructions"][i]),
                             state=None if rec["states"].size == 0 else rec["states"][i], step=int(rec["steps"][i]), raw=None)
        acts.append(np.asarray(adapter.act(obs, task), dtype=np.float64).reshape(-1))
    acts = np.stack(acts)
    d = min(acts.shape[1], ref.shape[1])
    dev = np.abs(acts[:, :d] - ref[:, :d])
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": a.model, "method": a.method or "rtn", "preset": a.preset.upper(), "scope": a.scope,
        "only": a.only, "exclude": a.exclude, "checkpoint": adapter.checkpoint, "quant_report": report,
        "record": str(a.record), "record_meta": meta, "n_steps": n, "action_dims": d,
        "definition": "mean over steps and the first 7 action dimensions of |a_quantized - a_fullprecision|, "
                      "actions in the environment's unnormalized units as returned by adapter.act; "
                      "identical observation sequence fed open-loop after adapter.reset",
        "mean_abs_dev": float(dev.mean()), "per_dim_mean_abs_dev": dev.mean(axis=0).tolist(),
        "max_abs_dev": float(dev.max()), "vqb_commit": git_commit(), "gpu": gpu_name(),
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    out.write_text(json.dumps(summary, indent=1))
    np.save(out.with_suffix(".actions.npy"), acts)
    print(f"{summary['method']}-{summary['preset']}-{a.scope}: mean |dev| = {dev.mean():.5f} over {n} steps -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("record")
    r.add_argument("--model", required=True); r.add_argument("--benchmark", default="libero")
    r.add_argument("--suite", default="libero_spatial"); r.add_argument("--checkpoint", default=None)
    r.add_argument("--task", type=int, default=0); r.add_argument("--episode", type=int, default=20)
    r.add_argument("--seed", type=int, default=0); r.add_argument("--out", required=True)
    p = sub.add_parser("replay")
    p.add_argument("--model", required=True); p.add_argument("--checkpoint", default=None)
    p.add_argument("--record", required=True); p.add_argument("--method", default=None)
    p.add_argument("--preset", required=True); p.add_argument("--scope", default="llm")
    p.add_argument("--only", nargs="*", default=None); p.add_argument("--exclude", nargs="*", default=None)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    (record if a.mode == "record" else replay)(a)


if __name__ == "__main__":
    main()
