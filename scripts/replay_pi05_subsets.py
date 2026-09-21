#!/usr/bin/env python
"""Same-observation replay of pi0.5 action-head subsets (arXiv revision P0).

Question: the closed-loop six-setting comparison (attention / adaRMS / their union / 126-layer
subset / full 167-layer head, all W4A4) shows non-additive failure and non-monotone recovery in
*task success*. Success is thresholded, so that alone does not say whether the *numerical* error
composes. This tool measures the policy's outputs on identical inputs:

  record   roll out the full-precision policy on held-out initial states (>= 20; the evaluation
           states are 0-19) and save every observation the policy received, the executed actions,
           and the RNG seed the adapter used for the episode's flow-matching noise;
  replay   load the model once, apply one quantization configuration exactly as ``vqb run``
           would, feed each recorded observation sequence open-loop with the *recorded* seed
           (so every configuration draws the same flow noise), and save
             - the executed (unnormalized) action per step,
             - the full predicted action chunk at every replanning step (normalized units),
             - the denoising state x_t at every flow step of every chunk,
             - per-call input statistics of the action expert's down_proj layers
               (raw inputs, observed before the activation quantizer);
  analyze  compare configurations against the baseline replay: executed-action MAE and signed
           bias per action dimension, chunk-level MAE, the additivity residual
           ||e_union - (e_attn + e_adaRMS)|| / ||e_union||, the 126-vs-167 error comparison,
           per-flow-step divergence, and down_proj input statistics.

    P=~/anaconda3/envs/lerobot/bin/python; R=results/replay/pi05_spatial
    $P scripts/replay_pi05_subsets.py record  --tasks 0 1 2 3 4 5 6 7 8 9 --episodes 20 21 --out $R/records
    for c in baseline baseline2 attention adarms union s126 s167; do
      $P scripts/replay_pi05_subsets.py replay --config $c --records $R/records --out $R/W4A4/$c.npz; done
    $P scripts/replay_pi05_subsets.py analyze --dir $R/W4A4 --out $R/W4A4/analysis.json
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

ADARMS = ["*layernorm.dense*", "*model.norm.dense*"]
# name -> (preset, scope, only patterns, expected quantized layer count)
CONFIGS = {
    "baseline": ("BASELINE", "e2e", None, 0),
    "baseline2": ("BASELINE", "e2e", None, 0),          # repeatability floor
    "attention": ("W4A4", "ah", ["*self_attn*"], 72),
    "adarms": ("W4A4", "ah", ADARMS, 36),
    "union": ("W4A4", "ah", ["*self_attn*", *ADARMS], 108),
    "s126": ("W4A4", "ah", ["*down_proj*", "*self_attn*", *ADARMS], 126),
    "s167": ("W4A4", "ah", None, 167),
    # recovery split: the 41 layers added from s126 to s167 are 36 gate/up projections + 5 others
    "s126_gateup": ("W4A4", "ah", ["*down_proj*", "*self_attn*", *ADARMS, "*gate_proj*", "*up_proj*"], 162),
    "s126_others": ("W4A4", "ah", ["*down_proj*", "*self_attn*", *ADARMS, "*gemma_expert.norm.dense",
                                   "*action_in_proj*", "*action_out_proj*", "*time_mlp_in*", "*time_mlp_out*"], 131),
}


def _run_args(argv):
    return build_parser().parse_args(["run", *argv, "--out", "/dev/null"])


def _setup(model, suite, preset, scope, only, checkpoint, seed):
    argv = ["--model", model, "--benchmark", "libero", "--suite", suite, "--preset", preset, "--scope", scope,
            "--seed", str(seed)]
    if checkpoint:
        argv += ["--checkpoint", checkpoint]
    if only:
        argv += ["--only", *only]
    args = _run_args(argv)
    args.checkpoint = resolve_checkpoint(args.model, args.checkpoint, args.benchmark, args.suite)
    runner = get_runner_cls(args.benchmark)(args.suite, episodes_per_task=50, seed=args.seed)
    adapter = build_adapter(args)
    if hasattr(runner, "configure_for"):
        runner.configure_for(adapter)
    return args, runner, adapter


def _copy(obj):
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if isinstance(obj, dict):
        return {k: _copy(v) for k, v in obj.items()}
    return obj


# ------------------------------------------------------------------------------------ record
def record(a):
    args, runner, adapter = _setup(a.model, a.suite, "BASELINE", "e2e", None, a.checkpoint, a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tasks = runner.tasks()
    for ti in a.tasks:
        task = tasks[ti]
        for ep in a.episodes:
            dest = out / f"t{ti}_e{ep}.npz"
            if dest.exists():
                print(f"skip {dest} (exists)"); continue
            obs_log, act_log, seeds, orig_act, orig_reset = [], [], [], adapter.act, adapter.reset

            def rec_reset(*ra, _orig=orig_reset, **rk):
                _orig(*ra, **rk); seeds.append(int(adapter._episode_seed))

            def rec_act(obs, *rest, _orig=orig_act, **kw):
                obs_log.append({"images": _copy(obs.images), "instruction": obs.instruction,
                                "state": None if obs.state is None else np.asarray(obs.state).copy(), "step": obs.step})
                o = _orig(obs, *rest, **kw); act_log.append(np.asarray(o, dtype=np.float64).reshape(-1).copy()); return o

            adapter.reset, adapter.act = rec_reset, rec_act
            try:
                summary = runner.run(adapter, None, tasks=[task], episodes=range(ep, ep + 1), log_every=1000)
            finally:
                adapter.reset, adapter.act = orig_reset, orig_act
            cams = sorted(obs_log[0]["images"])
            np.savez_compressed(
                dest, actions=np.stack(act_log), states=np.stack([o["state"] for o in obs_log]),
                steps=np.array([o["step"] for o in obs_log]), instructions=np.array([o["instruction"] for o in obs_log]),
                cameras=np.array(cams), **{f"img_{c}": np.stack([o["images"][c] for o in obs_log]) for c in cams},
                meta=json.dumps({"model": a.model, "suite": a.suite, "task_id": task.task_id, "task_name": task.task_name,
                                 "episode": ep, "runner_seed": a.seed, "episode_seed": seeds[-1],
                                 "checkpoint": adapter.checkpoint, "success": bool(round(summary["success_rate"]["point"])),
                                 "n_steps": len(obs_log), "n_action_steps": int(getattr(adapter, "n_action_steps", -1)),
                                 "vqb_commit": git_commit(), "gpu": gpu_name(),
                                 "created": _dt.datetime.now().isoformat(timespec="seconds")}))
            print(f"recorded task {ti} episode {ep}: {len(obs_log)} steps, success={summary['success_rate']['point']:.0f}, "
                  f"episode_seed={seeds[-1]} -> {dest}")


# ------------------------------------------------------------------------------------ replay
class _DownProjStats:
    """Per-call summary of the raw input of one linear layer: per-token absmax quantiles and |x| kurtosis."""

    def __init__(self):
        self.rows = []

    def __call__(self, module, args):
        x = args[0]
        if not hasattr(x, "shape") or x.shape[-1] != module.weight.shape[1]:
            return
        import torch
        with torch.no_grad():
            xa = x.detach().reshape(-1, x.shape[-1]).abs().float()
            tok = xa.amax(dim=1)
            mu, sd = xa.mean(), xa.std().clamp_min(1e-12)
            kurt = (((xa - mu) / sd) ** 4).mean()
            q = torch.quantile(tok, torch.tensor([0.5, 0.99], device=tok.device))
            self.rows.append([float(tok.mean()), float(q[0]), float(q[1]), float(tok.max()), float(kurt), int(tok.numel())])


def replay(a):
    import torch

    preset, scope, only, want_layers = CONFIGS[a.config]
    preset = a.preset or preset
    if a.only:
        only = a.only
    args, runner, adapter = _setup(a.model, a.suite, preset, scope, only, a.checkpoint, a.seed)
    report = apply_quantization(adapter, args) if preset != "BASELINE" else {}
    got = sum(v.get("layers", 0) for v in report.values() if isinstance(v, dict))
    if want_layers and got != want_layers:
        raise SystemExit(f"config {a.config}: quantized {got} layers, expected {want_layers}")
    policy = adapter.policy
    tasks = runner.tasks()

    chunks, flow = [], []
    orig_chunk = policy.predict_action_chunk

    def chunk_hook(batch, **kw):
        out = orig_chunk(batch, **kw)
        chunks.append(out.detach().float().cpu().numpy()[0]); return out

    policy.predict_action_chunk = chunk_hook
    orig_denoise = policy.model.denoise_step

    def denoise_hook(**kw):
        v = orig_denoise(**kw)
        flow.append((kw["x_t"].detach().float().cpu().numpy()[0], v.detach().float().cpu().numpy()[0],
                     float(kw["timestep"].reshape(-1)[0])))
        return v

    policy.model.denoise_step = denoise_hook
    stats, handles = {}, []
    for name, mod in policy.named_modules():
        if "gemma_expert" in name and name.endswith("mlp.down_proj"):
            st = _DownProjStats(); stats[name] = st
            handles.append(mod.register_forward_pre_hook(st, prepend=True))

    recs = sorted(Path(a.records).glob("t*_e*.npz"))
    if not recs:
        raise SystemExit(f"no records in {a.records}")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    arrays, index = {}, []
    for ri, rp in enumerate(recs):
        rec = np.load(rp, allow_pickle=False)
        meta = json.loads(str(rec["meta"]))
        task = tasks[meta["task_id"]]
        cams = [str(c) for c in rec["cameras"]]
        n = int(meta["n_steps"])
        chunks.clear(); flow.clear()
        for st in stats.values():
            st.rows.clear()
        adapter.reset(task)
        seed = int(meta["episode_seed"])
        adapter._episode_seed = seed
        torch.manual_seed(seed); np.random.seed(seed % (2 ** 32 - 1))
        acts = []
        with torch.no_grad():
            for i in range(n):
                obs = _b.Observation(images={c: rec[f"img_{c}"][i] for c in cams}, instruction=str(rec["instructions"][i]),
                                     state=rec["states"][i], step=int(rec["steps"][i]), raw=None)
                acts.append(np.asarray(adapter.act(obs, task), dtype=np.float64).reshape(-1))
        acts = np.stack(acts)
        ch = np.stack(chunks)                                  # (m, 50, 7)
        k = len(flow) // len(chunks)                           # flow steps per chunk
        xt = np.stack([f[0] for f in flow]).reshape(len(chunks), k, *flow[0][0].shape)
        vt = np.stack([f[1] for f in flow]).reshape(len(chunks), k, *flow[0][1].shape)
        arrays[f"r{ri}_actions"], arrays[f"r{ri}_chunks"], arrays[f"r{ri}_xt"], arrays[f"r{ri}_vt"] = acts, ch, xt, vt
        arrays[f"r{ri}_downproj"] = np.array([np.mean(st.rows, axis=0) if st.rows else np.full(6, np.nan) for st in stats.values()])
        dev = float(np.abs(acts[:, :7] - rec["actions"][:, :7]).mean())
        index.append({"record": rp.name, "task_id": meta["task_id"], "episode": meta["episode"], "n_steps": n,
                      "n_chunks": int(len(chunks)), "flow_steps": int(k), "recorded_success": meta["success"],
                      "dev_from_recorded_actions": dev})
        print(f"[{a.config}] {rp.name}: {n} steps, {len(chunks)} chunks x {k} flow steps, "
              f"mean|a - a_recorded| = {dev:.5f}")
    for h in handles:
        h.remove()
    summary = {"config": a.config, "model": a.model, "suite": a.suite, "preset": preset, "scope": scope, "only": only,
               "quantized_layers": got, "quant_report": report, "checkpoint": adapter.checkpoint,
               "downproj_layers": list(stats), "downproj_columns": ["tok_absmax_mean", "tok_absmax_p50", "tok_absmax_p99",
                                                                    "tok_absmax_max", "abs_kurtosis", "n_tokens"],
               "records": index, "vqb_commit": git_commit(), "gpu": gpu_name(),
               "created": _dt.datetime.now().isoformat(timespec="seconds")}
    np.savez_compressed(out, meta=json.dumps(summary), **arrays)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print(f"-> {out}")


# ------------------------------------------------------------------------------------ analyze
def _load(p):
    z = np.load(p, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return meta, z


def analyze(a):
    d = Path(a.dir)
    data = {p.stem: _load(p) for p in sorted(d.glob("*.npz"))}
    if "baseline" not in data:
        raise SystemExit("need baseline.npz")
    bmeta, bz = data["baseline"]
    R = len(bmeta["records"])
    res = {"records": bmeta["records"], "configs": {}}

    def per_record(cfg, key):
        return [data[cfg][1][f"r{ri}_{key}"] for ri in range(R)]

    B_act, B_ch, B_xt = per_record("baseline", "actions"), per_record("baseline", "chunks"), per_record("baseline", "xt")
    for cfg, (meta, z) in data.items():
        A, C, X = per_record(cfg, "actions"), per_record(cfg, "chunks"), per_record(cfg, "xt")
        act_dev = np.concatenate([np.abs(A[i][:, :7] - B_act[i][:, :7]) for i in range(R)])
        signed = np.concatenate([A[i][:, :7] - B_act[i][:, :7] for i in range(R)])
        ch_dev = np.concatenate([np.abs(C[i] - B_ch[i]).reshape(len(C[i]), -1).mean(1) for i in range(R)])
        # divergence per flow step (mean |x_t - x_t^base| over chunk x dims), averaged over chunks and records
        k = X[0].shape[1]
        step_dev = np.mean(np.concatenate([np.abs(X[i] - B_xt[i]).reshape(len(X[i]), k, -1).mean(2) for i in range(R)]), axis=0)
        dp = np.nanmean(np.stack([z[f"r{ri}_downproj"] for ri in range(R)]), axis=0)  # (layers, 6)
        res["configs"][cfg] = {
            "preset": meta["preset"], "quantized_layers": meta["quantized_layers"],
            "executed_action_mae": float(act_dev.mean()), "executed_action_mae_per_dim": act_dev.mean(0).tolist(),
            "executed_action_signed_mean_per_dim": signed.mean(0).tolist(),
            "chunk_mae_mean": float(ch_dev.mean()), "chunk_mae_median": float(np.median(ch_dev)),
            "flow_step_divergence": step_dev.tolist(),
            "downproj_tok_absmax_mean": float(dp[:, 0].mean()), "downproj_tok_absmax_p99": float(dp[:, 2].mean()),
            "downproj_abs_kurtosis": float(dp[:, 4].mean()),
            "dev_from_recorded_actions": float(np.mean([r["dev_from_recorded_actions"] for r in meta["records"]])),
        }
    # additivity: union error vs sum of the two isolated errors, per chunk (normalized chunk space)
    if all(c in data for c in ("attention", "adarms", "union")):
        E = {c: [per_record(c, "chunks")[i] - B_ch[i] for i in range(R)] for c in ("attention", "adarms", "union")}
        rows = []
        for i in range(R):
            for j in range(len(B_ch[i])):
                eu, ea, ed = E["union"][i][j].ravel(), E["attention"][i][j].ravel(), E["adarms"][i][j].ravel()
                s = ea + ed
                rows.append([np.linalg.norm(eu), np.linalg.norm(ea), np.linalg.norm(ed), np.linalg.norm(s),
                             np.linalg.norm(eu - s) / max(np.linalg.norm(eu), 1e-12),
                             float(np.dot(eu, s) / max(np.linalg.norm(eu) * np.linalg.norm(s), 1e-12))])
        rows = np.array(rows)
        res["additivity"] = {
            "n_chunks": int(len(rows)),
            "norm_union_over_sum_of_norms": {"median": float(np.median(rows[:, 0] / (rows[:, 1] + rows[:, 2]))),
                                             "iqr": np.percentile(rows[:, 0] / (rows[:, 1] + rows[:, 2]), [25, 75]).tolist()},
            "norm_union_over_norm_of_sum": {"median": float(np.median(rows[:, 0] / np.maximum(rows[:, 3], 1e-12)))},
            "residual_over_union": {"median": float(np.median(rows[:, 4])), "iqr": np.percentile(rows[:, 4], [25, 75]).tolist()},
            "cosine_union_vs_sum": {"median": float(np.median(rows[:, 5])), "iqr": np.percentile(rows[:, 5], [25, 75]).tolist()},
            "definition": "e_c = chunk_c - chunk_baseline on identical observations and identical flow noise; "
                          "sum = e_attention + e_adarms; residual = ||e_union - sum|| / ||e_union||",
        }
    if all(c in data for c in ("s126", "s167")):
        m126 = np.concatenate([np.abs(per_record("s126", "chunks")[i] - B_ch[i]).reshape(len(B_ch[i]), -1).mean(1) for i in range(R)])
        m167 = np.concatenate([np.abs(per_record("s167", "chunks")[i] - B_ch[i]).reshape(len(B_ch[i]), -1).mean(1) for i in range(R)])
        res["recovery"] = {"chunk_mae_126": float(m126.mean()), "chunk_mae_167": float(m167.mean()),
                           "frac_chunks_167_lower": float(np.mean(m167 < m126)), "n_chunks": int(len(m126))}
    if "baseline2" in data:
        res["repeatability_floor"] = res["configs"]["baseline2"]["executed_action_mae"]
    # scale of the full-precision executed actions (environment units), for relative readings of the deviations
    res["fp_mean_abs_action"] = float(np.mean(np.concatenate([np.abs(a[:, :7]) for a in B_act])))
    res["created"] = _dt.datetime.now().isoformat(timespec="seconds")
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "records"}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("record")
    r.add_argument("--model", default="pi05"); r.add_argument("--suite", default="libero_spatial")
    r.add_argument("--checkpoint", default=None); r.add_argument("--seed", type=int, default=0)
    r.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    r.add_argument("--episodes", type=int, nargs="+", default=[20]); r.add_argument("--out", required=True)
    p = sub.add_parser("replay")
    p.add_argument("--config", required=True, choices=sorted(CONFIGS)); p.add_argument("--preset", default=None)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--model", default="pi05"); p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--checkpoint", default=None); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--records", required=True); p.add_argument("--out", required=True)
    n = sub.add_parser("analyze"); n.add_argument("--dir", required=True); n.add_argument("--out", required=True)
    a = ap.parse_args()
    {"record": record, "replay": replay, "analyze": analyze}[a.mode](a)


if __name__ == "__main__":
    main()
