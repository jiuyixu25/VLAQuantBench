"""``vqb`` command line interface.

    vqb inspect   --model pi05 --checkpoint lerobot/pi05_libero [--device cpu]
    vqb run       --model pi05 --benchmark libero --suite libero_spatial --preset W4A8 --scope e2e
    vqb summarize results/ [--format md|latex|csv|json] [--ci]
    vqb matrix    configs/experiments/exp1_end2end.yaml [--print] [--only-model ...]
    vqb presets
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .components import quantize_scope, resolve_scope
from .quant.presets import PRESETS, parse_spec
from .registry import BENCHMARKS, MODELS, get_adapter_cls, get_runner_cls
from .results import RunHeader, RunWriter

log = logging.getLogger("vqb")
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_model_config(model: str) -> dict[str, Any]:
    path = CONFIG_DIR / "models" / f"{model}.yaml"
    if not path.exists():
        return {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def resolve_checkpoint(model: str, checkpoint: str | None, benchmark: str, suite: str) -> str:
    if checkpoint and checkpoint != "auto":
        return checkpoint
    if model == "remote":
        return "remote"  # the server knows its checkpoint; recorded from /info
    cfg = load_model_config(model)
    ckpts = cfg.get("checkpoints", {})
    bench = ckpts.get(benchmark)
    if isinstance(bench, dict):
        if suite in bench:
            return bench[suite]
        if "default" in bench:
            return bench["default"]
    elif isinstance(bench, str):
        return bench
    raise SystemExit(
        f"no checkpoint configured for model={model} benchmark={benchmark} suite={suite}; "
        f"pass --checkpoint or edit configs/models/{model}.yaml"
    )


def default_out_path(model: str, benchmark: str, suite: str, method: str, preset: str, scope: str,
                     lm_head: bool, seed: int = 0) -> Path:
    tag = f"{method}-{preset}-{scope}" + ("-lmhead" if lm_head else "") + (f"-seed{seed}" if seed else "")
    return REPO_ROOT / "results" / benchmark / suite / model / f"{tag}.jsonl"


def build_adapter(args) -> Any:
    cls = get_adapter_cls(args.model)
    cfg = load_model_config(args.model)
    extra = dict(cfg.get("adapter_kwargs", {}))
    for kv in args.model_kwargs or []:
        k, _, v = kv.partition("=")
        extra[k] = yaml.safe_load(v)
    dtype = args.dtype or cfg.get("dtype")
    adapter = cls(args.checkpoint, device=args.device, dtype=dtype, seed=args.seed, **extra)
    adapter.load()
    return adapter


def _collect_act_calib(adapter, runner, args) -> dict:
    """Run calibration episodes with the un-quantized policy and build per-layer
    smoothing/clip vectors for the components named in ``--act-calib``.

    Statistics are cached per (model, checkpoint, suite, components, episodes);
    the six cells of the calibration experiment share one collection run.
    """
    import torch

    from .quant.act_calib import LayerCalib, build_calibration, cache_path, collect_stats
    from .quant.apply import DEFAULT_EXCLUDE_PATTERNS, iter_quantizable

    comps = list(dict.fromkeys(args.act_calib))
    cmap = adapter.component_map()
    layers: dict = {}
    for c in comps:
        exclude = list(DEFAULT_EXCLUDE_PATTERNS) + list(cmap.exclude.get(c, []))
        allow = list(cmap.allow.get(c, []))
        for qual, mod in iter_quantizable(cmap.roots(c), exclude_patterns=exclude, allow_patterns=allow):
            layers[qual] = mod
    if not layers:
        raise SystemExit(f"--act-calib {comps}: no quantizable layers found")

    from .results import git_commit

    cache = cache_path(args.model, adapter.checkpoint, args.suite, comps, args.act_calib_episodes,
                       episodes_from=args.act_calib_from, tasks=args.act_calib_tasks, seed=args.seed,
                       commit=git_commit())
    if cache.exists():
        raw = torch.load(cache, weights_only=False)
        log.info("act-calib: loaded cached statistics for %d layers from %s", len(raw), cache)
    else:
        cal_tasks = runner.tasks()[: max(1, args.act_calib_tasks)]
        log.info(
            "act-calib: collecting statistics over %d episode(s) x %d task(s) with the un-quantized policy",
            args.act_calib_episodes, len(cal_tasks),
        )
        ep0 = args.act_calib_from
        raw = collect_stats(
            layers,
            lambda: runner.run(
                adapter, None, tasks=cal_tasks, episodes=range(ep0, ep0 + args.act_calib_episodes), log_every=100
            ),
        )
        torch.save(raw, cache)
        log.info("act-calib: cached statistics to %s", cache)
    bank = build_calibration(layers, raw, alpha=args.act_calib_alpha, clip_quantile=args.act_calib_quantile)
    log.info("act-calib: calibrating %d layers in components %s", len(bank), "+".join(comps))
    args._act_calib_provenance = {
        "components": comps, "episodes": args.act_calib_episodes, "episodes_from": args.act_calib_from,
        "tasks": args.act_calib_tasks, "seed": args.seed, "alpha": args.act_calib_alpha,
        "clip_quantile": args.act_calib_quantile, "cache": cache.name, "commit": git_commit(),
    }
    return bank


def apply_quantization(adapter, args, act_calib: dict | None = None) -> dict[str, Any]:
    if getattr(adapter, "name", "") == "remote":
        # quantization lives on the server; record what it reports
        return {"remote": adapter.url, **adapter.info.get("quant_report", {})}
    cmap = adapter.component_map()
    spec = parse_spec(args.preset)
    method = (args.method or "rtn").lower()
    if method == "rtn":
        # `serve` and other callers may not define the run-only flags
        for pattern in (getattr(args, "exclude", None) or []):
            for comp in resolve_scope(args.scope, cmap):
                cmap.exclude.setdefault(comp, []).append(pattern)
        if getattr(args, "only", None):
            # keep everything that does NOT match, i.e. quantize only the named layers
            for comp in resolve_scope(args.scope, cmap):
                cmap.exclude.setdefault(comp, []).append("*")
                cmap.allow.setdefault(comp, []).extend(getattr(args, "only", []))
        reports = quantize_scope(
            cmap, spec, scope=args.scope, include_conv=args.include_conv,
            quantize_lm_head=args.quantize_lm_head, compute_error=args.report_error,
            act_calib=act_calib,
        )
        out = {c: {"layers": r.n_layers, "params": r.n_params, "mean_rel_mse": r.mean_rel_mse()} for c, r in reports.items()}
        if act_calib:
            for c, r in reports.items():
                out[c]["calibrated_layers"] = sum(1 for l in r.layers if l.act_calibrated)
            out["act_calib"] = getattr(args, "_act_calib_provenance", None) or {
                "components": list(args.act_calib), "episodes": getattr(args, "act_calib_episodes", None)}
        return out
    from .methods import apply_method  # heavy deps, imported lazily

    if args.scope not in ("llm",):
        raise SystemExit(f"method {method!r} is an LLM PTQ method and only supports --scope llm")
    return apply_method(adapter, method, spec, calib=args.calib, real_kernel=args.real_kernel)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_presets(_: argparse.Namespace) -> None:
    for name, spec in PRESETS.items():
        print(f"{name:9s} {spec.describe()}")


def cmd_inspect(args: argparse.Namespace) -> None:
    args.checkpoint = resolve_checkpoint(args.model, args.checkpoint, args.benchmark or "libero", args.suite or "default")
    adapter = build_adapter(args)
    info = adapter.describe()
    print(json.dumps(info, indent=2))
    if args.list_layers:
        from .quant.apply import iter_quantizable

        cmap = adapter.component_map()
        for comp in cmap.present():
            print(f"\n== {comp} ==")
            for name, mod in iter_quantizable(cmap.roots(comp)):
                print(f"  {name:70s} {tuple(mod.weight.shape)}")


def cmd_run(args: argparse.Namespace) -> None:
    args.checkpoint = resolve_checkpoint(args.model, args.checkpoint, args.benchmark, args.suite)
    runner_cls = get_runner_cls(args.benchmark)
    runner_kwargs: dict[str, Any] = {}
    for kv in args.bench_kwargs or []:
        k, _, v = kv.partition("=")
        runner_kwargs[k] = yaml.safe_load(v)
    runner = runner_cls(
        args.suite, episodes_per_task=args.episodes, seed=args.seed, max_steps_override=args.max_steps,
        video_dir=args.video_dir, **runner_kwargs,
    )
    adapter = build_adapter(args)
    out_model = getattr(adapter, "served_name", None) if args.model == "remote" else args.model
    out = Path(args.out) if args.out else default_out_path(
        out_model or args.model, args.benchmark, args.suite, args.method or "rtn", args.preset.upper(),
        args.scope, args.quantize_lm_head, args.seed
    )
    if hasattr(runner, "configure_for"):
        runner.configure_for(adapter)  # adopt the model's official benchmark conventions before listing tasks
    act_calib = None
    if getattr(args, "act_calib", None):
        act_calib = _collect_act_calib(adapter, runner, args)
    quant_report = apply_quantization(adapter, args, act_calib=act_calib)
    model_name = getattr(adapter, "served_name", None) if args.model == "remote" else args.model
    header = RunHeader(
        model=model_name or args.model, checkpoint=adapter.checkpoint, benchmark=args.benchmark, suite=args.suite,
        preset=args.preset.upper(), scope=args.scope, method=(args.method or "rtn"),
        baseline_dtype=str(adapter.dtype).replace("torch.", ""), episodes_per_task=runner.episodes_per_task,
        seed=args.seed, quantize_lm_head=args.quantize_lm_head, include_conv=args.include_conv,
        quant_report=quant_report,
        extra={"adapter_options": adapter.options, "runner_options": runner_kwargs, "exclude": args.exclude, "only": args.only,
               "act_calib": (list(args.act_calib) if getattr(args, "act_calib", None) else None)},
    )
    tasks = None
    if args.tasks:
        wanted = {int(t) for t in args.tasks.split(",")}
        tasks = [t for t in runner.tasks() if t.task_id in wanted]
    episodes = range(args.episodes_from, runner.episodes_per_task) if args.episodes_from else None
    with RunWriter(out, header, resume=not args.no_resume) as writer:
        summary = runner.run(adapter, writer, tasks=tasks, episodes=episodes)
    if hasattr(runner, "close"):
        runner.close()
    summary["out"] = str(out)
    print(json.dumps(summary, indent=2))


def cmd_serve(args: argparse.Namespace) -> None:
    from .remote import PolicyServer

    args.checkpoint = resolve_checkpoint(args.model, args.checkpoint, args.benchmark or "libero", args.suite or "default")
    adapter = build_adapter(args)
    quant_report = apply_quantization(adapter, args)
    log.info("serving %s %s | %s %s -> %s", args.model, args.checkpoint, args.method, args.preset, json.dumps(quant_report))
    PolicyServer(adapter, quant_report, host=args.host, port=args.port).serve_forever()


def cmd_tasks(args: argparse.Namespace) -> None:
    runner_kwargs: dict[str, Any] = {}
    for kv in args.bench_kwargs or []:
        k, _, v = kv.partition("=")
        runner_kwargs[k] = yaml.safe_load(v)
    runner = get_runner_cls(args.benchmark)(args.suite, **runner_kwargs)
    tasks = runner.tasks()
    total = 0
    for t in tasks:
        n = len(runner.episodes_for(t))
        total += n
        print(f"{t.task_id:4d}  {t.task_name[:78]:78s}  eps={n:4d}  max_steps={t.max_steps}")
    print(f"# {len(tasks)} tasks, {total} episodes ({args.benchmark}/{args.suite})", file=sys.stderr)


def cmd_summarize(args: argparse.Namespace) -> None:
    from .summarize import collect, pivot, render_csv, render_latex, render_markdown, to_json

    cells = collect(args.root)
    if args.benchmark:
        cells = {k: v for k, v in cells.items() if k[0] == args.benchmark}
    if args.model:
        cells = {k: v for k, v in cells.items() if k[2] == args.model}
    if args.format == "json":
        print(to_json(cells))
        return
    rows = tuple(args.rows.split(","))
    fields, cols, table = pivot(cells, rows=rows)
    if args.format == "md":
        print(render_markdown(fields, cols, table, ci=args.ci))
    elif args.format == "latex":
        print(render_latex(fields, cols, table, ci=args.ci, caption=args.caption))
    else:
        print(render_csv(fields, cols, table))


def expand_matrix(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand an experiment YAML into a list of run dicts (see configs/experiments)."""
    jobs: list[dict[str, Any]] = []
    defaults = cfg.get("defaults", {})
    for entry in cfg["runs"]:
        e = {**defaults, **entry}
        models = e.pop("models", [e.pop("model", None)])
        benches = e.pop("benchmarks", [e.pop("benchmark", None)])
        presets = e.pop("presets", [e.pop("preset", "BASELINE")])
        scopes = e.pop("scopes", [e.pop("scope", "e2e")])
        methods = e.pop("methods", [e.pop("method", "rtn")])
        for model in models:
            for bench in benches:
                bname, suites = (bench["name"], bench.get("suites", ["default"])) if isinstance(bench, dict) else (bench, ["default"])
                for suite in suites:
                    for method in methods:
                        for preset in presets:
                            for scope in scopes:
                                jobs.append({**e, "model": model, "benchmark": bname, "suite": suite,
                                             "method": method, "preset": preset, "scope": scope})
    return jobs


def job_to_cmd(job: dict[str, Any]) -> list[str]:
    cmd = ["vqb", "run", "--model", job["model"], "--benchmark", job["benchmark"], "--suite", job["suite"],
           "--preset", str(job["preset"]), "--scope", job["scope"], "--method", job["method"]]
    for k in ("checkpoint", "episodes", "seed", "dtype", "device", "out", "max_steps", "calib"):
        if job.get(k) is not None:
            cmd += [f"--{k.replace('_', '-')}", str(job[k])]
    for flag in ("quantize_lm_head", "include_conv", "real_kernel", "no_resume"):
        if job.get(flag):
            cmd.append(f"--{flag.replace('_', '-')}")
    return cmd


def cmd_matrix(args: argparse.Namespace) -> None:
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    jobs = expand_matrix(cfg)
    if args.only_model:
        jobs = [j for j in jobs if j["model"] in args.only_model]
    if args.only_benchmark:
        jobs = [j for j in jobs if j["benchmark"] in args.only_benchmark]
    envs = {m: load_model_config(m).get("env") for m in {j["model"] for j in jobs}}
    if args.print or not args.execute:
        for j in jobs:
            env = envs.get(j["model"])
            prefix = f"conda run -n {env} " if env else ""
            print(prefix + " ".join(job_to_cmd(j)))
        print(f"# {len(jobs)} runs", file=sys.stderr)
        return
    import subprocess

    failures = []
    for i, j in enumerate(jobs, 1):
        cmd = job_to_cmd(j)
        env = envs.get(j["model"])
        if env and os.environ.get("CONDA_DEFAULT_ENV") != env:
            cmd = ["conda", "run", "--no-capture-output", "-n", env] + cmd
        log.info("[%d/%d] %s", i, len(jobs), " ".join(cmd))
        rc = subprocess.call(cmd)
        if rc != 0:
            failures.append((j, rc))
            if not args.keep_going:
                raise SystemExit(f"run failed (rc={rc}): {' '.join(cmd)}")
    if failures:
        print(f"{len(failures)} runs failed:", file=sys.stderr)
        for j, rc in failures:
            print(f"  rc={rc}: {' '.join(job_to_cmd(j))}", file=sys.stderr)


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vqb", description="VLAQuantBench CLI")
    p.add_argument("--version", action="version", version=f"vqb {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("presets", help="list quantization presets").set_defaults(func=cmd_presets)

    def add_model_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--model", required=True, choices=sorted(MODELS))
        sp.add_argument("--checkpoint", default="auto", help="HF id / path, or 'auto' to read configs/models/<model>.yaml")
        sp.add_argument("--device", default="cuda")
        sp.add_argument("--dtype", default=None, help="baseline precision override (bf16/fp16/fp32)")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--model-kwargs", nargs="*", metavar="K=V", help="extra adapter options")

    ins = sub.add_parser("inspect", help="load a model and print its component decomposition")
    add_model_args(ins)
    ins.add_argument("--benchmark", default=None, help="used only to resolve 'auto' checkpoints")
    ins.add_argument("--suite", default=None)
    ins.add_argument("--list-layers", action="store_true")
    ins.set_defaults(func=cmd_inspect)

    run = sub.add_parser("run", help="evaluate one (model, benchmark, suite, preset, scope) cell")
    add_model_args(run)
    run.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    run.add_argument("--suite", required=True)
    run.add_argument("--preset", default="BASELINE", help="W3 | W4 | W8 | W4A4 | W4A8 | W8A8 | W2 | BASELINE | custom e.g. W3G64")
    run.add_argument("--scope", default="e2e", help="e2e | ve | mp | llm | ah | llm+ah ...")
    run.add_argument("--method", default="rtn", help="rtn | awq | nf4 | int8 | smoothquant (LLM-only methods need --scope llm)")
    run.add_argument("--calib", default=None, help="calibration data override for LLM PTQ methods")
    run.add_argument("--real-kernel", action="store_true", help="execute with the method's real kernel instead of fake quant")
    run.add_argument("--quantize-lm-head", action="store_true")
    run.add_argument("--include-conv", action="store_true", help="also quantize Conv2d weights (patch embeddings)")
    run.add_argument("--act-calib", nargs="*", metavar="COMP", default=None,
                     help="calibrate activation quantization for these components (SmoothQuant-style "
                          "smoothing folded into weights + clipping) using closed-loop rollouts of the "
                          "un-quantized policy; other quantized components keep plain per-token absmax")
    run.add_argument("--act-calib-alpha", type=float, default=0.5,
                     help="SmoothQuant migration strength (default 0.5)")
    run.add_argument("--act-calib-quantile", type=float, default=0.999,
                     help="clip quantile: 0.999 (default) or 0.99 for harder clipping")
    run.add_argument("--act-calib-from", type=int, default=20,
                     help="first init-state index used for calibration rollouts (default 20: disjoint from the "
                          "evaluation states 0-19; the 2026-08 cells were collected with --act-calib-from 0)")
    run.add_argument("--act-calib-episodes", type=int, default=2,
                     help="episodes per calibration task (default 2)")
    run.add_argument("--act-calib-tasks", type=int, default=1,
                     help="number of suite tasks used for calibration (default 1)")
    run.add_argument("--only", nargs="*", metavar="GLOB", default=None,
                     help="quantize ONLY layers matching these patterns (the dual of --exclude); "
                          "use to isolate a layer group, e.g. --only '*layernorm.dense*'")
    run.add_argument("--exclude", nargs="*", metavar="GLOB", default=None,
                     help="keep layers matching these name patterns at baseline precision "
                          "(ablation, e.g. --exclude '*layernorm.dense*'); recorded in the run header")
    run.add_argument("--report-error", action="store_true", help="record per-layer weight reconstruction error")
    run.add_argument("--episodes", type=int, default=None, help="episodes per task (default: official protocol)")
    run.add_argument("--episodes-from", type=int, default=0)
    run.add_argument("--tasks", default=None, help="comma-separated task ids subset")
    run.add_argument("--max-steps", type=int, default=None)
    run.add_argument("--video-dir", default=None)
    run.add_argument("--bench-kwargs", nargs="*", metavar="K=V")
    run.add_argument("--out", default=None)
    run.add_argument("--no-resume", action="store_true")
    run.set_defaults(func=cmd_run)

    srv = sub.add_parser("serve", help="host an adapter (with quantization applied) for `--model remote` clients")
    add_model_args(srv)
    srv.add_argument("--benchmark", default=None, help="used to resolve 'auto' checkpoints")
    srv.add_argument("--suite", default=None)
    srv.add_argument("--preset", default="BASELINE")
    srv.add_argument("--scope", default="e2e")
    srv.add_argument("--method", default="rtn")
    srv.add_argument("--calib", default=None)
    srv.add_argument("--real-kernel", action="store_true")

    sm = sub.add_parser("summarize", help="aggregate results into tables")
    sm.add_argument("root", nargs="?", default=str(REPO_ROOT / "results"))
    sm.add_argument("--format", choices=("md", "latex", "csv", "json"), default="md")
    sm.add_argument("--rows", default="benchmark,suite,model,method,scope")
    sm.add_argument("--benchmark", default=None)
    sm.add_argument("--model", default=None)
    sm.add_argument("--ci", action="store_true", help="print 95%% confidence intervals")
    sm.add_argument("--caption", default="")
    sm.set_defaults(func=cmd_summarize)

    mx = sub.add_parser("matrix", help="expand an experiment yaml into runs")
    mx.add_argument("config")
    mx.add_argument("--print", action="store_true", help="print the commands (default unless --execute)")
    mx.add_argument("--execute", action="store_true", help="run sequentially (dispatching to each model's conda env)")
    mx.add_argument("--keep-going", action="store_true")
    mx.add_argument("--only-model", nargs="*")
    mx.add_argument("--only-benchmark", nargs="*")
    mx.set_defaults(func=cmd_matrix)

    lt = sub.add_parser("tasks", help="list a benchmark suite's official task table")
    lt.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    lt.add_argument("--suite", required=True)
    lt.add_argument("--bench-kwargs", nargs="*", metavar="K=V")
    lt.set_defaults(func=cmd_tasks)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
