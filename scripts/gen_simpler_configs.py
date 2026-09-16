#!/usr/bin/env python
"""Expand SimplerEnv's official evaluation scripts into JSON task tables.

The Visual-Matching (VM) and Variant-Aggregation (VA) protocols are defined
*only* by the nested bash loops in ``SimplerEnv/scripts/rt1_*.sh`` (the RT-1,
Octo and CogACT scripts pass identical environment arguments). Instead of
transcribing them by hand, this script runs every script through bash with
``python`` replaced by a function that prints its arguments, and stores each
``main_inference.py`` invocation's argument vector verbatim::

    configs/simpler/google_robot_vm.json
    configs/simpler/google_robot_va.json
    configs/simpler/widowx_vm.json

At run time :class:`vlaquantbench.benchmarks.simpler.SimplerRunner` feeds each
argument vector back into SimplerEnv's own ``get_args()`` parser and enumerates
episodes exactly like ``maniskill2_evaluator`` does. Re-run this script when
the pinned SimplerEnv commit changes (``--simpler third_party/SimplerEnv``).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIMPLER = ROOT / "third_party" / "SimplerEnv"

SUITES = {
    "google_robot_vm": {
        "pick_coke_can": "rt1_pick_coke_can_visual_matching.sh",
        "move_near": "rt1_move_near_visual_matching.sh",
        "drawer": "rt1_drawer_visual_matching.sh",
        "place_in_drawer": "rt1_put_in_drawer_visual_matching.sh",
    },
    "google_robot_va": {
        "pick_coke_can": "rt1_pick_coke_can_variant_agg.sh",
        "move_near": "rt1_move_near_variant_agg.sh",
        "drawer": "rt1_drawer_variant_agg.sh",
        "place_in_drawer": "rt1_put_in_drawer_variant_agg.sh",
    },
    "widowx_vm": {"bridge": "rt1x_bridge.sh"},
}

# invocation args that belong to the policy, not to the environment protocol
POLICY_ARGS = {"--policy-model", "--ckpt-path", "--policy-setup", "--octo-init-rng", "--logging-dir", "--action-scale"}


def expand(script: Path) -> list[list[str]]:
    """Run ``script`` with ``python`` stubbed out; return the argument vectors it would have executed."""
    stub = (
        'python() { printf "%s\\n" "$*"; }\n'
        'python3() { python "$@"; }\n'
        "export -f python python3\n"
        f"cd {shlex.quote(str(script.parent.parent))}\n"
        f"source {shlex.quote(str(script))}\n"
    )
    out = subprocess.run(["bash", "-c", stub], capture_output=True, text=True, check=True).stdout
    argvs = []
    for line in out.splitlines():
        if "main_inference.py" not in line:
            continue
        toks = shlex.split(line)
        toks = toks[toks.index("simpler_env/main_inference.py") + 1 :]
        argvs.append(_strip_policy_args(toks))
    return argvs


def _strip_policy_args(toks: list[str]) -> list[str]:
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t in POLICY_ARGS:
            i += 2
            continue
        out.append(t)
        i += 1
    return out


def dedupe_keep_order(items: list[list[str]]) -> list[list[str]]:
    seen, out = set(), []
    for it in items:
        key = tuple(it)
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out


def describe_variant(argv: list[str]) -> dict:
    """Human-readable tags used by the official metric aggregation."""
    d: dict = {}
    it = iter(range(len(argv)))
    for i in it:
        if argv[i] == "--env-name":
            d["env_name"] = argv[i + 1]
        elif argv[i] == "--scene-name":
            d["scene_name"] = argv[i + 1]
        elif argv[i] == "--rgb-overlay-path":
            d["overlay"] = os.path.basename(argv[i + 1])
        elif argv[i] == "--additional-env-build-kwargs":
            kws = []
            j = i + 1
            while j < len(argv) and not argv[j].startswith("--"):
                kws.append(argv[j])
                j += 1
            d["env_kwargs"] = kws
    env = d.get("env_name", "")
    d["camera_variant"] = "AltGoogleCamera" in env  # excluded from the official VA score
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simpler", default=str(DEFAULT_SIMPLER))
    ap.add_argument("--out", default=str(ROOT / "configs" / "simpler"))
    args = ap.parse_args()
    simpler = Path(args.simpler)
    commit = subprocess.run(["git", "-C", str(simpler), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    os.makedirs(args.out, exist_ok=True)
    for suite, families in SUITES.items():
        entries = []
        for family, script_name in families.items():
            script = simpler / "scripts" / script_name
            argvs = dedupe_keep_order(expand(script))  # the scripts loop over several checkpoints -> identical env args
            for argv in argvs:
                entries.append({"family": family, "script": script_name, "argv": argv, **describe_variant(argv)})
            print(f"{suite:16s} {family:16s} {len(argvs):4d} configs  ({script_name})")
        payload = {"source": "SimplerEnv/scripts", "commit": commit, "suite": suite, "configs": entries}
        with open(Path(args.out) / f"{suite}.json", "w") as fh:
            json.dump(payload, fh, indent=1)
        print(f"-> {Path(args.out) / f'{suite}.json'}: {len(entries)} configs")


if __name__ == "__main__":
    main()
