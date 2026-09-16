"""Lazy registries for model adapters and benchmark runners.

Adapters import heavy, mutually incompatible third-party stacks, so they are
only imported when requested by name.
"""

from __future__ import annotations

import importlib
from typing import Type

__all__ = ["MODELS", "BENCHMARKS", "get_adapter_cls", "get_runner_cls"]

#: registry key -> "module:Class"
MODELS: dict[str, str] = {
    "openvla": "vlaquantbench.models.openvla:OpenVLAAdapter",
    "openvla_oft": "vlaquantbench.models.openvla_oft:OpenVLAOFTAdapter",
    "pi0": "vlaquantbench.models.pi0_lerobot:Pi0Adapter",
    "pi05": "vlaquantbench.models.pi0_lerobot:Pi05Adapter",
    "cogact": "vlaquantbench.models.cogact:CogActAdapter",
    "internvla_m1": "vlaquantbench.models.internvla_m1:InternVLAM1Adapter",
    "xvla": "vlaquantbench.models.xvla:XVLAAdapter",
    "remote": "vlaquantbench.remote:RemoteAdapter",  # client for `vqb serve`
}

BENCHMARKS: dict[str, str] = {
    "libero": "vlaquantbench.benchmarks.libero:LiberoRunner",
    "simpler": "vlaquantbench.benchmarks.simpler:SimplerRunner",
    "calvin": "vlaquantbench.benchmarks.calvin:CalvinRunner",
    "vlabench": "vlaquantbench.benchmarks.vlabench:VLABenchRunner",
}


def _load(spec: str):
    mod_name, _, cls_name = spec.partition(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


def get_adapter_cls(name: str) -> Type:
    key = name.lower().replace("-", "_")
    if key not in MODELS:
        raise KeyError(f"unknown model {name!r}; available: {sorted(MODELS)}")
    return _load(MODELS[key])


def get_runner_cls(name: str) -> Type:
    key = name.lower().replace("-", "_")
    if key not in BENCHMARKS:
        raise KeyError(f"unknown benchmark {name!r}; available: {sorted(BENCHMARKS)}")
    return _load(BENCHMARKS[key])
