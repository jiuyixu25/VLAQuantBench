"""Adapter interface between a VLA implementation and the benchmark runners.

An adapter owns *everything model-specific*: loading the official checkpoint,
the official observation preprocessing for a given benchmark (image flips,
resizing, prompt templates, proprio formatting), action chunking /
un-normalisation, and the :class:`~vlaquantbench.components.ComponentMap`
that tells the quantizer where each functional component lives.

Runners only ever call :meth:`VLAAdapter.reset` once per episode and
:meth:`VLAAdapter.act` once per simulator step, passing the raw simulator
observation wrapped in an :class:`Observation`.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import torch
from torch import nn

from ..components import ComponentMap
from ..profiling import LatencyMeter

__all__ = ["Observation", "TaskSpec", "VLAAdapter", "resolve_dtype"]

_DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def resolve_dtype(name: str | torch.dtype | None, default: torch.dtype = torch.bfloat16) -> torch.dtype:
    if name is None:
        return default
    if isinstance(name, torch.dtype):
        return name
    try:
        return _DTYPES[name.lower()]
    except KeyError as e:
        raise ValueError(f"unknown dtype {name!r}; expected one of {sorted(_DTYPES)}") from e


@dataclass
class TaskSpec:
    benchmark: str
    suite: str
    task_id: int
    task_name: str
    instruction: str
    max_steps: int
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    """Raw simulator observation.

    ``images`` maps a camera name to an ``HxWx3`` uint8 RGB array **exactly as
    the simulator returned it** (runners never flip/resize; adapters apply the
    official preprocessing of their model). ``state`` is the benchmark's native
    proprioceptive vector (or ``None``). ``raw`` is the untouched simulator
    observation for adapters that need more than the common fields.
    """

    images: dict[str, np.ndarray]
    instruction: str
    state: np.ndarray | None = None
    step: int = 0
    raw: Any = None


class VLAAdapter(abc.ABC):
    """Base class for all model adapters."""

    #: registry key, e.g. ``"openvla_oft"``
    name: ClassVar[str] = "base"
    #: dtype the released checkpoint is meant to run in (the *baseline precision*)
    default_dtype: ClassVar[str] = "bf16"
    #: benchmarks this adapter implements preprocessing for
    supported_benchmarks: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str | torch.device = "cuda",
        dtype: str | torch.dtype | None = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        self.dtype = resolve_dtype(dtype, resolve_dtype(self.default_dtype))
        self.seed = seed
        self.options = kwargs
        self.model: nn.Module | None = None
        self._cmap: ComponentMap | None = None
        #: timing of *model inference calls* (observation -> action chunk). Adapters wrap the
        #: model call in ``with self.inference_meter.measure():``; runners reset it per episode.
        #: This is distinct from the per-env-step latency measured around ``act()``.
        self.inference_meter: LatencyMeter = LatencyMeter(warmup=2)

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def load(self) -> None:
        """Load the checkpoint onto ``self.device`` in ``self.dtype`` and set ``self.model``."""

    @abc.abstractmethod
    def build_component_map(self) -> ComponentMap:
        """Return the module roots of ``ve`` / ``mp`` / ``llm`` / ``ah``."""

    def component_map(self) -> ComponentMap:
        if self._cmap is None:
            if self.model is None:
                raise RuntimeError("call load() before component_map()")
            self._cmap = self.build_component_map()
            self._cmap.check_disjoint()
        return self._cmap

    def unload(self) -> None:
        self.model = None
        self._cmap = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # closed-loop interface
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def reset(self, task: TaskSpec) -> None:
        """Start a new episode (clear action queues / history, set the instruction)."""

    @abc.abstractmethod
    def act(self, obs: Observation, task: TaskSpec) -> np.ndarray:
        """Return **one** environment action for the current step.

        Adapters that predict action chunks must queue them internally and
        re-plan exactly as their official evaluation code does.
        """

    def new_inference_meter(self, warmup: int = 2) -> LatencyMeter:
        self.inference_meter = LatencyMeter(warmup=warmup)
        return self.inference_meter

    def peak_vram_gb(self) -> float | None:
        """Peak CUDA allocation of the policy process (remote adapters query their server)."""
        from ..profiling import peak_memory_gb

        return peak_memory_gb()

    # ------------------------------------------------------------------ #
    # optional hooks for the LLM-only PTQ experiment
    # ------------------------------------------------------------------ #
    def llm_causal_lm(self) -> nn.Module | None:
        """The HF-style causal LM backbone (``*ForCausalLM``) if the model has one."""
        return None

    def llm_tokenizer(self):
        return None

    # ------------------------------------------------------------------ #
    def describe(self) -> dict[str, Any]:
        cm = self.component_map()
        return {
            "model": self.name,
            "checkpoint": self.checkpoint,
            "dtype": str(self.dtype).replace("torch.", ""),
            "components": cm.param_table(),
            "present": cm.present(),
            "notes": cm.notes,
        }
