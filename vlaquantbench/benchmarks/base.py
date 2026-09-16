"""Common closed-loop evaluation loop shared by all benchmark runners.

A runner knows how to enumerate the official task list of a benchmark, build
an environment for a task, reset it to a reproducible initial state and
step it. The per-episode loop, bookkeeping, latency measurement and
resumable persistence live here so that every benchmark is evaluated in the
same way.
"""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass
from typing import Any, ClassVar, Iterable

import numpy as np

from ..models.base import Observation, TaskSpec, VLAAdapter
from ..profiling import LatencyMeter, reset_peak_memory
from ..results import EpisodeRecord, RunWriter
from ..stats import mean_summary, success_summary

__all__ = ["EpisodeOutcome", "BenchmarkRunner"]

log = logging.getLogger(__name__)


@dataclass
class EpisodeOutcome:
    success: bool
    steps: int
    subtasks_completed: int | None = None
    progress_score: float | None = None
    extra: dict[str, Any] | None = None


class BenchmarkRunner(abc.ABC):
    name: ClassVar[str] = "base"
    #: metric reported by the benchmark's official protocol
    primary_metric: ClassVar[str] = "success_rate"

    def __init__(
        self,
        suite: str,
        *,
        episodes_per_task: int | None = None,
        seed: int = 0,
        max_steps_override: int | None = None,
        video_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.suite = suite
        self.episodes_per_task = episodes_per_task or self.default_episodes_per_task()
        self.seed = seed
        self.max_steps_override = max_steps_override
        self.video_dir = video_dir
        self.options = kwargs

    # ------------------------------------------------------------------ #
    # to implement per benchmark
    # ------------------------------------------------------------------ #
    @classmethod
    @abc.abstractmethod
    def suites(cls) -> tuple[str, ...]:
        """Official suite / protocol names (e.g. ``libero_spatial``)."""

    @classmethod
    @abc.abstractmethod
    def default_episodes_per_task(cls) -> int:
        """Episode count per task prescribed by the official protocol."""

    @abc.abstractmethod
    def tasks(self) -> list[TaskSpec]:
        """Ordered list of tasks in ``self.suite``."""

    @abc.abstractmethod
    def run_episode(self, adapter: VLAAdapter, task: TaskSpec, episode: int, meter: LatencyMeter) -> EpisodeOutcome:
        """Roll out one episode; must call ``adapter.reset(task)`` first and wrap
        every ``adapter.act`` call in ``with meter.measure():``."""

    # ------------------------------------------------------------------ #
    # shared loop
    # ------------------------------------------------------------------ #
    def episode_seed(self, task: TaskSpec, episode: int) -> int:
        return int(self.seed * 1_000_003 + task.task_id * 1009 + episode)

    def run(
        self,
        adapter: VLAAdapter,
        writer: RunWriter | None = None,
        *,
        tasks: Iterable[TaskSpec] | None = None,
        episodes: Iterable[int] | None = None,
        log_every: int = 1,
    ) -> dict[str, Any]:
        task_list = list(tasks) if tasks is not None else self.tasks()
        records: list[EpisodeRecord] = []
        for task in task_list:
            ep_list = list(episodes) if episodes is not None else self.episodes_for(task)
            for ep in ep_list:
                if writer is not None and writer.is_done(task.suite, task.task_id, ep):
                    continue
                meter = LatencyMeter(warmup=3)
                infer = adapter.new_inference_meter(warmup=1)
                reset_peak_memory()
                t0 = time.perf_counter()
                outcome = self.run_episode(adapter, task, ep, meter)
                rec = EpisodeRecord(
                    suite=task.suite,
                    task_id=task.task_id,
                    task_name=task.task_name,
                    episode=ep,
                    seed=self.episode_seed(task, ep),
                    success=bool(outcome.success),
                    steps=int(outcome.steps),
                    wall_time_s=time.perf_counter() - t0,
                    subtasks_completed=outcome.subtasks_completed,
                    progress_score=outcome.progress_score,
                    policy_latency_mean_s=meter.mean(),
                    policy_latency_p95_s=meter.p95(),
                    inference_latency_mean_s=infer.mean(),
                    inference_latency_p95_s=infer.p95(),
                    n_inferences=infer._seen,
                    peak_vram_gb=adapter.peak_vram_gb(),
                    extra=outcome.extra or {},
                )
                records.append(rec)
                if writer is not None:
                    writer.add(rec)
                if log_every and (len(records) % log_every == 0):
                    sr = np.mean([r.success for r in records])
                    log.info(
                        "[%s/%s] task %d ep %d -> %s (%d steps) | running SR %.1f%% over %d eps",
                        self.name, task.suite, task.task_id, ep, "OK" if rec.success else "fail", rec.steps,
                        100 * sr, len(records),
                    )
        return self.summarize(records)

    def summarize(self, records: list[EpisodeRecord]) -> dict[str, Any]:
        out: dict[str, Any] = {"n_episodes": len(records)}
        if not records:
            return out
        out["success_rate"] = success_summary([r.success for r in records]).as_dict()
        if any(r.subtasks_completed is not None for r in records):
            out["avg_len"] = mean_summary([r.subtasks_completed or 0 for r in records]).as_dict()
        if any(r.progress_score is not None for r in records):
            out["progress_score"] = mean_summary([r.progress_score or 0.0 for r in records]).as_dict()
        lat = [r.policy_latency_mean_s for r in records if r.policy_latency_mean_s]
        if lat:
            out["policy_latency_mean_s"] = float(np.mean(lat))
        inf = [r.inference_latency_mean_s for r in records if r.inference_latency_mean_s]
        if inf:
            out["inference_latency_mean_s"] = float(np.mean(inf))
        vram = [r.peak_vram_gb for r in records if r.peak_vram_gb]
        if vram:
            out["peak_vram_gb"] = float(np.max(vram))
        return out

    # ------------------------------------------------------------------ #
    # helpers for subclasses
    # ------------------------------------------------------------------ #
    def episodes_for(self, task: TaskSpec) -> list[int]:
        """Episode indices for ``task`` (uniform by default; SIMPLER/CALVIN override)."""
        return list(range(self.episodes_per_task))

    def max_steps(self, default: int) -> int:
        return self.max_steps_override or default

    @staticmethod
    def make_obs(images: dict[str, np.ndarray], instruction: str, state: np.ndarray | None, step: int, raw: Any) -> Observation:
        return Observation(images=images, instruction=instruction, state=state, step=step, raw=raw)
