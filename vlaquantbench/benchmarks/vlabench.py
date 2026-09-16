"""VLABench runner (Zhang et al., 2025) - official VLA evaluation tracks.

Mirrors ``VLABench.evaluation.evaluator.base.Evaluator.evaluate_single_episode``:

* tasks and per-episode scene configurations come from
  ``VLABench/configs/evaluation/tracks/<track>.json`` (10 tasks, 50 pre-generated
  episodes each; track 2 ships 10 for ``insert_flower``); the env is built with
  ``load_env(task, episode_config=cfg, random_init=False, run_mode="eval")``
  so the layout, instruction, conditions and targets are deterministic;
* ``max_episode_length`` per task family from ``configs/task_config.json``
  (default 200; ``select_poker``/``select_painting`` 100, ``add_condiment`` 300);
* per step the policy sees ``rgb`` ``(4, 480, 480, 3)`` (cameras right, left,
  front, wrist), ``ee_state`` ``[pos(3), quat_wxyz(4), gripper(1)]``,
  ``instruction``, ``robot_frame`` and ``last_action``; end-effector policies
  return ``(pos, euler_xyz, gripper_state(2))`` which is converted to joint
  targets by the env's IK (``robot.get_qpos_from_ee_pos``) exactly as upstream;
* ``env.step`` is repeated up to ``max_substeps`` (breaking early once the
  joint targets are reached within ``tolerance``) and the episode ends when
  ``timestep.last()`` (all task conditions met). ``max_substeps`` is
  **model-dependent**: ``scripts/evaluate_policy.py`` uses 1, while X-VLA's own
  client uses 10 (absolute-pose chunks need time to be reached). Adapters
  declare ``vlabench_max_substeps``; the value used is recorded per episode;
* metrics per episode: ``success``, ``progress_score`` (fraction of satisfied
  conditions plus target entities ever grasped) and ``intention_score``
  (minimum EE-to-target distance below 0.1 m). The headline number of the
  benchmark is the mean progress score.

Upstream catches per-episode exceptions and silently drops those episodes; we
record them instead (``extra.error``) so the denominator stays visible.

Requires the ``VLABench`` package, its asset download (~5.6 GB) and
``MUJOCO_GL=egl`` exported *before* ``dm_control`` is imported.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ..models.base import TaskSpec, VLAAdapter
from ..profiling import LatencyMeter
from .base import BenchmarkRunner, EpisodeOutcome

__all__ = ["VLABenchRunner", "TRACKS"]

log = logging.getLogger(__name__)

TRACKS = (
    "track_1_in_distribution",
    "track_2_cross_category",
    "track_3_common_sense",
    "track_4_semantic_instruction",
    "track_6_unseen_texture",
)
DEFAULT_MAX_EPISODE_LENGTH = 200


class VLABenchRunner(BenchmarkRunner):
    name: ClassVar[str] = "vlabench"
    primary_metric: ClassVar[str] = "progress_score"

    def __init__(self, suite: str = "track_1_in_distribution", *, vlabench_root: str | None = None,
                 max_substeps: int | None = None, tolerance: float = 1e-2, intention_score_threshold: float = 0.1,
                 eval_unseen: bool = False, tasks_filter: str | None = None, **kwargs: Any):
        if suite not in self.suites():
            raise ValueError(f"unknown VLABench track {suite!r}; expected one of {self.suites()}")
        super().__init__(suite, **kwargs)
        self.vlabench_root = vlabench_root or os.environ.get("VLABENCH_ROOT")
        self.max_substeps = max_substeps
        self.tolerance = tolerance
        self.intention_score_threshold = intention_score_threshold
        self.eval_unseen = eval_unseen
        self.tasks_filter = set(tasks_filter.split(",")) if tasks_filter else None
        os.environ.setdefault("MUJOCO_GL", "egl")
        self._episode_config: dict[str, list] | None = None
        self._task_configs: dict[str, Any] | None = None

    @classmethod
    def suites(cls) -> tuple[str, ...]:
        return TRACKS

    @classmethod
    def default_episodes_per_task(cls) -> int:
        return 50

    #: fallback when neither the CLI nor the adapter specifies one (official script value)
    DEFAULT_MAX_SUBSTEPS: ClassVar[int] = 1

    def configure_for(self, adapter: VLAAdapter) -> None:
        if self.max_substeps is None:
            self.max_substeps = int(getattr(adapter, "vlabench_max_substeps", 0) or self.DEFAULT_MAX_SUBSTEPS)

    def run(self, adapter: VLAAdapter, writer=None, **kwargs):  # type: ignore[override]
        self.configure_for(adapter)
        tasks = kwargs.pop("tasks", None)
        task_list = list(tasks) if tasks is not None else self.tasks()
        buildable, skipped = [], []
        for t in task_list:
            try:
                env = self._build_env(t, 0)
                try:
                    env.close()
                except Exception:  # pragma: no cover
                    pass
                buildable.append(t)
            except Exception as e:  # noqa: BLE001
                # Some upstream assets do not compile under this MuJoCo build
                # (e.g. "mesh volume is too small" for a handful of dish/pan
                # meshes). Drop the task loudly instead of letting one broken
                # asset kill the whole (setting, track) cell; the summary
                # records exactly what was skipped so coverage stays auditable.
                reason = str(e).strip().splitlines()[-1][:200] if str(e).strip() else type(e).__name__
                log.warning("vlabench: skipping task %s — env build failed: %s", t.task_name, reason)
                skipped.append({"task": t.task_name, "reason": reason})
        summary = super().run(adapter, writer, tasks=buildable, **kwargs)
        summary["skipped_tasks"] = skipped
        summary["n_tasks_run"] = len(buildable)
        return summary

    # ------------------------------------------------------------------ #
    @staticmethod
    def _register_entities() -> None:
        """VLABench registers tasks/robots as an import side effect (``scripts/evaluate_policy.py``
        does ``from VLABench.tasks import *`` / ``from VLABench.robots import *``)."""
        import VLABench.robots  # noqa: F401  # type: ignore
        import VLABench.tasks  # noqa: F401  # type: ignore

    def _root(self) -> Path:
        if self.vlabench_root:
            return Path(self.vlabench_root)
        import VLABench  # type: ignore

        return Path(os.environ["VLABENCH_ROOT"]) if os.environ.get("VLABENCH_ROOT") else Path(VLABench.__file__).parent

    def _configs(self) -> dict[str, list]:
        if self._episode_config is None:
            path = self._root() / "configs" / "evaluation" / "tracks" / f"{self.suite}.json"
            with open(path) as fh:
                self._episode_config = json.load(fh)
        return self._episode_config

    def _task_config(self) -> dict[str, Any]:
        if self._task_configs is None:
            with open(self._root() / "configs" / "task_config.json") as fh:
                self._task_configs = json.load(fh)
        return self._task_configs

    def _max_episode_length(self, task_name: str) -> int:
        cfg = self._task_config()
        entry = cfg.get(f"{task_name}_series", {})
        return int(entry.get("evaluation", {}).get("max_episode_length", DEFAULT_MAX_EPISODE_LENGTH)
                   if isinstance(entry.get("evaluation"), dict) else entry.get("evaluation", DEFAULT_MAX_EPISODE_LENGTH))

    def tasks(self) -> list[TaskSpec]:
        specs = []
        for tid, (task_name, episodes) in enumerate(sorted(self._configs().items())):
            if self.tasks_filter and task_name not in self.tasks_filter:
                continue
            specs.append(
                TaskSpec(
                    benchmark=self.name,
                    suite=self.suite,
                    task_id=tid,
                    task_name=task_name,
                    instruction="",  # per-episode, read from the env
                    max_steps=self.max_steps(self._max_episode_length(task_name)),
                    extra={"n_episodes": len(episodes)},
                )
            )
        return specs

    def episodes_for(self, task: TaskSpec) -> list[int]:
        return list(range(min(int(task.extra["n_episodes"]), self.episodes_per_task)))

    # ------------------------------------------------------------------ #
    def _build_env(self, task: TaskSpec, episode: int):
        """Construct the dm_control env for one (task, episode) config."""
        self._register_entities()
        from VLABench.envs import load_env  # type: ignore

        cfg = self._configs()[task.task_name][episode]
        return load_env(task.task_name, episode_config=cfg, random_init=False, eval=self.eval_unseen, run_mode="eval")

    def run_episode(self, adapter: VLAAdapter, task: TaskSpec, episode: int, meter: LatencyMeter) -> EpisodeOutcome:
        self.configure_for(adapter)
        from VLABench.utils.utils import euler_to_quaternion, quaternion_to_euler  # type: ignore

        try:
            env = self._build_env(task, episode)
        except Exception as e:  # noqa: BLE001
            # per-episode scene configs pull different assets, so a task that
            # builds for episode 0 can still fail later; record it the way
            # upstream records dropped episodes (extra.error keeps the
            # denominator visible) instead of killing the whole cell
            reason = str(e).strip().splitlines()[-1][:200] if str(e).strip() else type(e).__name__
            log.warning("vlabench: %s ep %d env build failed: %s", task.task_name, episode, reason)
            return EpisodeOutcome(success=False, steps=0, progress_score=None,
                                  extra={"error": f"env build failed: {reason}"})
        try:
            env.reset()
            robot_frame = env.get_robot_frame_position()
            instruction = env.task.get_instruction()
            task.instruction = task.instruction or instruction
            adapter.reset(task)

            success = False
            last_action = None
            steps = 0
            while steps < task.max_steps:
                obs = env.get_observation(require_pcd=False)
                obs["instruction"] = env.task.get_instruction()
                ee_state = np.asarray(obs["ee_state"])
                if last_action is None:
                    last_action = np.concatenate([ee_state[:3], quaternion_to_euler(ee_state[3:7])])
                obs["robot_frame"] = robot_frame
                obs["last_action"] = last_action

                raw = {"rgb": np.asarray(obs["rgb"]), "ee_state": ee_state,
                       "robot_frame": np.asarray(robot_frame), "last_action": np.asarray(last_action),
                       "q_state": np.asarray(obs.get("q_state", [])), "instruction": obs["instruction"]}
                images = {f"cam{i}": np.asarray(obs["rgb"])[i] for i in range(len(obs["rgb"]))}
                observation = self.make_obs(images, obs["instruction"], ee_state, steps, raw)
                with meter.measure():
                    out = adapter.act(observation, task)

                pos, euler, gripper_state = out
                last_action = np.concatenate([np.asarray(pos), np.asarray(euler)])
                quat = euler_to_quaternion(*np.asarray(euler))
                _, qpos = env.robot.get_qpos_from_ee_pos(physics=env.physics, pos=np.asarray(pos), quat=quat)
                action = np.concatenate([qpos, np.asarray(gripper_state)])

                for _ in range(self.max_substeps):
                    timestep = env.step(action)
                    if timestep.last():
                        success = True
                        break
                    current_qpos = np.array(env.task.robot.get_qpos(env.physics)).reshape(-1)
                    diff = current_qpos - np.asarray(action)[:7]
                    if np.max(diff) < self.tolerance and np.min(diff) > -self.tolerance:
                        break
                steps += 1
                if success:
                    break

            intention = float(env.get_intention_score(threshold=self.intention_score_threshold))
            progress = float(env.get_task_progress())
            return EpisodeOutcome(
                success=success, steps=steps, progress_score=progress,
                extra={"intention_score": intention, "max_steps": task.max_steps, "track": self.suite,
                       "max_substeps": self.max_substeps},
            )
        finally:
            try:
                env.close()
            except Exception:  # pragma: no cover
                pass

    def close(self) -> None:
        return None
