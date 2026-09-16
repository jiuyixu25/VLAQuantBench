"""SIMPLER runner (Li et al., 2024) - Google-robot Visual Matching (VM) and
Variant Aggregation (VA) protocols, plus the WidowX/Bridge VM protocol.

The task tables ``configs/simpler/<suite>.json`` are generated from the
official ``SimplerEnv/scripts/rt1_*.sh`` sweeps by
``scripts/gen_simpler_configs.py``; every entry is the verbatim argument vector
of one ``main_inference.py`` invocation. At run time the vector is parsed by
SimplerEnv's own ``get_args()`` and episodes are enumerated exactly like
``maniskill2_evaluator`` (robot x/y/quaternion grid x object xy-grid or
episode ids). Environments are built with ``build_maniskill2_env`` (overlay
cameras, ray tracing, extra build kwargs as in the scripts).

Episode loop (``run_maniskill2_eval_single_episode``): the policy is queried
at every control step (3 Hz Google robot / 5 Hz WidowX) with the RGB frame of
the overhead (Google) or 3rd-view (WidowX) camera and the *current* language
instruction (which changes after a sub-task advance); the episode runs until
the ``TimeLimit`` truncation and **success is the env's ``done`` flag at the
last step** (official semantics - a drawer re-closing before the time limit
flips success back to failure). Adapters may instead request the X-VLA client
semantics (``simpler_stop_on_done=True``, ``simpler_max_steps_factor=2``).

Actions: the adapter returns the 7-D vector that the env's control mode
expects. Default control mode (official): Google robot
``arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner``
(``[world_vector(3), rot_axangle(3), gripper(1)]``, gripper +1 close / -1 open,
sticky-gripper logic is the adapter's job); WidowX
``arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos``. Adapters may
declare ``simpler_control_mode`` (X-VLA needs the absolute base-pose mode of
the ``255isWhite/SimplerEnv`` fork).

Metric: per family (pick_coke_can / move_near / drawer / place_in_drawer)
the official aggregation is an equal-weight mean over variants; camera
variants are excluded from the VA score (``camera_variant``); the suite score
is the mean over families. Records carry ``family``/``variant``/``scored`` so
``vqb summarize`` can reproduce it.

Requirements: SimplerEnv + ManiSkill2_real2sim (sapien 2.2.2, Vulkan, Python
3.10/3.11, numpy < 2). Set ``DISPLAY=""`` for headless Vulkan.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ..models.base import TaskSpec, VLAAdapter
from ..profiling import LatencyMeter
from .base import BenchmarkRunner, EpisodeOutcome

__all__ = ["SimplerRunner"]

log = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "simpler"


@contextmanager
def _argv(args: list[str]):
    old = sys.argv
    sys.argv = ["main_inference.py", *args]
    try:
        yield
    finally:
        sys.argv = old


class SimplerRunner(BenchmarkRunner):
    name: ClassVar[str] = "simpler"
    primary_metric: ClassVar[str] = "success_rate"

    def __init__(self, suite: str, *, simpler_root: str | None = None, control_mode: str | None = None,
                 stop_on_done: bool | None = None, max_steps_factor: int | None = None, **kwargs: Any):
        if suite not in self.suites():
            raise ValueError(f"unknown SIMPLER suite {suite!r}; expected one of {self.suites()}")
        super().__init__(suite, **kwargs)
        self.simpler_root = simpler_root or os.environ.get("SIMPLER_DIR")
        self.control_mode = control_mode
        self.stop_on_done = stop_on_done
        self.max_steps_factor = max_steps_factor
        self._table: list[dict[str, Any]] | None = None
        self._parsed: dict[int, Any] = {}
        self._env_cache: dict[int, Any] = {}
        os.environ.setdefault("DISPLAY", "")

    @classmethod
    def suites(cls) -> tuple[str, ...]:
        return ("google_robot_vm", "google_robot_va", "widowx_vm")

    @classmethod
    def default_episodes_per_task(cls) -> int:
        return 10**6  # per-config counts come from the official sweep; see episodes_for()

    # ------------------------------------------------------------------ #
    def configure_for(self, adapter: VLAAdapter) -> None:
        if self.control_mode is None:
            self.control_mode = getattr(adapter, "simpler_control_mode", None)
        if self.stop_on_done is None:
            self.stop_on_done = bool(getattr(adapter, "simpler_stop_on_done", False))
        if self.max_steps_factor is None:
            self.max_steps_factor = int(getattr(adapter, "simpler_max_steps_factor", 1) or 1)

    def run(self, adapter: VLAAdapter, writer=None, **kwargs):  # type: ignore[override]
        self.configure_for(adapter)
        return super().run(adapter, writer, **kwargs)

    # ------------------------------------------------------------------ #
    def _load_table(self) -> list[dict[str, Any]]:
        if self._table is None:
            with open(CONFIG_DIR / f"{self.suite}.json") as fh:
                self._table = json.load(fh)["configs"]
        return self._table

    def _args(self, task_id: int):
        if task_id not in self._parsed:
            from simpler_env.evaluation.argparse import get_args  # type: ignore

            argv = list(self._load_table()[task_id]["argv"])
            if self.simpler_root:  # overlay paths in the scripts are relative to the SimplerEnv repo
                argv = [os.path.join(self.simpler_root, a[2:]) if a.startswith("./ManiSkill2_real2sim") else a for a in argv]
            with _argv(argv):
                self._parsed[task_id] = get_args()
        return self._parsed[task_id]

    def _reset_combos(self, args) -> list[tuple[float, float, Any, dict[str, Any]]]:
        combos = []
        for rx in args.robot_init_xs:
            for ry in args.robot_init_ys:
                for rq in args.robot_init_quats:
                    if args.obj_variation_mode == "xy":
                        for ox in args.obj_init_xs:
                            for oy in args.obj_init_ys:
                                combos.append((rx, ry, rq, {"init_xy": np.array([ox, oy])}))
                    elif args.obj_variation_mode == "episode":
                        for ep in range(args.obj_episode_range[0], args.obj_episode_range[1]):
                            combos.append((rx, ry, rq, {"episode_id": ep}))
                    else:  # pragma: no cover
                        raise NotImplementedError(args.obj_variation_mode)
        return combos

    def tasks(self) -> list[TaskSpec]:
        specs = []
        for tid, entry in enumerate(self._load_table()):
            args = self._args(tid)
            n = len(self._reset_combos(args))
            tag = "/".join(filter(None, [entry.get("scene_name"), entry.get("overlay"), " ".join(entry.get("env_kwargs", []))]))
            specs.append(
                TaskSpec(
                    benchmark=self.name,
                    suite=self.suite,
                    task_id=tid,
                    task_name=f"{entry['family']}/{entry['env_name']}/{tag}",
                    instruction="",  # filled from env.get_language_instruction() at reset
                    max_steps=self.max_steps(int(args.max_episode_steps) * int(self.max_steps_factor or 1)),
                    extra={"family": entry["family"], "env_name": entry["env_name"], "variant": tag,
                           "camera_variant": bool(entry.get("camera_variant", False)), "n_episodes": n,
                           "scored": not (self.suite.endswith("_va") and entry.get("camera_variant", False))},
                )
            )
        return specs

    def episodes_for(self, task: TaskSpec) -> list[int]:
        n = int(task.extra["n_episodes"])
        return list(range(min(n, self.episodes_per_task)))

    # ------------------------------------------------------------------ #
    def _make_env(self, task: TaskSpec):
        if task.task_id in self._env_cache:
            return self._env_cache[task.task_id]
        from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode  # type: ignore

        args = self._args(task.task_id)
        control_mode = self.control_mode or get_robot_control_mode(args.robot, "rt1")
        kwargs = dict(args.additional_env_build_kwargs or {})
        if args.enable_raytracing:  # exactly as run_maniskill2_eval_single_episode
            kwargs = {"shader_dir": "rt", **kwargs}
        env = build_maniskill2_env(
            args.env_name,
            obs_mode="rgbd",
            robot=args.robot,
            sim_freq=args.sim_freq,
            control_mode=control_mode,
            control_freq=args.control_freq,
            max_episode_steps=args.max_episode_steps,
            scene_name=args.scene_name,
            camera_cfgs={"add_segmentation": True},
            rgb_overlay_path=args.rgb_overlay_path,
            **kwargs,
        )
        for old in self._env_cache.values():
            try:
                old.close()
            except Exception:  # pragma: no cover
                pass
        self._env_cache = {task.task_id: env}
        return env

    @staticmethod
    def _tcp_pos_in_base(obs) -> np.ndarray:
        from sapien.core import Pose  # type: ignore

        base = obs["agent"]["base_pose"]
        tcp = obs["extra"]["tcp_pose"]
        return np.asarray((Pose(p=base[:3], q=base[3:]).inv() * Pose(p=tcp[:3], q=tcp[3:])).p)

    def run_episode(self, adapter: VLAAdapter, task: TaskSpec, episode: int, meter: LatencyMeter) -> EpisodeOutcome:
        from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict  # type: ignore

        args = self._args(task.task_id)
        env = self._make_env(task)
        rx, ry, rq, obj_opt = self._reset_combos(args)[episode]
        options = {"robot_init_options": {"init_xy": np.array([rx, ry]), "init_rot_quat": rq}, "obj_init_options": obj_opt}
        obs, _ = env.reset(seed=self.episode_seed(task, episode), options=options)
        instruction = env.get_language_instruction()
        task.instruction = task.instruction or instruction
        adapter.reset(task)

        camera = args.obs_camera_name
        done = truncated = False
        steps = 0
        info: dict[str, Any] = {}
        while not truncated and steps < task.max_steps:
            image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=camera)
            raw = {
                "agent": {k: np.asarray(v) for k, v in obs["agent"].items()},
                "extra": {k: np.asarray(v) for k, v in obs["extra"].items() if not isinstance(v, dict)},
                "tcp_pos_base": self._tcp_pos_in_base(obs),
                "is_final_subtask": bool(env.is_final_subtask()),
            }
            observation = self.make_obs({"main": image, "overhead": image}, instruction, None, steps, raw)
            with meter.measure():
                action = np.asarray(adapter.act(observation, task), dtype=np.float64).reshape(-1)
            obs, _, done, truncated, info = env.step(action)
            steps += 1
            instruction = env.get_language_instruction()
            if done and self.stop_on_done:
                break
        return EpisodeOutcome(
            success=bool(done),
            steps=steps,
            extra={"family": task.extra["family"], "variant": task.extra["variant"], "scored": task.extra["scored"],
                   "env_name": task.extra["env_name"], "max_steps": task.max_steps,
                   "episode_stats": {k: (float(v) if isinstance(v, (int, float, np.generic, bool)) else str(v))
                                     for k, v in (info.get("episode_stats", {}) or {}).items()}},
        )

    def close(self) -> None:
        for env in self._env_cache.values():
            try:
                env.close()
            except Exception:  # pragma: no cover
                pass
        self._env_cache.clear()
