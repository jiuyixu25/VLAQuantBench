"""LIBERO runner (Liu et al., 2023) following the protocol used by OpenVLA /
OpenVLA-OFT / pi0 / pi0.5 evaluations:

* suites ``libero_spatial`` / ``libero_object`` / ``libero_goal`` / ``libero_10``
  (10 tasks each), 50 episodes per task taken from the official initial-state
  files (``task_suite.get_task_init_states``);
* ``OffScreenRenderEnv`` at 256x256, ``num_steps_wait = 10`` dummy steps after
  reset so that objects settle, then at most 220 / 280 / 300 / 520 policy steps;
* LIBERO renders images upside down; the official evaluation scripts rotate
  both cameras by 180 degrees before feeding the policy. The runner applies
  that rotation (``flip_images=True``) so adapters receive upright images;
* action: 7-D ``[dx, dy, dz, droll, dpitch, dyaw, gripper]`` with gripper
  ``-1 = open`` / ``+1 = close`` (adapters must output this convention);
* success = the environment's ``done`` flag (``env.check_success()``).

State exposed to adapters (``Observation.state``): the common 8-D proprio
vector ``[eef_pos(3), eef_axis_angle(3), gripper_qpos(2)]``; the raw robosuite
observation dict and the live env are available as ``Observation.raw["obs"]``
/ ``Observation.raw["env"]`` (X-VLA reads controller state from the latter).

Model-specific official protocols differ in details (LeRobot renders 360x360
and allows 280 steps on libero_spatial; X-VLA uses absolute pose control and
800/900-step horizons). Adapters may declare ``libero_max_steps``,
``libero_camera_size`` and ``libero_action_mode``; the runner honours them
unless ``--bench-kwargs honor_adapter_protocol=false`` / explicit overrides
are given, and records the values used in every episode record.
"""

from __future__ import annotations

import logging
import math
from typing import Any, ClassVar

import numpy as np

from ..models.base import TaskSpec, VLAAdapter
from ..profiling import LatencyMeter
from .base import BenchmarkRunner, EpisodeOutcome

__all__ = ["LiberoRunner", "MAX_STEPS", "quat2axisangle", "libero_proprio"]

log = logging.getLogger(__name__)

MAX_STEPS: dict[str, int] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
NUM_STEPS_WAIT = 10
DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
IMAGE_SIZE = 256


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """robosuite convention ``(x, y, z, w)`` -> axis-angle (3,)."""
    q = np.asarray(quat, dtype=np.float64).copy()
    if q[3] > 1.0:
        q[3] = 1.0
    elif q[3] < -1.0:
        q[3] = -1.0
    den = math.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def libero_proprio(obs: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ]
    )


class LiberoRunner(BenchmarkRunner):
    name: ClassVar[str] = "libero"
    primary_metric: ClassVar[str] = "success_rate"

    def __init__(
        self,
        suite: str,
        *,
        flip_images: bool = True,
        camera_size: int | None = None,
        action_mode: str | None = None,
        honor_adapter_protocol: bool = True,
        **kwargs: Any,
    ):
        if suite not in self.suites():
            raise ValueError(f"unknown LIBERO suite {suite!r}; expected one of {self.suites()}")
        super().__init__(suite, **kwargs)
        self.flip_images = flip_images
        self.camera_size = camera_size  # None -> adapter preference or IMAGE_SIZE
        self.action_mode = action_mode  # None -> adapter preference or "delta"
        #: if True, an adapter may declare its model's official LIBERO protocol
        #: (``libero_max_steps``, ``libero_camera_size``, ``libero_action_mode``); the
        #: values actually used are recorded in every episode record.
        self.honor_adapter_protocol = honor_adapter_protocol
        self._adapter_max_steps: dict[str, int] = {}
        self._benchmark = None
        self._env_cache: dict[int, Any] = {}
        self._init_states_cache: dict[int, Any] = {}

    # ------------------------------------------------------------------ #
    def configure_for(self, adapter: VLAAdapter) -> None:
        """Adopt the adapter's official LIBERO conventions unless overridden explicitly."""
        if not self.honor_adapter_protocol:
            return
        if self.camera_size is None:
            self.camera_size = getattr(adapter, "libero_camera_size", None) or IMAGE_SIZE
        if self.action_mode is None:
            self.action_mode = getattr(adapter, "libero_action_mode", None) or "delta"
        if self.max_steps_override is None:
            self._adapter_max_steps = dict(getattr(adapter, "libero_max_steps", {}) or {})

    def _effective(self) -> None:
        if self.camera_size is None:
            self.camera_size = IMAGE_SIZE
        if self.action_mode is None:
            self.action_mode = "delta"

    @classmethod
    def suites(cls) -> tuple[str, ...]:
        return ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")

    @classmethod
    def default_episodes_per_task(cls) -> int:
        return 50

    # ------------------------------------------------------------------ #
    def _suite(self):
        if self._benchmark is None:
            from libero.libero import benchmark  # type: ignore

            self._benchmark = benchmark.get_benchmark_dict()[self.suite]()
        return self._benchmark

    def tasks(self) -> list[TaskSpec]:
        suite = self._suite()
        specs = []
        for tid in range(suite.n_tasks):
            task = suite.get_task(tid)
            specs.append(
                TaskSpec(
                    benchmark=self.name,
                    suite=self.suite,
                    task_id=tid,
                    task_name=task.name,
                    instruction=task.language,
                    max_steps=self.max_steps(self._adapter_max_steps.get(self.suite, MAX_STEPS[self.suite])),
                    extra={"bddl_file": task.bddl_file, "problem_folder": task.problem_folder,
                           "init_states_file": task.init_states_file},
                )
            )
        return specs

    def run(self, adapter: VLAAdapter, writer=None, **kwargs):  # type: ignore[override]
        self.configure_for(adapter)
        self._effective()
        return super().run(adapter, writer, **kwargs)

    def _init_states(self, task: TaskSpec):
        """Official 50 initial states per task, loaded directly (``torch.load`` needs
        ``weights_only=False`` on torch >= 2.6, which LIBERO's own helper omits)."""
        if task.task_id not in self._init_states_cache:
            import os

            import torch
            from libero.libero import get_libero_path  # type: ignore

            path = os.path.join(get_libero_path("init_states"), task.extra["problem_folder"], task.extra["init_states_file"])
            self._init_states_cache = {task.task_id: torch.load(path, weights_only=False)}
        return self._init_states_cache[task.task_id]

    def _make_env(self, task: TaskSpec):
        if task.task_id in self._env_cache:
            return self._env_cache[task.task_id]
        from libero.libero import get_libero_path  # type: ignore
        from libero.libero.envs import OffScreenRenderEnv  # type: ignore
        import os

        bddl = os.path.join(get_libero_path("bddl_files"), task.extra["problem_folder"], task.extra["bddl_file"])
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=self.camera_size, camera_widths=self.camera_size)
        env.seed(self.seed)
        # keep a single live env (MuJoCo contexts are expensive); close the previous one
        for old in self._env_cache.values():
            try:
                old.close()
            except Exception:  # pragma: no cover
                pass
        self._env_cache = {task.task_id: env}
        return env

    def _images(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        agent = obs["agentview_image"]
        wrist = obs["robot0_eye_in_hand_image"]
        if self.flip_images:
            agent = agent[::-1, ::-1]
            wrist = wrist[::-1, ::-1]
        return {
            "agentview": np.ascontiguousarray(agent),
            "wrist": np.ascontiguousarray(wrist),
        }

    def run_episode(self, adapter: VLAAdapter, task: TaskSpec, episode: int, meter: LatencyMeter) -> EpisodeOutcome:
        self._effective()
        env = self._make_env(task)
        init_states = self._init_states(task)
        env.reset()
        obs = env.set_init_state(init_states[episode % len(init_states)])
        adapter.reset(task)

        frames = [] if self.video_dir else None
        t = 0
        steps = 0
        success = False
        max_t = task.max_steps + NUM_STEPS_WAIT
        while t < max_t:
            if t < NUM_STEPS_WAIT:
                obs, _, done, _ = env.step(DUMMY_ACTION)
                t += 1
                if t == NUM_STEPS_WAIT and self.action_mode == "absolute":
                    for robot in env.env.robots:  # OSC_POSE consumes absolute [pos, axis-angle] targets
                        robot.controller.use_delta = False
                continue
            images = self._images(obs)
            ctrl = env.env.robots[0].controller
            raw = {
                "obs": {k: np.asarray(v) for k, v in obs.items()},
                "env": env,
                "controller": {"ee_pos": np.asarray(ctrl.ee_pos), "ee_ori_mat": np.asarray(ctrl.ee_ori_mat)},
            }
            observation = self.make_obs(images, task.instruction, libero_proprio(obs), steps, raw)
            with meter.measure():
                action = adapter.act(observation, task)
            action = np.asarray(action, dtype=np.float64).reshape(-1)
            if action.shape[0] != 7:
                raise ValueError(f"LIBERO expects a 7-D action, adapter returned shape {action.shape}")
            obs, _, done, _ = env.step(action.tolist())
            if frames is not None:
                frames.append(images["agentview"])
            t += 1
            steps += 1
            if done:
                success = True
                break
        if frames is not None:
            self._save_video(frames, task, episode, success)
        return EpisodeOutcome(
            success=success,
            steps=steps,
            extra={"max_steps": task.max_steps, "camera_size": self.camera_size, "action_mode": self.action_mode,
                   "init_state_id": episode % len(init_states)},
        )

    def _save_video(self, frames: list[np.ndarray], task: TaskSpec, episode: int, success: bool) -> None:
        try:
            import imageio.v2 as imageio  # type: ignore
            import os

            os.makedirs(self.video_dir, exist_ok=True)  # type: ignore[arg-type]
            path = os.path.join(self.video_dir, f"{task.suite}_t{task.task_id}_e{episode}_{'ok' if success else 'fail'}.mp4")  # type: ignore[arg-type]
            imageio.mimwrite(path, frames, fps=30)
        except Exception as e:  # pragma: no cover
            log.warning("could not save video: %s", e)

    def close(self) -> None:
        for env in self._env_cache.values():
            try:
                env.close()
            except Exception:  # pragma: no cover
                pass
        self._env_cache.clear()
