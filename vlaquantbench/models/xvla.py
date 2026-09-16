"""X-VLA adapter (Zheng et al., 2025) - Florence-2 (DaViT + BART encoder) VLM and a
24-block soft-prompted flow-matching action transformer, 0.9B params, FP32.

The official deployment is a FastAPI server (``deploy.py``) queried by one
thin client per benchmark (``evaluation/{libero,simpler,calvin,vlabench}``).
This adapter runs the same model call in-process and reproduces each client's
observation / proprio / action conventions, which differ per benchmark:

==========  ================================================================================
LIBERO      domain 3; image0 = agent view rotated 180 deg, image1 = wrist view *not* rotated;
            proprio = [ee_pos, rot6d(column-stacked) of controller.ee_ori_mat, 0] + zeros(10) at the
            first query, then open-loop (last predicted pose); all 30 rows executed; absolute
            pose control (``controller.use_delta=False``); horizon 800 (900 for libero_10);
            action = [xyz, axis-angle(rot6d), 1 if g>0.5 else -1]
SIMPLER     domain 1 (Google robot); single overhead image; proprio = zeros(20); executes
            ``action[::2][:10]`` (place-in-drawer: ``[:6]`` VM / ``[:10]`` VA); xyz are deltas
            integrated on the current TCP position; action = [xyz_abs, euler_xyz(rot6d), 1 if g>thr else -1]
            with per-task thresholds; needs the ``255isWhite/SimplerEnv`` fork's absolute base-pose controller
CALVIN      domain 2; image0 = rgb_static, image1 = rgb_gripper; proprio from robot_obs once per chain,
            then open-loop; executes the first 20 of 30 rows; action = (xyz, quat_xyzw(rot6d), 1 if g<0.8 else -1)
VLABench    domain 8; images rgb[0], rgb[2], rgb[-1]; proprio from ee_state (robot frame); all 30 rows;
            returns (pos_world, euler_xyz, gripper_state) for the VLABench evaluator's IK
==========  ================================================================================

Component map (attribute paths verified against the repo, commit 6bc2513):

====  ==================================================================
ve    ``vlm.vision_tower`` (DaViT, 96 Linear + patch-embed / depthwise Conv2d)
mp    absent as ``nn.Linear`` - ``vlm.image_projection`` is a bare Parameter matmul; kept at baseline
llm   ``vlm.language_model`` (BART-large encoder, 72 Linear; no decoder / lm_head)
ah    ``transformer`` (24 blocks + ``vlm_proj`` + ``aux_visual_proj``, 98 Linear); ``action_encoder`` /
      ``action_decoder`` are ``DomainAwareLinear`` embedding tables (not nn.Linear) and stay at baseline
====  ==================================================================
"""

from __future__ import annotations

import logging
import os
import sys
from collections import deque
from typing import Any, ClassVar

import numpy as np
import torch

from ..components import ComponentMap
from . import _rot as R
from .base import Observation, TaskSpec, VLAAdapter

log = logging.getLogger(__name__)

REPO_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "third_party", "X-VLA")
DOMAIN_IDS = {"libero": 3, "simpler": 1, "simpler_widowx": 0, "calvin": 2, "vlabench": 8}
CALVIN_BASE = np.array([0.0, -0.4, 0.78])  # not used for CALVIN; VLABench robot base offset below
VLABENCH_BASE = np.array([0.0, -0.4, 0.78])


class XVLAAdapter(VLAAdapter):
    name: ClassVar[str] = "xvla"
    default_dtype: ClassVar[str] = "fp32"
    supported_benchmarks: ClassVar[tuple[str, ...]] = ("libero", "simpler", "calvin", "vlabench")

    # official X-VLA LIBERO client protocol
    libero_max_steps: ClassVar[dict[str, int]] = {
        "libero_spatial": 800, "libero_object": 800, "libero_goal": 800, "libero_10": 900, "libero_90": 800,
    }
    libero_camera_size: ClassVar[int] = 256
    libero_action_mode: ClassVar[str] = "absolute"
    # official X-VLA SIMPLER client protocol (fork with absolute base-pose control)
    simpler_control_mode: ClassVar[str] = (
        "arm_pd_ee_base_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
    )
    simpler_max_steps_factor: ClassVar[int] = 2
    # official X-VLA CALVIN client uses EP_LEN = 720 (CALVIN's own protocol is 360)
    calvin_ep_len: ClassVar[int] = 720
    # official X-VLA VLABench client uses max_substeps = 10 (VLABench's own script uses 1);
    # absolute-pose chunks need the extra physics steps to actually reach each target
    vlabench_max_substeps: ClassVar[int] = 10

    def __init__(self, checkpoint: str, *, device="cuda", dtype=None, seed: int = 0, **kwargs: Any):
        super().__init__(checkpoint, device=device, dtype=dtype, seed=seed, **kwargs)
        self.repo = kwargs.get("repo") or os.environ.get("XVLA_REPO", REPO_ROOT)
        self.source = kwargs.get("source") or ("repo" if os.path.isdir(os.path.join(self.repo, "models")) else "hub")
        self.steps = int(kwargs.get("steps", 10))
        self.processor = None
        self._plan: deque = deque()
        self._proprio: np.ndarray | None = None
        self._current_xyz: np.ndarray | None = None
        self._episode_seed = seed
        self._bench = None

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        if self.source == "repo":
            if sys.path[:1] != [self.repo]:
                sys.path.insert(0, self.repo)
            from models.modeling_xvla import XVLA  # type: ignore
            from models.processing_xvla import XVLAProcessor  # type: ignore

            model = XVLA.from_pretrained(self.checkpoint, trust_remote_code=True, torch_dtype=torch.float32)
            self.processor = XVLAProcessor.from_pretrained(self.checkpoint)
        else:
            from transformers import AutoModel, AutoProcessor  # type: ignore

            model = AutoModel.from_pretrained(self.checkpoint, trust_remote_code=True, torch_dtype=torch.float32)
            self.processor = AutoProcessor.from_pretrained(self.checkpoint, trust_remote_code=True)
        # deploy.py: .to(device).to(float32) - the released server runs fp32 end to end
        model = model.to(self.device).to(self.dtype)
        model.eval()
        self.model = model
        log.info("X-VLA loaded from %s (%s code, dtype=%s, flow steps=%d)", self.checkpoint, self.source, self.dtype, self.steps)

    def build_component_map(self) -> ComponentMap:
        m = self.model
        cm = ComponentMap()
        cm.add("ve", "vlm.vision_tower", m.vlm.vision_tower)
        cm.add("llm", "vlm.language_model", m.vlm.language_model)
        cm.add("ah", "transformer", m.transformer)
        cm.notes["mp"] = "image_projection is a bare Parameter (x @ W), not nn.Linear; kept at baseline precision"
        cm.notes["llm"] = "BART-large encoder only (Florence-2); no lm_head"
        cm.notes["ah"] = "soft-prompted transformer incl. vlm_proj/aux_visual_proj; DomainAwareLinear encoder/decoder are embedding tables"
        return cm

    # ------------------------------------------------------------------ #
    def reset(self, task: TaskSpec) -> None:
        self._plan.clear()
        self._proprio = None
        self._current_xyz = None
        self._bench = task.benchmark
        self._episode_seed = (self._episode_seed * 7919 + 1) % (2**31 - 1)
        torch.manual_seed(self._episode_seed)

    @torch.no_grad()
    def _infer(self, images: list[np.ndarray], instruction: str, proprio: np.ndarray, domain_id: int) -> np.ndarray:
        from PIL import Image

        pil = [Image.fromarray(np.ascontiguousarray(im)) for im in images]
        inputs = self.processor(pil, instruction)
        dtype = next(self.model.parameters()).dtype
        inputs = {k: (v.to(self.device, dtype=dtype) if v.is_floating_point() else v.to(self.device)) for k, v in inputs.items()}
        inputs["proprio"] = torch.as_tensor(np.asarray(proprio, dtype=np.float32)).unsqueeze(0).to(self.device, dtype=dtype)
        inputs["domain_id"] = torch.tensor([domain_id], dtype=torch.long, device=self.device)
        with self.inference_meter.measure():
            action = self.model.generate_actions(**inputs, steps=self.steps)
        return action.squeeze(0).float().cpu().numpy()  # [30, 20]

    def act(self, obs: Observation, task: TaskSpec):
        b = task.benchmark
        if b == "libero":
            return self._act_libero(obs)
        if b == "simpler":
            return self._act_simpler(obs, task)
        if b == "calvin":
            return self._act_calvin(obs)
        if b == "vlabench":
            return self._act_vlabench(obs)
        raise ValueError(f"X-VLA adapter has no protocol for benchmark {b!r}")

    # ------------------------------------------------------------------ #
    # LIBERO  (evaluation/libero/libero_client.py)
    # ------------------------------------------------------------------ #
    def _act_libero(self, obs: Observation) -> np.ndarray:
        if not self._plan:
            raw = obs.raw["obs"]
            ctrl = obs.raw["controller"]  # {"ee_pos", "ee_ori_mat"} filled by the LIBERO runner
            if self._proprio is None:
                ee6 = R.mat_to_rot6d_cols(np.asarray(ctrl["ee_ori_mat"]))
                p = np.concatenate([np.asarray(ctrl["ee_pos"]), ee6, [0.0]]).astype(np.float32)
                self._proprio = np.concatenate([p, np.zeros_like(p)])
            images = [obs.images["agentview"], raw["robot0_eye_in_hand_image"]]  # wrist view is NOT rotated by the client
            action = self._infer(images, obs.instruction, self._proprio, DOMAIN_IDS["libero"])
            self._proprio[:9] = action[-1, :9]
            for row in action:  # all 30 rows, one per env step
                aa = R.robosuite_mat_to_axisangle(R.rot6d_cols_to_mat(row[3:9]))
                self._plan.append(np.concatenate([row[:3], aa, [row[9]]]))
        a = np.asarray(self._plan.popleft(), dtype=np.float64)
        a[-1] = 1.0 if a[-1] > 0.5 else -1.0
        return a

    # ------------------------------------------------------------------ #
    # SIMPLER Google robot (evaluation/simpler/google-{VM,VA}/client_*.py)
    # ------------------------------------------------------------------ #
    _SIMPLER_THRESH = {  # (task, protocol) -> gripper threshold ; chunk length for place-in-drawer
        ("pick_coke_can", "vm"): 0.25, ("pick_coke_can", "va"): 0.25,
        ("move_near", "vm"): 0.25, ("move_near", "va"): 0.25,
        ("drawer", "vm"): 0.35, ("drawer", "va"): 0.25,
        ("place_in_drawer", "vm"): 0.28, ("place_in_drawer", "va"): 0.3,
    }

    def _act_simpler(self, obs: Observation, task: TaskSpec) -> np.ndarray:
        family = task.extra.get("family", "pick_coke_can")
        protocol = "va" if "va" in task.suite else "vm"
        thr = self._SIMPLER_THRESH.get((family, protocol), 0.25)
        n_exec = 6 if (family == "place_in_drawer" and protocol == "vm") else 10
        if self._current_xyz is None:
            self._current_xyz = np.asarray(obs.raw["tcp_pos_base"], dtype=np.float64)
        if not self._plan:
            image = obs.images["overhead"]
            action = self._infer([image], obs.instruction, np.zeros(20, dtype=np.float32), DOMAIN_IDS["simpler"])
            seq = action[::2][:n_exec].copy()
            seq[:, :3] += self._current_xyz
            self._plan.extend(seq)
        row = np.asarray(self._plan.popleft(), dtype=np.float64)
        euler = R.mat_to_euler_xyz(R.rot6d_rows_to_mat(row[3:9]))
        final = np.concatenate([row[:3], euler, [1.0 if row[9] > thr else -1.0]])
        self._current_xyz = final[:3].copy()
        return final

    # ------------------------------------------------------------------ #
    # CALVIN (evaluation/calvin/calvin_client.py); reset() is called once per 5-task chain
    # ------------------------------------------------------------------ #
    def _act_calvin(self, obs: Observation):
        robot_obs = np.asarray(obs.raw["robot_obs"], dtype=np.float64)
        if self._proprio is None:
            p = np.concatenate([robot_obs[:3], R.mat_to_rot6d_rows(R.euler_xyz_to_mat(robot_obs[3:6])), [float(robot_obs[-1] > 0.0)]])
            self._proprio = np.concatenate([p, np.zeros_like(p)]).astype(np.float32)
        if not self._plan:
            images = [obs.images["rgb_static"], obs.images["rgb_gripper"]]
            action = self._infer(images, obs.instruction, self._proprio, DOMAIN_IDS["calvin"])
            self._plan.extend(action[:20])
        row = np.asarray(self._plan.popleft(), dtype=np.float64)
        self._proprio[:10] = row[:10]
        quat = R.mat_to_quat_xyzw(R.rot6d_rows_to_mat(row[3:9]))
        return (row[:3], quat, 1 if row[9] < 0.8 else -1)

    # ------------------------------------------------------------------ #
    # VLABench (evaluation/vlabench/vlabench_client.py)
    # ------------------------------------------------------------------ #
    def _act_vlabench(self, obs: Observation):
        if not self._plan:
            rgb = obs.raw["rgb"]
            images = [rgb[0], rgb[2], rgb[-1]]
            ee = np.asarray(obs.raw["ee_state"], dtype=np.float64).reshape(-1)
            pos, quat, grip = ee[:3] - VLABENCH_BASE, ee[3:7], ee[7:8]
            r6 = R.mat_to_rot6d_rows(R.quat_xyzw_to_mat(quat))  # the client passes the quaternion as-is (scalar_first=False)
            p = np.concatenate([pos, r6, grip]).astype(np.float32)
            proprio = np.concatenate([p, np.zeros_like(p)])
            action = self._infer(images, obs.instruction, proprio, DOMAIN_IDS["vlabench"])
            for row in action:
                euler = R.mat_to_euler_xyz(R.rot6d_rows_to_mat(row[3:9]))
                self._plan.append(np.concatenate([row[:3], euler, [row[9]]]))
        row = np.asarray(self._plan.popleft(), dtype=np.float64)
        pos = row[:3] + VLABENCH_BASE
        gripper_state = np.ones(2) * 0.04 if row[-1] <= 0.5 else np.zeros(2)
        return (pos, row[3:6], gripper_state)
