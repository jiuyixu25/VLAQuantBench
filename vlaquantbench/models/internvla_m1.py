"""InternVLA-M1 adapter (Chen et al., 2025) - Qwen2.5-VL 3B spatial VLM + DINOv2
side encoder + layer-wise Q-Former + DiT action head, BF16.

The official deployment splits the model into a websocket server
(``deployment/model_server/server_policy_M1.py``) and per-benchmark clients;
this adapter runs the same computation in-process and reproduces the clients'
conventions:

* **LIBERO** (``examples/LIBERO/model2libero_interface.py``): both cameras
  rotated 180 deg then ``cv2.INTER_AREA``-resized to 224; ``batch_images =
  [[agentview, wrist]]``; the checkpoint's ``CoT_prompt`` wraps the
  instruction; no proprio; a chunk of ``future_action_window_size + 1``
  actions (8) is executed open-loop; un-normalisation uses ``min``/``max``
  (not q01/q99) with the gripper binarised at 0.5, then
  ``gripper = 1 - 2 * open`` for the robosuite convention;
* **SIMPLER** (``examples/SimplerEnv/model2simpler_interface.py``): single
  camera resized to 224, q01/q99 un-normalisation, ``AdaptiveEnsembler``
  (horizon 2 google / 7 widowx, alpha 0.1) so exactly one action is executed
  per control step, ``euler2axangle`` rotations and the sticky-gripper logic.

Component map (verified against the repo, commit 21e6e8f):

====  =====================================================================
ve    ``qwen_vl_interface.model.model.visual`` (Qwen2.5-VL ViT) + ``dino_encoder.body``
mp    ``...visual.merger`` (Qwen patch merger) + ``dino_pro``
llm   ``qwen_vl_interface.model.model.language_model`` (Qwen2.5 3B; the tied ``lm_head`` is excluded)
ah    ``action_model.net`` (DiT-B) + ``layer_qformer`` (produces the 64 condition tokens)
====  =====================================================================

``layer_qformer.layers[*].cross_attn`` is an ``nn.MultiheadAttention`` whose
Q/K/V projection is a raw ``in_proj_weight`` parameter and whose ``out_proj``
is bypassed by the functional path - neither is an ``nn.Linear`` we can
quantize, so the Q-Former's attention projections stay at baseline precision
(recorded in ``ComponentMap.notes``); its MLP and input projection are
quantized.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, ClassVar

import numpy as np
import torch

from ..components import ComponentMap
from .base import Observation, TaskSpec, VLAAdapter

log = logging.getLogger(__name__)

REPO_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "third_party", "InternVLA-M1")


class InternVLAM1Adapter(VLAAdapter):
    name: ClassVar[str] = "internvla_m1"
    default_dtype: ClassVar[str] = "bf16"
    supported_benchmarks: ClassVar[tuple[str, ...]] = ("libero", "simpler")
    #: official LIBERO client protocol (identical to OpenVLA's)
    libero_max_steps: ClassVar[dict[str, int]] = {
        "libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400,
    }
    libero_camera_size: ClassVar[int] = 256
    libero_action_mode: ClassVar[str] = "delta"

    def __init__(self, checkpoint: str, *, device="cuda", dtype=None, seed: int = 0, **kwargs: Any):
        super().__init__(checkpoint, device=device, dtype=dtype, seed=seed, **kwargs)
        self.repo = kwargs.get("repo") or os.environ.get("INTERNVLA_M1_REPO", REPO_ROOT)
        self.cfg_scale = float(kwargs.get("cfg_scale", 1.5))
        self.num_ddim_steps = int(kwargs.get("num_ddim_steps", 10))
        self.policy_setup = kwargs.get("policy_setup", "google_robot")
        self.image_size = tuple(kwargs.get("image_size", (224, 224)))
        self.unnorm_key = kwargs.get("unnorm_key")
        self.norm_stats: dict[str, Any] = {}
        self.chunk_size = 8
        self._plan: np.ndarray | None = None
        self._step = 0
        self._ensembler = None
        self._sticky_action = 0.0
        self._sticky_count = 0
        self._is_sticky = False
        self._prev_gripper: float | None = None

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        if not os.path.isdir(os.path.join(self.repo, "InternVLA")):
            raise FileNotFoundError(
                f"InternVLA-M1 repository not found at {self.repo!r}; clone "
                "https://github.com/InternRobotics/InternVLA-M1 there or pass --model-kwargs repo=/path/to/InternVLA-M1"
            )
        if self.repo not in sys.path:
            sys.path.insert(0, self.repo)
        from InternVLA.model.framework.M1 import InternVLA_M1  # type: ignore
        from InternVLA.model.framework.share_tools import read_mode_config  # type: ignore

        model = InternVLA_M1.from_pretrained(self.checkpoint)
        if self.dtype == torch.bfloat16:
            model = model.to(torch.bfloat16)  # --use_bf16 of the official server
        self.model = model.to(self.device).eval()
        cfg, norm_stats = read_mode_config(self.checkpoint)
        self.norm_stats = norm_stats
        self.chunk_size = int(cfg["framework"]["action_model"]["future_action_window_size"]) + 1
        log.info("InternVLA-M1 loaded from %s (dtype=%s, chunk=%d, ddim=%d)", self.checkpoint, self.dtype,
                 self.chunk_size, self.num_ddim_steps)

    def build_component_map(self) -> ComponentMap:
        m = self.model
        qwen = m.qwen_vl_interface.model.model
        cm = ComponentMap()
        cm.add("ve", "qwen.visual", qwen.visual)
        cm.add("ve", "dino_encoder.body", m.dino_encoder.body)
        cm.add("mp", "dino_pro", m.dino_pro)
        cm.add("mp", "qwen.visual.merger", qwen.visual.merger)
        cm.add("llm", "qwen.language_model", qwen.language_model)
        cm.add("ah", "action_model.net", m.action_model.net)
        cm.add("ah", "layer_qformer", m.layer_qformer)
        # the Qwen patch merger lives inside visual.*; move it to mp by excluding it from ve
        cm.exclude["ve"] = ["*visual.merger*"]  # the merger is the projector -> belongs to mp
        cm.notes["ve"] = "Qwen2.5-VL ViT (merger excluded, see mp) + DINOv2 side encoder"
        cm.notes["mp"] = "dino_pro (DINO -> LLM width) + the Qwen patch merger (excluded from ve so it is quantized exactly once, as mp)"
        cm.notes["llm"] = "Qwen2.5 3B text backbone; lm_head is tied to embed_tokens and excluded"
        cm.notes["ah"] = ("DiT-B + layer-wise Q-Former. The Q-Former cross-attention is an nn.MultiheadAttention "
                          "(raw in_proj_weight, functional out_proj) and cannot be swapped per-Linear; it stays at "
                          "baseline precision, its proj/MLP are quantized.")
        return cm

    def llm_causal_lm(self):
        return None  # Qwen2_5_VLTextModel is not a *ForCausalLM; the LLM PTQ experiment uses OpenVLA

    # ------------------------------------------------------------------ #
    def reset(self, task: TaskSpec) -> None:
        self._plan = None
        self._step = 0
        self._sticky_action, self._sticky_count, self._is_sticky = 0.0, 0, False
        self._prev_gripper = None
        torch.manual_seed(self.seed + task.task_id * 1009)
        if task.benchmark == "simpler":
            sys.path.insert(0, self.repo) if self.repo not in sys.path else None
            from examples.SimplerEnv.adaptive_ensemble import AdaptiveEnsembler  # type: ignore

            horizon = 2 if self.policy_setup == "google_robot" else 7
            self._ensembler = AdaptiveEnsembler(pred_action_horizon=horizon, adaptive_ensemble_alpha=0.1)
            self._ensembler.reset()

    def _resize(self, image: np.ndarray) -> np.ndarray:
        import cv2 as cv  # type: ignore

        return cv.resize(np.ascontiguousarray(image), tuple(self.image_size), interpolation=cv.INTER_AREA)

    def _check_unnorm_key(self, benchmark: str) -> str:
        if self.unnorm_key:
            return self.unnorm_key
        if len(self.norm_stats) == 1:
            return next(iter(self.norm_stats))
        key = "fractal20220817_data" if self.policy_setup == "google_robot" else "bridge_dataset"
        if key not in self.norm_stats:
            raise KeyError(f"un-norm key {key!r} not in norm_stats (have {sorted(self.norm_stats)})")
        return key

    @torch.no_grad()
    def _infer(self, images: list[np.ndarray], instruction: str) -> np.ndarray:
        from PIL import Image

        pil = [Image.fromarray(self._resize(im)) for im in images]
        with self.inference_meter.measure():
            out = self.model.predict_action(
                batch_images=[pil], instructions=[instruction], cfg_scale=self.cfg_scale,
                use_ddim=True, num_ddim_steps=self.num_ddim_steps,
            )
        return np.asarray(out["normalized_actions"])[0]  # [chunk, 7]

    @staticmethod
    def _unnormalize(normalized: np.ndarray, stats: dict, *, use_quantiles: bool) -> np.ndarray:
        lo_key, hi_key = ("q01", "q99") if use_quantiles else ("min", "max")
        low, high = np.asarray(stats[lo_key]), np.asarray(stats[hi_key])
        mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)))
        a = np.clip(np.asarray(normalized, dtype=np.float64), -1, 1)
        a[:, 6] = np.where(a[:, 6] < 0.5, 0, 1)
        return np.where(mask, 0.5 * (a + 1) * (high - low) + low, a)

    # ------------------------------------------------------------------ #
    def act(self, obs: Observation, task: TaskSpec) -> np.ndarray:
        if task.benchmark == "libero":
            return self._act_libero(obs, task)
        if task.benchmark == "simpler":
            return self._act_simpler(obs, task)
        raise ValueError(f"InternVLA-M1 adapter has no protocol for benchmark {task.benchmark!r}")

    def _act_libero(self, obs: Observation, task: TaskSpec) -> np.ndarray:
        stats = self.norm_stats[self._check_unnorm_key("libero")]["action"]
        if self._step % self.chunk_size == 0:
            raw = obs.raw["obs"]
            images = [obs.images["agentview"], raw["robot0_eye_in_hand_image"][::-1, ::-1]]
            self._plan = self._unnormalize(self._infer(images, obs.instruction), stats, use_quantiles=False)
        a = np.asarray(self._plan)[self._step % self.chunk_size]  # type: ignore[index]
        self._step += 1
        action = np.concatenate([a[:3], a[3:6], [1.0 - 2.0 * float(a[6])]])  # open=1 -> -1 (robosuite: +1 closes)
        return action.astype(np.float64)

    def _act_simpler(self, obs: Observation, task: TaskSpec) -> np.ndarray:
        from transforms3d.euler import euler2axangle  # type: ignore

        stats = self.norm_stats[self._check_unnorm_key("simpler")]["action"]
        image = obs.images.get("main") if "main" in obs.images else next(iter(obs.images.values()))
        chunk = self._unnormalize(self._infer([image], obs.instruction), stats, use_quantiles=True)
        raw = self._ensembler.ensemble_action(chunk) if self._ensembler is not None else chunk[0]
        raw = np.asarray(raw, dtype=np.float64).reshape(-1)
        axis, angle = euler2axangle(*raw[3:6])
        rot = axis * angle
        gripper = float(raw[6])
        if self.policy_setup == "google_robot":
            g = self._sticky_google(gripper)
        else:
            g = 2.0 * (gripper > 0.5) - 1.0
        return np.concatenate([raw[:3], rot, [g]]).astype(np.float64)

    def _sticky_google(self, open_gripper: float) -> float:
        """Sticky relative gripper of the official clients (10-step latch)."""
        if self._prev_gripper is None:
            self._prev_gripper = open_gripper
        relative = self._prev_gripper - open_gripper
        if self._is_sticky:
            relative = self._sticky_action
            self._sticky_count += 1
            if self._sticky_count >= 10:
                self._is_sticky = False
                self._sticky_action = 0.0
                self._sticky_count = 0
        elif abs(relative) > 0.5:
            self._is_sticky = True
            self._sticky_action = relative
            self._sticky_count = 1
        self._prev_gripper = open_gripper
        return float(relative)
