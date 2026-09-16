"""π0 and π0.5 adapters (LeRobot PyTorch port, ``lerobot>=0.5``).

Official inference pipeline reproduced:

* ``PI0Policy`` / ``PI05Policy`` loaded with the checkpoint's own config
  (``compile_model`` disabled so that fake-quant hooks are not baked into a
  compiled graph), the checkpoint's pre-/post-processor pipelines
  (normalisation stats, state tokenisation, PaliGemma tokenizer), and
  ``policy.select_action`` which owns the action-chunk queue
  (``n_action_steps`` actions executed open-loop before re-planning).
* The LIBERO environment processor of LeRobot rotates images by 180 degrees
  and builds the 8-D state ``[eef_pos, eef_axis_angle, gripper_qpos]`` - the
  runner already provides exactly that, so it is not re-applied.
* Flow-matching noise is drawn from the global torch RNG; we seed it per
  episode so that a JSONL record is reproducible.

Component map (attribute paths verified against lerobot 0.5.2):

====  ===================================================================
ve    ``model.paligemma_with_expert.paligemma.model.vision_tower``  (SigLIP-so400m, 27 layers)
mp    ``model.paligemma_with_expert.paligemma.model.multi_modal_projector``
llm   ``model.paligemma_with_expert.paligemma.model.language_model`` (Gemma-2B, 18 layers; ``lm_head`` is a sibling and never executed)
ah    ``model.paligemma_with_expert.gemma_expert.model`` (Gemma-300M action expert incl. π0.5 adaRMS ``dense`` layers) + ``action_in_proj`` / ``action_out_proj`` / time MLPs (+ ``state_proj`` for π0)
====  ===================================================================

The dead ``gemma_expert.lm_head`` (263M params, never called) is left alone.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np
import torch

from ..components import ComponentMap
from .base import Observation, TaskSpec, VLAAdapter

log = logging.getLogger(__name__)

OBS_STATE = "observation.state"
OBS_IMAGES = "observation.images"
ACTION = "action"


class _LeRobotPiAdapter(VLAAdapter):
    policy_cls_name: ClassVar[str] = ""
    config_cls_path: ClassVar[tuple[str, str]] = ("", "")
    supported_benchmarks: ClassVar[tuple[str, ...]] = ("libero",)
    default_dtype: ClassVar[str] = "bf16"

    #: LeRobot's own LIBERO protocol (lerobot/envs/libero.py TASK_SUITE_MAX_STEPS) and render size.
    libero_max_steps: ClassVar[dict[str, int]] = {
        "libero_spatial": 280, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400,
    }
    libero_camera_size: ClassVar[int | None] = 360
    libero_action_mode: ClassVar[str] = "delta"
    #: LIBERO camera name -> LeRobot image key (see LiberoEnv camera_name_mapping)
    image_keys: ClassVar[dict[str, str]] = {"agentview": "image", "wrist": "image2"}

    def __init__(self, checkpoint: str, *, device="cuda", dtype=None, seed: int = 0, **kwargs: Any):
        super().__init__(checkpoint, device=device, dtype=dtype, seed=seed, **kwargs)
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self._episode_seed = seed

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        import importlib

        from lerobot.configs.policies import PreTrainedConfig  # type: ignore
        from lerobot.policies.factory import make_pre_post_processors  # type: ignore

        mod_name, cls_name = self.config_cls_path
        policy_mod = importlib.import_module(mod_name)
        policy_cls = getattr(policy_mod, self.policy_cls_name)

        cfg = PreTrainedConfig.from_pretrained(self.checkpoint)
        cfg.compile_model = False  # keep eager: hooks must not be frozen into a compiled graph
        cfg.gradient_checkpointing = False  # training-only flag carried by some checkpoints
        cfg.device = str(self.device)
        if self.options.get("num_inference_steps"):
            cfg.num_inference_steps = int(self.options["num_inference_steps"])
        # Replan every 10 env steps, overriding the checkpoint's saved n_action_steps
        # (50). Executing the full 50-step chunk open-loop is a known ~10-25-point
        # LIBERO regression and not the protocol the reported numbers use — with 10,
        # the community reproduces pi0.5's published scores
        # (huggingface/lerobot#2114; also the documented eval command's setting).
        cfg.n_action_steps = int(self.options.get("n_action_steps", 10))
        # the released checkpoints declare their own precision (bf16 with fp32 vision/norms); we keep it
        if self.options.get("force_dtype"):
            cfg.dtype = self.options["force_dtype"]
        policy = policy_cls.from_pretrained(self.checkpoint, config=cfg)
        policy.to(self.device)
        policy.eval()
        self.policy = policy
        self.model = policy.model  # the PI0Pytorch / PI05Pytorch module tree
        self.dtype = torch.bfloat16 if str(cfg.dtype).endswith("bfloat16") else torch.float32

        pre, post = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=self.checkpoint,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        self.preprocessor, self.postprocessor = pre, post
        self.n_action_steps = cfg.n_action_steps
        log.info(
            "%s loaded from %s (dtype=%s, n_action_steps=%d, num_inference_steps=%d)",
            self.name, self.checkpoint, cfg.dtype, cfg.n_action_steps, cfg.num_inference_steps,
        )

    def build_component_map(self) -> ComponentMap:
        m = self.model
        pwe = m.paligemma_with_expert
        cm = ComponentMap()
        cm.add("ve", "paligemma.vision_tower", pwe.paligemma.model.vision_tower)
        cm.add("mp", "paligemma.multi_modal_projector", pwe.paligemma.model.multi_modal_projector)
        cm.add("llm", "paligemma.language_model", pwe.paligemma.model.language_model)
        cm.add("ah", "gemma_expert", pwe.gemma_expert.model)
        for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out",
                     "state_proj", "action_time_mlp_in", "action_time_mlp_out"):
            mod = getattr(m, name, None)
            if isinstance(mod, torch.nn.Module):
                cm.add("ah", name, mod)
        cm.notes["llm"] = "Gemma-2B prefix model; paligemma.lm_head is tied/unused and excluded"
        cm.notes["ah"] = "Gemma-300M action expert (incl. adaRMS dense layers for pi0.5) + action/time/state projections; dead gemma_expert.lm_head excluded"
        return cm

    def llm_causal_lm(self):
        return None  # PiGemmaModel is not a *ForCausalLM; LLM PTQ methods are evaluated on OpenVLA

    # ------------------------------------------------------------------ #
    def reset(self, task: TaskSpec) -> None:
        self.policy.reset()
        self._episode_seed = (self._episode_seed * 7919 + 1) % (2**31 - 1)
        torch.manual_seed(self._episode_seed)
        np.random.seed(self._episode_seed % (2**32 - 1))

    def _batch(self, obs: Observation) -> dict[str, Any]:
        batch: dict[str, Any] = {}
        for cam, key in self.image_keys.items():
            if cam not in obs.images:
                continue
            img = torch.from_numpy(np.ascontiguousarray(obs.images[cam]))  # HxWx3 uint8, upright
            img = img.permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 255.0
            batch[f"{OBS_IMAGES}.{key}"] = img
        if obs.state is None:
            raise ValueError("LeRobot pi0/pi0.5 adapters need the 8-D LIBERO proprio state")
        batch[OBS_STATE] = torch.as_tensor(np.asarray(obs.state, dtype=np.float32)).unsqueeze(0)
        batch["task"] = [obs.instruction]
        return batch

    @torch.no_grad()
    def act(self, obs: Observation, task: TaskSpec) -> np.ndarray:
        batch = self.preprocessor(self._batch(obs))
        replanning = len(self.policy._action_queue) == 0
        with torch.inference_mode():
            if replanning:
                with self.inference_meter.measure():
                    action = self.policy.select_action(batch)
            else:
                action = self.policy.select_action(batch)
        action = self.postprocessor(action)
        return action.to("cpu").numpy().reshape(-1)[:7].astype(np.float64)


class Pi0Adapter(_LeRobotPiAdapter):
    name = "pi0"
    policy_cls_name = "PI0Policy"
    config_cls_path = ("lerobot.policies.pi0.modeling_pi0", "PI0Policy")


class Pi05Adapter(_LeRobotPiAdapter):
    name = "pi05"
    policy_cls_name = "PI05Policy"
    config_cls_path = ("lerobot.policies.pi05.modeling_pi05", "PI05Policy")
