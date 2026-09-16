"""OpenVLA adapter (Kim et al., 2024) - original autoregressive action-token model.

Official LIBERO pipeline (``openvla/openvla`` ``run_libero_eval.py``):
180-degree rotated agent-view image -> JPEG round-trip + Lanczos resize to
224 -> 0.9 center crop -> prompt ``"In: What action should the robot take to
{task}?\\nOut:"`` -> ``vla.predict_action(**inputs, unnorm_key, do_sample=False)``
(7 action tokens decoded with ``generate``) -> gripper binarised and inverted
-> one action per env step (no chunking).

The checkpoints ``openvla/openvla-7b-finetuned-libero-*`` use the remote code
of ``openvla/openvla-7b``. Attention: the moojink transformers fork used by
OpenVLA-OFT makes ``LlamaSdpaAttention`` *bidirectional*, which silently
corrupts autoregressive decoding; this adapter therefore loads the model with
``flash_attention_2`` (what the official eval hard-codes) when flash-attn is
installed and falls back to ``eager`` otherwise - never SDPA.

Component map: ``ve`` = DINOv2-L + SigLIP-so400m featurizers, ``mp`` =
fused MLP projector, ``llm`` = LLaMA-2-7B (``lm_head`` kept at baseline by
default, ``--quantize-lm-head`` to include the action-token decoder head);
no ``ah``.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, ClassVar

import numpy as np
import torch

from ..components import ComponentMap
from .base import Observation, TaskSpec, VLAAdapter
from .openvla_oft import _prepend_repo, resolve_unnorm_key

log = logging.getLogger(__name__)


def _pick_attn_impl(requested: str | None) -> str:
    if requested:
        return requested
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        log.warning("flash-attn not installed: loading OpenVLA with eager attention (slower, numerically identical)")
        return "eager"


class OpenVLAAdapter(VLAAdapter):
    name: ClassVar[str] = "openvla"
    default_dtype: ClassVar[str] = "bf16"
    supported_benchmarks: ClassVar[tuple[str, ...]] = ("libero",)
    libero_max_steps: ClassVar[dict[str, int]] = {
        "libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400,
    }
    libero_camera_size: ClassVar[int] = 256
    libero_action_mode: ClassVar[str] = "delta"

    def __init__(self, checkpoint: str, *, device="cuda", dtype=None, seed: int = 0, **kwargs: Any):
        super().__init__(checkpoint, device=device, dtype=dtype, seed=seed, **kwargs)
        self.repo = _prepend_repo(kwargs.get("repo"))  # for the official resize / crop helpers
        self.center_crop = bool(kwargs.get("center_crop", True))
        self.attn_impl = _pick_attn_impl(kwargs.get("attn_implementation"))
        self.vla = None
        self.processor = None
        self.unnorm_key = ""

    def load(self) -> None:
        from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore

        from experiments.robot.robot_utils import set_seed_everywhere  # type: ignore

        set_seed_everywhere(int(self.options.get("model_seed", 7)))
        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.checkpoint,
            attn_implementation=self.attn_impl,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        self.vla.eval()
        self.processor = AutoProcessor.from_pretrained(self.checkpoint, trust_remote_code=True)
        self.model = self.vla
        self.dtype = torch.bfloat16
        log.info("OpenVLA loaded from %s (attn=%s)", self.checkpoint, self.attn_impl)

    def build_component_map(self) -> ComponentMap:
        v = self.vla
        cm = ComponentMap()
        cm.add("ve", "vision_backbone.featurizer", v.vision_backbone.featurizer)
        cm.add("ve", "vision_backbone.fused_featurizer", v.vision_backbone.fused_featurizer)
        cm.add("mp", "projector", v.projector)
        cm.add("llm", "language_model", v.language_model)
        cm.notes["llm"] = "LLaMA-2-7B; lm_head decodes the 7 action tokens and is kept at baseline unless --quantize-lm-head"
        cm.notes["ah"] = "absent (discrete action tokens)"
        return cm

    def llm_causal_lm(self):
        return self.vla.language_model

    def llm_tokenizer(self):
        return self.processor.tokenizer

    # ------------------------------------------------------------------ #
    def reset(self, task: TaskSpec) -> None:
        self.unnorm_key = resolve_unnorm_key(self.vla.norm_stats, task.suite)

    def act(self, obs: Observation, task: TaskSpec) -> np.ndarray:
        from PIL import Image

        from experiments.robot import openvla_utils as U  # type: ignore
        from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action  # type: ignore

        img = U.resize_image_for_policy(obs.images["agentview"], U.OPENVLA_IMAGE_SIZE)
        image = Image.fromarray(img).convert("RGB")
        if self.center_crop:
            image = U.center_crop_image(image)
        prompt = f"In: What action should the robot take to {obs.instruction.lower()}?\nOut:"
        inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)
        with torch.no_grad(), self.inference_meter.measure():
            action = self.vla.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)
        return np.asarray(action, dtype=np.float64)
