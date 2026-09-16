"""OpenVLA-OFT adapter (Kim et al., 2025) - official LIBERO inference pipeline.

Everything model-specific is delegated to the upstream repository's own
evaluation helpers (``experiments/robot/openvla_utils.py``), so preprocessing
(JPEG round-trip + Lanczos resize to 224, 0.9 center crop, prompt template,
proprio normalisation), the parallel-decoding ``predict_action`` with the L1
regression head, un-normalisation with ``libero_*_no_noops`` statistics and
the 8-step open-loop action chunk are exactly those of the official release.

Requirements: the ``openvla-oft`` environment (moojink transformers fork,
timm 0.9.10, tensorflow for the resize/crop ops) with the upstream repo
available at ``repo`` (default ``third_party/openvla-oft``); the repo root is
put first on ``sys.path`` so that ``experiments.robot.*`` and ``prismatic``
resolve to the pinned upstream code.

Component map:

====  ==========================================================================
ve    ``vision_backbone.featurizer`` (DINOv2-L) + ``vision_backbone.fused_featurizer`` (SigLIP-so400m)
mp    ``projector`` (fused MLP 2176->8704->4096->4096) + ``proprio_projector`` (8->4096->4096, projects proprio into the LLM token space)
llm   ``language_model`` (LLaMA-2-7B; ``lm_head`` excluded - its logits are discarded by the L1 path)
ah    ``action_head`` (L1RegressionActionHead MLPResNet 28672->4096->...->7)  [+ ``noisy_action_projector`` for the diffusion variant]
====  ==========================================================================
"""

from __future__ import annotations

import logging
import os
import sys
from collections import deque
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import torch

from ..components import ComponentMap
from .base import Observation, TaskSpec, VLAAdapter

log = logging.getLogger(__name__)

REPO_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "third_party", "openvla-oft")


def _prepend_repo(repo: str | None) -> str:
    repo = repo or os.environ.get("OPENVLA_OFT_REPO", REPO_ROOT)
    if not os.path.isdir(os.path.join(repo, "experiments", "robot")):
        raise FileNotFoundError(
            f"openvla-oft repository not found at {repo!r}; clone https://github.com/moojink/openvla-oft there "
            "or pass --model-kwargs repo=/path/to/openvla-oft"
        )
    if sys.path[:1] != [repo]:
        sys.path.insert(0, repo)
    return repo


def resolve_unnorm_key(norm_stats: dict, suite: str) -> str:
    """Mirror of ``check_unnorm_key`` in the official ``run_libero_eval.py``."""
    key = suite
    if key not in norm_stats and f"{key}_no_noops" in norm_stats:
        key = f"{key}_no_noops"
    if key not in norm_stats:
        raise KeyError(f"un-normalisation key for {suite!r} not in norm_stats (have {sorted(norm_stats)})")
    return key


class OpenVLAOFTAdapter(VLAAdapter):
    name: ClassVar[str] = "openvla_oft"
    default_dtype: ClassVar[str] = "bf16"
    supported_benchmarks: ClassVar[tuple[str, ...]] = ("libero",)
    #: official protocol of the OFT LIBERO evaluation
    libero_max_steps: ClassVar[dict[str, int]] = {
        "libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400,
    }
    libero_camera_size: ClassVar[int] = 256
    libero_action_mode: ClassVar[str] = "delta"

    def __init__(self, checkpoint: str, *, device="cuda", dtype=None, seed: int = 0, **kwargs: Any):
        super().__init__(checkpoint, device=device, dtype=dtype, seed=seed, **kwargs)
        self.repo = _prepend_repo(kwargs.get("repo"))
        self.cfg = SimpleNamespace(
            model_family="openvla",
            pretrained_checkpoint=checkpoint,
            load_in_8bit=False,
            load_in_4bit=False,
            use_l1_regression=bool(kwargs.get("use_l1_regression", True)),
            use_diffusion=bool(kwargs.get("use_diffusion", False)),
            num_diffusion_steps_train=int(kwargs.get("num_diffusion_steps_train", 50)),
            num_diffusion_steps_inference=int(kwargs.get("num_diffusion_steps_inference", 50)),
            use_film=bool(kwargs.get("use_film", False)),
            num_images_in_input=int(kwargs.get("num_images_in_input", 2)),
            use_proprio=bool(kwargs.get("use_proprio", True)),
            center_crop=bool(kwargs.get("center_crop", True)),
            num_open_loop_steps=int(kwargs.get("num_open_loop_steps", 8)),
            lora_rank=int(kwargs.get("lora_rank", 32)),
            unnorm_key="",
        )
        self.vla = None
        self.processor = None
        self.action_head = None
        self.proprio_projector = None
        self.noisy_action_projector = None
        self._queue: deque = deque(maxlen=self.cfg.num_open_loop_steps)

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        from experiments.robot import openvla_utils as U  # type: ignore
        from experiments.robot.robot_utils import set_seed_everywhere  # type: ignore

        set_seed_everywhere(int(self.options.get("model_seed", 7)))  # official default seed
        self.U = U
        self.vla = U.get_vla(self.cfg)
        # The HF-cache snapshots of these checkpoints may carry a locally
        # patched ``modeling_prismatic.py`` (another project on this machine
        # extends ``predict_action`` to also return KV caches). Upstream's
        # ``get_vla_action`` unpacks exactly two values, so normalize the
        # arity here instead of touching the shared cache.
        _orig_predict = self.vla.predict_action

        def _predict_2tuple(*a, **k):
            out = _orig_predict(*a, **k)
            return tuple(out[:2]) if isinstance(out, tuple) and len(out) > 2 else out

        self.vla.predict_action = _predict_2tuple
        self.processor = U.get_processor(self.cfg)
        if self.cfg.use_proprio:
            self.proprio_projector = U.get_proprio_projector(self.cfg, self.vla.llm_dim, proprio_dim=8)
        if self.cfg.use_l1_regression or self.cfg.use_diffusion:
            self.action_head = U.get_action_head(self.cfg, self.vla.llm_dim)
        if self.cfg.use_diffusion:
            self.noisy_action_projector = U.get_noisy_action_projector(self.cfg, self.vla.llm_dim)
        self.model = self.vla
        self.dtype = torch.bfloat16
        log.info("OpenVLA-OFT loaded from %s (images=%d, proprio=%s, head=%s)", self.checkpoint,
                 self.cfg.num_images_in_input, self.cfg.use_proprio, type(self.action_head).__name__)

    def build_component_map(self) -> ComponentMap:
        v = self.vla
        cm = ComponentMap()
        vb = v.vision_backbone
        if hasattr(vb, "vision_backbone"):  # FiLM wrapper
            vb = vb.vision_backbone
        cm.add("ve", "vision_backbone.featurizer", vb.featurizer)
        cm.add("ve", "vision_backbone.fused_featurizer", vb.fused_featurizer)
        cm.add("mp", "projector", v.projector)
        if self.proprio_projector is not None:
            cm.add("mp", "proprio_projector", self.proprio_projector)
        cm.add("llm", "language_model", v.language_model)
        if self.action_head is not None:
            cm.add("ah", "action_head", self.action_head)
        if self.noisy_action_projector is not None:
            cm.add("ah", "noisy_action_projector", self.noisy_action_projector)
        cm.notes["ve"] = "DINOv2-L + SigLIP-so400m (timm); the last block of each ViT and SigLIP attn_pool are dead weights"
        cm.notes["mp"] = "fused MLP projector + proprio projector (both map inputs into the LLM token space)"
        cm.notes["llm"] = "LLaMA-2-7B; lm_head kept at baseline (logits unused by the L1 head)"
        cm.notes["ah"] = "L1 regression MLPResNet head over the 56 action-token hidden states"
        return cm

    def llm_causal_lm(self):
        return self.vla.language_model

    def llm_tokenizer(self):
        return self.processor.tokenizer

    # ------------------------------------------------------------------ #
    def reset(self, task: TaskSpec) -> None:
        self._queue.clear()
        self.cfg.unnorm_key = resolve_unnorm_key(self.vla.norm_stats, task.suite)

    def act(self, obs: Observation, task: TaskSpec) -> np.ndarray:
        U = self.U
        if not self._queue:
            observation = {
                "full_image": U.resize_image_for_policy(obs.images["agentview"], U.OPENVLA_IMAGE_SIZE),
                "state": np.asarray(obs.state, dtype=np.float64),
            }
            if self.cfg.num_images_in_input > 1:
                observation["wrist_image"] = U.resize_image_for_policy(obs.images["wrist"], U.OPENVLA_IMAGE_SIZE)
            with torch.no_grad(), self.inference_meter.measure():
                actions = U.get_vla_action(
                    self.cfg, self.vla, self.processor, observation, obs.instruction,
                    action_head=self.action_head, proprio_projector=self.proprio_projector,
                    noisy_action_projector=self.noisy_action_projector, use_film=self.cfg.use_film,
                )
            self._queue.extend(actions)
        action = np.asarray(self._queue.popleft(), dtype=np.float64)
        return self.process_action(action)

    def process_action(self, action: np.ndarray) -> np.ndarray:
        """``process_action`` of the official eval: [0,1] gripper -> binarised [-1,1], then inverted for LIBERO."""
        from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action  # type: ignore

        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)
        return np.asarray(action, dtype=np.float64)
