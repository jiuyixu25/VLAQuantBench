"""CogACT adapter (Li et al., 2024) - Prismatic VLM (DINOv2 + SigLIP + LLaMA-2-7B)
with a diffusion-transformer (DiT) action model, FP32 release, SIMPLER only.

The adapter *wraps the official SIMPLER policy wrapper*
``sim_cogact/cogact_policy.py::CogACTInference`` rather than re-implementing it,
so prompt construction, image transform, DDIM sampling (``use_ddim=True``,
``num_ddim_steps=10``, ``cfg_scale=1.5``), action un-normalisation, the
adaptive action ensembler and the sticky-gripper logic are exactly upstream's.
One action is executed per control step (the model is queried every step).

Component map (verified against the repo, commit b174a1b):

====  ==================================================================
ve    ``vlm.vision_backbone.dino_featurizer`` (DINOv2-L) + ``.siglip_featurizer`` (SigLIP-so400m)
mp    ``vlm.projector`` (fused MLP 2176->8704->4096->4096)
llm   ``vlm.llm_backbone.llm`` (LLaMA-2-7B; ``lm_head`` runs but its logits are discarded)
ah    ``action_model.net`` (DiT-B/S/L incl. x/z/t embedders and the final layer)
====  ==================================================================

Notes: the VLM always runs under ``torch.autocast(bf16)`` while the DiT runs in
its weight dtype, so quantized Linears see bf16 activations in the VLM and
fp32 in the head - that is upstream behaviour and is kept. ``--dtype bf16``
casts ``vlm`` only (the ``use_bf16`` flag of the official wrapper).
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

REPO_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "third_party", "CogACT")


class CogActAdapter(VLAAdapter):
    name: ClassVar[str] = "cogact"
    default_dtype: ClassVar[str] = "fp32"
    supported_benchmarks: ClassVar[tuple[str, ...]] = ("simpler",)
    #: official SIMPLER protocol (the CogACT scripts pass the same env args as RT-1's)
    simpler_max_steps_factor: ClassVar[int] = 1

    def __init__(self, checkpoint: str = "CogACT/CogACT-Base", *, device="cuda", dtype=None, seed: int = 0, **kwargs: Any):
        super().__init__(checkpoint, device=device, dtype=dtype, seed=seed, **kwargs)
        self.repo = kwargs.get("repo") or os.environ.get("COGACT_REPO", REPO_ROOT)
        self.policy_setup = kwargs.get("policy_setup", "google_robot")
        self.action_model_type = kwargs.get("action_model_type", "DiT-B")
        self.cfg_scale = float(kwargs.get("cfg_scale", 1.5))
        self.num_ddim_steps = int(kwargs.get("num_ddim_steps", 10))
        self.action_scale = float(kwargs.get("action_scale", 1.0))
        self.policy = None

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        if not os.path.isdir(os.path.join(self.repo, "sim_cogact")):
            raise FileNotFoundError(
                f"CogACT repository not found at {self.repo!r}; clone https://github.com/microsoft/CogACT there "
                "or pass --model-kwargs repo=/path/to/CogACT"
            )
        if self.repo not in sys.path:
            sys.path.insert(0, self.repo)
        from sim_cogact.cogact_policy import CogACTInference  # type: ignore

        self.policy = CogACTInference(
            saved_model_path=self.checkpoint,
            policy_setup=self.policy_setup,
            action_model_type=self.action_model_type,
            cfg_scale=self.cfg_scale,
            use_ddim=True,
            num_ddim_steps=self.num_ddim_steps,
            action_scale=self.action_scale,
            use_bf16=(self.dtype == torch.bfloat16),
        )
        self.model = self.policy.vla
        log.info("CogACT loaded from %s (%s, %s, ddim=%d, cfg=%.1f, dtype=%s)", self.checkpoint, self.policy_setup,
                 self.action_model_type, self.num_ddim_steps, self.cfg_scale, self.dtype)

    def build_component_map(self) -> ComponentMap:
        vla = self.model
        cm = ComponentMap()
        vb = vla.vlm.vision_backbone
        cm.add("ve", "vlm.vision_backbone.dino_featurizer", vb.dino_featurizer)
        cm.add("ve", "vlm.vision_backbone.siglip_featurizer", vb.siglip_featurizer)
        cm.add("mp", "vlm.projector", vla.vlm.projector)
        cm.add("llm", "vlm.llm_backbone.llm", vla.vlm.llm_backbone.llm)
        cm.add("ah", "action_model.net", vla.action_model.net)
        cm.notes["ve"] = "DINOv2-L + SigLIP-so400m; last block of each ViT and SigLIP attn_pool are dead weights"
        cm.notes["llm"] = "LLaMA-2-7B; the cognition feature is the last hidden state, lm_head logits are discarded"
        cm.notes["ah"] = "DiT action model (x/z/t embedders, blocks, final layer); history_embedder is dead"
        return cm

    def llm_causal_lm(self):
        return self.model.vlm.llm_backbone.llm

    def llm_tokenizer(self):
        return self.model.vlm.llm_backbone.tokenizer

    # ------------------------------------------------------------------ #
    def reset(self, task: TaskSpec) -> None:
        # CogACT's DDIM noise comes from the global RNG; seed per episode for reproducibility
        torch.manual_seed(self.seed + task.task_id * 1009)
        self.policy.reset(task.instruction)

    def act(self, obs: Observation, task: TaskSpec) -> np.ndarray:
        image = obs.images.get("main") if "main" in obs.images else next(iter(obs.images.values()))
        with self.inference_meter.measure():
            _, action = self.policy.step(np.ascontiguousarray(image), obs.instruction)
        return np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]).astype(np.float64)
