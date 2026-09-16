#!/usr/bin/env bash
# OpenVLA + OpenVLA-OFT + LIBERO, plus the Experiment-3 PTQ libraries.
#
#   bash scripts/env/setup_openvla_oft.sh [env_name]
#
# Pins that matter: torch 2.2.0 / torchvision 0.17.0 / triton 2.2.0, the
# moojink transformers fork (bidirectional SDPA needed by OFT's parallel
# decoding), timm 0.9.10 (the remote code hard-fails on other versions),
# tensorflow 2.15 (image resize/crop ops used by the official preprocessing).
# ALWAYS install extras with --no-deps: bitsandbytes>=0.44 pulls a newer torch
# and silently breaks the fork.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ENV="${1:-openvla-oft}"

conda_env "$ENV" 3.10
clone_pinned https://github.com/moojink/openvla-oft.git       openvla-oft e4287e94541f459edc4feabc4e181f537cd569a8
clone_pinned https://github.com/Lifelong-Robot-Learning/LIBERO.git LIBERO  8f1084e3132a39270c3a13ebe37270a43ece2a01
clone_pinned https://github.com/mit-han-lab/llm-awq.git       llm-awq     d6e797a42b9ef7778de8ee2352116e0f48a78d61
clone_pinned https://github.com/mit-han-lab/smoothquant.git   smoothquant c61476d728e42ae0d8a35e7e78494edcac3237b5

run "$ENV" python -m pip install -e "$THIRD_PARTY/openvla-oft"
run "$ENV" python -m pip install -r "$THIRD_PARTY/openvla-oft/experiments/robot/libero/libero_requirements.txt"
run "$ENV" python -m pip install --no-deps -e "$THIRD_PARTY/LIBERO"
run "$ENV" python -m pip install "flash-attn==2.5.5" --no-build-isolation || \
  echo "  !! flash-attn not installed: OpenVLA falls back to eager attention (slower, same numerics)"

# Experiment 3 (LLM PTQ). --no-deps everywhere so torch/transformers stay pinned.
run "$ENV" python -m pip install --no-deps "bitsandbytes==0.43.3" datasets zstandard
run "$ENV" python -m pip install --no-deps -e "$THIRD_PARTY/llm-awq"
run "$ENV" python -m pip install --no-deps -e "$THIRD_PARTY/smoothquant"

write_libero_config "$THIRD_PARTY/LIBERO/libero/libero"
install_vqb "$ENV"
run "$ENV" python - <<'PY'
import torch, transformers, timm, inspect
from transformers.models.llama import modeling_llama as m
assert torch.__version__.startswith("2.2"), f"torch must stay at 2.2.x, got {torch.__version__}"
assert "is_causal=False" in inspect.getsource(m.LlamaSdpaAttention.forward), "the moojink transformers fork is not installed"
print(f"OK torch {torch.__version__} | transformers {transformers.__version__} (fork) | timm {timm.__version__}")
PY
echo "done: MUJOCO_GL=egl vqb run --model openvla_oft --benchmark libero --suite libero_spatial --preset W4"
