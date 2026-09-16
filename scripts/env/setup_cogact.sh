#!/usr/bin/env bash
# CogACT (Prismatic VLM + DiT action head) + SIMPLER.
#   bash scripts/env/setup_cogact.sh [env_name]
#
# CogACT's `prismatic` package comes from the arnoldland/openvla fork; the
# released checkpoints are fp32 (~30 GB), so a 24 GB card needs --dtype bf16
# (which casts the VLM only, exactly like the official `use_bf16` flag).
# The LLaMA-2 tokenizer/config come from the gated meta-llama/Llama-2-7b-hf repo:
# run `huggingface-cli login` first.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ENV="${1:-cogact}"

conda_env "$ENV" 3.10
clone_pinned https://github.com/microsoft/CogACT.git CogACT b174a1b86deedfab4d198d935207e7bb0527994e
run "$ENV" python -m pip install -e "$THIRD_PARTY/CogACT"
setup_simpler "$ENV"
install_vqb "$ENV"
run "$ENV" python - <<'PY'
import torch, transformers, timm
import simpler_env  # noqa: F401
assert timm.__version__ == "0.9.10", "CogACT's vision backbones require timm 0.9.10"
print(f"OK torch {torch.__version__} | transformers {transformers.__version__} | timm {timm.__version__}")
PY
echo "done: DISPLAY= vqb run --model cogact --benchmark simpler --suite google_robot_vm --preset W4 --dtype bf16"
