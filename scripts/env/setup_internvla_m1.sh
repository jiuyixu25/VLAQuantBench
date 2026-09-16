#!/usr/bin/env bash
# InternVLA-M1 (Qwen2.5-VL + DINOv2 + Q-Former + DiT) + LIBERO / SIMPLER.
#   bash scripts/env/setup_internvla_m1.sh [env_name]
#
# `from_pretrained` needs a LOCAL run directory (<dir>/checkpoints/*.pt +
# config.yaml + dataset_statistics.json), not a Hub id -- download first:
#   hf download InternRobotics/InternVLA-M1-LIBERO-Spatial --local-dir playground/InternVLA-M1-LIBERO-Spatial
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ENV="${1:-internvla-m1}"

conda_env "$ENV" 3.10
clone_pinned https://github.com/InternRobotics/InternVLA-M1.git InternVLA-M1 21e6e8f4bd42bee269dd021a118133501e9d8ede
run "$ENV" python -m pip install -r "$THIRD_PARTY/InternVLA-M1/requirements.txt"
run "$ENV" python -m pip install timm transforms3d opencv-python
run "$ENV" python -m pip install flash-attn --no-build-isolation   # hard-coded in the model code
clone_pinned https://github.com/Lifelong-Robot-Learning/LIBERO.git LIBERO 8f1084e3132a39270c3a13ebe37270a43ece2a01
run "$ENV" python -m pip install --no-deps -e "$THIRD_PARTY/LIBERO"
run "$ENV" python -m pip install robosuite==1.4.1 bddl easydict cloudpickle gym "imageio[ffmpeg]"
write_libero_config "$THIRD_PARTY/LIBERO/libero/libero"
install_vqb "$ENV"
run "$ENV" python -c "
import torch, transformers, flash_attn
assert transformers.__version__ >= '4.52', 'the checkpoint key layout needs transformers >= 4.52'
print(f'OK torch {torch.__version__} | transformers {transformers.__version__} | flash-attn {flash_attn.__version__}')"
echo "done: MUJOCO_GL=egl vqb run --model internvla_m1 --benchmark libero --suite libero_spatial --preset W4"
