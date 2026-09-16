#!/usr/bin/env bash
# pi0 / pi0.5 (LeRobot PyTorch port) + LIBERO.
#   bash scripts/env/setup_lerobot.sh [env_name]
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ENV="${1:-lerobot}"

conda_env "$ENV" 3.12
run "$ENV" python -m pip install "lerobot[pi]==0.5.2" "hf_libero"
clone_pinned https://github.com/Lifelong-Robot-Learning/LIBERO.git LIBERO 8f1084e3132a39270c3a13ebe37270a43ece2a01
write_libero_config "$THIRD_PARTY/LIBERO/libero/libero"
install_vqb "$ENV"
run "$ENV" python - <<'PY'
import lerobot, torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy   # noqa: F401
from lerobot.policies.factory import make_pre_post_processors  # noqa: F401
print(f"OK lerobot {lerobot.__version__} | torch {torch.__version__}")
PY
echo "done: MUJOCO_GL=egl vqb run --model pi05 --benchmark libero --suite libero_spatial --preset W4A8"
