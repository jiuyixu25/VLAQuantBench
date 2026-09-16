#!/usr/bin/env bash
# X-VLA (Florence-2 + soft-prompted flow-matching transformer), FP32.
# X-VLA supports all four benchmarks; it is served over `vqb serve` so the
# simulator can live in its own environment.
#   bash scripts/env/setup_xvla.sh [env_name]
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ENV="${1:-xvla-stable}"

conda_env "$ENV" 3.10
clone_pinned https://github.com/2toINF/X-VLA.git X-VLA 6bc2513f5f1cbec715cc668b414392a6cae5c671
run "$ENV" python -m pip install -r "$THIRD_PARTY/X-VLA/requirements.txt"
install_vqb "$ENV"
run "$ENV" python - <<'PY'
import transformers, torch, timm, fastapi, uvicorn, json_numpy, cv2  # noqa: F401
assert transformers.__version__ <= "4.51.3", "X-VLA pins transformers <= 4.51.3"
print(f"OK torch {torch.__version__} | transformers {transformers.__version__} | timm {timm.__version__}")
PY
echo "done: vqb serve --model xvla --benchmark calvin --preset W4 --port 8010"
