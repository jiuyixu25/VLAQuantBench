# Shared helpers for the per-model environment setup scripts.
# Usage: source scripts/env/common.sh
set -euo pipefail
VQB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
THIRD_PARTY="$VQB_ROOT/third_party"
mkdir -p "$THIRD_PARTY"

# clone_pinned <url> <dir> <commit>
clone_pinned() {
  local url="$1" dir="$2" commit="$3"
  if [ ! -d "$THIRD_PARTY/$dir/.git" ]; then
    git clone "$url" "$THIRD_PARTY/$dir"
  fi
  git -C "$THIRD_PARTY/$dir" fetch --depth 50 origin "$commit" 2>/dev/null || git -C "$THIRD_PARTY/$dir" fetch origin
  git -C "$THIRD_PARTY/$dir" checkout -q "$commit"
  echo "  $dir @ $commit"
}

# conda_env <name> <python-version>
conda_env() {
  local name="$1" py="$2"
  if ! conda env list | awk '{print $1}' | grep -qx "$name"; then
    conda create -y -n "$name" "python=$py"
  fi
  echo "  conda env: $name (python $py)"
}

# run <env> <command...>   -- run a command inside a conda env
run() { local env="$1"; shift; conda run --no-capture-output -n "$env" "$@"; }

# install_vqb <env>  -- install the benchmark core package (no heavy deps)
install_vqb() { run "$1" python -m pip install -e "$VQB_ROOT"; }

# LIBERO needs ~/.libero/config.yaml; create it non-interactively (its __init__ prompts otherwise)
write_libero_config() {
  local libero_pkg="$1"
  mkdir -p "$HOME/.libero"
  if [ ! -f "$HOME/.libero/config.yaml" ]; then
    cat > "$HOME/.libero/config.yaml" <<YAML
benchmark_root: $libero_pkg
bddl_files: $libero_pkg/bddl_files
init_states: $libero_pkg/init_files
datasets: $libero_pkg/../datasets
assets: $libero_pkg/assets
YAML
    echo "  wrote ~/.libero/config.yaml -> $libero_pkg"
  fi
}

# setup_simpler <env>  -- SimplerEnv + ManiSkill2_real2sim (sapien 2.2, Vulkan, numpy < 2)
setup_simpler() {
  local env="$1"
  clone_pinned https://github.com/simpler-env/SimplerEnv.git SimplerEnv 06accaca93535902d408da4855f21cece12bceb7
  git -C "$THIRD_PARTY/SimplerEnv" submodule update --init --recursive ManiSkill2_real2sim
  run "$env" python -m pip install "numpy==1.24.4"
  run "$env" python -m pip install -e "$THIRD_PARTY/SimplerEnv/ManiSkill2_real2sim"
  run "$env" python -m pip install -e "$THIRD_PARTY/SimplerEnv"
  echo "  SIMPLER needs the NVIDIA Vulkan ICD; if envs fail with"
  echo "  'vk::Instance::enumeratePhysicalDevices: ErrorInitializationFailed', install libvulkan1 and"
  echo "  copy the ICD files from $THIRD_PARTY/SimplerEnv/ManiSkill2_real2sim/docker/ into /usr/share/vulkan/icd.d/."
  echo "  export SIMPLER_DIR=$THIRD_PARTY/SimplerEnv   # so the overlay image paths resolve"
}
