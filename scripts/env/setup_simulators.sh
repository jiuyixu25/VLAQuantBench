#!/usr/bin/env bash
# Simulator-only environments for the benchmarks that cannot share a stack with a
# policy. Pair them with `vqb serve` (see README "Installation").
#
#   bash scripts/env/setup_simulators.sh calvin     # PyBullet / python 3.8
#   bash scripts/env/setup_simulators.sh vlabench   # mujoco + dm_control
#   bash scripts/env/setup_simulators.sh simpler    # sapien 2.2 + Vulkan
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
WHICH="${1:?usage: setup_simulators.sh calvin|vlabench|simpler [env_name]}"
ENV="${2:-$WHICH}"

case "$WHICH" in
  calvin)
    # calvin_models (and its unbuildable `pyhash`) is NOT needed: the two functions
    # the protocol requires are vendored in vlaquantbench/benchmarks/_calvin_official.py.
    conda_env "$ENV" 3.8
    clone_pinned https://github.com/mees/calvin.git calvin fa03f01f19c65920e18cf37398a9ce859274af76
    git -C "$THIRD_PARTY/calvin" submodule update --init --recursive calvin_env
    run "$ENV" python -m pip install -e "$THIRD_PARTY/calvin/calvin_env"
    run "$ENV" python -m pip install "numpy<1.24" pybullet numpy-quaternion numba opencv-python \
        hydra-core omegaconf gym scipy pandas rich cloudpickle gitpython
    install_vqb "$ENV"
    run "$ENV" python -c "
from vlaquantbench.benchmarks.calvin import CalvinRunner
r = CalvinRunner('ABC_D'); print('tasks:', len(r.tasks())); r._make_env(); print('CALVIN env OK')"
    echo "done: vqb run --model remote --model-kwargs url=http://127.0.0.1:8010 --benchmark calvin --suite ABC_D"
    ;;
  vlabench)
    conda_env "$ENV" 3.10
    clone_pinned https://github.com/OpenMOSS/VLABench.git VLABench cf588fe60c0c7282174fe979f5913170cfe69017
    run "$ENV" python -m pip install -e "$THIRD_PARTY/VLABench"
    # `rrt-algorithms` ships rrt_algorithms/rrt/ WITHOUT __init__.py, so find_packages()
    # drops it and `from rrt_algorithms.rrt.rrt import RRT` fails. Add it, then install.
    tmp="$(mktemp -d)"
    git clone -q https://github.com/motion-planning/rrt-algorithms.git "$tmp/rrt"
    git -C "$tmp/rrt" checkout -q d61b21b12888f6c420aa205221a690dfe2d61456
    find "$tmp/rrt/rrt_algorithms" -type d -exec touch {}/__init__.py \;
    run "$ENV" python -m pip install --no-deps "$tmp/rrt"
    rm -rf "$tmp"
    echo "  downloading VLABench assets (~5.6 GB) ..."
    run "$ENV" python "$THIRD_PARTY/VLABench/scripts/download_assets.py"
    install_vqb "$ENV"
    MUJOCO_GL=egl run "$ENV" python -c "
import VLABench.tasks, VLABench.robots
from vlaquantbench.benchmarks.vlabench import VLABenchRunner
print('tasks:', [t.task_name for t in VLABenchRunner('track_1_in_distribution').tasks()])"
    echo "done: export MUJOCO_GL=egl VLABENCH_ROOT=... && vqb run --model remote ... --benchmark vlabench"
    ;;
  simpler)
    conda_env "$ENV" 3.10
    setup_simpler "$ENV"
    install_vqb "$ENV"
    echo "done: regenerate the task tables with  python scripts/gen_simpler_configs.py"
    ;;
  *) echo "unknown target $WHICH"; exit 1 ;;
esac
