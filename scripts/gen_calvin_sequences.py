#!/usr/bin/env python
"""Materialise CALVIN's official LH-MTLC evaluation set once, with the official code.

`calvin_agent.evaluation.multistep_sequences.get_sequences(1000)` and
`calvin_agent.evaluation.utils.get_env_state_for_initial_condition` (which
depends on the `pyhash` package that no longer builds on modern Pythons) are
run here inside an environment that has `calvin_models` installed, and the
result - 1000 x (robot_obs, scene_obs, 5 sub-tasks) - is written to
`configs/calvin/eval_sequences_1000.json`. The runner then only needs
`calvin_env`.

    conda activate calvin_venv && python scripts/gen_calvin_sequences.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    from calvin_agent.evaluation.multistep_sequences import get_sequences  # type: ignore
    from calvin_agent.evaluation.utils import get_env_state_for_initial_condition  # type: ignore

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    seqs = get_sequences(n)
    out = []
    for initial_state, eval_sequence in seqs:
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        out.append(
            {
                "initial_state": {k: (v if isinstance(v, str) else str(v)) for k, v in initial_state.items()},
                "robot_obs": np.asarray(robot_obs, dtype=np.float64).tolist(),
                "scene_obs": np.asarray(scene_obs, dtype=np.float64).tolist(),
                "tasks": list(eval_sequence),
            }
        )
    path = ROOT / "configs" / "calvin" / f"eval_sequences_{n}.json"
    with open(path, "w") as fh:
        json.dump({"source": "calvin_agent.evaluation.multistep_sequences.get_sequences + get_env_state_for_initial_condition",
                   "num_sequences": n, "sequences": out}, fh)
    print(f"wrote {len(out)} sequences to {path}")


if __name__ == "__main__":
    main()
