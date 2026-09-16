"""Guards on protocol constants that are easy to drift from the upstream code."""

import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_libero_official_horizons():
    from vlaquantbench.benchmarks.libero import MAX_STEPS, NUM_STEPS_WAIT

    assert MAX_STEPS == {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
                         "libero_10": 520, "libero_90": 400}
    assert NUM_STEPS_WAIT == 10


def test_adapter_declared_protocols_match_their_upstream_clients():
    from vlaquantbench.models.openvla_oft import OpenVLAOFTAdapter
    from vlaquantbench.models.pi0_lerobot import Pi05Adapter
    from vlaquantbench.models.xvla import XVLAAdapter

    # OpenVLA-OFT: run_libero_eval.py TASK_MAX_STEPS
    assert OpenVLAOFTAdapter.libero_max_steps["libero_spatial"] == 220
    assert OpenVLAOFTAdapter.libero_camera_size == 256 and OpenVLAOFTAdapter.libero_action_mode == "delta"
    # LeRobot envs/libero.py TASK_SUITE_MAX_STEPS (spatial differs from OpenVLA's!)
    assert Pi05Adapter.libero_max_steps["libero_spatial"] == 280
    assert Pi05Adapter.libero_camera_size == 360
    # X-VLA libero_client.py LIBERO_DATASETS_HORIZON + absolute pose control
    assert XVLAAdapter.libero_max_steps["libero_spatial"] == 800
    assert XVLAAdapter.libero_max_steps["libero_10"] == 900
    assert XVLAAdapter.libero_action_mode == "absolute"
    assert XVLAAdapter.calvin_ep_len == 720          # X-VLA's client; CALVIN's own protocol is 360
    assert XVLAAdapter.vlabench_max_substeps == 10   # VLABench's own script uses 1


def test_calvin_official_constants():
    from vlaquantbench.benchmarks.calvin import EP_LEN, NUM_SEQUENCES

    assert EP_LEN == 360 and NUM_SEQUENCES == 1000


def test_calvin_sequences_are_deterministic_and_well_formed():
    from vlaquantbench.benchmarks._calvin_official import get_env_state_for_initial_condition, get_sequences

    seqs = get_sequences(20)
    assert len(seqs) == 20
    for initial_state, chain in seqs:
        assert len(chain) == 5 and len(set(chain)) == 5
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        assert robot_obs.shape == (15,) and scene_obs.shape == (24,)
    # deterministic across calls
    assert [c for _, c in get_sequences(20)] == [c for _, c in seqs]


def test_pyhash_replacement_matches_pyhash_test_vectors():
    """`hasher` must reproduce pyhash.fnv1_32 (UTF-16-LE, seed 0) exactly, or every
    CALVIN initial block layout would differ from the official evaluation set."""
    from vlaquantbench.benchmarks._calvin_official import fnv1_32, hasher

    assert fnv1_32(b"test") == 3698262380          # pyfasthash tests/test_fnv1.py bytes_hash
    assert hasher("test") == 3910690890            # ... unicode_hash
    assert fnv1_32(b"test", seed=3698262380) == 660137056  # ... seed_hash


def test_simpler_task_tables_match_the_official_sweeps():
    vm = json.loads((ROOT / "configs" / "simpler" / "google_robot_vm.json").read_text())
    va = json.loads((ROOT / "configs" / "simpler" / "google_robot_va.json").read_text())
    fam = lambda d: {c["family"] for c in d["configs"]}
    assert fam(vm) == fam(va) == {"pick_coke_can", "move_near", "drawer", "place_in_drawer"}
    # visual matching sweeps 4 urdf_versions; coke can additionally 3 can orientations
    assert sum(1 for c in vm["configs"] if c["family"] == "pick_coke_can") == 12
    assert sum(1 for c in vm["configs"] if c["family"] == "drawer") == 216
    # camera variants exist only in VA and are excluded from its score
    assert not any(c["camera_variant"] for c in vm["configs"])
    assert sum(1 for c in va["configs"] if c["camera_variant"]) == 8
    assert all("--policy-model" not in c["argv"] for c in vm["configs"])  # policy args stripped


@pytest.mark.parametrize("name", [p.stem for p in (ROOT / "configs" / "models").glob("*.yaml")])
def test_model_configs_are_wellformed(name):
    from vlaquantbench.registry import MODELS

    cfg = yaml.safe_load((ROOT / "configs" / "models" / f"{name}.yaml").read_text())
    assert name in MODELS, f"{name}.yaml has no registry entry"
    for bench, entry in (cfg.get("checkpoints") or {}).items():
        assert isinstance(entry, (str, dict)) and entry, f"{name}: bad checkpoint entry for {bench}"


def test_experiment_matrices_reference_known_models_and_benchmarks():
    from vlaquantbench.cli import expand_matrix
    from vlaquantbench.quant.presets import parse_spec
    from vlaquantbench.registry import BENCHMARKS, MODELS

    for path in sorted((ROOT / "configs" / "experiments").glob("*.yaml")):
        jobs = expand_matrix(yaml.safe_load(path.read_text()))
        assert jobs, f"{path.name} expands to nothing"
        for j in jobs:
            assert j["model"] in MODELS, f"{path.name}: unknown model {j['model']}"
            assert j["benchmark"] in BENCHMARKS, f"{path.name}: unknown benchmark {j['benchmark']}"
            parse_spec(j["preset"])  # must parse
