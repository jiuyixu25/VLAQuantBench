import json

from vlaquantbench.results import EpisodeRecord, RunHeader, RunWriter, read_run
from vlaquantbench.stats import bootstrap_ci, wilson_ci


def test_wilson_basic():
    ci = wilson_ci(45, 50)
    assert abs(ci.point - 0.9) < 1e-9 and 0.78 < ci.low < 0.82 and 0.95 < ci.high < 0.97
    assert wilson_ci(0, 50).low == 0.0 and wilson_ci(50, 50).high == 1.0
    assert wilson_ci(0, 0).n == 0


def test_bootstrap_mean():
    ci = bootstrap_ci([4, 5, 3, 5, 4, 4, 5, 3], n_boot=2000)
    assert ci.low <= ci.point <= ci.high and 3.5 < ci.point < 4.5


def test_run_writer_resume(tmp_path):
    p = tmp_path / "run.jsonl"
    h = RunHeader(model="toy", checkpoint="x", benchmark="libero", suite="libero_spatial", preset="W4", scope="e2e")
    with RunWriter(p, h) as w:
        w.add(EpisodeRecord(suite="libero_spatial", task_id=0, episode=0, success=True, steps=10))
        w.add(EpisodeRecord(suite="libero_spatial", task_id=0, episode=1, success=False, steps=220))
    with RunWriter(p, h, resume=True) as w:
        assert w.is_done("libero_spatial", 0, 1) and not w.is_done("libero_spatial", 0, 2)
        w.add(EpisodeRecord(suite="libero_spatial", task_id=0, episode=2, success=True, steps=50))
    header, eps = read_run(p)
    assert header.preset == "W4" and len(eps) == 3 and sum(e.success for e in eps) == 2
    # every line is valid json
    assert all(json.loads(l) for l in p.read_text().splitlines())


def test_run_writer_refuses_mismatched_config(tmp_path):
    p = tmp_path / "run.jsonl"
    RunWriter(p, RunHeader(model="toy", preset="W4")).close()
    try:
        RunWriter(p, RunHeader(model="toy", preset="W3"))
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")


def test_seed_is_part_of_a_runs_identity(tmp_path):
    """Two seeds must not share a results file: the episode keys collide, so the
    second seed would resume the first and silently record nothing."""
    from vlaquantbench.cli import default_out_path

    a = default_out_path("m", "libero", "libero_spatial", "rtn", "W3", "ah", False, seed=0)
    b = default_out_path("m", "libero", "libero_spatial", "rtn", "W3", "ah", False, seed=1)
    assert a != b and a.name == "rtn-W3-ah.jsonl" and b.name == "rtn-W3-ah-seed1.jsonl"

    p = tmp_path / "run.jsonl"
    RunWriter(p, RunHeader(model="m", preset="W3", seed=0)).close()
    try:
        RunWriter(p, RunHeader(model="m", preset="W3", seed=1))
    except RuntimeError:
        pass
    else:
        raise AssertionError("appending a different seed to the same file must be refused")
