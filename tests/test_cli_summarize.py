import json
import subprocess
import sys

from vlaquantbench.cli import build_parser, expand_matrix, job_to_cmd
from vlaquantbench.results import EpisodeRecord, RunHeader, RunWriter
from vlaquantbench.summarize import collect, pivot, render_latex, render_markdown


def _fake_run(root, suite, preset, k, n=50, model="toy"):
    p = root / "libero" / suite / model / f"rtn-{preset}-e2e.jsonl"
    with RunWriter(p, RunHeader(model=model, benchmark="libero", suite=suite, preset=preset, scope="e2e")) as w:
        for i in range(n):
            w.add(EpisodeRecord(suite=suite, task_id=i % 10, episode=i // 10, success=i < k, steps=100))


def test_collect_pivot_and_render(tmp_path):
    for suite, k in (("libero_spatial", 45), ("libero_object", 50), ("libero_goal", 40), ("libero_10", 30)):
        _fake_run(tmp_path, suite, "W4", k)
        _fake_run(tmp_path, suite, "BASELINE", k + 0)
    cells = collect(tmp_path)
    avg = cells[("libero", "libero_avg4", "toy", "rtn", "e2e", "W4", False)]
    assert abs(avg.interval.point - (0.9 + 1.0 + 0.8 + 0.6) / 4) < 1e-9
    fields, cols, table = pivot(cells)
    assert cols == ["BASELINE", "W4"]
    md = render_markdown(fields, cols, table, ci=True)
    assert "libero_avg4" in md and "[" in md
    tex = render_latex(fields, cols, table, caption="x")
    assert "\\toprule" in tex and "libero\\_10" in tex


def test_matrix_expansion():
    cfg = {
        "defaults": {"episodes": 50},
        "runs": [
            {"models": ["pi05", "openvla_oft"], "benchmarks": [{"name": "libero", "suites": ["libero_spatial", "libero_10"]}],
             "presets": ["BASELINE", "W4"], "scopes": ["e2e"]},
            {"model": "xvla", "benchmark": "calvin", "presets": ["W4A8"], "scopes": ["llm", "ah"]},
        ],
    }
    jobs = expand_matrix(cfg)
    assert len(jobs) == 2 * 2 * 2 + 2
    cmd = job_to_cmd(jobs[0])
    assert cmd[:2] == ["vqb", "run"] and "--episodes" in cmd and "50" in cmd


def test_cli_presets_runs():
    out = subprocess.run([sys.executable, "-m", "vlaquantbench.cli", "presets"], capture_output=True, text=True, check=True)
    assert "W4A8" in out.stdout and "w4g128asym+a8pt" in out.stdout


def test_parser_run_defaults():
    args = build_parser().parse_args(["run", "--model", "pi05", "--benchmark", "libero", "--suite", "libero_10"])
    assert args.preset == "BASELINE" and args.scope == "e2e" and args.method == "rtn"


def test_simpler_official_aggregation_weights_variants_equally(tmp_path):
    """One variant with many episodes must not dominate a variant with few."""
    from vlaquantbench.results import EpisodeRecord, RunHeader, RunWriter
    from vlaquantbench.summarize import collect

    p = tmp_path / "simpler" / "google_robot_va" / "m" / "rtn-W4-e2e.jsonl"
    h = RunHeader(model="m", benchmark="simpler", suite="google_robot_va", preset="W4", scope="e2e")
    with RunWriter(p, h) as w:
        # family A: variant a1 = 100 episodes all fail, variant a2 = 2 episodes all succeed
        for i in range(100):
            w.add(EpisodeRecord(suite="google_robot_va", task_id=0, episode=i, success=False,
                                extra={"family": "A", "variant": "a1", "scored": True}))
        for i in range(2):
            w.add(EpisodeRecord(suite="google_robot_va", task_id=1, episode=i, success=True,
                                extra={"family": "A", "variant": "a2", "scored": True}))
        # a camera variant that must be ignored entirely
        for i in range(50):
            w.add(EpisodeRecord(suite="google_robot_va", task_id=2, episode=i, success=True,
                                extra={"family": "A", "variant": "cam", "scored": False}))
    cells = collect(tmp_path)
    plain = cells[("simpler", "google_robot_va", "m", "rtn", "e2e", "W4", False)]
    official = cells[("simpler", "google_robot_va_official", "m", "rtn", "e2e", "W4", False)]
    assert abs(plain.interval.point - 52 / 152) < 1e-9      # episode average, incl. camera variant
    assert abs(official.interval.point - 0.5) < 1e-9        # (0% + 100%) / 2, camera variant dropped
