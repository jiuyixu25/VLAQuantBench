import torch
from torch import nn

from vlaquantbench.quant.act_calib import build_calibration, collect_stats
from vlaquantbench.quant.apply import quantize_modules, remove_activation_hooks
from vlaquantbench.quant.presets import parse_spec


def _toy(outlier_channel=7, scale=80.0, n_calls=6):
    torch.manual_seed(0)
    lin = nn.Linear(64, 32, bias=False)
    xs = []
    for _ in range(n_calls):
        x = torch.randn(16, 64)
        x[:, outlier_channel] *= scale  # one channel dominates per-token absmax
        xs.append(x)
    return lin, xs


def test_smoothing_is_exact_without_quantization():
    lin, xs = _toy()
    ref = lin(xs[0])
    stats = collect_stats({"lin": lin}, lambda: [lin(x) for x in xs])
    bank = build_calibration({"lin": lin}, stats)
    s = bank["lin"].smooth
    w_scaled = lin.weight.detach() * s
    got = (xs[0] / s) @ w_scaled.T
    assert torch.allclose(ref, got, atol=1e-4), (ref - got).abs().max()


def test_calibration_reduces_w4a4_output_error():
    spec = parse_spec("W4A4")

    lin_plain, xs = _toy()
    lin_cal = nn.Linear(64, 32, bias=False)
    lin_cal.weight.data.copy_(lin_plain.weight.data)
    ref = [lin_plain(x).detach() for x in xs]

    stats = collect_stats({"lin": lin_cal}, lambda: [lin_cal(x) for x in xs])
    bank = build_calibration({"lin": lin_cal}, stats)

    quantize_modules([("lin", lin_plain)], spec)
    quantize_modules([("lin", lin_cal)], spec, act_calib={"lin": bank["lin"]})

    err_plain = torch.stack([(lin_plain(x) - r).norm() / r.norm() for x, r in zip(xs, ref)]).mean()
    err_cal = torch.stack([(lin_cal(x) - r).norm() / r.norm() for x, r in zip(xs, ref)]).mean()
    remove_activation_hooks([("lin", lin_plain), ("lin", lin_cal)])

    # with one dominating channel, per-token absmax wastes the whole grid on it;
    # smoothing+clipping must recover at least half the error
    assert err_cal < 0.5 * err_plain, (err_plain.item(), err_cal.item())


def test_calib_requires_activation_quant():
    lin, xs = _toy()
    stats = collect_stats({"lin": lin}, lambda: [lin(x) for x in xs])
    bank = build_calibration({"lin": lin}, stats)
    try:
        quantize_modules([("lin", lin)], parse_spec("W4"), act_calib={"lin": bank["lin"]})
    except ValueError:
        pass
    else:
        raise AssertionError("weight-only spec with act_calib should raise")


def test_quantile_observer_runs_on_the_input_device_and_survives_large_batches():
    """The q tensor must be built on the input's device (torch < 2.3 raises otherwise) and
    torch.quantile rejects inputs above 2**24 elements; both are silent-crash risks during a
    calibration run that has already spent GPU time collecting rollouts."""
    import torch

    from vlaquantbench.quant.act_calib import _Observer

    lin = torch.nn.Linear(8, 4)
    obs = _Observer()
    obs(lin, (torch.randn(2_100_000, 8),))          # 16.8M elements: over the quantile limit
    assert obs.calls == 1
    assert obs.q_hi is not None and obs.q_hi.shape == (8,)
    assert torch.all(obs.q_hi <= obs.absmax + 1e-5)
    assert torch.all(obs.q_99 <= obs.q_hi + 1e-5)


def test_cache_key_covers_every_input_that_changes_the_statistics():
    """Two collections that roll out different init states, seeds, or code must never share a
    cache entry; the 2026-08 key covered only (model, checkpoint, suite, components, episodes)."""
    from vlaquantbench.quant.act_calib import cache_path

    base = cache_path("pi05", "ckpt", "libero_spatial", ["ah"], 2, episodes_from=20, tasks=1, seed=0, commit="abc")
    assert cache_path("pi05", "ckpt", "libero_spatial", ["ah"], 2, episodes_from=22, tasks=1, seed=0, commit="abc") != base
    assert cache_path("pi05", "ckpt", "libero_spatial", ["ah"], 2, episodes_from=20, tasks=1, seed=1, commit="abc") != base
    assert cache_path("pi05", "ckpt", "libero_spatial", ["ah"], 2, episodes_from=20, tasks=1, seed=0, commit="def") != base
    both = cache_path("pi05", "ckpt", "libero_spatial", ["ah", "llm"], 2, episodes_from=20, tasks=1, seed=0, commit="abc")
    assert both != base
    assert cache_path("pi05", "ckpt", "libero_spatial", ["llm", "ah"], 2, episodes_from=20, tasks=1, seed=0, commit="abc") == both
