import math

import pytest
import torch

from vlaquantbench.quant import (
    ActQuantSpec,
    WeightQuantSpec,
    fake_quant_act_per_token,
    fake_quant_weight,
    parse_spec,
    weight_quant_error,
)
from vlaquantbench.quant.fake_quant import fake_quant_asymmetric, fake_quant_symmetric

torch.manual_seed(0)


def _n_levels_per_row(x: torch.Tensor) -> list[int]:
    return [x[i].unique().numel() for i in range(x.shape[0])]


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_symmetric_grid_has_at_most_2b_levels(bits):
    x = torch.randn(16, 4096)
    q = fake_quant_symmetric(x, bits)
    assert max(_n_levels_per_row(q)) <= 2**bits
    # symmetric: zero is representable and max magnitude is preserved
    assert torch.allclose(q.abs().amax(dim=-1), x.abs().amax(dim=-1), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_asymmetric_grid_has_at_most_2b_levels_and_covers_range(bits):
    x = torch.randn(16, 4096) + 3.0  # skewed, all-positive
    q = fake_quant_asymmetric(x, bits)
    assert max(_n_levels_per_row(q)) <= 2**bits
    # min and max of every row are grid points (up to rounding of the zero-point)
    s = (x.amax(-1) - x.amin(-1)) / (2**bits - 1)
    assert torch.all((q.amax(-1) - x.amax(-1)).abs() <= s + 1e-5)
    assert torch.all((q.amin(-1) - x.amin(-1)).abs() <= s + 1e-5)


def test_asymmetric_beats_symmetric_on_skewed_data():
    x = torch.rand(64, 1024) * 2 + 1  # in [1, 3]
    err_sym = (fake_quant_symmetric(x, 3) - x).pow(2).mean()
    err_asym = (fake_quant_asymmetric(x, 3) - x).pow(2).mean()
    assert err_asym < err_sym / 4


def test_error_decreases_with_bits():
    w = torch.randn(256, 1024)
    prev = math.inf
    for bits in (2, 3, 4, 8):
        err = weight_quant_error(w, WeightQuantSpec(bits=bits, granularity="per_channel"))["rel_mse"]
        assert err < prev
        prev = err
    assert prev < 1e-4  # 8-bit per-channel is essentially lossless


def test_per_group_ragged_tail_is_not_zero_padded():
    # in_features = 4304 (SigLIP-so400m MLP) is not divisible by 128 -> tail group of 80
    w = torch.randn(8, 4304)
    spec = WeightQuantSpec(bits=4, granularity="per_group", group_size=128)
    q = fake_quant_weight(w, spec)
    assert q.shape == w.shape
    # the tail is quantized as its own group: it should be reconstructed as well as a full group
    head_err = (q[:, :4224] - w[:, :4224]).pow(2).mean()
    tail_err = (q[:, 4224:] - w[:, 4224:]).pow(2).mean()
    assert tail_err < 3 * head_err
    # and the tail actually is quantized (not passed through)
    assert not torch.equal(q[:, 4224:], w[:, 4224:])


def test_per_group_has_lower_error_than_per_channel():
    w = torch.randn(128, 2048) * torch.rand(1, 2048) * 4  # heterogeneous column scales
    pc = weight_quant_error(w, WeightQuantSpec(bits=3, granularity="per_channel", symmetric=False))["rel_mse"]
    pg = weight_quant_error(w, WeightQuantSpec(bits=3, granularity="per_group", group_size=128, symmetric=False))["rel_mse"]
    assert pg < pc


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_dtype_and_shape_are_preserved(dtype):
    w = torch.randn(32, 256).to(dtype)
    q = fake_quant_weight(w, WeightQuantSpec(bits=4, granularity="per_group", group_size=128, symmetric=False))
    assert q.dtype == dtype and q.shape == w.shape
    conv = torch.randn(16, 3, 14, 14).to(dtype)
    qc = fake_quant_weight(conv, WeightQuantSpec(bits=8, granularity="per_channel"))
    assert qc.shape == conv.shape and qc.dtype == dtype


def test_16bit_is_identity():
    w = torch.randn(4, 8)
    assert torch.equal(fake_quant_weight(w, WeightQuantSpec(bits=16)), w)
    x = torch.randn(2, 3, 8)
    assert fake_quant_act_per_token(x, 16) is x


def test_act_per_token_scale_is_per_row():
    x = torch.randn(2, 5, 64)
    x[0, 0] *= 100  # a huge token must not affect other tokens' grids
    q = fake_quant_act_per_token(x, 8)
    assert q.shape == x.shape
    err_other = (q[0, 1:] - x[0, 1:]).abs().max()
    assert err_other < 0.02  # 8-bit of O(1) values


def test_constant_rows_do_not_nan():
    w = torch.zeros(4, 256)
    w[1] = 2.0
    for sym in (True, False):
        q = fake_quant_weight(w, WeightQuantSpec(bits=3, granularity="per_group", group_size=128, symmetric=sym))
        assert torch.isfinite(q).all()
        assert torch.allclose(q[1], w[1], atol=1e-5)


def test_parse_spec_table7_defaults():
    s = parse_spec("W4")
    assert s.weight.granularity == "per_group" and s.weight.group_size == 128 and not s.weight.symmetric
    assert s.act is None
    s = parse_spec("W8")
    assert s.weight.granularity == "per_channel" and s.weight.symmetric
    s = parse_spec("W4A8")
    assert s.act == ActQuantSpec(bits=8, granularity="per_token", symmetric=True)
    s = parse_spec("w3g64sym")
    assert s.weight.group_size == 64 and s.weight.symmetric
    assert parse_spec("W16").is_baseline and parse_spec("baseline").is_baseline
    assert parse_spec("W2").weight.bits == 2
    with pytest.raises(ValueError):
        parse_spec("A8")
