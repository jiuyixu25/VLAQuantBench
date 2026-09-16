import pytest
import torch

from vlaquantbench.methods.awq import rtn_scales_zeros
from vlaquantbench.quant.fake_quant import WeightQuantSpec, fake_quant_weight


def test_rtn_scales_zeros_matches_the_protocol_quantizer():
    """The scales/zeros handed to the AWQ kernel must reproduce the benchmark's own W4 grid."""
    torch.manual_seed(0)
    w = torch.randn(64, 512)
    deq, scales, zeros = rtn_scales_zeros(w, bits=4, group_size=128)
    ref = fake_quant_weight(w, WeightQuantSpec(bits=4, granularity="per_group", group_size=128, symmetric=False))
    assert torch.allclose(deq, ref, atol=1e-6)
    assert scales.shape == (64, 4) and zeros.shape == (64, 4)
    assert (zeros >= 0).all() and (zeros <= 15).all()
    # dequantization identity used by WQLinear_GEMM.from_linear: round(w_hat/s) + z in [0, 15]
    q = torch.round(deq.reshape(64, 4, 128) / scales[..., None]) + zeros[..., None]
    assert q.min() >= 0 and q.max() <= 15


def test_ragged_input_dim_is_rejected_for_kernel_packing():
    with pytest.raises(ValueError, match="divisible"):
        rtn_scales_zeros(torch.randn(8, 4304), bits=4, group_size=128)


def test_awq_search_cache_tag_is_checkpoint_specific():
    """Two checkpoints of the same model family must not share a cached AWQ search."""
    from types import SimpleNamespace

    from vlaquantbench.methods import _cache_tag

    a = SimpleNamespace(name="openvla_oft", checkpoint="moojink/openvla-7b-oft-finetuned-libero-spatial")
    b = SimpleNamespace(name="openvla_oft", checkpoint="moojink/openvla-7b-oft-finetuned-libero-10")
    c = SimpleNamespace(name="openvla", checkpoint=a.checkpoint)
    assert _cache_tag(a) != _cache_tag(b) != _cache_tag(c) and _cache_tag(a) != _cache_tag(c)
    assert _cache_tag(a) == _cache_tag(SimpleNamespace(name=a.name, checkpoint=a.checkpoint))
    assert _cache_tag(a).startswith("openvla_oft-")
