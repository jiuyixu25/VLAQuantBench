import pytest
import torch
from torch import nn

from vlaquantbench.methods.int8_gemm import W8A8Linear, int_mm_supported, replace_linear_with_int8

cuda = pytest.mark.skipif(not torch.cuda.is_available() or not int_mm_supported(), reason="needs CUDA int8 GEMM")


@cuda
@pytest.mark.parametrize("rows", [1, 7, 32, 300])
def test_w8a8_matches_fp_reference(rows):
    torch.manual_seed(0)
    lin = nn.Linear(4096, 1024, bias=True).cuda().to(torch.bfloat16)
    x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
    ref = lin(x).float()
    q = W8A8Linear(lin).cuda()
    out = q(x).float()
    rel = (out - ref).norm() / ref.norm()
    assert out.shape == ref.shape and rel < 0.03, rel


@cuda
def test_replace_and_3d_input():
    m = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64)).cuda().half()
    x = torch.randn(2, 5, 64, device="cuda", dtype=torch.half)
    ref = m(x).float()
    n = replace_linear_with_int8(m)
    assert n == 2 and isinstance(m[0], W8A8Linear)
    out = m(x).float()
    assert out.shape == ref.shape and ((out - ref).norm() / ref.norm()) < 0.05
