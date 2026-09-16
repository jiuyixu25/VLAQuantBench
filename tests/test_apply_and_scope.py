import pytest
import torch
from torch import nn

from vlaquantbench import ComponentMap, parse_spec, quantize_scope, resolve_scope
from vlaquantbench.components import clear_activation_quant
from vlaquantbench.quant import count_linear_params, iter_quantizable, quantize_modules, remove_activation_hooks


class ToyVLA(nn.Module):
    """A miniature VLM-based VLA with the four components."""

    def __init__(self):
        super().__init__()
        self.vision = nn.Sequential(nn.Conv2d(3, 8, 4, 4), nn.Flatten(2))  # patch embed conv
        self.vision_blocks = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 8))
        self.projector = nn.Linear(8, 16)
        self.embed_tokens = nn.Embedding(100, 16)
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, 16)) for _ in range(2)])
        self.lm_head = nn.Linear(16, 100, bias=False)
        self.action_head = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 7))

    def forward(self, img, ids):
        v = self.vision(img).transpose(1, 2)  # B, N, 8
        v = self.vision_blocks(v)
        v = self.projector(v)
        h = torch.cat([v, self.embed_tokens(ids)], dim=1)
        for blk in self.layers:
            h = h + blk(h)
        return self.lm_head(h[:, -1]), self.action_head(h[:, -1])


def cmap_of(m: ToyVLA) -> ComponentMap:
    cm = ComponentMap()
    cm.add("ve", "vision", m.vision)
    cm.add("ve", "vision_blocks", m.vision_blocks)
    cm.add("mp", "projector", m.projector)
    cm.add("llm", "layers", m.layers)
    cm.add("llm", "lm_head", m.lm_head)  # excluded by default pattern
    cm.add("ah", "action_head", m.action_head)
    return cm


def test_iter_quantizable_skips_embeddings_conv_and_lm_head():
    m = ToyVLA()
    names = [n for n, _ in iter_quantizable([("m", m)])]
    assert "m.lm_head" not in names
    assert not any("embed" in n for n in names)
    assert not any("vision.0" in n for n in names)  # Conv2d not included by default
    assert "m.projector" in names and "m.layers.0.0" in names and "m.action_head.2" in names
    names_conv = [n for n, _ in iter_quantizable([("m", m)], include_conv=True)]
    assert "m.vision.0" in names_conv


def test_weight_only_changes_weights_in_place_and_keeps_module_types():
    m = ToyVLA()
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    rep = quantize_modules([("", m.layers)], parse_spec("W4"))
    assert rep.n_layers == 4
    assert all(type(mod) in (nn.Linear, nn.SiLU, nn.Sequential, nn.ModuleList) for mod in m.layers.modules())
    for n, p in m.named_parameters():
        if n.startswith("layers.") and n.endswith("weight"):
            assert not torch.equal(p, before[n])
            assert p.unique().numel() <= 16 * 32  # at most 16 levels per group-row
        else:
            assert torch.equal(p, before[n])


def test_double_weight_quantization_raises():
    m = ToyVLA()
    quantize_modules([("", m.layers)], parse_spec("W4"))
    with pytest.raises(RuntimeError):
        quantize_modules([("", m.layers)], parse_spec("W3"))


def test_activation_hook_changes_outputs_and_can_be_removed():
    torch.manual_seed(0)
    m = ToyVLA().eval()
    img, ids = torch.randn(2, 3, 16, 16), torch.randint(0, 100, (2, 5))
    with torch.no_grad():
        ref_logits, ref_act = m(img, ids)
        spec = parse_spec("W16A4")  # activation only (W16 = weights untouched)
        assert spec.weight is None and spec.act.bits == 4
        quantize_modules([("", m.layers)], spec)
        q_logits, q_act = m(img, ids)
        assert not torch.allclose(q_act, ref_act)
        n = remove_activation_hooks([("", m.layers)])
        assert n == 4
        r_logits, r_act = m(img, ids)
        assert torch.allclose(r_act, ref_act)


def test_scope_resolution_and_component_isolation():
    m = ToyVLA()
    cm = cmap_of(m)
    assert cm.present() == ("ve", "mp", "llm", "ah")
    assert resolve_scope("e2e", cm) == ("ve", "mp", "llm", "ah")
    assert resolve_scope("llm+ah", cm) == ("llm", "ah")
    with pytest.raises(ValueError):
        resolve_scope("foo", cm)

    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    reports = quantize_scope(cm, parse_spec("W3"), scope="ah")
    assert set(reports) == {"ah"} and reports["ah"].n_layers == 2
    for n, p in m.named_parameters():
        changed = not torch.equal(p, before[n])
        assert changed == (n.startswith("action_head") and n.endswith("weight")), n


def test_lm_head_is_protected_unless_requested():
    m = ToyVLA()
    w0 = m.lm_head.weight.detach().clone()
    quantize_scope(cmap_of(m), parse_spec("W4"), scope="llm")
    assert torch.equal(m.lm_head.weight, w0)
    m2 = ToyVLA()
    rep = quantize_scope(cmap_of(m2), parse_spec("W4"), scope="llm", quantize_lm_head=True)
    assert any(l.name.endswith("lm_head") for l in rep["llm"].layers)


def test_overlapping_components_are_rejected():
    m = ToyVLA()
    cm = ComponentMap()
    cm.add("llm", "layers", m.layers)
    cm.add("ah", "layers0", m.layers[0])
    with pytest.raises(ValueError):
        cm.check_disjoint()


def test_param_table_counts_effective_layers():
    m = ToyVLA()
    cm = cmap_of(m)
    t = cm.param_table()
    assert t["ah"]["quantized_layers"] == 2
    assert t["ah"]["quantized_params"] == 16 * 16 + 16 * 7
    assert t["llm"]["quantized_layers"] == 4  # lm_head is excluded by default
    assert t["llm"]["params"] == sum(p.numel() for p in m.layers.parameters()) + m.lm_head.weight.numel()
    n_l, n_p = count_linear_params(m.action_head)
    assert n_l == 2 and n_p == 16 * 16 + 16 * 7


def test_nested_component_is_disjoint_via_exclusion():
    """A projector nested inside the vision tower can be given to `mp` with an exclusion."""
    m = ToyVLA()
    cm = ComponentMap()
    cm.add("ve", "vision", m.vision_blocks)          # contains .0 and .2
    cm.add("mp", "inner", m.vision_blocks[2])        # nested inside the ve root
    with pytest.raises(ValueError, match="disjoint"):
        cm.check_disjoint()
    cm.exclude["ve"] = ["*vision.2*"]
    cm.check_disjoint()  # no longer overlapping
    w_ve = m.vision_blocks[0].weight.detach().clone()
    w_mp = m.vision_blocks[2].weight.detach().clone()
    quantize_scope(cm, parse_spec("W3"), scope="mp")
    assert torch.equal(m.vision_blocks[0].weight, w_ve)      # ve layer untouched
    assert not torch.equal(m.vision_blocks[2].weight, w_mp)  # mp layer quantized


def test_e2e_w4a8_runs_forward_and_clear_hooks():
    torch.manual_seed(0)
    m = ToyVLA().eval()
    cm = cmap_of(m)
    quantize_scope(cm, parse_spec("W4A8"), scope="e2e")
    with torch.no_grad():
        out = m(torch.randn(1, 3, 16, 16), torch.randint(0, 100, (1, 4)))
    assert all(torch.isfinite(o).all() for o in out)
    assert clear_activation_quant(cm) == 9  # 2 ve + 1 mp + 4 llm + 2 ah linears
