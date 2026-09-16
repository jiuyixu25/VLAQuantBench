"""Named quantization presets (the paper's Table 7) and a small spec grammar.

Preset grammar (case-insensitive)::

    W<bits>[G<group>][A<bits>][SYM|ASYM]

Examples: ``W4`` ``W3`` ``W8`` ``W4A8`` ``W4A4`` ``W8A8`` ``W2`` ``W3G64``
``W4A8SYM`` ``W16`` (= baseline, no-op).

Defaults when the modifiers are omitted follow the benchmark protocol:

==========  =====================================  =====================
weights     bits <= 4                              per-group g=128, asymmetric
weights     bits >= 5                              per-channel, symmetric
acts        any                                    per-token, symmetric, dynamic
==========  =====================================  =====================
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .fake_quant import ActQuantSpec, WeightQuantSpec

__all__ = ["QuantSpec", "parse_spec", "PRESETS", "BASELINE", "DEFAULT_GROUP_SIZE"]

DEFAULT_GROUP_SIZE = 128
_LOW_BIT_THRESHOLD = 4  # <= 4 bits -> per-group asymmetric (Table 7)

_PATTERN = re.compile(
    r"^W(?P<w>\d{1,2})(?:G(?P<g>\d+))?(?:A(?P<a>\d{1,2}))?(?P<sym>SYM|ASYM)?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QuantSpec:
    """A complete (weight, activation) quantization setting."""

    name: str
    weight: WeightQuantSpec | None = None  # None -> weights untouched
    act: ActQuantSpec | None = None  # None -> activations untouched

    @property
    def is_baseline(self) -> bool:
        return self.weight is None and self.act is None

    def describe(self) -> str:
        parts = []
        if self.weight:
            parts.append(self.weight.short())
        if self.act:
            parts.append(self.act.short())
        return "+".join(parts) if parts else "baseline"


def parse_spec(text: str, *, default_group_size: int = DEFAULT_GROUP_SIZE) -> QuantSpec:
    """Parse a preset string such as ``"W4A8"`` into a :class:`QuantSpec`."""
    key = text.strip().replace("-", "").replace("_", "").upper()
    if key in ("BASELINE", "FP", "BF16", "FP32", "FP16", "NONE", "W16"):
        return BASELINE
    m = _PATTERN.match(key)
    if not m:
        raise ValueError(
            f"cannot parse quantization spec {text!r}; expected e.g. W4, W3G64, W4A8, W8A8SYM"
        )
    w_bits = int(m.group("w"))
    a_bits = int(m.group("a")) if m.group("a") else None
    explicit_group = int(m.group("g")) if m.group("g") else None
    sym_flag = m.group("sym").upper() if m.group("sym") else None

    weight: WeightQuantSpec | None
    if w_bits >= 16:
        weight = None
    elif explicit_group == 0:
        # G0 = per-channel at any bit-width (one scale per output row). Exists for
        # granularity ablations; the presets pair low bits with per-group scales.
        weight = WeightQuantSpec(
            bits=w_bits,
            granularity="per_channel",
            symmetric=(sym_flag != "ASYM") if sym_flag else True,
        )
    elif w_bits <= _LOW_BIT_THRESHOLD or explicit_group is not None:
        weight = WeightQuantSpec(
            bits=w_bits,
            granularity="per_group",
            group_size=explicit_group or default_group_size,
            symmetric=(sym_flag == "SYM") if sym_flag else False,
        )
    else:
        weight = WeightQuantSpec(
            bits=w_bits,
            granularity="per_channel",
            symmetric=(sym_flag != "ASYM") if sym_flag else True,
        )
    act = ActQuantSpec(bits=a_bits, granularity="per_token", symmetric=True) if a_bits and a_bits < 16 else None
    return QuantSpec(name=key, weight=weight, act=act)


BASELINE = QuantSpec(name="BASELINE")

#: The six settings evaluated in the paper plus W2 (added for the released benchmark).
PRESETS: dict[str, QuantSpec] = {
    name: parse_spec(name) for name in ("W2", "W3", "W4", "W8", "W4A4", "W4A8", "W8A8")
}
PRESETS["BASELINE"] = BASELINE
