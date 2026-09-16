"""Component decomposition of a VLA policy and quantization *scopes*.

Every VLM-based VLA is decomposed into four functional components:

``ve``  vision encoder(s)            ``mp``  multimodal projector
``llm`` LLM / language backbone      ``ah``  action decoding head

An adapter (:mod:`vlaquantbench.models`) fills a :class:`ComponentMap` with the
*module roots* of each component. A **scope** selects which components are
quantized while the rest stay at baseline precision:

* ``e2e``                    every component present in the model
* ``ve`` / ``mp`` / ``llm`` / ``ah``   component-isolated
* ``llm+ah`` (any ``+``-joined subset)  custom combination
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from torch import nn

from .quant.apply import (
    DEFAULT_EXCLUDE_PATTERNS,
    QuantReport,
    quantize_modules,
    remove_activation_hooks,
)
from .quant.presets import QuantSpec

__all__ = ["COMPONENTS", "ComponentMap", "resolve_scope", "quantize_scope"]

COMPONENTS: tuple[str, ...] = ("ve", "mp", "llm", "ah")


@dataclass
class ComponentMap:
    """Module roots per component. Roots are ``(name, module)`` pairs.

    ``exclude`` holds extra fnmatch patterns per component that must stay at
    baseline precision (in addition to :data:`DEFAULT_EXCLUDE_PATTERNS`);
    ``allow`` lists exceptions to the exclude patterns (e.g. ``*lm_head*`` to
    quantize the vocabulary head after all).
    """

    ve: list[tuple[str, nn.Module]] = field(default_factory=list)
    mp: list[tuple[str, nn.Module]] = field(default_factory=list)
    llm: list[tuple[str, nn.Module]] = field(default_factory=list)
    ah: list[tuple[str, nn.Module]] = field(default_factory=list)
    exclude: dict[str, list[str]] = field(default_factory=dict)
    allow: dict[str, list[str]] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)

    def roots(self, component: str) -> list[tuple[str, nn.Module]]:
        if component not in COMPONENTS:
            raise KeyError(f"unknown component {component!r}; expected one of {COMPONENTS}")
        return getattr(self, component)

    def present(self) -> tuple[str, ...]:
        return tuple(c for c in COMPONENTS if self.roots(c))

    def add(self, component: str, name: str, module: nn.Module) -> None:
        self.roots(component).append((name, module))

    def param_table(self, *, include_conv: bool = False) -> dict[str, dict[str, int]]:
        """Per-component totals: all params under the roots, and the layers that
        would actually be quantized (after exclusion / allow patterns)."""
        from .quant.apply import DEFAULT_EXCLUDE_PATTERNS, iter_quantizable

        table: dict[str, dict[str, int]] = {}
        for c in COMPONENTS:
            total = 0
            seen: set[int] = set()
            for _, root in self.roots(c):
                if id(root) in seen:
                    continue
                seen.add(id(root))
                total += sum(p.numel() for p in root.parameters())
            exclude = list(DEFAULT_EXCLUDE_PATTERNS) + list(self.exclude.get(c, []))
            allow = list(self.allow.get(c, []))
            layers = list(
                iter_quantizable(self.roots(c), include_conv=include_conv, exclude_patterns=exclude, allow_patterns=allow)
            )
            table[c] = {
                "params": total,
                "quantized_layers": len(layers),
                "quantized_params": sum(m.weight.numel() for _, m in layers),
            }
        return table

    def effective_layers(self, component: str, *, include_conv: bool = True) -> dict[int, str]:
        """``{id(module): qualified_name}`` of the layers that quantizing ``component`` would touch.

        Exclusion / allow patterns are applied, so a module that is filtered out
        of a component (e.g. a projector nested inside the vision tower) does
        not count as belonging to it.
        """
        from .quant.apply import DEFAULT_EXCLUDE_PATTERNS, iter_quantizable

        exclude = list(DEFAULT_EXCLUDE_PATTERNS) + list(self.exclude.get(component, []))
        allow = list(self.allow.get(component, []))
        return {
            id(mod): name
            for name, mod in iter_quantizable(
                self.roots(component), include_conv=include_conv, exclude_patterns=exclude, allow_patterns=allow
            )
        }

    def check_disjoint(self) -> None:
        """Raise if the same layer would be quantized by two different components."""
        owner: dict[int, tuple[str, str]] = {}
        for c in COMPONENTS:
            if not self.roots(c):
                continue
            for mid, name in self.effective_layers(c).items():
                prev = owner.setdefault(mid, (c, name))
                if prev[0] != c:
                    raise ValueError(
                        f"layer {name!r} (also {prev[1]!r}) would be quantized by both {prev[0]!r} and {c!r}; "
                        f"components must be disjoint - use ComponentMap.exclude[{prev[0]!r}] to give it to {c!r}"
                    )


def resolve_scope(scope: str, cmap: ComponentMap) -> tuple[str, ...]:
    """Translate a scope string into the tuple of components to quantize."""
    key = scope.strip().lower().replace(" ", "")
    if key in ("e2e", "end2end", "end-to-end", "all", "full"):
        return cmap.present()
    parts = tuple(dict.fromkeys(key.split("+")))
    for p in parts:
        if p not in COMPONENTS:
            raise ValueError(f"unknown component {p!r} in scope {scope!r}; expected {COMPONENTS} or 'e2e'")
        if not cmap.roots(p):
            raise ValueError(f"component {p!r} is absent from this model (present: {cmap.present()})")
    return parts


def quantize_scope(
    cmap: ComponentMap,
    spec: QuantSpec,
    scope: str = "e2e",
    *,
    include_conv: bool = False,
    quantize_lm_head: bool = False,
    compute_error: bool = False,
    act_calib: dict | None = None,
) -> dict[str, QuantReport]:
    """Quantize the components selected by ``scope``. Returns per-component reports.

    ``act_calib`` maps qualified layer names to :class:`LayerCalib`; matching
    layers get smoothing folded into their weights and a calibrated activation
    hook instead of the plain per-token absmax one.
    """
    reports: dict[str, QuantReport] = {}
    cmap.check_disjoint()
    for c in resolve_scope(scope, cmap):
        exclude: list[str] = list(DEFAULT_EXCLUDE_PATTERNS) + list(cmap.exclude.get(c, []))
        allow: list[str] = list(cmap.allow.get(c, []))
        if quantize_lm_head and c == "llm":
            allow.append("*lm_head*")
        reports[c] = quantize_modules(
            cmap.roots(c),
            spec,
            include_conv=include_conv,
            exclude_patterns=exclude,
            allow_patterns=allow,
            compute_error=compute_error,
            act_calib=act_calib,
        )
    return reports


def clear_activation_quant(cmap: ComponentMap) -> int:
    n = 0
    for c in COMPONENTS:
        n += remove_activation_hooks(cmap.roots(c))
    return n
