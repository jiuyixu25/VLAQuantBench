"""VLAQuantBench: post-training quantization of vision-language-action models
evaluated entirely under closed-loop task success."""

from .components import COMPONENTS, ComponentMap, quantize_scope, resolve_scope
from .quant import BASELINE, PRESETS, QuantSpec, parse_spec, quantize_modules

__version__ = "0.1.0"
__all__ = [
    "COMPONENTS",
    "ComponentMap",
    "QuantSpec",
    "BASELINE",
    "PRESETS",
    "parse_spec",
    "quantize_modules",
    "quantize_scope",
    "resolve_scope",
    "__version__",
]
