"""Model-agnostic quantised inference kernels.

This package is the reusable IP: nothing in it knows about Inkling, Qwen or any
other model. A model supplies shapes and a layer plan; these kernels do the
work.

The point of the split is that the same code has to serve several very
different checkpoints:

  * Inkling 2 Mini  - 24 layers, a middle block run three times, 128 experts,
                      top-8, experts in NVFP4 (layers 0-19) and MXFP8 (20-23)
  * Qwen 3.8 27B    - a plain MoE stack, no looping, different expert counts
  * dense models    - no experts at all

so the kernel is parameterised by a Format (how a weight block is packed) and
driven by a Plan (which layers run, and how many times). Everything else is
shared.
"""
from .formats import Format, NVFP4, MXFP8, FP8_E4M3_BLOCK, BF16, get_format
from .plans import LayerPlan, Plan
from .execution import LayerFootprint, Budget, budget_for, compare
from .quant_ref import (
    dequant_block_numpy, quantized_gemv_numpy, e4m3_to_float, e2m1_code_values,
)

__all__ = [
    "Format", "NVFP4", "MXFP8", "FP8_E4M3_BLOCK", "BF16", "get_format",
    "LayerPlan", "Plan",
    "LayerFootprint", "Budget", "budget_for", "compare",
    "dequant_block_numpy", "quantized_gemv_numpy", "e4m3_to_float", "e2m1_code_values",
]
