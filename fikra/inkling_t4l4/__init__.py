"""Inkling 2 Mini on T4 / L4.

Ships no weights. Loads the released Apache 2.0 checkpoint
(thinkingmachines/Inkling-2-Mini) and measures it so results compare across
people and machines.

WHY THIS EXISTS
The released checkpoint is 19.53 GB: experts are NVFP4 on layers 0-19 and
MXFP8 on 20-23, while attention, norms, router and embeddings are bf16 (the
attention-residual queries and MoE gate bias are fp32). T4 (sm75) has no FP4
or FP8 hardware and L4 (sm89) has FP8 but no MX block scaling, so both need
the expert weights unpacked in software. Only the expert tensors are
quantised, which confines that work to the MoE path.
"""
from .benchmark import BenchResult, run_decode_benchmark, quality_on_manifest, load_manifest

__all__ = ["BenchResult", "run_decode_benchmark", "quality_on_manifest", "load_manifest"]
