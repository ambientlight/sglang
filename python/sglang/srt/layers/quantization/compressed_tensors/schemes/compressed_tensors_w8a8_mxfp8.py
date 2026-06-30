# SPDX-License-Identifier: Apache-2.0
"""MXFP8 weight-only linear scheme for SM120 (MiniMax-M3).

M3's non-expert layers (attention q/k/v/o, dense MLP, shared experts) are
compressed-tensors `mxfp8-quantized`: FP8-E4M3 weights + UE8M0 group-32 block
scales, no input quant in the checkpoint. sglang has no SM120 native MXFP8
linear GEMM (the fp8 path is HIP/gfx95-gated, DeepGEMM excludes SM120, no
MarlinMxfp8), but `mxfp8_native.dot_scaled_mxfp8_blockscaled_linear` (a pure
Triton `tl.dot_scaled` kernel, originally written for CDNA4) runs correctly on
SM120 — validated cos 0.9993 vs a BF16 reference on M3's 6144x6144 shape. This
scheme loads the packed E4M3 weight + E8M0 scale and dispatches that GEMM.

Selected by `compressed_tensors.get_scheme` via `_is_mxfp8_weight_only` on SM120.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import torch

from sglang.srt.layers.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsLinearScheme,
)

__all__ = ["CompressedTensorsW8A8Mxfp8"]

_MXFP8_BLOCK = 32  # E8M0 group size


class CompressedTensorsW8A8Mxfp8(CompressedTensorsLinearScheme):
    """MXFP8 (E4M3 weight + UE8M0 group-32 scale) linear via Triton dot_scaled."""

    def __init__(self, strategy: str):
        self.strategy = strategy

    @classmethod
    def get_min_capability(cls) -> int:
        return 120  # RTX PRO 6000 Blackwell (SM120) — the Triton dot_scaled path

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size: int,
        output_partition_sizes: List[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # WEIGHT: FP8 E4M3, [N, K]
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE: UE8M0 stored as raw uint8 bytes, [N, K // 32]. The
        # checkpoint already holds the E8M0 exponent bytes (e.g. 127 == 2^0),
        # which is exactly what dot_scaled consumes — declare the param as uint8
        # so the HF loader copies the bytes verbatim (a float32 param would copy
        # by VALUE, turning byte 115 into 115.0 and corrupting the encoding).
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // _MXFP8_BLOCK,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer) -> None:
        # Scales are already raw UE8M0 bytes (uint8) from the checkpoint — no
        # re-encode. Just materialize both as plain Parameters.
        layer.weight = torch.nn.Parameter(layer.weight.data, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(
            layer.weight_scale.data, requires_grad=False
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from sglang.srt.layers.quantization.mxfp8_native import (
            dot_scaled_mxfp8_blockscaled_linear,
        )

        out = dot_scaled_mxfp8_blockscaled_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            output_dtype=x.dtype,
        )
        if bias is not None:
            out = out + bias
        return out
