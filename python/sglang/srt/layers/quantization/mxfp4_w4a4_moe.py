"""Native MXFP4 x MXFP4 (W4A4-mx) MoE method for SM120 — no re-quant, no calibration.

DeepSeek-V4-Flash ships routed experts as native MXFP4 (E2M1 packed int8
weights + E8M0 32-element block scales). This method loads those tensors as-is
and runs the experts with FlashInfer's *fused* SwiGLU CuTe-DSL MoE kernels
(``launch_sm120_moe(quant_mode="mxfp4")``) — MXFP4 weights x MXFP4 activations,
E8M0 self-scaling.

Contrast:
  * W4A8-mx (``mxfp4_w4a8_moe.py``): MXFP4 weights x MXFP8 activations via the
    un-fused grouped-GEMM cubin. Correct (cos 0.9986) but FP8-class throughput.
  * This (W4A4-mx): MXFP4 weights x MXFP4 activations via the *fused* kernels
    (the same ones the retired NVFP4 path used to hit 759 tok/s @16w), but
    native MXFP4 — no MXFP4->BF16->NVFP4 re-quant, no activation calibration.

Weight loading (create_weights + the native-MXFP4 read) is shared verbatim with
the W4A8-mx method; only the weight-scale swizzle TARGET and the forward differ:
  * W4A8 swizzles weight scales into the grouped-GEMM 128x4 layout.
  * W4A4 (here) swizzles into the 128x4 layout THEN converts to the MMA layout
    that the fused kernel's ``_get_weight_views`` expects (it reverses the MMA
    layout via ``convert_sf_from_mma_layout``).

Validated on real DeepSeek-V4-Flash weights (rtx-pro-6000-bench/spikes): fused
W4A4-mx vs BF16 cos ~0.975 (the FP4-activation floor) across micro/static/
dynamic backends, M=1..256 incl skewed >64-rows/expert routing.

Selected by ``fp8.py`` when: SM120 + native MXFP4 experts + the env flag
``SGLANG_MXFP4_W4A4=1``. NVFP4/Marlin and W4A8-mx paths are left untouched.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils import log_info_on_rank0, set_weight_attrs
from sglang.srt.utils.common import is_sm120_supported

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput

logger = logging.getLogger(__name__)

_FP4_BLOCK_K = 32  # E8M0 block size


class Mxfp4W4A4MoEMethod:
    """Native MXFP4 weights x MXFP4 activations MoE (SM120 fused CuTe-DSL)."""

    def __init__(self, fp8_method, prefix: str):
        self._fp8 = fp8_method
        self.prefix = prefix

    # ------------------------------------------------------------------ #
    def create_moe_runner(self, layer, moe_runner_config):
        # The fused launch is invoked directly in apply(); the runner is only
        # kept so MoE config (routed_scaling_factor / swiglu_limit) is reachable.
        from sglang.srt.layers.moe.moe_runner import MoeRunner

        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    # ------------------------------------------------------------------ #
    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Register native-MXFP4 expert buffers sized to the checkpoint
        (identical to the W4A8-mx method): E2M1-packed int8 weights + (float32
        placeholder) E8M0 block scales, so the HF weight loader fills them
        directly — no FP8 dequant, no buffer-size mismatch."""
        from sglang.srt.layers.moe.fused_moe_triton import (
            FusedMoeWeightScaleSupported,
        )

        w13_weight = Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        w2_weight = Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Scale buffers are float32 (the HF loader casts checkpoint F8_E8M0 -> f32
        # losslessly; E8M0 are exact powers of two). process_weights_after_loading
        # re-encodes to E8M0 + swizzles + converts to the MMA layout.
        w13_weight_scale = Parameter(
            torch.ones(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // _FP4_BLOCK_K,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        w2_weight_scale = Parameter(
            torch.ones(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // _FP4_BLOCK_K,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        w13_weight_scale.format_ue8m0 = False
        w2_weight_scale.format_ue8m0 = False
        scale_attrs = dict(extra_weight_attrs)
        scale_attrs["quant_method"] = FusedMoeWeightScaleSupported.BLOCK.value
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, scale_attrs)
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, scale_attrs)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_e8m0_u8(scale_f32: torch.Tensor) -> torch.Tensor:
        """float32 scale magnitude -> E8M0 byte (uint8). Lossless for the
        power-of-two values the checkpoint stores."""
        return scale_f32.to(torch.float8_e8m0fnu).view(torch.uint8)

    def process_weights_after_loading(self, layer: Module) -> None:
        if not is_sm120_supported():
            raise RuntimeError(
                "Mxfp4W4A4MoEMethod requires SM120 (RTX PRO 6000 Blackwell)."
            )
        if getattr(layer, "_w4a4_weights_built", False):
            return

        # Reuse the W4A8-mx weight-scale 128x4 swizzle, then convert to the MMA
        # layout the fused kernel's _get_weight_views expects (it calls
        # convert_sf_from_mma_layout to reverse it back to 2D-swizzled storage).
        from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
        from sglang.srt.layers.moe.fused_moe_triton.mxfp4_w4a8_sm120 import (
            swizzle_weight_scale_mxf4,
        )

        w13 = layer.w13_weight.data  # (E, 2I, H/2) int8, native [w1|w3]
        w2 = layer.w2_weight.data  # (E, H, I/2)  int8
        w13_s = layer.w13_weight_scale_inv.data
        w2_s = layer.w2_weight_scale_inv.data

        E = w13.shape[0]
        twoI = w13.shape[1]
        Hhalf = w13.shape[2]
        H = Hhalf * 2
        Ihalf = w2.shape[2]
        I = Ihalf * 2

        # The fused SM120 W4A4 kernel's gate/up tile split expects [up, gate] =
        # [w3, w1] order along the 2I dim, but DeepSeek-V4-Flash loads W13 as
        # [w1=gate, w3=up]. Swap the two I-row halves of BOTH the packed weights
        # and the (still 3D, pre-swizzle) scales, per expert. (Mirrors the retired
        # NVFP4 marlin path's reorder_w1w3_to_w3w1; w2 has no gate/up split.)
        Irows = twoI // 2
        w13 = torch.cat([w13[:, Irows:, :], w13[:, :Irows, :]], dim=1).contiguous()
        w13_s = torch.cat([w13_s[:, Irows:, :], w13_s[:, :Irows, :]], dim=1).contiguous()

        # E8M0 bytes (uint8), unswizzled: w13_s (E,2I,H/32), w2_s (E,H,I/32)
        w13_s_u8 = self._to_e8m0_u8(w13_s.to(torch.float32))
        w2_s_u8 = self._to_e8m0_u8(w2_s.to(torch.float32))

        # 1) 128x4 block-scale swizzle (per expert, validated reference path).
        w13_s_sw = swizzle_weight_scale_mxf4(w13_s_u8, E, twoI, H)
        w2_s_sw = swizzle_weight_scale_mxf4(w2_s_u8, E, H, I)

        # 2) Convert the swizzled scales into the MMA layout (6D physical order)
        #    that the fused kernel reads. convert_sf_to_mma_layout expects the
        #    128x4-swizzled 2D scales and reshapes them; flatten experts into the
        #    leading row dim (num_groups=E) exactly like the validated spike.
        w13_sf_mma = convert_sf_to_mma_layout(
            w13_s_sw.reshape(E * w13_s_sw.shape[1], w13_s_sw.shape[2]),
            m=twoI, k=H, num_groups=E, sf_vec_size=_FP4_BLOCK_K,
        )
        w2_sf_mma = convert_sf_to_mma_layout(
            w2_s_sw.reshape(E * w2_s_sw.shape[1], w2_s_sw.shape[2]),
            m=H, k=I, num_groups=E, sf_vec_size=_FP4_BLOCK_K,
        )

        layer.w13_weight = Parameter(
            w13.view(torch.uint8).contiguous(), requires_grad=False
        )
        layer.w2_weight = Parameter(
            w2.view(torch.uint8).contiguous(), requires_grad=False
        )
        layer.w13_weight_scale_inv = Parameter(
            w13_sf_mma.contiguous(), requires_grad=False
        )
        layer.w2_weight_scale_inv = Parameter(
            w2_sf_mma.contiguous(), requires_grad=False
        )
        layer._w4a4_H = H
        layer._w4a4_I = I
        layer._w4a4_E = E
        layer._w4a4_weights_built = True

        # Pre-materialize the FlashInfer weight views ONCE here (at load time),
        # not lazily inside apply()/CUDA-graph capture. _get_weight_views builds
        # contiguous swizzled scale storage (~48 MB/layer) and caches it in
        # _WEIGHT_CACHE keyed by data_ptr; if first built during capture those
        # ~2 GB (×43 layers) land in the graph-private pool. Building them now
        # keeps them as ordinary pre-capture model state (mirrors NVFP4 marlin).
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
            _get_weight_views,
        )

        ones_e = torch.ones(E, device=layer.w13_weight.device, dtype=torch.float32)
        layer._w4a4_alpha = ones_e  # GEMM alpha = 1 (MXF4 self-scales); reused
        layer._w4a4_weight_views = _get_weight_views(
            w1_fp4=layer.w13_weight,
            w1_blockscale=layer.w13_weight_scale_inv,
            w2_fp4=layer.w2_weight,
            w2_blockscale=layer.w2_weight_scale_inv,
            w1_alphas=ones_e,
            w2_alphas=ones_e,
            n=I,
            k=H,
            activation_precision="fp4",
            quant_mode="mxfp4",
        )

        # Release the large per-layer transients (f32 scale upcasts, the
        # torch.cat weight copies ~2 GB each, swizzle/MMA temps). The final
        # Parameters above hold .contiguous() copies, so every intermediate is
        # dead here. Without this the caching allocator keeps ~8-10 GB/GPU of
        # high-water reserve across 43 layers, starving CUDA-graph capture and
        # long-context attention (mirrors the NVFP4 marlin cleanup).
        del w13, w2, w13_s, w2_s
        del w13_s_u8, w2_s_u8, w13_s_sw, w2_s_sw, w13_sf_mma, w2_sf_mma
        torch.cuda.empty_cache()

        log_info_on_rank0(
            logger,
            f"SM120 W4A4-mx: native MXFP4 experts ready "
            f"(E={E}, H={H}, I={I}, fused MXFP4xMXFP4, no re-quant, "
            f"no calibration; layer: {self.prefix})",
        )

    # ------------------------------------------------------------------ #
    def apply(
        self,
        layer: Module,
        dispatch_output: "DispatchOutput",
    ) -> "CombineInput":
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
            launch_sm120_moe,
            _get_cached_workspace,
            select_sm120_moe_backend,
        )
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardCombineInput,
        )
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        topk_output = dispatch_output.topk_output
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise ValueError(f"Unsupported topk output format: {topk_output.format}")

        hidden_states = dispatch_output.hidden_states
        M = hidden_states.shape[0]
        H = layer._w4a4_H
        I = layer._w4a4_I
        E_local = layer._w4a4_E
        # Global expert count drives topk-id range; local count sizes the expert
        # buffers. With expert parallelism global > local and the kernel remaps
        # global topk ids -> local via its global_to_local table (static path).
        E_global = int(getattr(layer, "num_experts", E_local))
        top_k = topk_output.topk_ids.shape[1]
        dev = hidden_states.device

        cfg = getattr(self.runner, "config", None)

        # MXF4 self-scales: no FC1/FC2 activation global scale. Reuse the alpha
        # ones tensor built at load time (GEMM alpha = 1, fc2_input_scale = 1).
        ones = layer._w4a4_alpha

        # Pre-materialize ONE capped STATIC workspace per (layer-config, top_k)
        # so decode batches use a workspace allocated OUTSIDE CUDA-graph capture
        # that never grows with live routed_rows. The cap (>= the static/dynamic
        # cutover, 640) covers all static-path batches; mirrors the NVFP4 marlin
        # SGLANG_NVFP4_STATIC_WS_CAP pattern. _WORKSPACE_CACHE is process-global
        # by shape, so the first layer allocates and the rest hit the cache.
        #
        # Larger batches (prefill) select the "dynamic" backend or need a static
        # workspace with > cap rows; for those we pass _workspace=None and let
        # launch_sm120_moe allocate the correct one (passing a static workspace
        # would force the static backend with insufficient capacity).
        ws_cap = int(os.environ.get("SGLANG_MXFP4_STATIC_WS_CAP", "640"))
        routed_rows = M * top_k
        ws_key = getattr(layer, "_w4a4_ws_key", None)
        if ws_key != top_k:
            layer._w4a4_static_ws = _get_cached_workspace(
                backend="static",
                state_E=E_local,
                weight_E=E_global,
                routed_rows=ws_cap,
                k=H,
                n=I,
                num_topk=top_k,
                device=dev,
                quant_mode="mxfp4",
                activation="silu",
            )
            layer._w4a4_ws_key = top_k

        backend = select_sm120_moe_backend(
            num_tokens=M, num_topk=top_k, quant_mode="mxfp4"
        )
        # EP forces static (kernel needs global->local remap); match the launcher.
        use_static = backend == "static" and routed_rows <= ws_cap
        workspace = layer._w4a4_static_ws if use_static else None

        output = torch.empty(M, H, dtype=hidden_states.dtype, device=dev)
        launch_sm120_moe(
            a=hidden_states,
            topk_ids=topk_output.topk_ids,
            topk_weights=topk_output.topk_weights,
            w1_weight=layer.w13_weight,
            w1_weight_sf=layer.w13_weight_scale_inv,
            w1_alpha=ones,
            fc2_input_scale=ones,
            w2_weight=layer.w2_weight,
            w2_weight_sf=layer.w2_weight_scale_inv,
            w2_alpha=ones,
            num_experts=E_global,
            top_k=top_k,
            num_local_experts=E_local,
            scatter_output=output,
            input_scales_are_reciprocal=False,
            fast_math=True,
            activation="silu",
            quant_mode="mxfp4",
            _weight_views=layer._w4a4_weight_views,
            _workspace=workspace,
        )

        routed_scaling_factor = (
            getattr(cfg, "routed_scaling_factor", None) if cfg else None
        )
        if routed_scaling_factor is not None and routed_scaling_factor != 1.0:
            output = output * routed_scaling_factor
        return StandardCombineInput(hidden_states=output)
