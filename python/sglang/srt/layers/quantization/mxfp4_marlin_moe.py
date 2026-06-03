from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch
from torch.nn import Module

from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils import log_info_on_rank0, set_weight_attrs
from sglang.srt.utils.common import is_sm90_supported, is_sm120_supported

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput

logger = logging.getLogger(__name__)


class Mxfp4MarlinMoEMethod:
    """MXFP4 (E8M0 scales) MoE quantization method using the Marlin backend."""

    def __init__(self, fp8_method, prefix: str):
        self._fp8 = fp8_method
        self.prefix = prefix

    def create_moe_runner(self, layer, moe_runner_config):
        from sglang.srt.layers.moe.moe_runner import MoeRunner

        # When using FP8 buffers for NVFP4, use triton runner as fallback
        # (the NVFP4 apply() bypasses the runner anyway)
        if (
            os.environ.get("SGLANG_FP4_MOE_NVFP4", "0") == "1"
            and os.environ.get("SGLANG_DSV4_FP4_EXPERTS", "1") == "0"
        ):
            self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)
        else:
            self.runner = MoeRunner(MoeRunnerBackend.MARLIN, moe_runner_config)

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # When loading FP8 checkpoint for NVFP4 re-quantization,
        # delegate to FP8 base to create correctly-sized FP8 buffers.
        if (
            os.environ.get("SGLANG_FP4_MOE_NVFP4", "0") == "1"
            and os.environ.get("SGLANG_DSV4_FP4_EXPERTS", "1") == "0"
        ):
            self._fp8.create_weights(
                layer, num_experts, hidden_size,
                intermediate_size_per_partition, params_dtype,
                **extra_weight_attrs,
            )
            return

        from sglang.srt.layers.moe.fused_moe_triton import (
            FusedMoeWeightScaleSupported,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // fp4_block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // fp4_block_k,
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

    def process_weights_after_loading(self, layer: Module) -> None:
        from sglang.srt.layers.quantization.marlin_utils import (
            check_moe_marlin_supports_layer,
        )
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            prepare_moe_mxfp4_layer_for_marlin,
        )

        # Let the FP8 base method handle ROCm normalization, etc.
        self._fp8.process_weights_after_loading(layer)

        if getattr(layer, "_mega_moe_weights_built", False):
            return

        if not is_sm90_supported() and not is_sm120_supported():
            raise RuntimeError(
                "DeepSeekV4 MXFP4 Marlin fallback requires Hopper/SM90 or above."
            )

        # SM120: Skip Marlin repacking, keep original weight format
        # for Triton dequant kernel (Marlin kernel produces NaN on SM120)
        if is_sm120_supported():
            from torch.nn import Parameter

            w13 = layer.w13_weight.data
            w2 = layer.w2_weight.data
            w13_s = layer.w13_weight_scale_inv.data
            w2_s = layer.w2_weight_scale_inv.data

            # ── NVFP4 (W4A4) B12x tensor-core path (direct static/dynamic kernel) ──
            if os.environ.get("SGLANG_FP4_MOE_NVFP4", "0") == "1":
                log_info_on_rank0(
                    logger,
                    f"SM120 NVFP4: preparing direct SM120 MoE "
                    f"(layer: {self.prefix})...",
                )
                try:
                    from flashinfer import nvfp4_quantize
                    from flashinfer.quantization.fp4_quantization import SfLayout
                    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
                    from sglang.srt.layers.quantization.utils import swizzle_blockscale

                    E_local = w13.shape[0]
                    I_per_part = w13.shape[1] // 2
                    is_fp8_input = w13.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)

                    if is_fp8_input:
                        # FP8 checkpoint: dequantize FP8 → bfloat16 using block scales
                        K = w13.shape[2]  # FP8 is not packed
                        log_info_on_rank0(logger,
                            f"  FP8→NVFP4: w13={w13.shape} {w13.dtype}, w2={w2.shape}, "
                            f"E={E_local}, I={I_per_part}, K={K}")

                        def _dequant_fp8_expert(w_fp8, s_fp8):
                            """Dequantize block-quantized FP8 to bfloat16."""
                            w_f32 = w_fp8.to(torch.float32)
                            if s_fp8 is not None and s_fp8.numel() > 1:
                                # Block-quantized: s has shape matching weight blocks
                                # Broadcast scales to weight shape
                                if s_fp8.dim() == 1:
                                    # Per-expert scalar scale
                                    w_f32 = w_f32 * s_fp8.item()
                                else:
                                    # Block scales: [rows // block_h, cols // block_w]
                                    bh = w_fp8.shape[0] // s_fp8.shape[0]
                                    bw = w_fp8.shape[1] // s_fp8.shape[1]
                                    s_expanded = s_fp8.repeat_interleave(bh, dim=0).repeat_interleave(bw, dim=1)
                                    w_f32 = w_f32 * s_expanded[:w_fp8.shape[0], :w_fp8.shape[1]]
                            elif s_fp8 is not None:
                                w_f32 = w_f32 * s_fp8.item()
                            return w_f32.to(torch.bfloat16)
                    else:
                        # MXFP4 checkpoint: original path
                        K = w13.shape[2] * 2  # MXFP4 is packed
                        log_info_on_rank0(logger,
                            f"  MXFP4→NVFP4: w13={w13.shape} {w13.dtype}, w2={w2.shape}, "
                            f"E={E_local}, I={I_per_part}, K={K}")

                        from flashinfer import mxfp4_dequantize

                        def _to_e8m0(s):
                            if s.dtype == torch.float32:
                                e8m0 = torch.zeros(s.shape, dtype=torch.uint8,
                                                   device=s.device)
                                nz = s > 0
                                e8m0[nz] = (torch.log2(s[nz]).round().to(torch.int32)
                                            + 127).clamp(0, 255).to(torch.uint8)
                                return e8m0.view(torch.float8_e8m0fnu)
                            return s

                    dev = w13.device
                    gs_one = torch.tensor(1.0, dtype=torch.float32, device=dev)

                    # Re-quantize W13 via nvfp4_quantize(gs=1.0)
                    # Direct SM120 static W4A4 kernel expects [up, gate] order.
                    # W13 is loaded as [w1=gate, w3=up] → swap to [w3=up, w1=gate]
                    log_info_on_rank0(logger, "  Re-quantizing W13 as [up, gate]...")
                    w13_nv_list, w13_sf_list = [], []
                    for e in range(E_local):
                        # Swap gate/up halves before quantization
                        w1_e = w13[e, :I_per_part]    # gate
                        w3_e = w13[e, I_per_part:]    # up
                        w13_e = torch.cat([w3_e, w1_e], dim=0)

                        if is_fp8_input:
                            s1_e = w13_s[e, :I_per_part] if w13_s.dim() >= 3 else w13_s[e]
                            s3_e = w13_s[e, I_per_part:] if w13_s.dim() >= 3 and w13_s.shape[1] > 1 else w13_s[e]
                            s13_e = torch.cat([s3_e, s1_e], dim=0) if w13_s.dim() >= 3 and w13_s.shape[1] > 1 else s1_e
                            w_bf16 = _dequant_fp8_expert(w13_e, s13_e)
                        else:
                            s1_e = w13_s[e, :I_per_part]
                            s3_e = w13_s[e, I_per_part:]
                            s13_e = torch.cat([s3_e, s1_e], dim=0)
                            s_e8m0 = _to_e8m0(s13_e)
                            w_float = mxfp4_dequantize(
                                w13_e.cpu().view(torch.uint8), s_e8m0.cpu()
                            )
                            w_bf16 = w_float.to(dev).to(torch.bfloat16)

                        nv_d, nv_sf = nvfp4_quantize(
                            w_bf16, gs_one, sfLayout=SfLayout.layout_128x4
                        )
                        w13_nv_list.append(nv_d)
                        w13_sf_list.append(nv_sf)
                        del w_bf16
                    w13_fp4 = torch.stack(w13_nv_list)
                    w13_sf_raw = torch.stack(w13_sf_list)
                    del w13_nv_list, w13_sf_list

                    log_info_on_rank0(logger, "  Re-quantizing W2...")
                    w2_nv_list, w2_sf_list = [], []
                    for e in range(E_local):
                        if is_fp8_input:
                            s_e = w2_s[e] if w2_s.dim() >= 2 else w2_s
                            w_bf16 = _dequant_fp8_expert(w2[e], s_e)
                        else:
                            s_e8m0 = _to_e8m0(w2_s[e])
                            w_float = mxfp4_dequantize(
                                w2[e].cpu().view(torch.uint8), s_e8m0.cpu()
                            )
                            w_bf16 = w_float.to(dev).to(torch.bfloat16)

                        nv_d, nv_sf = nvfp4_quantize(
                            w_bf16, gs_one, sfLayout=SfLayout.layout_128x4
                        )
                        w2_nv_list.append(nv_d)
                        w2_sf_list.append(nv_sf)
                        del w_bf16
                    w2_fp4 = torch.stack(w2_nv_list)
                    w2_sf_raw = torch.stack(w2_sf_list)
                    del w2_nv_list, w2_sf_list

                    # Free originals
                    del w13, w2, w13_s, w2_s
                    layer.w13_weight = None
                    layer.w2_weight = None
                    layer.w13_weight_scale_inv = None
                    layer.w2_weight_scale_inv = None
                    torch.cuda.empty_cache()

                    # Convert block scales: swizzle → MMA layout
                    # nvfp4_quantize outputs swizzled 128x4 scales
                    w13_sf_e4m3 = w13_sf_raw.view(torch.float8_e4m3fn)
                    w13_rows = w13_fp4.shape[1]
                    w13_sf_mma = convert_sf_to_mma_layout(
                        w13_sf_e4m3.view(torch.uint8).reshape(
                            E_local * w13_sf_e4m3.shape[1], w13_sf_e4m3.shape[2]
                        ),
                        m=w13_rows, k=K, num_groups=E_local,
                    )
                    w2_sf_e4m3 = w2_sf_raw.view(torch.float8_e4m3fn)
                    w2_rows = w2_fp4.shape[1]
                    w2_sf_mma = convert_sf_to_mma_layout(
                        w2_sf_e4m3.view(torch.uint8).reshape(
                            E_local * w2_sf_e4m3.shape[1], w2_sf_e4m3.shape[2]
                        ),
                        m=w2_rows, k=I_per_part, num_groups=E_local,
                    )
                    del w13_sf_raw, w2_sf_raw, w13_sf_e4m3, w2_sf_e4m3

                    log_info_on_rank0(logger,
                        f"  w13_fp4={w13_fp4.shape}, w13_sf_mma={w13_sf_mma.shape}")

                    # Store for apply()
                    layer._nvfp4_w13_fp4 = w13_fp4
                    layer._nvfp4_w2_fp4 = w2_fp4
                    layer._nvfp4_w13_sf = w13_sf_mma
                    layer._nvfp4_w2_sf = w2_sf_mma
                    layer._nvfp4_I = I_per_part
                    layer._nvfp4_K = K
                    layer._nvfp4_E = E_local
                    layer._nvfp4_calibrated = False

                    layer._dsv4_mxfp4_backend = "sm120_nvfp4"
                    log_info_on_rank0(
                        logger,
                        f"SM120 NVFP4: direct MoE ready "
                        f"(E_local={E_local}, needs calibration, "
                        f"layer: {self.prefix})",
                    )
                    return
                except Exception as e:
                    import traceback
                    log_info_on_rank0(
                        logger,
                        f"SM120 NVFP4: FlashInfer B12x init failed, "
                        f"falling back: {e}\n{traceback.format_exc()}",
                    )

            # ── Fallback: v3 GEMV scalar dequant path ──
            log_info_on_rank0(
                logger,
                f"SM120 detected: using MXFP4 MoE GEMV fallback "
                f"(layer: {self.prefix})...",
            )
            # Normalize scales to float32 for direct use in dequant
            if w13_s.dtype == torch.float8_e8m0fnu:
                pass  # already in e8m0 format, will convert at runtime
            elif w13_s.dtype in (torch.uint8, torch.int8):
                layer.w13_weight_scale_inv = Parameter(
                    w13_s.view(torch.uint8)
                    .view(torch.float8_e8m0fnu)
                    .to(torch.float32),
                    requires_grad=False,
                )
                layer.w2_weight_scale_inv = Parameter(
                    w2_s.view(torch.uint8).view(torch.float8_e8m0fnu).to(torch.float32),
                    requires_grad=False,
                )
            layer._dsv4_mxfp4_backend = "sm120_triton"
            return

        if not check_moe_marlin_supports_layer(layer, 32):
            raise RuntimeError(
                "Current DeepSeekV4 MoE layer does not satisfy Marlin constraints."
            )

        # NOTE: the Marlin MoE runner consumes w13 in the checkpoint's
        # native ``[w1; w3]`` order -- see ``silu_and_mul`` in
        # fused_marlin_moe.py which expects ``gate = intermediate[:, :N]``
        # (first half) and ``up = intermediate[:, N:]`` (second half).
        # Unlike the flashinfer trtllm_fp4 kernel (which wants [w3, w1]),
        # we must *not* call ``reorder_w1w3_to_w3w1`` here.

        log_info_on_rank0(
            logger,
            f"Preparing DeepSeekV4 MXFP4 experts for Marlin backend "
            f"(layer: {self.prefix})...",
        )
        prepare_moe_mxfp4_layer_for_marlin(layer)
        layer._dsv4_mxfp4_backend = "marlin"

    NVFP4_DENOM = 6.0 * 448.0  # FP4_MAX * FP8_MAX = 2688.0

    @torch.no_grad()
    def _calibrate_nvfp4(self, layer, hidden_states, topk_output):
        """One-shot activation calibration for NVFP4 scales.

        Measures amax of hidden_states for FC1 input scale.
        Uses a conservative heuristic for FC2 input scale.
        With ws2=1.0, alphas equal raw activation scales.
        """
        dev = hidden_states.device
        E_local = layer._nvfp4_E
        margin = 1.10

        # FC1 input: amax of hidden_states
        a1_amax = hidden_states.float().abs().amax().clamp(min=1e-6)
        a1_raw = margin * a1_amax / self.NVFP4_DENOM

        # a1_scale_quant = 1/a1_raw (reciprocal for fp4_quantize)
        layer._nvfp4_a1_quant = (1.0 / a1_raw).to(torch.float32)

        # FC2 input: SiLU(gate) * up intermediate activation
        # Measured on DeepSeek-V4-Flash: z_amax ≈ 3× x_amax for random input.
        # Real workloads may vary per layer, but 3-10× is the typical range.
        # Use 5× as conservative estimate. Exact calibration would need
        # actual forward pass through the Triton GEMV path.
        a2_raw = (5.0 * a1_raw).clamp(min=1e-12)

        layer._nvfp4_a2_quant = (1.0 / a2_raw).to(torch.float32).to(dev)

        # With ws2 = 1.0 (from nvfp4_quantize(gs=1.0)):
        # GEMM alpha = a_raw * ws2 = a_raw
        # FC1 activation scale = a1_raw (raw, not reciprocal)
        # FC2 activation scale = a2_raw (raw, not reciprocal)
        E_local = layer._nvfp4_E
        layer._nvfp4_fc1_scale = a1_raw.expand(E_local).to(dev).contiguous()
        layer._nvfp4_fc2_scale = a2_raw.expand(E_local).to(dev).contiguous()
        layer._nvfp4_w1_alpha = a1_raw.expand(E_local).to(dev).contiguous()
        layer._nvfp4_w2_alpha = a2_raw.expand(E_local).to(dev).contiguous()

        layer._nvfp4_calibrated = True
        log_info_on_rank0(
            logger,
            f"NVFP4 calibrated: a1_amax={a1_amax.item():.4f}, "
            f"a1_raw={a1_raw.item():.6e}, a2_raw={a2_raw.item():.6e}, "
            f"a1_quant={layer._nvfp4_a1_quant.item():.4f}, "
            f"a2_quant={layer._nvfp4_a2_quant.item():.4f}, "
            f"layer={self.prefix}",
        )

    def apply(
        self,
        layer: Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        topk_output = dispatch_output.topk_output
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise ValueError(f"Unsupported topk output format: {topk_output.format}")

        # SM120 NVFP4 (W4A4): Direct static/dynamic kernel with separate scales
        if getattr(layer, "_dsv4_mxfp4_backend", None) == "sm120_nvfp4":
            hidden_states = dispatch_output.hidden_states

            # Calibrate on first forward pass
            if not getattr(layer, "_nvfp4_calibrated", False):
                self._calibrate_nvfp4(layer, hidden_states, topk_output)

            from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
                launch_sm120_static_moe,
                launch_sm120_dynamic_moe,
                select_sm120_moe_backend,
                _get_weight_views,
                _get_cached_workspace,
            )

            M = hidden_states.shape[0]
            K = layer._nvfp4_K
            I = layer._nvfp4_I
            E_local = layer._nvfp4_E
            top_k = topk_output.topk_ids.shape[1]
            routed_rows = M * top_k

            output = torch.empty(M, K, dtype=hidden_states.dtype,
                                 device=hidden_states.device)

            # Build weight views with GEMM alphas (= a_raw * ws2 = a_raw since ws2=1)
            weight_views = _get_weight_views(
                w1_fp4=layer._nvfp4_w13_fp4,
                w1_blockscale=layer._nvfp4_w13_sf,
                w2_fp4=layer._nvfp4_w2_fp4,
                w2_blockscale=layer._nvfp4_w2_sf,
                w1_alphas=layer._nvfp4_w1_alpha,  # GEMM alpha (a1_raw * ws2)
                w2_alphas=layer._nvfp4_w2_alpha,  # GEMM alpha (a2_raw * ws2)
                n=I, k=K,
                activation_precision="fp4",
            )

            backend = select_sm120_moe_backend(
                num_tokens=M, num_topk=top_k, activation_precision="fp4",
            )
            if backend == "dynamic" and E_local != 256:
                backend = "static"

            # Cap static workspace routed_rows to a fixed size so the workspace
            # never grows and the RT kernel cache key stays stable.
            # Controlled by SGLANG_NVFP4_STATIC_WS_CAP (default 256).
            _ws_cap = int(os.environ.get("SGLANG_NVFP4_STATIC_WS_CAP", "256"))
            ws_routed_rows = max(routed_rows, _ws_cap) if backend == "static" else routed_rows

            workspace = _get_cached_workspace(
                backend=backend, state_E=E_local, weight_E=256,
                routed_rows=ws_routed_rows, k=K, n=I, num_topk=top_k,
                device=hidden_states.device, activation_precision="fp4",
                quant_mode="nvfp4", activation="silu",
            )

            launch_fn = (launch_sm120_dynamic_moe if backend == "dynamic"
                         else launch_sm120_static_moe)
            launch_fn(
                workspace=workspace,
                weights=weight_views,
                a=hidden_states,
                topk_ids=topk_output.topk_ids,
                topk_weights=topk_output.topk_weights,
                input_gs=layer._nvfp4_fc1_scale,         # FC1 activation scale ONLY
                down_input_scale=layer._nvfp4_fc2_scale,  # FC2 activation scale ONLY
                scatter_output=output,
                num_experts=256,
                num_tokens=M, k=K, n=I, top_k=top_k,
                input_scales_are_reciprocal=False,
                fast_math=True,
                activation="silu",
                activation_precision="fp4",
            )
            return StandardCombineInput(hidden_states=output)

        # SM120: use Triton fused dequant+GEMM (Marlin kernel produces NaN on SM120)
        if getattr(layer, "_dsv4_mxfp4_backend", None) == "sm120_triton":
            from sglang.srt.layers.moe.fused_moe_triton.mxfp4_moe_sm120_triton import (
                mxfp4_moe_forward_triton,
            )

            hidden_states = dispatch_output.hidden_states
            w13 = layer.w13_weight.data
            w2 = layer.w2_weight.data
            w13_scale = layer.w13_weight_scale_inv.data
            w2_scale = layer.w2_weight_scale_inv.data
            intermediate_size = w13.shape[1] // 2
            hidden_size = w13.shape[2] * 2

            output = mxfp4_moe_forward_triton(
                hidden_states=hidden_states,
                w13_packed=w13,
                w2_packed=w2,
                w13_scale=w13_scale,
                w2_scale=w2_scale,
                topk_ids=topk_output.topk_ids,
                topk_weights=topk_output.topk_weights,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                routed_scaling_factor=(
                    self.runner.config.routed_scaling_factor
                    if hasattr(self.runner, "config")
                    else None
                ),
                clamp_limit=(
                    self.runner.config.swiglu_limit
                    if hasattr(self.runner, "config")
                    else None
                ),
            )
            return StandardCombineInput(hidden_states=output)

        quant_info = MarlinMoeQuantInfo(
            w13_qweight=layer.w13_weight,
            w2_qweight=layer.w2_weight,
            w13_scales=layer.w13_weight_scale,
            w2_scales=layer.w2_weight_scale,
            w13_g_idx_sort_indices=None,
            w2_g_idx_sort_indices=None,
            weight_bits=4,
            is_k_full=True,
        )
        runner_output = self.runner.run(dispatch_output, quant_info=quant_info)

        return StandardCombineInput(hidden_states=runner_output.hidden_states)
