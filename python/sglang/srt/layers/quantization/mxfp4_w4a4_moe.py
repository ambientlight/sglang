"""MXFP4 x MXFP4 fused MoE method for SM120.

DeepSeek-V4-Flash ships routed experts as MXFP4 (E2M1 packed int8 weights +
E8M0 32-element block scales). This method loads them as-is and runs the experts
with FlashInfer's fused SwiGLU CuTe-DSL MoE kernels
(``launch_sm120_moe(quant_mode="mxfp4")``): MXFP4 weights x MXFP4 activations,
E8M0 self-scaling.

Weight loading is shared with the W4A8 method (``mxfp4_w4a8_moe.py``); only the
weight-scale swizzle target differs: W4A4 swizzles into the 128x4 layout then
converts to the MMA layout the fused kernel's ``_get_weight_views`` expects.

Selected by ``fp8.py`` on SM120 + MXFP4 experts + FlashInfer exposing
``launch_sm120_moe``.
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
    """MXFP4 weights x MXFP4 activations MoE (SM120 fused CuTe-DSL).

    Usable two ways:
      * directly as a FusedMoE quant method (DSV4 path), or
      * wrapped by `Mxfp4W4A4MoEScheme` as a compressed-tensors MoE scheme
        (MiniMax-M3 path, where the checkpoint is compressed-tensors format).
    """

    def __init__(self, fp8_method, prefix: str):
        self._fp8 = fp8_method
        self.prefix = prefix
        # Checkpoint MoE-scale encoding differs by model, and it MUST match how the
        # HF loader copies the tensor into the scale Parameter (below):
        #   * MiniMax-M3 (compressed-tensors, wrapped by Mxfp4W4A4MoEScheme with
        #     fp8_method=None): scales are RAW UE8M0 bytes stored as uint8 — the
        #     param must be uint8 so copy_ is byte-verbatim.
        #   * DeepSeek-V4-Flash (this method used directly, real fp8_method):
        #     scales are float8_e8m0fnu (F8_E8M0) — a float dtype. The param must
        #     be float32 and process_weights re-encodes via _to_e8m0_u8; a uint8
        #     param would NUMERIC-cast 2^-6=0.0156 -> 0, zeroing every sub-unit
        #     scale and producing token-salad output.
        # Discriminator: the M3 scheme constructs with fp8_method=None. Overridable
        # via SGLANG_MXFP4_SCALE_RAW_U8={0,1} for checkpoints that break the rule.
        _env = os.environ.get("SGLANG_MXFP4_SCALE_RAW_U8")
        if _env is not None:
            self.scale_raw_u8 = _env == "1"
        else:
            self.scale_raw_u8 = fp8_method is None

    def create_moe_runner(self, layer, moe_runner_config):
        # The fused launch runs in apply(); the runner only carries MoE config
        # (routed_scaling_factor / swiglu_limit).
        from sglang.srt.layers.moe.moe_runner import MoeRunner

        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
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

        # Scale param dtype MUST match the checkpoint encoding (see __init__):
        #   * M3 raw-UE8M0 (scale_raw_u8=True): uint8, HF loader copies bytes
        #     verbatim (byte 121 == 2^(121-127) == 2^-6), consumed after swizzle.
        #   * DSV4 F8_E8M0 (scale_raw_u8=False): float32 placeholder — E8M0 are
        #     exact powers of two so the HF loader casts the float8 scales
        #     losslessly; process_weights_after_loading re-encodes via _to_e8m0_u8.
        # Getting this wrong corrupts every scale (uint8 param + float ckpt casts
        # 2^-6 -> 0 == token-salad; float param + raw-u8 ckpt reads bytes as values).
        _scale_dtype = torch.uint8 if self.scale_raw_u8 else torch.float32
        _scale_init = torch.zeros if self.scale_raw_u8 else torch.ones
        w13_weight_scale = Parameter(
            _scale_init(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // _FP4_BLOCK_K,
                dtype=_scale_dtype,
            ),
            requires_grad=False,
        )
        w2_weight_scale = Parameter(
            _scale_init(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // _FP4_BLOCK_K,
                dtype=_scale_dtype,
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

    @staticmethod
    def _to_e8m0_u8(scale_f32: torch.Tensor) -> torch.Tensor:
        """float32 scale magnitude -> E8M0 byte (uint8)."""
        return scale_f32.to(torch.float8_e8m0fnu).view(torch.uint8)

    def process_weights_after_loading(self, layer: Module) -> None:
        if not is_sm120_supported():
            raise RuntimeError(
                "Mxfp4W4A4MoEMethod requires SM120 (RTX PRO 6000 Blackwell)."
            )
        if getattr(layer, "_w4a4_weights_built", False):
            return

        # 128x4 weight-scale swizzle (shared with W4A8), then convert to the MMA
        # layout the fused kernel's _get_weight_views expects.
        from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

        from sglang.srt.layers.quantization.mxfp4_sm120_common import (
            swizzle_weight_scale_mxf4,
        )

        w13 = layer.w13_weight.data  # (E, 2I, H/2) int8, checkpoint [w1|w3]
        w2 = layer.w2_weight.data  # (E, H, I/2)  int8
        w13_s = layer.w13_weight_scale_inv.data
        w2_s = layer.w2_weight_scale_inv.data

        E = w13.shape[0]
        twoI = w13.shape[1]
        Hhalf = w13.shape[2]
        H = Hhalf * 2
        Ihalf = w2.shape[2]
        intermediate = Ihalf * 2

        # Fused kernel expects gate/up as [w3, w1] (up, gate), but the checkpoint
        # loads W13 as [w1, w3]; swap the I-row halves of weights and 3D scales.
        # SGLANG_M3_NO_GATEUP_SWAP=1 disables the swap (A/B probe for the asymmetric
        # clamp's gate/up orientation — the harness can't discriminate it).
        Irows = twoI // 2
        _no_swap = os.environ.get("SGLANG_M3_NO_GATEUP_SWAP", "0") == "1"
        if not _no_swap:
            w13 = torch.cat(
                [w13[:, Irows:, :], w13[:, :Irows, :]], dim=1
            ).contiguous()
            w13_s = torch.cat(
                [w13_s[:, Irows:, :], w13_s[:, :Irows, :]], dim=1
            ).contiguous()

        # Convert the loaded scales to raw UE8M0 bytes for the swizzle. The [w3,w1]
        # reorder above is a byte-/element-wise gather, so it preserves either
        # encoding. Which conversion is correct depends on the checkpoint dtype
        # (see __init__ scale_raw_u8):
        #   * M3 (scale_raw_u8=True): the param already holds raw UE8M0 bytes —
        #     take them verbatim (a float re-encode would corrupt them).
        #   * DSV4 (scale_raw_u8=False): the param holds float8_e8m0fnu magnitudes
        #     loaded into float32 — re-encode to E8M0 bytes via _to_e8m0_u8 (a
        #     verbatim uint8 view would read the float mantissa as a byte value).
        if self.scale_raw_u8:
            w13_s_u8 = w13_s.to(torch.uint8).contiguous()
            w2_s_u8 = w2_s.to(torch.uint8).contiguous()
        else:
            w13_s_u8 = self._to_e8m0_u8(w13_s.to(torch.float32))
            w2_s_u8 = self._to_e8m0_u8(w2_s.to(torch.float32))

        # 128x4 block-scale swizzle (per expert), then convert to the MMA layout
        # the fused kernel reads (experts flattened into the leading row dim).
        w13_s_sw = swizzle_weight_scale_mxf4(w13_s_u8, E, twoI, H)
        w2_s_sw = swizzle_weight_scale_mxf4(w2_s_u8, E, H, intermediate)
        w13_sf_mma = convert_sf_to_mma_layout(
            w13_s_sw.reshape(E * w13_s_sw.shape[1], w13_s_sw.shape[2]),
            m=twoI,
            k=H,
            num_groups=E,
            sf_vec_size=_FP4_BLOCK_K,
        )
        w2_sf_mma = convert_sf_to_mma_layout(
            w2_s_sw.reshape(E * w2_s_sw.shape[1], w2_s_sw.shape[2]),
            m=H,
            k=intermediate,
            num_groups=E,
            sf_vec_size=_FP4_BLOCK_K,
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
        layer._w4a4_I = intermediate
        layer._w4a4_E = E
        layer._w4a4_weights_built = True

        # Build the FlashInfer weight views now (load time), not lazily under
        # CUDA-graph capture, so the ~48 MB/layer scale storage stays out of the
        # graph-private pool.
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
            _get_weight_views,
        )

        ones_e = torch.ones(E, device=layer.w13_weight.device, dtype=torch.float32)
        layer._w4a4_alpha = ones_e  # GEMM alpha = 1 (MXFP4 self-scales)
        layer._w4a4_weight_views = _get_weight_views(
            w1_fp4=layer.w13_weight,
            w1_blockscale=layer.w13_weight_scale_inv,
            w2_fp4=layer.w2_weight,
            w2_blockscale=layer.w2_weight_scale_inv,
            w1_alphas=ones_e,
            w2_alphas=ones_e,
            n=intermediate,
            k=H,
            activation_precision="fp4",
            quant_mode="mxfp4",
        )

        # Free the per-layer transients; the final Parameters hold contiguous
        # copies. Without this the allocator's reserve starves CUDA-graph capture.
        del w13, w2, w13_s, w2_s
        del w13_s_u8, w2_s_u8, w13_s_sw, w2_s_sw, w13_sf_mma, w2_sf_mma
        torch.cuda.empty_cache()

        log_info_on_rank0(
            logger,
            f"SM120 MXFP4 W4A4 experts ready "
            f"(E={E}, H={H}, I={intermediate}; layer: {self.prefix})",
        )

    def apply(
        self,
        layer: Module,
        dispatch_output: "DispatchOutput",
    ) -> "CombineInput":
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
            _get_cached_workspace,
            launch_sm120_moe,
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
        intermediate = layer._w4a4_I
        E_local = layer._w4a4_E
        # Global expert count bounds topk-id range; local count sizes the expert
        # buffers. Under EP global > local and the kernel remaps global->local.
        E_global = int(getattr(layer, "num_experts", E_local))
        top_k = topk_output.topk_ids.shape[1]
        dev = hidden_states.device

        cfg = getattr(self.runner, "config", None)

        # MXFP4 self-scales: no activation global scale (GEMM alpha = 1).
        ones = layer._w4a4_alpha

        # M3 clamped SwiGLU-OAI: pull (alpha, limit, beta) from the runner config.
        # Defaults match M3 (alpha=1.702, limit=7.0, beta=1.0). When the config
        # carries no clamp (limit None) fall back to plain "silu".
        _limit = getattr(cfg, "gemm1_clamp_limit", None) if cfg else None
        _alpha = getattr(cfg, "gemm1_alpha", None) if cfg else None
        if _limit is not None:
            act = "swigluoai"
            sw_alpha = float(_alpha) if _alpha is not None else 1.702
            sw_limit = float(_limit)
            sw_beta = float(getattr(cfg, "gemm1_beta", 1.0) or 1.0)
        else:
            act = "silu"
            sw_alpha, sw_limit, sw_beta = 1.702, 7.0, 1.0

        # Decode reuses one small capped static workspace (allocated outside
        # CUDA-graph capture so the decode graph never grows it). Larger prefill
        # batches are CHUNKED over M into <=ws_cap-row sub-launches that reuse the
        # same capped workspace (see the chunking rationale below).
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
                n=intermediate,
                num_topk=top_k,
                device=dev,
                quant_mode="mxfp4",
                activation=act,
            )
            layer._w4a4_ws_key = top_k

        # M3 (H=6144) makes the dynamic kernel unusable on SM120: its tiling
        # overflows the 101376-byte smem budget (115712 > 101376 on sm_120a — a
        # pre-existing limit, independent of the SwiGLU clamp; launching it throws
        # cudaErrorInvalidValue). The static kernel is smem-safe (it tiles 128x128
        # over M) and validated at M3 dims, but its workspace pins per-expert
        # capacity at max_rows == routed_rows, i.e. packed_input is
        # [E_local, routed_rows, k//2] — at a full 4096-token prefill chunk that
        # is ~12 GB/layer, an instant OOM. So for any batch above the cap we keep
        # the static kernel but CHUNK over M: each sub-launch routes <=ws_cap rows
        # through the one small cached workspace, writing its own output slice.
        # Decode/small batch (routed_rows<=ws_cap) is the single-launch fast path.
        # SGLANG_M3_FORCE_STATIC=0 opts the large batch back into the (currently
        # broken) dynamic kernel for once its tile is reduced upstream.
        force_static = os.environ.get("SGLANG_M3_FORCE_STATIC", "1") == "1"

        # Zero-init (not empty): if launch_sm120_moe's scatter does not write
        # every row (e.g. a token whose routed tiles do not cover it), an
        # uninitialized row would leak garbage into the residual stream and make
        # greedy decoding NON-deterministic. Zeroing makes any such gap a
        # well-defined no-op instead of random memory.
        output = torch.zeros(M, H, dtype=hidden_states.dtype, device=dev)

        def _launch(a_slice, ids_slice, w_slice, out_slice, ws):
            launch_sm120_moe(
                a=a_slice,
                topk_ids=ids_slice,
                topk_weights=w_slice,
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
                scatter_output=out_slice,
                input_scales_are_reciprocal=False,
                fast_math=True,
                activation=act,
                quant_mode="mxfp4",
                swiglu_alpha=sw_alpha,
                swiglu_limit=sw_limit,
                swiglu_beta=sw_beta,
                _weight_views=layer._w4a4_weight_views,
                _workspace=ws,
            )

        # Sub-launch over M so each static launch stays within the capped
        # workspace. NOTE: a power-of-2 BUCKETED-M variant (zero-padding each
        # chunk to a fixed `m` to bound CuteDSL per-`m` JIT compiles) was tried
        # here and caused a cudaErrorIllegalAddress inside the static MoE kernel
        # under live agentic load (faulting launch confirmed at this call site via
        # CUDA_LAUNCH_BLOCKING=1). The padding concentrated padded rows onto
        # expert 0 (ids=0) and passed a fresh out buffer, exercising a workspace
        # capacity/scatter path the exact-M loop never hit. Reverted to the
        # ORIGINAL exact-M chunk loop (known-good, never crashed). The JIT-thrash
        # mitigation is shelved pending a padding scheme that respects the static
        # workspace contract (likely: size a per-bucket workspace, not reuse the
        # 640-row cap). force_static=0 keeps the (currently broken) dynamic path.
        if routed_rows <= ws_cap or not force_static:
            # Single launch: decode/small batch reuses the capped workspace; with
            # force_static off, _workspace=None lets the dispatcher pick (dynamic).
            ws = layer._w4a4_static_ws if routed_rows <= ws_cap else None
            _launch(
                hidden_states,
                topk_output.topk_ids,
                topk_output.topk_weights,
                output,
                ws,
            )
        else:
            # Large prefill: chunk M so each static sub-launch routes <=ws_cap rows
            # through the one capped workspace, writing its own output slice.
            chunk_M = max(1, ws_cap // top_k)
            for start in range(0, M, chunk_M):
                end = min(start + chunk_M, M)
                _launch(
                    hidden_states[start:end],
                    topk_output.topk_ids[start:end],
                    topk_output.topk_weights[start:end],
                    output[start:end],
                    layer._w4a4_static_ws,
                )

        # NOTE: M3 applies routed_scaling_factor (2.0) inside TopK
        # (apply_routed_scaling_factor_on_output=True -> topk_weights *= rsf), so it
        # is ALREADY baked into topk_weights the kernel combined. Do NOT re-multiply
        # here (the DSV4 template did; for M3 that would double-apply the 2x).
        return StandardCombineInput(hidden_states=output)


def _flashinfer_has_native_mxfp4() -> bool:
    """True iff the active flashinfer exposes the SM120 native-MXFP4 MoE path."""
    try:
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (  # noqa: F401
            _normalize_quant_mode,
            launch_sm120_moe,
        )

        _normalize_quant_mode("mxfp4")  # raises if this flashinfer lacks mxfp4
        return True
    except Exception:
        return False


class Mxfp4W4A4MoEScheme:
    """compressed-tensors MoE scheme wrapping the native SM120 MXFP4 method.

    MiniMax-M3's olka-fi checkpoint is compressed-tensors `mixed-precision`
    (routed experts tagged `mxfp4-pack-quantized`). sglang routes it through
    `CompressedTensorsFusedMoEMethod` -> `layer.scheme`, so this thin scheme
    adapter delegates the four scheme hooks to `Mxfp4W4A4MoEMethod` (which does
    the E8M0 swizzle + `launch_sm120_moe(quant_mode="mxfp4", activation=
    "swigluoai")`). Selected by `get_moe_scheme` on SM120 + native flashinfer.
    """

    def __init__(self, prefix: str = ""):
        self._method = Mxfp4W4A4MoEMethod(fp8_method=None, prefix=prefix)

    @classmethod
    def get_min_capability(cls) -> int:
        return 120  # RTX PRO 6000 Blackwell (SM120)

    def create_weights(self, *args, **kwargs):
        return self._method.create_weights(*args, **kwargs)

    def create_moe_runner(self, layer, moe_runner_config):
        return self._method.create_moe_runner(layer, moe_runner_config)

    def process_weights_after_loading(self, layer):
        return self._method.process_weights_after_loading(layer)

    def apply_weights(self, layer, dispatch_output):
        return self._method.apply(layer, dispatch_output)

