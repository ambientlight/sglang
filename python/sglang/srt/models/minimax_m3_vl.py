# SPDX-License-Identifier: Apache-2.0
# MiniMax M3 VL — vision tower + M3 (mixed sparse/dense MoE) text backbone.

import logging
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.distributed import (
    get_moe_expert_parallel_world_size,
    get_pp_group,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.utils.common import get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.minimax_m3 import (
    MiniMaxM3Model,
    MiniMaxM3SparseForCausalLM,
    build_minimax_fused_qkv_index,
    get_spec_layer_idx_from_weight_name,
)
from sglang.srt.models.minimax_vl_common import (
    CLIPVisionConfig,
    MiniMaxVLVisionModel,
    get_image_feature,
    get_video_feature,
    load_vision_weight,
    merge_vit_qkv_weights,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, get_device_sm, is_cuda, log_info_on_rank0
from sglang.srt.utils.hf_transformers_utils import get_rope_config

logger = logging.getLogger(__name__)


_is_cuda = is_cuda()
_device_sm = get_device_sm()


class MiniMaxM3SparseForConditionalGeneration(nn.Module):
    """MiniMax M3 VL: shared vision tower + M3 LLM with mixed sparse/dense attention.

    Always loaded as the mixed sparse/dense backbone: which layers are sparse
    vs dense is decided by ``config.text_config.sparse_attention_config``. A
    checkpoint that omits ``sparse_attention_config`` will produce a pure-dense
    model.
    """

    # Fused-module -> unfused checkpoint components, so a quant config that targets
    # the unfused projections (e.g. compressed-tensors mxfp8) can resolve the fused
    # GEMMs the model builds. The lightning-indexer fuses q+k only (no value when
    # disable_index_value). Without this, scheme resolution for index_qkv_proj
    # raises "Unable to find matching target".
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "index_qkv_proj": ["index_q_proj", "index_k_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder

        self.num_fused_shared_experts = 0
        self._determine_num_fused_shared_experts()

        vision_config_raw = config.vision_config
        assert vision_config_raw is not None, "vision_config is required"
        if hasattr(vision_config_raw, "to_dict"):
            vision_config_dict = vision_config_raw.to_dict()
        else:
            vision_config_dict = vision_config_raw
        vision_config = CLIPVisionConfig.from_dict(vision_config_dict)
        self.vision_config = vision_config

        text_hidden_size = getattr(config.text_config, "hidden_size", None)
        assert text_hidden_size is not None, "text_hidden_size is required"
        projector_hidden_size = getattr(config, "projector_hidden_size", None)

        # Vision model skips quantization: CLIP dimensions (head_dim=80) are not
        # compatible with MXFP8 kernel alignment requirements (128).
        self.vision_tower = MiniMaxVLVisionModel(
            config=vision_config,
            text_hidden_size=text_hidden_size,
            projector_hidden_size=projector_hidden_size,
            quant_config=None,
            prefix=add_prefix("vision_tower", prefix),
        )

        # Language model: M3 (with optional sparse attention).
        # The unified MiniMaxM3Model reads ``text_config.sparse_attention_config``
        # to decide per-layer whether to construct dense or sparse attention,
        # so no branching is needed here.
        text_config = config.text_config
        self.model = MiniMaxM3Model(
            config=text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model.model", prefix),
        )

        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                text_config.vocab_size,
                text_config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("language_model.lm_head", prefix),
                use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
            )
        else:
            self.lm_head = PPMissingLayer()

        _, text_rope_scaling = get_rope_config(text_config)
        self.is_mrope_enabled = (
            text_rope_scaling is not None and "mrope_section" in text_rope_scaling
        )

        self.logits_processor = LogitsProcessor(text_config)

    def _determine_num_fused_shared_experts(self) -> None:
        text_config = self.config.text_config
        if get_global_server_args().disable_shared_experts_fusion:
            return

        disable_reason = None
        if not getattr(text_config, "n_shared_experts", None):
            disable_reason = "No shared experts are defined in the config."
        elif not _is_cuda:
            disable_reason = "Shared experts fusion currently requires CUDA devices."
        elif _is_cuda and (_device_sm is not None) and (_device_sm < 80):
            disable_reason = "Shared experts fusion requires SM80 or newer GPUs."
        elif get_moe_expert_parallel_world_size() > 1:
            disable_reason = (
                "Shared experts fusion is not supported together with expert "
                "parallelism yet."
            )
        elif get_moe_a2a_backend().is_deepep():
            disable_reason = (
                "Shared experts fusion is not supported when Deepep MoE backend "
                "is enabled."
            )

        if disable_reason is not None:
            get_global_server_args().disable_shared_experts_fusion = True
            log_info_on_rank0(
                logger,
                f"{disable_reason} Shared experts fusion optimization is disabled.",
            )
            return

        self.num_fused_shared_experts = text_config.n_shared_experts
        assert (
            self.num_fused_shared_experts == 1
        ), "Only 1 fused shared expert is supported"
        log_info_on_rank0(logger, "Shared experts fusion optimization enabled.")

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        # EP looks up this hook on the top-level arch class to build expert-location
        # metadata (else ExpertLocationDispatchInfo.init_new asserts). The VL config
        # nests the LM config under text_config, so delegate there; fall back to
        # config itself when text_config is absent (LM config passed directly).
        text_config = getattr(config, "text_config", None) or config
        return MiniMaxM3SparseForCausalLM.get_model_config_for_expert_location(
            text_config
        )

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens(
            input_ids, mm_inputs
        )

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        return get_image_feature(self.vision_tower, items, self.use_data_parallel)

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        return get_video_feature(self.vision_tower, items, self.use_data_parallel)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if self.pp_group.is_last_rank and not get_embedding:
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
            )
        return hidden_states

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load checkpoint weights for the vision tower and the M3 LLM.

        M3 LLM differs from M2 in:
        - MoE path is ``mlp.experts.*`` (not ``block_sparse_moe.experts.*``);
          checkpoints saved with the M2 naming are remapped on the fly.
        - Optional shared experts fusion: ``mlp.shared_experts`` is mapped onto
          a synthetic ``mlp.experts.{num_local_experts}`` slot.
        - PP layer skipping via ``get_layer_id``.
        - MTP / spec-decode layers are skipped.
        """
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        # ``.qkv_proj`` (with the leading dot) prevents matching e.g.
        # ``index_q_proj`` in the sparse-attention branch.
        llm_stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        # Mirror the LLM's fused index projection (see MiniMaxM3.load_weights):
        # restack the separate index_q/k/v projections into one index_qkv_proj.
        # The leading "." makes these match only the index_*_proj weights.
        if (
            getattr(self.config.text_config, "sparse_attention_config", None)
            is not None
        ):
            llm_stacked_params_mapping += [
                (".index_qkv_proj", ".index_q_proj", "q"),
                (".index_qkv_proj", ".index_k_proj", "k"),
                (".index_qkv_proj", ".index_v_proj", "v"),
            ]

        num_experts = getattr(self.config.text_config, "num_local_experts", 0)
        expert_params_mapping = (
            FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=num_experts + self.num_fused_shared_experts,
            )
            if num_experts > 0
            else []
        )

        params_dict = dict(self.named_parameters())
        vit_qkv_weights: dict = {}
        vit_qkv_biases: dict = {}

        # Load-completeness accounting for the routed experts. ``expert_loads``
        # counts source tensors that actually reached a registered destination
        # param; ``dropped_experts`` collects ones that matched an expert source
        # name but resolved to NO destination (the silent-skip signature). See
        # the post-load guard below.
        load_stats = {"expert_loads": 0, "dropped_experts": []}

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # HF checkpoints may include a root model. prefix; normalize it
            # before dispatch so text self_attn q/k/v does not enter VIT QKV merge.
            if name.startswith("model."):
                name = name[len("model.") :]

            if name.startswith("language_model."):
                self._load_llm_weight(
                    name[len("language_model.") :],
                    loaded_weight,
                    params_dict,
                    llm_stacked_params_mapping,
                    expert_params_mapping,
                    load_stats,
                )
                continue

            load_vision_weight(
                name, loaded_weight, params_dict, vit_qkv_weights, vit_qkv_biases
            )

        merge_vit_qkv_weights(vit_qkv_weights, vit_qkv_biases, params_dict)

        # Fuse main qkv_proj + sparse index_qkv_proj into one GEMM per sparse
        # attention layer (see MiniMaxM3.load_weights for the rationale).
        build_minimax_fused_qkv_index(self)

        # ---- Load-completeness guard (Quest-5 root-cause bug class) ----
        # The multi-day garble hunt ended at a SILENT expert-weight skip: the
        # checkpoint's ``...experts.N.wX.weight_packed`` names produced
        # destination params (``w13_weight_packed``) that matched nothing the
        # bridge registered, so ``if new_name not in params_dict: continue``
        # dropped EVERY routed-expert weight and the MoE ran on uninitialized
        # ``torch.empty()`` memory -- fluent but prompt-independent, non-
        # deterministic output. Every isolated kernel test was green because none
        # exercised this production load path. Convert that silent failure into a
        # loud one: a checkpoint that exposes expert tensors but lands none (or
        # drops any) of them must NOT boot a server that serves garbage.
        if num_experts > 0:
            if load_stats["dropped_experts"]:
                dropped = load_stats["dropped_experts"]
                raise RuntimeError(
                    f"[minimax_m3 load_weights] {len(dropped)} routed-expert weight(s) "
                    f"matched an expert source name but resolved to NO registered "
                    f"destination param -- they would be SILENTLY skipped, leaving the "
                    f"MoE on uninitialized memory (fluent-but-random, prompt-independent "
                    f"output). This is the weight_packed/weight_scale naming-contract bug. "
                    f"Examples: {dropped[:6]}"
                )
            if load_stats["expert_loads"] == 0:
                raise RuntimeError(
                    f"[minimax_m3 load_weights] num_experts={num_experts} but ZERO "
                    f"routed-expert weights were loaded -- the checkpoint exposed no "
                    f"'experts.N.wX' tensors to this loader (prefix/naming mismatch). "
                    f"The MoE would run on uninitialized memory; refusing to serve."
                )

    def _load_llm_weight(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict,
        llm_stacked_params_mapping: list,
        expert_params_mapping: list,
        load_stats: dict = None,
    ) -> None:
        # Older checkpoints used the M2-style ``block_sparse_moe`` naming.
        if "block_sparse_moe" in name:
            name = name.replace("block_sparse_moe", "mlp")

        layer_id = get_layer_id(name)
        if layer_id is not None and (
            layer_id < self.model.start_layer or layer_id >= self.model.end_layer
        ):
            return

        if self.num_fused_shared_experts > 0 and "mlp.shared_experts" in name:
            name = name.replace(
                "mlp.shared_experts",
                f"mlp.experts.{self.config.text_config.num_local_experts}",
            )
            name = name.replace("gate_proj", "w1")
            name = name.replace("down_proj", "w2")
            name = name.replace("up_proj", "w3")

        if (
            get_spec_layer_idx_from_weight_name(self.config.text_config, name)
            is not None
        ):
            return

        for param_name, weight_name, shard_id in llm_stacked_params_mapping:
            if weight_name not in name:
                continue
            if "mlp.experts." in name:
                # Experts are handled by expert_params_mapping below.
                continue
            new_name = name.replace(weight_name, param_name)
            if new_name.endswith(".bias") and new_name not in params_dict:
                continue
            if new_name not in params_dict:
                continue
            param = params_dict[new_name]
            param.weight_loader(param, loaded_weight, shard_id)
            return

        # MXFP4 routed experts: the olka-fi checkpoint stores each expert
        # projection as ``...experts.N.wX.weight_packed`` (E2M1 nibbles) and
        # ``...experts.N.wX.weight_scale`` (E8M0 bytes). FusedMoE's
        # make_expert_params_mapping targets the fused params ``experts.w13_*`` /
        # ``experts.w2_*`` by appending the suffix that follows ``wX.`` — i.e.
        # ``weight`` -> ``w13_weight`` and ``weight_scale_inv`` -> the scale param.
        # Without this rename the produced names (``w13_weight_packed`` /
        # ``w13_weight_scale``) match NO registered param, so every expert weight
        # is SILENTLY skipped and the MoE runs on uninitialized memory (fluent but
        # prompt-independent, non-deterministic output). Map the checkpoint
        # suffixes onto the bridge's param names. Scoped to ``mlp.experts.`` so the
        # MXFP8 linears (which legitimately use ``weight``/``weight_scale``) are
        # untouched.
        if "mlp.experts." in name:
            if name.endswith(".weight_packed"):
                name = name[: -len(".weight_packed")] + ".weight"
            elif name.endswith(".weight_scale"):
                name = name[: -len(".weight_scale")] + ".weight_scale_inv"

        is_expert_weight = False
        for mapping in expert_params_mapping:
            param_name, weight_name, expert_id, shard_id = mapping
            if weight_name not in name:
                continue
            is_expert_weight = True
            new_name = name.replace(weight_name, param_name)
            if new_name not in params_dict:
                continue
            param = params_dict[new_name]
            param.weight_loader(
                param,
                loaded_weight,
                new_name,
                shard_id=shard_id,
                expert_id=expert_id,
            )
            if load_stats is not None:
                load_stats["expert_loads"] += 1
            return
        if is_expert_weight:
            # Recognized as a routed-expert tensor by SOURCE name, but NO
            # mapping produced a destination param that exists in params_dict.
            # This is exactly the silent-skip that left the MoE uninitialized;
            # record it so the post-load guard can refuse to serve.
            if load_stats is not None:
                load_stats["dropped_experts"].append(name)
            return

        if name.endswith(".bias") and name not in params_dict:
            return
        remapped = maybe_remap_kv_scale_name(name, params_dict)
        if remapped is None:
            return
        if remapped not in params_dict:
            logger.warning(f"Parameter {remapped} not found in params_dict")
            return
        param = params_dict[remapped]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        try:
            weight_loader(param, loaded_weight)
        except Exception as e:
            logger.warning(f"Error loading weight {remapped}: {e}")


EntryClass = [MiniMaxM3SparseForConditionalGeneration]
