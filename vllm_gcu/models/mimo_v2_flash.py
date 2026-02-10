# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn

from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention
from vllm.config import (
    CacheConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
    ColumnParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import MixtureOfExperts, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    tensor_model_parallel_all_gather,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from typing import Optional
from vllm.model_executor.layers.linear import divide
from torch.nn.parameter import Parameter
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    BlockQuantScaleParameter,
    PackedColumnParameter,
    PackedvLLMParameter,
    PerTensorScaleParameter,
    RowvLLMParameter,
)

from unittest.mock import patch
from vllm.config.model import _STR_DTYPE_TO_TORCH_DTYPE

logger = init_logger(__name__)


def str_dtype_to_torch_dtype(type: str):
    return _STR_DTYPE_TO_TORCH_DTYPE.get(type)


def validate_fp8_block_shape(
    layer: torch.nn.Module,
    input_size: int,
    output_size: int,
    input_size_per_partition: int,
    output_partition_sizes: list[int],
    block_size: list[int],
) -> None:
    """Validate block quantization shapes for tensor parallelism."""
    from vllm.distributed import get_tensor_model_parallel_world_size

    if getattr(layer, "allow_fp8_block_shape_mismatch", False):
        logger.debug(
            "Skipping FP8 block shape validation for layer %s due to detected" " mismatch allowance.",
            getattr(layer, "prefix", "<unknown>"),
        )
        return
    tp_size = getattr(layer, "tp_size", get_tensor_model_parallel_world_size())
    block_n, block_k = block_size[0], block_size[1]

    # Required by row parallel
    if (
        tp_size > 1
        and input_size // input_size_per_partition == tp_size
        and input_size_per_partition % block_k != 0
    ):
        raise ValueError(
            f"Weight input_size_per_partition = {input_size_per_partition} "
            f"is not divisible by weight quantization block_k = {block_k}."
        )
    # Required by column parallel or enabling merged weights
    is_tp_split = tp_size > 1 and output_size // sum(output_partition_sizes) == tp_size
    is_merged_gemm = len(output_partition_sizes) > 1
    if is_tp_split or is_merged_gemm:
        sizes_to_check = output_partition_sizes
        if not is_tp_split and is_merged_gemm:
            # In case of merged matrices, we allow the last
            # matrix to not be a multiple of block size
            sizes_to_check = output_partition_sizes[:-1]
        for output_partition_size in sizes_to_check:
            if output_partition_size % block_n != 0:
                raise ValueError(
                    f"Weight output_partition_size = "
                    f"{output_partition_size} is not divisible by "
                    f"weight quantization block_n = {block_n}."
                )


class ColumnParallelLinearGCU(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        self.quant_config = quant_config
        # Divide the weight matrix along the last dimension.
        self.tp_rank = get_tensor_model_parallel_rank() if not disable_tp else 0
        self.tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        # If QKV or MergedColumn, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(output_size, self.tp_size) for output_size in self.output_sizes
            ]
        self.allow_fp8_block_shape_mismatch = False
        self._maybe_allow_fp8_block_shape_mismatch()
        super().__init__(
            input_size,
            output_size,
            bias,
            False,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

    def _maybe_allow_fp8_block_shape_mismatch(self) -> None:
        quant_config = getattr(self, "quant_config", None)
        weight_block = getattr(quant_config, "weight_block_size", None)
        if weight_block is None or len(weight_block) < 1 or len(self.output_partition_sizes) <= 1:
            return

        try:
            block_n = int(weight_block[0])
        except (ValueError, TypeError):
            return

        if block_n <= 0:
            return

        if any(size % block_n != 0 for size in self.output_partition_sizes):
            self.allow_fp8_block_shape_mismatch = True
            logger.debug(
                "Allowing FP8 block shape mismatch for %s (block_n=%d, partitions=%s)",
                getattr(self, "prefix", "<unknown>"),
                block_n,
                self.output_partition_sizes,
            )


class QKVParallelLinearGCU(ColumnParallelLinearGCU):
    """Linear layers for the attention's QKV transformation.

    Linear layers for the linear transformation of the query, key, and value
    vectors in the attention layer. The weight matrix is concatenated along
    the output dimension. The layer is parallelized along the head dimension.
    When the number of key/value heads is smaller than the number of query
    heads (e.g., multi-query/grouped-query attention), the key/value head may
    be replicated while the query heads are partitioned.

    Args:
        hidden_size: input hidden state size of the transformer.
        head_size: size of each attention head.
        total_num_heads: total number of attention query heads.
        total_num_kv_heads: total number of attention key/value heads. If
                            None, assume total_num_kv_heads = total_num_heads.
        bias: If true, add bias.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: If true, weights matrix won't be sharded through tp rank.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        v_head_size: int | None = None,
    ):
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.v_head_size = v_head_size if v_head_size is not None else head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        # Divide the weight matrix along the last dimension.
        tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        input_size = self.hidden_size
        output_size = (
            self.num_heads * self.head_size
            + self.num_kv_heads * self.head_size
            + self.num_kv_heads * self.v_head_size
        ) * tp_size
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,  # q_proj
            self.num_kv_heads * self.head_size * tp_size,  # k_proj
            self.num_kv_heads * self.v_head_size * tp_size,  # v_proj
        ]

        with patch(
            "vllm.model_executor.layers.quantization.fp8.validate_fp8_block_shape", validate_fp8_block_shape
        ):
            super().__init__(
                input_size=input_size,
                output_size=output_size,
                bias=bias,
                skip_bias_add=skip_bias_add,
                params_dtype=params_dtype,
                quant_config=quant_config,
                prefix=prefix,
                return_bias=return_bias,
                disable_tp=disable_tp,
            )

    def _get_shard_offset_mapping(self, loaded_shard_id: str):
        shard_offset_mapping = {
            "q": 0,
            "k": self.num_heads * self.head_size,
            "v": (self.num_heads + self.num_kv_heads) * self.head_size,
            "total": (self.num_heads + self.num_kv_heads) * self.head_size
            + self.num_kv_heads * self.v_head_size,
        }
        return shard_offset_mapping.get(loaded_shard_id)

    def _get_shard_size_mapping(self, loaded_shard_id: str):
        shard_size_mapping = {
            "q": self.num_heads * self.head_size,
            "k": self.num_kv_heads * self.head_size,
            "v": self.num_kv_heads * self.v_head_size,
        }
        return shard_size_mapping.get(loaded_shard_id)

    def _load_fused_module_from_checkpoint(self, param: BasevLLMParameter, loaded_weight: torch.Tensor):
        """
        Handle special case for models where QKV layers are already
        fused on disk. In this case, we have no shard id. This function
        determines the shard id by splitting these layers and then calls
        the weight loader using the shard id.

        An example of a model with these fused layers:
        https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
        """
        shard_offsets = [
            # (shard_id, shard_offset, shard_size)
            ("q", 0, self.total_num_heads * self.head_size),
            (
                "k",
                self.total_num_heads * self.head_size,
                self.total_num_kv_heads * self.head_size,
            ),
            (
                "v",
                (self.total_num_heads + self.total_num_kv_heads) * self.head_size,
                self.total_num_kv_heads * self.v_head_size,
            ),
        ]

        for shard_id, shard_offset, shard_size in shard_offsets:
            # Special case for Quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            if (
                isinstance(param, (PackedColumnParameter, PackedvLLMParameter))
                and param.packed_dim == param.output_dim
            ):
                shard_size, shard_offset = param.adjust_shard_indexes_for_packing(
                    shard_size=shard_size, shard_offset=shard_offset
                )

            loaded_weight_shard = loaded_weight.narrow(param.output_dim, shard_offset, shard_size)
            self.weight_loader_v2(param, loaded_weight_shard, shard_id)

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ):
        if loaded_shard_id is None:  # special case for certain models
            if isinstance(param, PerTensorScaleParameter):
                param.load_qkv_weight(loaded_weight=loaded_weight, shard_id=0, tp_rank=self.tp_rank)
                return
            elif type(param) in (RowvLLMParameter, BasevLLMParameter):
                param.load_qkv_weight(loaded_weight=loaded_weight, tp_rank=self.tp_rank)
                return
            # TODO: @dsikka - move to parameter.py
            self._load_fused_module_from_checkpoint(param, loaded_weight)
            return

        assert loaded_shard_id in ["q", "k", "v"]

        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)

        # Note(simon): This is needed for Qwen3's fp8 quantization.
        if isinstance(param, BlockQuantScaleParameter):
            assert self.quant_method is not None
            # Assume the weight block size has been set by quant method
            assert hasattr(self, "weight_block_size")
            weight_block_size = self.weight_block_size
            assert weight_block_size is not None
            block_n, _ = weight_block_size[0], weight_block_size[1]
            shard_offset = (shard_offset + block_n - 1) // block_n
            shard_size = (shard_size + block_n - 1) // block_n

        param.load_qkv_weight(
            loaded_weight=loaded_weight,
            num_heads=self.num_kv_head_replicas,
            shard_id=loaded_shard_id,
            shard_offset=shard_offset,
            shard_size=shard_size,
            tp_rank=self.tp_rank,
        )

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ):
        # Special case for GGUF
        # initialize GGUF param after we know the quantize type
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            idx_map = {"q": 0, "k": 1, "v": 2}
            if loaded_shard_id is not None:
                param.data[idx_map[loaded_shard_id]].copy_(loaded_weight)
                param.shard_weight_type[loaded_shard_id] = loaded_weight.item()
            else:
                param.shard_weight_type = {k: loaded_weight.item() for k in idx_map}
            return

        if is_gguf_weight:
            output_dim = getattr(param, "output_dim", None)
            shard_size = loaded_weight.size(output_dim) // self.tp_size
            start_idx = self.tp_rank * shard_size

            if loaded_shard_id is not None:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
                param.shard_id.append(loaded_shard_id)
                param.shard_id_map[loaded_shard_id] = len(param.data_container)
                param.data_container.append(loaded_weight)
                return

        param_data = param.data
        output_dim = getattr(param, "output_dim", None)

        # Special case for per-tensor scales in fused case.
        needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)

        if loaded_shard_id is None:
            # Loaded weight is already fused on disk (qkv).
            # (e.g., Phi-3's qkv_proj).
            if output_dim is None:
                if needs_scalar_to_array:
                    param_data, loaded_weight = adjust_scalar_to_fused_array(param_data, loaded_weight, 0)

                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return
            shard_offsets = [
                # (shard_id, shard_offset, shard_size)
                ("q", 0, self.total_num_heads * self.head_size),
                (
                    "k",
                    self.total_num_heads * self.head_size,
                    self.total_num_kv_heads * self.head_size,
                ),
                (
                    "v",
                    (self.total_num_heads + self.total_num_kv_heads) * self.head_size,
                    self.total_num_kv_heads * self.v_head_size,
                ),
            ]
            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)

            packed_dim = getattr(param, "packed_dim", None)
            for shard_id, shard_offset, shard_size in shard_offsets:
                # Special case for Quantized Weights.
                # If quantized, we need to adjust the offset and size to account
                # for the packing.
                if packed_dim == output_dim:
                    shard_size = shard_size // param.packed_factor
                    shard_offset = shard_offset // param.packed_factor

                    # Special case for Marlin.
                    shard_size, shard_offset = adjust_marlin_shard(param, shard_size, shard_offset)

                if use_bitsandbytes_4bit:
                    orig_qkv_offsets = {
                        "q": (0, self.total_num_heads * self.head_size),
                        "k": (
                            self.total_num_heads * self.head_size,
                            self.total_num_kv_heads * self.head_size,
                        ),
                        "v": (
                            (self.total_num_heads + self.total_num_kv_heads) * self.head_size,
                            self.total_num_kv_heads * self.v_head_size,
                        ),
                        "total": (
                            (self.total_num_heads + self.total_num_kv_heads) * self.head_size
                            + self.total_num_kv_heads * self.v_head_size,
                            0,
                        ),
                    }

                    shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                        param, orig_qkv_offsets, shard_id
                    )

                loaded_weight_shard = loaded_weight.narrow(output_dim, shard_offset, shard_size)
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        assert loaded_shard_id in ["q", "k", "v"]

        # If output dim is defined, use the default loading process.
        if output_dim is not None:
            if loaded_shard_id == "q":
                shard_offset = 0
                shard_size = self.num_heads * self.head_size
            elif loaded_shard_id == "k":
                shard_offset = self.num_heads * self.head_size
                shard_size = self.num_kv_heads * self.head_size
            elif loaded_shard_id == "v":
                shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size
                shard_size = self.num_kv_heads * self.v_head_size
            # Special case for Quantized Weights.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            packed_dim = getattr(param, "packed_dim", None)
            if packed_dim == output_dim:
                shard_size = shard_size // param.packed_factor
                shard_offset = shard_offset // param.packed_factor

                # Special case for Marlin.
                shard_size, shard_offset = adjust_marlin_shard(param, shard_size, shard_offset)

            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
            is_sharded_weight = getattr(param, "is_sharded_weight", False)
            # bitsandbytes loads the weights of the specific portion
            # no need to narrow
            is_sharded_weight = is_sharded_weight or use_bitsandbytes_4bit

            if use_bitsandbytes_4bit:
                orig_qkv_offsets = {
                    "q": (0, self.num_heads * self.head_size),
                    "k": (
                        self.num_heads * self.head_size,
                        self.num_kv_heads * self.head_size,
                    ),
                    "v": (
                        (self.num_heads + self.num_kv_heads) * self.head_size,
                        self.num_kv_heads * self.v_head_size,
                    ),
                    "total": (
                        (self.num_heads + self.num_kv_heads) * self.head_size
                        + self.num_kv_heads * self.v_head_size,
                        0,
                    ),
                }
                shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                    param, orig_qkv_offsets, loaded_shard_id
                )

            param_data = param_data.narrow(output_dim, shard_offset, shard_size)
            if loaded_shard_id == "q":
                shard_rank = self.tp_rank
            else:
                shard_rank = self.tp_rank // self.num_kv_head_replicas
            start_idx = shard_rank * shard_size

            if not is_sharded_weight:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        # Special case for per-tensor scales in fused case.
        elif needs_scalar_to_array:
            param_data, loaded_weight = adjust_scalar_to_fused_array(
                param_data, loaded_weight, loaded_shard_id
            )
        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "QKVParallelLinear, assume the weight is the same "
                    "for all partitions."
                )

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


class MiMoV2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MiMoV2MoE(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        is_nextn: bool = False,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.n_routed_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. " "Only silu is supported for now."
            )

        vllm_config = get_current_vllm_config()
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = self.physical_expert_start + self.n_local_physical_experts

        dtype = getattr(config, "moe_router_dtype", "float32")
        self.gate_dtype = str_dtype_to_torch_dtype(dtype)
        self.gate = nn.Linear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            dtype=self.gate_dtype,
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=self.gate_dtype)
        )

        self.experts = FusedMoE(
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=True,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            e_score_correction_bias=self.gate.e_score_correction_bias,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            scoring_func="sigmoid",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.dim() <= 2, "MiMoV2MoE only supports 1D or 2D inputs"
        is_input_1d = hidden_states.dim() == 1
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        if self.gate_dtype is not None:
            gate_input = hidden_states.to(self.gate_dtype)
        else:
            gate_input = hidden_states
        router_logits = self.gate(gate_input)
        final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=router_logits)

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(final_hidden_states, 0)
            final_hidden_states = final_hidden_states[:num_tokens]

        return final_hidden_states.squeeze(0) if is_input_1d else final_hidden_states


class MiMoV2Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        v_head_dim: int | None = None,
        sliding_window_size: int = -1,
        attention_bias: bool = False,
        add_swa_attention_sink_bias: bool = False,
        layer_id: int = 0,
        rope_theta: float = 1000000,
        max_position_embeddings: int = 32768,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        partial_rotary_factor: float = 1.0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_id = layer_id
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = head_dim

        self.v_head_dim = v_head_dim if v_head_dim is not None else head_dim

        self.q_size = self.num_heads * self.head_dim
        self.k_size = self.num_kv_heads * self.head_dim
        self.v_size = self.num_kv_heads * self.v_head_dim

        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinearGCU(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            v_head_size=self.v_head_dim,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.v_head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=True,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling={"rope_type": "default"},
            partial_rotary_factor=partial_rotary_factor,
        )

        self.attention_sink_bias = (
            torch.nn.Parameter(torch.empty(self.num_heads), requires_grad=False)
            if add_swa_attention_sink_bias
            else None
        )

        sliding_window = sliding_window_size if sliding_window_size > -1 else None
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            attn_type=AttentionType.DECODER,
            prefix=f"{prefix}.attn",
            sinks=self.attention_sink_bias,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)

        v = v.view(-1, self.num_kv_heads, self.v_head_dim)
        v = torch.nn.functional.pad(v, [0, self.head_dim - self.v_head_dim], value=0)
        v = v.view(-1, self.num_kv_heads * self.head_dim)

        attn_output = self.attn(q, k, v)

        attn_output = attn_output.view(-1, self.num_heads, self.head_dim)[..., : self.v_head_dim].reshape(
            -1, self.num_heads * self.v_head_dim
        )

        output, _ = self.o_proj(attn_output)
        return output


class MiMoV2FlashDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        layer_id = extract_layer_index(prefix)

        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_id = layer_id

        rope_theta = getattr(config, "rope_theta", 1000000)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)

        if self.is_compressed_softmax_layer():
            self.self_attn = MiMoV2Attention(
                hidden_size=self.hidden_size,
                num_heads=config.swa_num_attention_heads,
                num_kv_heads=config.swa_num_key_value_heads,
                head_dim=config.swa_head_dim,
                v_head_dim=getattr(config, "swa_v_head_dim", None),
                sliding_window_size=config.sliding_window_size,
                attention_bias=config.attention_bias,
                add_swa_attention_sink_bias=getattr(config, "add_swa_attention_sink_bias", False),
                layer_id=layer_id,
                rope_theta=getattr(config, "swa_rope_theta", rope_theta),
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                partial_rotary_factor=getattr(config, "partial_rotary_factor", 1.0),
                prefix=f"{prefix}.self_attn",
            )
        else:
            self.self_attn = MiMoV2Attention(
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                v_head_dim=getattr(config, "v_head_dim", None),
                sliding_window_size=-1,  # normal attention
                attention_bias=config.attention_bias,
                layer_id=layer_id,
                rope_theta=rope_theta,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                partial_rotary_factor=getattr(config, "partial_rotary_factor", 1.0),
                prefix=f"{prefix}.self_attn",
            )

        self.is_layer_sparse = self.is_moe_layer(layer_id)
        if self.is_layer_sparse:
            self.mlp = MiMoV2MoE(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = MiMoV2MLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def is_moe_layer(self, layer_idx: int) -> bool:
        return (
            hasattr(self.config, "moe_layer_freq")
            and layer_idx >= 0
            and not isinstance(self.config.moe_layer_freq, int)
            and self.config.moe_layer_freq[layer_idx]
        )

    def is_compressed_softmax_layer(self) -> bool:
        return self.config.hybrid_layer_pattern[self.layer_id] == 1


class MiMoV2Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config.get_text_config()
        quant_config = vllm_config.quant_config
        eplb_config = vllm_config.parallel_config.eplb_config

        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size
        self.num_redundant_experts = eplb_config.num_redundant_experts
        self.v_scale = getattr(config, "attention_value_scale", None)

        if get_pp_group().is_first_rank or (config.tie_word_embeddings and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MiMoV2FlashDecoderLayer(
                vllm_config=vllm_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        else:
            self.norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
            if "mtp" in name:
                continue

            if self.quant_config is not None:
                cache_scale_name = self.quant_config.get_cache_scale(name)
                if cache_scale_name is not None and cache_scale_name in params_dict:
                    param = params_dict[cache_scale_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)

                    kv_scale = loaded_weight
                    if kv_scale.dim() > 0 and kv_scale.numel() > 1:
                        kv_scale = kv_scale.view(-1)[0]

                    weight_loader(param, kv_scale)
                    loaded_params.add(cache_scale_name)
                    continue

            expert_matched = False
            for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                if weight_name not in name:
                    continue

                name_rewritten = name.replace(weight_name, param_name)

                if is_pp_missing_parameter(name_rewritten, self):
                    continue

                if (
                    name_rewritten.endswith(".bias") or name_rewritten.endswith("_bias")
                ) and name_rewritten not in params_dict:
                    continue

                if name_rewritten not in params_dict:
                    continue

                param = params_dict[name_rewritten]
                weight_loader = param.weight_loader

                weight_loader(
                    param,
                    loaded_weight,
                    name_rewritten,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                loaded_params.add(name_rewritten)
                expert_matched = True
                break

            if expert_matched:
                continue

            stacked_matched = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name_rewritten = name.replace(weight_name, param_name)

                if name_rewritten.endswith(".bias") and name_rewritten not in params_dict:
                    continue

                if is_pp_missing_parameter(name_rewritten, self):
                    continue

                if name_rewritten not in params_dict:
                    continue

                param = params_dict[name_rewritten]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)

                if param_name == "qkv_proj" and shard_id == "v":
                    v_scale = (
                        self.v_scale
                        if self.v_scale is not None
                        else getattr(self.config, "attention_value_scale", None)
                    )
                    if v_scale is not None and (name.endswith("weight_scale_inv") or name.endswith(".bias")):
                        loaded_weight *= float(v_scale)
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name_rewritten)

                stacked_matched = True
                break

            if stacked_matched:
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue

            orig_name = name
            mapped_name = maybe_remap_kv_scale_name(name, params_dict)
            name = mapped_name if mapped_name is not None else orig_name

            if name not in params_dict:
                continue

            param = params_dict[name]

            if "attention_sink_bias" in name:
                total_heads = loaded_weight.shape[0]
                heads_per_rank = total_heads // tp_size
                head_start = tp_rank * heads_per_rank
                narrow_weight = loaded_weight.narrow(0, head_start, heads_per_rank)

                param.data.copy_(narrow_weight)
                loaded_params.add(name)
            else:
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params


class MiMoV2FlashForCausalLM(nn.Module, SupportsPP, MixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.model = MiMoV2Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
