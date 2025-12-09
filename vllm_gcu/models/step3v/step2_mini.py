# SPDX-License-Identifier: Apache-2.0
"""Inference-only Jurassic model."""
import inspect
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch import nn

from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (
    get_dp_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)

from vllm.distributed import (tensor_model_parallel_all_gather,
                              tensor_model_parallel_gather)
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.v1.outputs import SamplerOutput, PoolerOutput
from vllm.v1.sample.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.v1.pool.metadata import PoolingMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
)
from .step1 import Step1MoEMLP

logger = init_logger(__name__)


# 全局共享的CUDA graph memory pool，类似model_runner.py中的实现
_graph_memory_pool: Optional[Tuple[int, int]] = None

def _prune_hidden_states(
    hidden_states: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    # NOTE(kzawora): The if guard is needed for Gaudi - in some scenarios
    # (warmup, profile_run) we might not have selected_token_indices,
    # so we skip pruning.
    if sampling_metadata.selected_token_indices is not None:
        return hidden_states.index_select(
            0, sampling_metadata.selected_token_indices)
    else:
        return hidden_states


def _apply_logits_processors(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    found_logits_processors = False
    logits_processed = 0
    logits_row_ids_and_logits_row_futures = []
    for seq_group in sampling_metadata.seq_groups:
        seq_ids = seq_group.seq_ids
        sampling_params = seq_group.sampling_params
        logits_processors = sampling_params.logits_processors
        if logits_processors:
            found_logits_processors = True

            for seq_id, logits_row_idx in zip(seq_ids,
                                              seq_group.sample_indices):
                logits_row = logits[logits_row_idx]
                past_tokens_ids = seq_group.seq_data[seq_id].output_token_ids
                prompt_tokens_ids = seq_group.seq_data[seq_id].prompt_token_ids

                if _logits_processor_threadpool is not None:
                    logits_row_ids_and_logits_row_futures.append(
                        (logits_row_idx,
                         _logits_processor_threadpool.submit(
                             _apply_logits_processors_single_seq, logits_row,
                             logits_processors, past_tokens_ids,
                             prompt_tokens_ids)))
                else:
                    logits[logits_row_idx] = \
                        _apply_logits_processors_single_seq(
                            logits_row, logits_processors, past_tokens_ids,
                            prompt_tokens_ids)

        logits_processed += len(seq_group.sample_indices) + len(
            seq_group.prompt_logprob_indices)

    for logits_row_idx, future in logits_row_ids_and_logits_row_futures:
        logits[logits_row_idx] = future.result()

    if found_logits_processors:
        # verifies that no rows in logits were missed unexpectedly
        assert logits_processed == logits.shape[0]
    return logits


def _apply_logits_processors_single_seq(logits_row, logits_processors,
                                        past_tokens_ids,
                                        prompt_tokens_ids) -> torch.Tensor:
    for logits_processor in logits_processors:
        parameters = inspect.signature(logits_processor).parameters
        if len(parameters) == 3:
            logits_row = logits_processor(prompt_tokens_ids, past_tokens_ids,
                                          logits_row)
        else:
            logits_row = logits_processor(past_tokens_ids, logits_row)
    return logits_row


class LogitsProcessor(nn.Module):
    """Process logits and apply logits processors from sampling metadata.

    This layer does the following:
    1. Gather logits from model hidden_states.
    2. Scale logits if needed.
    3. Apply logits processors (if any).
    """

    def __init__(
        self,
        vocab_size: int,
        org_vocab_size: Optional[int] = None,
        scale: float = 1.0,
        logits_as_input: bool = False,
        soft_cap: Optional[float] = None,
        need_fp32_logits: bool = False,
    ) -> None:
        """
        Args:
            scale: A scaling factor to apply to the logits.
        """
        super().__init__()
        self.scale = scale
        self.vocab_size = vocab_size
        # Whether the input is logits (default is hidden states).
        self.logits_as_input = logits_as_input
        # original vocabulary size (without LoRA).
        self.org_vocab_size = org_vocab_size or vocab_size
        # Soft cap the logits. Used in Gemma 2.
        self.soft_cap = soft_cap
        # Whether to use gather or all-gather to gather the logits.
        self.use_all_gather = current_platform.use_all_gather()
        self.need_fp32_logits = need_fp32_logits

    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        sampling_metadata: Optional[SamplingMetadata] = None,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.logits_as_input:
            logits = hidden_states
        else:
            if sampling_metadata is not None:
                hidden_states = _prune_hidden_states(
                    hidden_states, sampling_metadata
                )

            # Get the logits for the next tokens.
            logits = self._get_logits(hidden_states, lm_head, embedding_bias)
        if logits is not None:
            if self.soft_cap is not None:
                logits = logits / self.soft_cap
                logits = torch.tanh(logits)
                logits = logits * self.soft_cap

            if self.scale != 1.0:
                logits *= self.scale

            # Apply logits processors (if any).
            if (
                sampling_metadata is not None
                and sampling_metadata.seq_groups is not None
            ):
                logits = _apply_logits_processors(logits, sampling_metadata)

        return logits

    def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """gather/all-gather the logits tensor across model parallel group."""
        if self.use_all_gather:
            # Gather is not supported for some devices such as TPUs.
            # Use all-gather instead.
            # NOTE(woosuk): Here, the outputs of every device should not be None
            # because XLA requires strict SPMD among all devices. Every device
            # should execute the same operations after gathering the logits.
            logits = tensor_model_parallel_all_gather(logits)
        else:
            # None may be returned for rank > 0
            logits = tensor_model_parallel_gather(logits)
        return logits

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        # Get the logits for the next tokens.
        if self.need_fp32_logits:
            logits = torch.ops.OptimusMoe.matmul_fp32(
                hidden_states, lm_head.weight.t()
            )
        else:
            logits = lm_head.quant_method.apply(
                lm_head, hidden_states, bias=embedding_bias
            )

        # Gather logits for TP
        logits = self._gather_logits(logits)

        # Remove paddings in vocab (if any).
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        return logits

    def extra_repr(self) -> str:
        s = f"vocab_size={self.vocab_size}"
        s += f", forg_vocab_size={self.org_vocab_size}"
        s += f", scale={self.scale}, logits_as_input={self.logits_as_input}"
        return s


class FusedMoEBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()

        if self.tp_size > config.moe_num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.moe_num_experts}."
            )

        assert config.moe_dynamic_exp_p == 1, "Only support dynamic exp p=1"

        self.experts = FusedMoE(
            num_experts=config.moe_num_experts,
            top_k=config.moe_top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_expert_weight,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.moe_num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)

        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states
            )

        return final_hidden_states.view(orig_shape)


class Step2MiniMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
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
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()
        self.prefix = prefix
        self.hidden_size = hidden_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        intermediate_act = self.act_fn(gate_up)
        output, _ = self.down_proj(intermediate_act)
        return output


class Step2MiniAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        norm_eps: float,
        rope_theta: int,
        share_q_dim: Optional[int] = None,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embedding: int = 8192,
        head_dim: int = 256,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.q_size = share_q_dim if share_q_dim else self.head_dim

        self.qkv_proj = ReplicatedLinear(
            hidden_size,
            self.q_size + self.kv_size * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.inter_norm = RMSNorm(self.q_size, eps=norm_eps)
        self.wq = ColumnParallelLinear(
            self.q_size,
            self.head_dim * self.total_num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq",
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embedding,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        scaling = self.head_dim**-0.5
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            scaling,
            self.num_kv_heads,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )
        self.prefix = prefix

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.inter_norm(q.contiguous())
        q = self.wq(q)[0]
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        residual, _ = self.o_proj(attn_output)
        return residual


class Step2MiniDecoderLayer(nn.Module):

    def __init__(
        self,
        config: ModelConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        use_fused_moe: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = config.hf_config
        self.hidden_size = config.hidden_size
        rope_scaling = getattr(config, "rope_scaling", None)

        self.self_attn = Step2MiniAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=1,
            cache_config=cache_config,
            quant_config=quant_config,
            norm_eps=config.rms_norm_eps,
            max_position_embedding=config.max_position_embedding,
            head_dim=config.head_dim,
            share_q_dim=config.share_q_dim,
            rope_theta=config.rope_theta,
            rope_scaling=rope_scaling,
            prefix=f"{prefix}.self_attn",
        )
        self.use_moe = False

        layer_idx = int(prefix.split("layers.")[1].split(".")[0])
        moe_layers_enum = getattr(config, "moe_layers_enum", None)
        if moe_layers_enum is not None:
            moe_layers_idx = [
                int(i) for i in moe_layers_enum.strip().split(",")
            ]
        else:
            # Default to 1dense.
            moe_layers_idx = [i for i in range(1, config.num_hidden_layers)]

        if layer_idx in moe_layers_idx:
            if not use_fused_moe:
                self.moe = Step1MoEMLP(
                    config.moe_num_experts,
                    config.moe_top_k,
                    config.moe_dynamic_exp_p,
                    hidden_size=self.hidden_size,
                    intermediate_size=config.moe_intermediate_size,
                    hidden_act="silu",
                    quant_config=quant_config,
                    norm_expert_weight=config.norm_expert_weight,
                    prefix=f"{prefix}.moe",
                )
            else:
                self.moe = FusedMoEBlock(
                    config=config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.moe",
                )
            self.share_expert = Step2MiniMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.share_expert_dim,
                hidden_act="silu",
                quant_config=quant_config,
                prefix=f"{prefix}.share_expert",
            )
            self.use_moe = True
        else:
            self.mlp = Step2MiniMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act="silu",
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.use_fused_moe = use_fused_moe
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.prefix = prefix

        # CUDA Graph parameters - 简化版本，使用共享memory pool
        # self.should_capture_graph = (
        #     get_dp_group().world_size > 1 and current_platform.is_cuda_alike()
        # )
        # disable step cuda graph by harry.zhao
        self.should_capture_graph = False
        # self.cuda_graphs_captured = False
        # self.graph_runners_fwd1: dict[int, Tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        # self.graph_runners_fwd2: dict[int, Tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        # self.graph_runners_fwd3: dict[int, Tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        # self.max_graph_tokens = 64
        # self.graph_token_step = 32
        # self.decoder_captured_sizes = list(range(self.graph_token_step,
        #                                      self.max_graph_tokens + 1,
        #                                      self.graph_token_step)) if self.should_capture_graph else []

    @torch.inference_mode()
    def _capture_cuda_graph(
        self,
        device: torch.device,
        hs_dtype: torch.dtype,
        pos_dtype: torch.dtype,
    ):
        global _graph_memory_pool
        if self.cuda_graphs_captured or not self.should_capture_graph:
            return

        # 使用全局共享的memory pool
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(stream):
            for total_tokens in reversed(self.decoder_captured_sizes):
                # --- Capture forward_1 ---
                graph_fwd1 = torch.cuda.CUDAGraph()

                # 创建输入buffers
                static_positions = torch.ones(
                    (total_tokens,), dtype=pos_dtype, device=device
                )
                static_hidden_states = torch.randn(
                    (total_tokens, self.hidden_size),
                    dtype=hs_dtype,
                    device=device,
                )

                # Warmup forward_1
                _, _, _ = self._forward_1_impl(
                    static_positions, static_hidden_states
                )

                # Capture forward_1 - 使用torch.cuda.graph()和共享memory pool
                with torch.cuda.graph(
                    graph_fwd1, pool=_graph_memory_pool, stream=stream
                ):
                    static_q_fwd1, static_k_fwd1, static_v_fwd1 = (
                        self._forward_1_impl(
                            static_positions, static_hidden_states
                        )
                    )

                # 更新全局memory pool
                if _graph_memory_pool is None:
                    _graph_memory_pool = graph_fwd1.pool()

                self.graph_runners_fwd1[total_tokens] = (
                    graph_fwd1,
                    static_positions,
                    static_hidden_states,
                    static_q_fwd1,
                    static_k_fwd1,
                    static_v_fwd1,
                )

                # --- Capture forward_2 ---
                graph_fwd2 = torch.cuda.CUDAGraph()

                # 创建输入buffers
                attn_output_size = (
                    self.self_attn.num_heads * self.self_attn.head_dim
                )
                static_attn_output = torch.randn(
                    (total_tokens, attn_output_size),
                    dtype=hs_dtype,
                    device=device,
                )
                static_residual = torch.randn(
                    (total_tokens, self.hidden_size),
                    dtype=hs_dtype,
                    device=device,
                )

                # Warmup forward_2
                _, _ = self._forward_2_impl(static_attn_output, static_residual)

                # Capture forward_2 - 使用torch.cuda.graph()和共享memory pool
                with torch.cuda.graph(
                    graph_fwd2, pool=_graph_memory_pool, stream=stream
                ):
                    static_hs_out_fwd2, static_residual_out_fwd2 = (
                        self._forward_2_impl(
                            static_attn_output, static_residual
                        )
                    )

                self.graph_runners_fwd2[total_tokens] = (
                    graph_fwd2,
                    static_attn_output,
                    static_residual,
                    static_hs_out_fwd2,
                    static_residual_out_fwd2,
                )

                # --- Capture forward_3 ---
                graph_fwd3 = torch.cuda.CUDAGraph()

                # 创建输入buffers (重用之前的)
                static_hidden_states_fwd3 = torch.randn(
                    (total_tokens, self.hidden_size),
                    dtype=hs_dtype,
                    device=device,
                )
                static_residual_fwd3 = torch.randn(
                    (total_tokens, self.hidden_size),
                    dtype=hs_dtype,
                    device=device,
                )

                # Warmup forward_3
                _, _ = self._forward_3_impl(
                    static_hidden_states_fwd3, static_residual_fwd3
                )

                # Capture forward_3 - 使用torch.cuda.graph()和共享memory pool
                with torch.cuda.graph(
                    graph_fwd3, pool=_graph_memory_pool, stream=stream
                ):
                    static_ffn_output_fwd3, static_router_logits_fwd3 = (
                        self._forward_3_impl(
                            static_hidden_states_fwd3, static_residual_fwd3
                        )
                    )

                self.graph_runners_fwd3[total_tokens] = (
                    graph_fwd3,
                    static_hidden_states_fwd3,
                    static_residual_fwd3,
                    static_ffn_output_fwd3,
                    static_router_logits_fwd3,
                )

        torch.cuda.current_stream().wait_stream(stream)
        self.cuda_graphs_captured = True

    def _ensure_cuda_graphs_captured(
        self,
        device: torch.device,
        hs_dtype: torch.dtype,
        pos_dtype: torch.dtype,
    ):
        if not self.cuda_graphs_captured and self.should_capture_graph:
            self._capture_cuda_graph(device, hs_dtype, pos_dtype)

    # Separate implementation logic from graph handling
    def _forward_1_impl(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ):
        hidden_states = self.input_layernorm(hidden_states)
        # q, _ = self.self_attn.q_proj(hidden_states)
        # kv, _ = self.self_attn.kv_proj(hidden_states)
        # k, v = kv.split([self.self_attn.kv_size, self.self_attn.kv_size], dim=-1)
        qkv, _ = self.self_attn.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [
                self.self_attn.q_size,
                self.self_attn.kv_size,
                self.self_attn.kv_size,
            ],
            dim=-1,
        )

        q = self.self_attn.inter_norm(q.contiguous())
        q = self.self_attn.wq(q)[0]
        q, k = self.self_attn.rotary_emb(positions, q, k)
        return q, k, v

    def forward_1(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        if self.should_capture_graph:
            self._ensure_cuda_graphs_captured(
                hidden_states.device, hidden_states.dtype, positions.dtype
            )

            graph_key = (
                (hidden_states.shape[0] + self.graph_token_step - 1)
                // self.graph_token_step
                * self.graph_token_step
            )
            graph_data = (
                self.graph_runners_fwd1.get(graph_key)
                if self.cuda_graphs_captured
                else None
            )
            use_graph = (
                graph_data is not None
                and hidden_states.shape[0] <= self.max_graph_tokens
            )

            if use_graph:
                (
                    graph,
                    static_pos_view,
                    static_hs_view,
                    static_q,
                    static_k,
                    static_v,
                ) = graph_data
                actual_tokens = hidden_states.shape[0]
                static_pos_view[:actual_tokens].copy_(positions)
                static_hs_view[:actual_tokens].copy_(hidden_states)
                graph.replay()
                return (
                    static_q[:actual_tokens],
                    static_k[:actual_tokens],
                    static_v[:actual_tokens],
                )

        # Fallback to eager execution
        return self._forward_1_impl(positions, hidden_states)

    # Separate implementation logic from graph handling
    def _forward_2_impl(
        self, attn_output: torch.Tensor, residual: torch.Tensor
    ):
        hidden_states, _ = self.self_attn.o_proj(attn_output)
        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return hidden_states, residual

    def forward_2(self, attn_output: torch.Tensor, residual: torch.Tensor):
        if self.should_capture_graph:
            graph_key = (
                (attn_output.shape[0] + self.graph_token_step - 1)
                // self.graph_token_step
                * self.graph_token_step
            )
            graph_data = (
                self.graph_runners_fwd2.get(graph_key)
                if self.cuda_graphs_captured
                else None
            )
            use_graph = (
                graph_data is not None
                and attn_output.shape[0] <= self.max_graph_tokens
            )

            if use_graph:
                (
                    graph,
                    static_attn_output_view,
                    static_residual_view,
                    static_hs_out,
                    static_residual_out,
                ) = graph_data
                actual_tokens = attn_output.shape[0]
                static_attn_output_view[:actual_tokens].copy_(attn_output)
                static_residual_view[:actual_tokens].copy_(residual)
                graph.replay()
                return (
                    static_hs_out[:actual_tokens],
                    static_residual_out[:actual_tokens],
                )

        # Fallback to eager execution
        return self._forward_2_impl(attn_output, residual)

    # Separate implementation logic from graph handling
    def _forward_3_impl(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ):
        if self.use_moe:
            ffn_output = self.share_expert(hidden_states)
            router_logits, _ = self.moe.gate(hidden_states)
        else:
            ffn_output = self.mlp(hidden_states)
            router_logits = None
        return (
            ffn_output + residual,
            router_logits,
        )  # Base output before potential MoE addition

    def forward_3(self, hidden_states: torch.Tensor, residual: torch.Tensor):
        if self.should_capture_graph:
            graph_key = (
                (hidden_states.shape[0] + self.graph_token_step - 1)
                // self.graph_token_step
                * self.graph_token_step
            )
            graph_data = (
                self.graph_runners_fwd3.get(graph_key)
                if self.cuda_graphs_captured
                else None
            )
            use_graph = (
                graph_data is not None
                and hidden_states.shape[0] <= self.max_graph_tokens
            )

            if use_graph:
                (
                    graph,
                    static_hs_view,
                    static_residual_view,
                    static_ffn_output,
                    static_router_logits,
                ) = graph_data
                actual_tokens = hidden_states.shape[0]
                static_hs_view[:actual_tokens].copy_(hidden_states)
                static_residual_view[:actual_tokens].copy_(residual)
                graph.replay()
                return static_ffn_output[:actual_tokens], (
                    static_router_logits[:actual_tokens]
                    if static_router_logits is not None
                    else None
                )

        # Fallback to eager execution
        return self._forward_3_impl(hidden_states, residual)

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if self.should_capture_graph:
            # 使用graph pool的分阶段forward
            residual = hidden_states
            q, k, v = self.forward_1(positions, hidden_states)

            attn_output = self.self_attn.attn(q, k, v)

            hidden_states, residual = self.forward_2(attn_output, residual)

            ffn_output_plus_residual, router_logits = self.forward_3(
                hidden_states, residual
            )
            print(f"========== hidden_states: {hidden_states.shape}")
            if self.use_moe:
                moe_output = self.moe.experts(hidden_states, router_logits)
                hidden_states = ffn_output_plus_residual + moe_output
            else:
                hidden_states = ffn_output_plus_residual

            return hidden_states
        else:
            # 原始的forward实现
            return self.forward_old(positions, hidden_states)

    def forward_old(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.use_moe:
            share_output = self.share_expert(hidden_states)
            moe_output = self.moe(hidden_states)
            ffn_output = share_output + moe_output
        else:
            ffn_output = self.mlp(hidden_states)

        hidden_states = ffn_output + residual
        return hidden_states


class Step2MiniModel(nn.Module):

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        use_fused_moe: bool = False,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.vocab_size = config.vocab_size
        self.config = config
        self.use_fused_moe = use_fused_moe

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Step2MiniDecoderLayer(
                config=vllm_config.model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                use_fused_moe=self.use_fused_moe,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states"], config.hidden_size
            )
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states = layer(positions, hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                }
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


@support_torch_compile
class Step3FlashModel(Step2MiniModel):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        use_fused_moe = True
        if vllm_config.quant_config.get_name() == "groupwise_quant":
            use_fused_moe = False
        super().__init__(vllm_config, prefix, use_fused_moe=use_fused_moe)


class Step2MiniPretrainedModel(nn.Module, SupportsPP):

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        qkv_params_mapping = [
            # (param_name, shard_name, relative_start_idx, relative_end_idx)
            (
                ".qkv_proj",
                ".q_proj",
                0,
                self.config.share_q_dim
                / (self.config.share_q_dim + self.config.head_dim * 2),
            ),
            (
                ".qkv_proj",
                ".k_proj",
                self.config.share_q_dim
                / (self.config.share_q_dim + self.config.head_dim * 2),
                (self.config.share_q_dim + self.config.head_dim)
                / (self.config.share_q_dim + self.config.head_dim * 2),
            ),
            (
                ".qkv_proj",
                ".v_proj",
                (self.config.share_q_dim + self.config.head_dim)
                / (self.config.share_q_dim + self.config.head_dim * 2),
                (self.config.share_q_dim + self.config.head_dim * 2)
                / (self.config.share_q_dim + self.config.head_dim * 2),
            ),
        ]
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        params_need_to_load = set()

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.model.config.moe_num_experts,
        )

        if self.model.use_fused_moe:
            expert_params_mapping = [
                (".moe.experts.w13_weight", ".moe.gate_proj.weight", "w1"),
                (".moe.experts.w13_weight", ".moe.up_proj.weight", "w3"),
                (".moe.experts.w2_weight", ".moe.down_proj.weight", "w2"),
            ]
        else:
            expert_params_mapping = []

        disable_moe_stacked_params = [data[1] for data in expert_params_mapping]

        for name, loaded_weight in weights:
            # continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if any(
                    disable_moe_stacked_param in name
                    for disable_moe_stacked_param in disable_moe_stacked_params
                ):
                    continue
                name = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    for expert_id in range(loaded_weight.shape[0]):
                        loaded_weight_expert = loaded_weight[expert_id]
                        weight_loader(
                            param,
                            loaded_weight_expert,
                            name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    loaded_params.add(name)
                    break
                else:
                    for (
                        param_name,
                        weight_name,
                        start_idx,
                        end_idx,
                    ) in qkv_params_mapping:
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(name, self):
                            continue
                        param = params_dict[name]
                        dim = param.shape[param.output_dim]
                        begin_idx = int(start_idx * dim)
                        end_idx = int(end_idx * dim)
                        param_slice = param.narrow(
                            param.output_dim, begin_idx, end_idx - begin_idx
                        )
                        param_slice.copy_(loaded_weight)
                        loaded_params.add(name)
                        break
                    else:
                        if is_pp_missing_parameter(name, self):
                            continue
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                        loaded_params.add(name)
        for name in params_dict:
            params_need_to_load.add(name)
        if params_need_to_load != loaded_params:
            param_name_example = list(params_need_to_load - loaded_params)[0]
            raise RuntimeError(
                f"Some parameters like {param_name_example} are not in the checkpoint and will falsely use random initialization"
            )


class Step2MiniForCausalLM(Step2MiniPretrainedModel):

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        lora_config = vllm_config.lora_config
        self.config = config

        # FIXME: hack for step3 flash model
        if self.config.num_hidden_layers == 42:
            self.model = Step2MiniModel(vllm_config=vllm_config, prefix=prefix)
        else:
            self.model = Step3FlashModel(vllm_config=vllm_config, prefix=prefix)

        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            if lora_config:
                self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=(
                    DEFAULT_VOCAB_PADDING_SIZE
                    if not lora_config
                    else lora_config.lora_vocab_padding_size
                ),
            )
            self.logits_processor = LogitsProcessor(
                self.unpadded_vocab_size,
                config.vocab_size,
                need_fp32_logits=False,
            )
            self.sampler = Sampler()
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ):

        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self, hidden_states: torch.Tensor,
        #sampling_metadata: SamplingMetadata
    ) -> torch.Tensor:
        logits = self.logits_processor(
            self.lm_head, hidden_states,
            #sampling_metadata
        )
        return logits

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens


class Step2MiniForSequenceClassification(Step2MiniPretrainedModel):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = Step2MiniModel(vllm_config, prefix=prefix)

        if get_pp_group().is_last_rank:
            self.score = ReplicatedLinear(
                self.config.hidden_size, self.config.num_labels, bias=False
            )
            pooler_config = vllm_config.model_config.pooler_config

            self._pooler = Pooler.from_config_with_defaults(
                pooler_config,
                pooling_type=PoolingType.LAST,
                normalize=False,
                softmax=False,
            )
        else:
            self._pooler = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> SamplerOutput:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        logits, _ = self.score(hidden_states)
        ret = self._pooler(logits, pooling_metadata)
        return ret

    def sequence_flops(self, input_length, context_length):
        output_flops = (
            1 * self.config.hidden_size * self.config.num_labels * 2.0 / 1e12
        )
        return (
            super().sequence_flops(input_length, context_length) + output_flops
        )
