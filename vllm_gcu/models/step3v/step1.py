# SPDX-License-Identifier: Apache-2.0
import math
import os
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from vllm.attention import Attention
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)

from vllm.model_executor.layers.activation import SiluAndMul

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)

from vllm.v1.sample.logits_processor import LogitsProcessor
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)

from vllm.v1.outputs import SamplerOutput, PoolerOutput
from vllm.v1.sample.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from safetensors.torch import safe_open
from vllm.v1.pool.metadata import PoolingMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

from .linear import RowParallelMoELinear, MergedColumnParallelMoELinear, OptimusSiluAndMul, OptimusRMSNorm, dynamic_fp8_pertensor_quantize

DISABLE_SEQUENCE_PARALLEL = (
    True  # FIXME: os.getenv("DISABLE_SEQUENCE_PARALLEL", "0") == "1"
)
SEQUENCE_PARALLEL_THRESHOLD = (
    512
    if os.getenv("SEQUENCE_PARALLEL_THRESHOLD", "0") == "0"
    else int(os.getenv("SEQUENCE_PARALLEL_THRESHOLD"))
)
GEMM_COMM_OVERLAP_RATIO = 0.5
MLP_BATCH_SIZE = 8192


def _get_alibi_slopes(n_heads):
    n = 2 ** math.floor(math.log2(n_heads))  # nearest 2**n to n_heads
    m0 = 2.0 ** (-8.0 / n)
    slopes = np.power(m0, np.arange(1, n + 1))
    if n < n_heads:
        m1 = 2.0 ** (-4.0 / n)
        mm = np.power(m1, np.arange(1, 1 + 2 * (n_heads - n), 2))
        slopes = np.concatenate([slopes, mm])
    return slopes


def _get_ntk_alibi_slopes(max_pos_interp_ratio, slopes):
    if max_pos_interp_ratio == 1.0:
        return slopes
    smax, smin = slopes.max(), slopes.min()
    D0 = np.log2(smax) - np.log2(smin)
    W1 = (np.log2(smax) - np.log2(slopes)) / D0
    ratios = np.power(max_pos_interp_ratio, W1)
    return slopes / (ratios**0.5)


def fp8_input_scales_loader(path: str):
    with safe_open(path, framework="pt") as f:
        for name in f.keys():  # noqa: SIM118
            param = f.get_slice(name)
            yield name, param


class Step1MoEMLP(nn.Module):

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        top_p: float,
        hidden_size: int,
        intermediate_size: int,
        hidden_act="",
        quant_config: Optional[QuantizationConfig] = None,
        norm_expert_weight=True,
        prefix: str = "",
        enable_cudagraph: bool = False,
    ):
        super().__init__()
        self.gate = ReplicatedLinear(
            input_size=hidden_size,
            output_size=num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        self.top_k = top_k
        self.top_p = top_p
        self.num_experts = num_experts
        tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size()
        )
        assert intermediate_size % tensor_model_parallel_world_size == 0
        self.gate_up_proj = MergedColumnParallelMoELinear(
            num_experts,
            hidden_size,
            [intermediate_size] * 2,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        if (intermediate_size / tensor_model_parallel_world_size) % 64 == 0:
            self.act_fn = OptimusSiluAndMul()
        else:
            self.act_fn = SiluAndMul()
        self.down_proj = RowParallelMoELinear(
            num_experts,
            intermediate_size,
            hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.tp_rank = get_tensor_model_parallel_rank()
        self.quant_config = quant_config
        self.norm_expert_weight = norm_expert_weight
        self.enable_cudagraph = enable_cudagraph
        self.need_fp32_gate = False

    def get_expert_output(
        self,
        inputs: torch.Tensor,
        expert_token_cnt: torch.Tensor,
        token_nums: int,
    ):
        if (
            self.quant_config
            and getattr(self.gate_up_proj.quant_method, "quant_config", None)
            and getattr(self.down_proj.quant_method, "quant_config", None)
        ):
            if (
                inputs.size(0) <= 1024
                and self.gate_up_proj.quant_method.quant_config.weight_bits == 8
                and self.down_proj.quant_method.quant_config.weight_bits == 8
            ):
                if self.enable_cudagraph:
                    tmp = torch.ops.Optimus.MoeFpAIntBGemm(
                        inputs,
                        self.gate_up_proj.qweight,
                        self.gate_up_proj.qweight.dtype,
                        self.gate_up_proj.scales,
                        expert_token_cnt,
                        token_nums,
                        None,
                    )
                    tmp = self.act_fn(tmp)
                    tmp = torch.ops.Optimus.MoeFpAIntBGemm(
                        tmp,
                        self.down_proj.qweight,
                        self.down_proj.qweight.dtype,
                        self.down_proj.scales,
                        expert_token_cnt,
                        token_nums,
                        None,
                    )
                    return tmp
                else:
                    quant_output_ = torch.ops.OptimusMoe.moe_ffn_quant(
                        inputs,
                        self.gate_up_proj.qweight.dtype,
                        self.gate_up_proj.qweight,
                        self.gate_up_proj.scales,
                        self.down_proj.qweight,
                        self.down_proj.scales,
                        expert_token_cnt,
                        token_nums,
                        out=inputs,
                    )
                    return quant_output_
            else:
                expert_token_cnt = expert_token_cnt.to("cpu").tolist()
                start = 0
                end = 0
                if (
                    getattr(
                        self.gate_up_proj.quant_method, "quant_config", None
                    )
                    and self.gate_up_proj.quant_method.quant_config.weight_bits
                    == 6
                ):
                    output = torch.empty_like(
                        inputs, dtype=torch.bfloat16, device=inputs.device
                    )
                else:
                    output = inputs
                for i in range(len(expert_token_cnt)):
                    cur_token_cnt = expert_token_cnt[i]
                    if cur_token_cnt <= 0:
                        continue
                    end += cur_token_cnt
                    tmp = self.gate_up_proj(inputs[start:end], expert_idx=i)
                    tmp = self.act_fn(tmp)
                    tmp = self.down_proj(
                        tmp, expert_idx=i, output=output[start:end]
                    )
                    start += cur_token_cnt
                return output
        else:
            moe_output = torch.ops.OptimusMoe.moe_ffn(
                inputs,
                self.gate_up_proj.weight,
                self.down_proj.weight,
                expert_token_cnt,
                token_nums,
            )
            return moe_output

    def forward(
        self,
        x,
        residual=None,
        layernorm=None,
        disable_allreduce=False,
        user_output=None,
    ):
        if layernorm is not None:
            x = layernorm(
                x,
                fp16_out=(
                    getattr(
                        self.gate_up_proj.quant_method, "quant_config", None
                    )
                    and self.gate_up_proj.quant_method.quant_config.weight_bits
                    == 6
                    if self.gate_up_proj.quant_method
                    else False
                ),
            )
        x_shape = x.shape
        if self.need_fp32_gate:
            if (
                getattr(self.gate_up_proj.quant_method, "quant_config", None)
                and self.gate_up_proj.quant_method.quant_config.weight_bits == 6
            ):
                logits = torch.ops.OptimusMoe.matmul_fp32(
                    x.to(torch.bfloat16), self.gate.weight.t()
                )
            else:
                logits = torch.ops.OptimusMoe.matmul_fp32(
                    x, self.gate.weight.t()
                )
        else:
            logits = self.gate(x)[0]
        if self.top_p < 1.0:
            from optimus import moe_expert_histogram as optimus_moe_expert_histogram
            from optimus import moe_scatter as optimus_moe_scatter
            from optimus import moe_gather as optimus_moe_gather
            top_k_index, expert_weight, scatter_index = (
                torch.ops.OptimusMoe.topk_topp_gating(
                    logits, self.top_k, self.top_p, self.norm_expert_weight
                )
            )
            expert_token_cnt = optimus_moe_expert_histogram(
                top_k_index, self.num_experts
            )
            scatter_index = torch.ops.OptimusMoe.index_compute(
                top_k_index, expert_token_cnt, out=scatter_index
            )
            mid_output = optimus_moe_scatter(x, scatter_index)
            expert_output = self.get_expert_output(
                mid_output, expert_token_cnt, x_shape[0]
            )
            output = optimus_moe_gather(
                expert_output, scatter_index, expert_weight
            )
        else:
            from optimus import moe_scatter as optimus_moe_scatter
            from optimus import moe_gather as optimus_moe_gather
            expert_weight, expert_token_cnt, scatter_index = (
                torch.ops.OptimusMoe.gating_histogram_index(
                    logits, self.top_k, 1.0, self.norm_expert_weight
                )
            )

            mid_output = optimus_moe_scatter(x, scatter_index)
            expert_output = self.get_expert_output(
                mid_output, expert_token_cnt, x_shape[0]
            )
            output = optimus_moe_gather(
                expert_output, scatter_index, expert_weight
            )
        if self.tp_rank == 0 and residual is not None:
            output += residual
        if not disable_allreduce:
            output = tensor_model_parallel_all_reduce(output)
        if user_output is not None:
            user_output.copy_(output)
        return output


class Step1MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        use_optimus_silu: bool = True,
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
        if use_optimus_silu:
            self.act_fn = OptimusSiluAndMul()
        else:
            self.act_fn = SiluAndMul()

    def forward(
        self,
        x,
        residual=None,
        layernorm=None,
        disable_allreduce=False,
        user_output=None,
    ):
        if layernorm is not None:
            x = layernorm(
                x,
                fp16_out=(
                    self.gate_up_proj.quant_method.quant_config.weight_bits == 6
                    if getattr(
                        self.gate_up_proj.quant_method, "quant_config", None
                    )
                    else False
                ),
            )
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        residual, _ = self.down_proj(
            x, residual, output=user_output, disable_allreduce=disable_allreduce
        )
        return residual


class Step1Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        slopes: Optional[List[float]] = None,
        max_pos_interp_ratio: float = 1.0,
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
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
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

        # Create the alibi slopes and slice them.
        tp_rank = get_tensor_model_parallel_rank()
        head_start = tp_rank * self.num_heads
        head_end = (tp_rank + 1) * self.num_heads
        if slopes is None:
            alibi_slopes = _get_alibi_slopes(self.total_num_heads)
            alibi_slopes = _get_ntk_alibi_slopes(
                max_pos_interp_ratio, alibi_slopes
            )
            alibi_slopes = alibi_slopes[head_start:head_end]
        else:
            assert len(slopes) == self.total_num_heads
            alibi_slopes = _get_ntk_alibi_slopes(
                max_pos_interp_ratio, slopes
            ).tolist()
            alibi_slopes = slopes[head_start:head_end]

        scaling = self.head_dim**-0.5
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            scaling,
            self.num_kv_heads,
            alibi_slopes,
            alibi_sqrt=True,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        layernorm: Optional[nn.Module] = None,
        disable_allreduce=False,
        user_output=None,
    ) -> torch.Tensor:
        del positions  # Unused.
        hidden_states = (
            layernorm(
                hidden_states,
                fp16_out=(
                    self.qkv_proj.quant_method.quant_config.weight_bits == 6
                    if getattr(self.qkv_proj.quant_method, "quant_config", None)
                    else False
                ),
            )
            if layernorm
            else hidden_states
        )
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v)
        residual, _ = self.o_proj(
            attn_output,
            residual,
            disable_allreduce=disable_allreduce,
            output=user_output,
        )
        return residual


class Step1DecoderLayer(nn.Module):

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.enable_cudagraph = not model_config.enforce_eager
        config = model_config.hf_config
        self.hidden_size = config.hidden_size
        self.self_attn = Step1Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_attention_groups,
            slopes=config.alibi_slopes,
            max_pos_interp_ratio=config.max_pos_interp_ratio,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        layer_idx = int(prefix.split("layers.")[1].split(".")[0])
        self.use_moe = (
            config.use_moe
            and (layer_idx + config.moe_layer_offset) % config.moe_every_n_layer
            == 0
        )

        if self.use_moe:
            self.moe = Step1MoEMLP(
                config.moe_num_experts,
                config.moe_top_k,
                config.moe_dynamic_exp_p,
                hidden_size=self.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                hidden_act="silu",
                quant_config=quant_config,
                prefix=f"{prefix}.moe",
                enable_cudagraph=self.enable_cudagraph,
            )
        else:
            self.mlp = Step1MLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act="silu",
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        ln_cls = OptimusRMSNorm if config.hidden_size % 64 == 0 else RMSNorm
        self.input_layernorm = ln_cls(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = ln_cls(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Self Attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            residual=hidden_states,
            layernorm=self.input_layernorm,
        )

        # Fully Connected
        def ffn_switch():
            return self.moe if self.use_moe else self.mlp

        hidden_states = ffn_switch()(
            hidden_states, hidden_states, self.post_attention_layernorm
        )
        return hidden_states


# @support_torch_compile
class Step1Model(nn.Module):

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        assert lora_config is None
        self.config = config
        self.allgather_dtype = None  # FIXME(ys): disable fp8 allgather
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Step1DecoderLayer(
                model_config=vllm_config.model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        ln_cls = OptimusRMSNorm if config.hidden_size % 64 == 0 else RMSNorm
        if get_pp_group().is_last_rank:
            self.norm = ln_cls(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states"], config.hidden_size
            )
        )

        self.sequence_parallel_threshold = (
            None if DISABLE_SEQUENCE_PARALLEL else SEQUENCE_PARALLEL_THRESHOLD
        )
        self.overlap_ratio = GEMM_COMM_OVERLAP_RATIO
        self.mlp_batch_size = MLP_BATCH_SIZE
        self.tp_size = get_tensor_model_parallel_world_size()
        self.use_moe = config.use_moe

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().world_size > 1:
            return self.forward_pp(
                input_ids, positions, intermediate_tensors, inputs_embeds
            )
        else:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            if self.use_moe:
                return self.forward_hidden_states_moe(hidden_states, positions)
            else:
                return self.forward_hidden_states(hidden_states, positions)

    def forward_pp(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
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

    def forward_hidden_states_moe(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        S = hidden_states.shape[0]
        if (
            self.tp_size > 1
            and self.sequence_parallel_threshold is not None
            and self.sequence_parallel_threshold < S
        ):
            if self.tp_size > 8:
                return self.forward_overlap_v2(hidden_states, positions)
            else:
                # TODO(xwx): overlap mlp layer of MoE model
                return self.forward_split_ffn(hidden_states, positions)
        else:
            for i in range(len(self.layers)):
                layer = self.layers[i]
                hidden_states = layer(
                    positions,
                    hidden_states,
                )
            hidden_states = self.norm(hidden_states)
            return hidden_states

    def forward_overlap_v2(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        del positions
        S = hidden_states.shape[0]
        tp_size = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()
        if S % tp_size != 0:
            # pad to multiple of tp_size with 0
            pad_len = tp_size - S % tp_size
            hidden_states = torch.cat(
                [
                    hidden_states,
                    torch.zeros(
                        pad_len,
                        hidden_states.shape[1],
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    ),
                ]
            )
            S = hidden_states.shape[0]
        else:
            pad_len = 0
        assert S % tp_size == 0
        hidden_states = hidden_states.view(S, -1)
        dim_0 = int(
            (S * self.overlap_ratio + tp_size - 1) // tp_size * tp_size
        )  # round up to multiple of tp_size
        buffer = torch.empty(
            S * int(self.config.intermediate_size / tp_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        mlp_buffer = buffer.view(S, -1)
        if tp_size > 8:
            assert (
                tp_size % 8 == 0
            ), f"tp_size should be an integer multiple of 8,but cur tp_size={tp_size}"
            kv_repeat = tp_size // 8
        else:
            kv_repeat = 1
        qkv_buffer = buffer[
            : S
            * int(
                (
                    self.config.num_attention_heads
                    + self.config.num_attention_groups * kv_repeat * 2
                )
                // tp_size
                * (self.config.hidden_size // self.config.num_attention_heads)
            )
        ].view(S, -1)
        chunk_size = S // tp_size
        residual = torch.empty(
            chunk_size,
            self.config.hidden_size,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        chunk_size_0 = dim_0 // tp_size
        chunk_size_1 = chunk_size - chunk_size_0
        residual_intersect_0 = residual[:chunk_size_0]
        residual_intersect_1 = residual[chunk_size_0:]
        hidden_states_intersect_0 = hidden_states[
            rank * chunk_size_0 : (rank + 1) * chunk_size_0
        ]
        hidden_states_intersect_1 = hidden_states[
            dim_0 + rank * chunk_size_1 : dim_0 + (rank + 1) * chunk_size_1
        ]
        s1 = torch.cuda.Stream(device=residual.device)

        for i in range(len(self.layers)):
            layer = self.layers[i]
            ffn = layer.moe if layer.use_moe else layer.mlp
            # Attention Forward
            residual_intersect_0.copy_(hidden_states_intersect_0)
            layer.input_layernorm(
                hidden_states[:dim_0], output=hidden_states[:dim_0]
            )
            layer.self_attn.qkv_proj(
                hidden_states[:dim_0], output=qkv_buffer[:dim_0]
            )
            with torch.cuda.stream(s1):
                residual_intersect_1.copy_(hidden_states_intersect_1)
                layer.input_layernorm(
                    hidden_states[dim_0:], output=hidden_states[dim_0:]
                )
                layer.self_attn.qkv_proj(
                    hidden_states[dim_0:], output=qkv_buffer[dim_0:]
                )
            torch.cuda.current_stream().wait_stream(s1)
            q, k, v = qkv_buffer.view(S, -1).split(
                [
                    layer.self_attn.q_size,
                    layer.self_attn.kv_size,
                    layer.self_attn.kv_size,
                ],
                dim=-1,
            )
            if pad_len > 0:
                attn_output = layer.self_attn.attn(
                    q[:-pad_len], k[:-pad_len], v[:-pad_len]
                )
                attn_output = torch.cat(
                    [
                        attn_output,
                        torch.zeros(
                            pad_len,
                            attn_output.shape[1],
                            dtype=attn_output.dtype,
                            device=attn_output.device,
                        ),
                    ],
                    dim=0,
                )
            else:
                attn_output = layer.self_attn.attn(q, k, v)
            hidden_states = hidden_states.view(S, -1)
            layer.self_attn.o_proj(
                attn_output[:dim_0],
                output=hidden_states[:dim_0],
                disable_allreduce=True,
            )
            hidden_states_intersect_0.add_(residual_intersect_0)
            torch.distributed.all_reduce(
                hidden_states[:dim_0],
                group=get_tensor_model_parallel_group().device_group,
            )
            with torch.cuda.stream(s1):
                layer.self_attn.o_proj(
                    attn_output[dim_0:],
                    output=hidden_states[dim_0:],
                    disable_allreduce=True,
                )
                hidden_states_intersect_1.add_(residual_intersect_1)
                torch.distributed.all_reduce(
                    hidden_states[dim_0:],
                    group=get_tensor_model_parallel_group().device_group,
                )
            del attn_output
            residual_intersect_0.copy_(hidden_states_intersect_0)
            layer.post_attention_layernorm(
                hidden_states[:dim_0], output=hidden_states[:dim_0]
            )
            num_batch_size = (
                dim_0 + self.mlp_batch_size - 1
            ) // self.mlp_batch_size
            for idx in range(num_batch_size):
                start = idx * self.mlp_batch_size
                end = min((idx + 1) * self.mlp_batch_size, dim_0)
                ffn(
                    hidden_states[start:end],
                    disable_allreduce=True,
                    user_output=hidden_states[start:end],
                )
                hidden_states_intersect_0.add_(residual_intersect_0)
            torch.distributed.all_reduce(
                hidden_states[:dim_0],
                group=get_tensor_model_parallel_group().device_group,
            )
            with torch.cuda.stream(s1):
                residual_intersect_1.copy_(hidden_states_intersect_1)
                layer.post_attention_layernorm(
                    hidden_states[dim_0:], output=hidden_states[dim_0:]
                )
                num_batch_size = (
                    S - dim_0 + self.mlp_batch_size - 1
                ) // self.mlp_batch_size
                for idx in range(num_batch_size):
                    start = dim_0 + idx * self.mlp_batch_size
                    end = dim_0 + min(
                        (idx + 1) * self.mlp_batch_size, S - dim_0
                    )
                    ffn(
                        hidden_states[start:end],
                        disable_allreduce=True,
                        user_output=hidden_states[start:end],
                    )
                hidden_states_intersect_1.add_(residual_intersect_1)
                torch.distributed.all_reduce(
                    hidden_states[dim_0:],
                    group=get_tensor_model_parallel_group().device_group,
                )
        torch.cuda.current_stream().wait_stream(s1)
        del buffer, mlp_buffer, qkv_buffer, residual
        self.norm(hidden_states, output=hidden_states)
        return hidden_states[: S - pad_len]

    def forward_split_ffn(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[0]
        tp_size = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()
        chunk_size = self.config.hidden_size // tp_size
        residual = torch.empty(
            seq_len,
            chunk_size,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states_intersect_0 = hidden_states.narrow(
                1, rank * chunk_size, chunk_size
            )
            residual.copy_(hidden_states_intersect_0)

            layer.input_layernorm(hidden_states, output=hidden_states)
            layer.self_attn(
                positions,
                hidden_states,
                residual=None,
                layernorm=None,
                disable_allreduce=True,
                user_output=hidden_states,
            )

            hidden_states_intersect_0.add_(residual)
            torch.distributed.all_reduce(
                hidden_states,
                group=get_tensor_model_parallel_group().device_group,
            )
            residual.copy_(hidden_states_intersect_0)

            layer.post_attention_layernorm(hidden_states, output=hidden_states)
            num_batch_size = (
                seq_len + self.mlp_batch_size - 1
            ) // self.mlp_batch_size
            hidden_states = hidden_states.view(seq_len, -1)
            for idx in range(num_batch_size):
                start = idx * self.mlp_batch_size
                end = min((idx + 1) * self.mlp_batch_size, seq_len)
                if layer.use_moe:
                    hidden_states[start:end] = layer.moe(
                        hidden_states[start:end], disable_allreduce=True
                    )
                else:
                    layer.mlp(
                        hidden_states[start:end],
                        disable_allreduce=True,
                        user_output=hidden_states[start:end],
                    )

            hidden_states_intersect_0.add_(residual)
            torch.distributed.all_reduce(
                hidden_states,
                group=get_tensor_model_parallel_group().device_group,
            )

        del residual
        self.norm(hidden_states, output=hidden_states)
        return hidden_states

    def forward_hidden_states(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        S = hidden_states.shape[0]
        if (
            self.tp_size > 1
            and self.sequence_parallel_threshold is not None
            and self.sequence_parallel_threshold < S
        ):
            tp_size = get_tensor_model_parallel_world_size()
            rank = get_tensor_model_parallel_rank()
            if S % tp_size != 0:
                # pad to multiple of tp_size with 0
                pad_len = tp_size - S % tp_size
                hidden_states = torch.cat(
                    [
                        hidden_states,
                        torch.zeros(
                            pad_len,
                            hidden_states.shape[1],
                            dtype=hidden_states.dtype,
                            device=hidden_states.device,
                        ),
                    ]
                )
                S = hidden_states.shape[0]
            else:
                pad_len = 0

            assert S % tp_size == 0
            chunk_size = S // tp_size
            residual = torch.empty(
                chunk_size,
                self.config.hidden_size,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            dim_0 = int(
                (S * self.overlap_ratio + tp_size - 1) // tp_size * tp_size
            )  # round up to multiple of tp_size
            dim_1 = S - dim_0

            mlp_dim = int(self.config.intermediate_size / tp_size)
            qkv_dim = int(
                (
                    self.config.num_attention_heads
                    + self.config.num_attention_groups * 2
                )
                // tp_size
                * (self.config.hidden_size // self.config.num_attention_heads)
            )

            if self.allgather_dtype is not None:
                fp8_dim = int(self.config.hidden_size / 2)
                max_buffer_dim = max(mlp_dim, qkv_dim, fp8_dim)
            else:
                max_buffer_dim = max(mlp_dim, qkv_dim)

            buffer = torch.empty(
                S * max_buffer_dim,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

            buffer_0 = buffer[: dim_0 * max_buffer_dim]
            buffer_1 = buffer[dim_0 * max_buffer_dim :]
            mlp_buffer_0 = buffer_0[: dim_0 * mlp_dim].view(dim_0, -1)
            mlp_buffer_1 = buffer_1[: dim_1 * mlp_dim].view(dim_1, -1)
            qkv_buffer = buffer[
                dim_0 * max_buffer_dim
                - dim_0 * qkv_dim : dim_0 * max_buffer_dim
                + dim_1 * qkv_dim
            ].view(S, -1)

            chunk_size_0 = dim_0 // tp_size
            chunk_size_1 = chunk_size - chunk_size_0
            residual_intersect_0 = residual[:chunk_size_0]
            hidden_states_0 = hidden_states[:dim_0]
            hidden_states_1 = hidden_states[dim_0:]
            hidden_states_intersect_0 = hidden_states[
                rank * chunk_size_0 : (rank + 1) * chunk_size_0
            ]
            residual_intersect_1 = residual[chunk_size_0:]
            hidden_states_intersect_1 = hidden_states[
                dim_0 + rank * chunk_size_1 : dim_0 + (rank + 1) * chunk_size_1
            ]

            if self.allgather_dtype is not None:
                hidden_states_fp8_0 = buffer_0[: dim_0 * fp8_dim]
                hidden_states_fp8_0 = torch.ops.OptimusFp8.as_type(
                    hidden_states_fp8_0, torch.uint8
                ).reshape(dim_0, self.config.hidden_size)
                hidden_states_fp8_1 = buffer_1[: dim_1 * fp8_dim]
                hidden_states_fp8_1 = torch.ops.OptimusFp8.as_type(
                    hidden_states_fp8_1, torch.uint8
                ).reshape(dim_1, self.config.hidden_size)
                hidden_states_fp8_intersect_0 = hidden_states_fp8_0[
                    rank * chunk_size_0 : (rank + 1) * chunk_size_0
                ]
                hidden_states_fp8_intersect_1 = hidden_states_fp8_1[
                    rank * chunk_size_1 : (rank + 1) * chunk_size_1
                ]

            s1 = torch.cuda.Stream(device=residual.device)

            for i in range(len(self.layers)):
                layer = self.layers[i]

                # Attention Forward
                if i == 0:
                    residual_intersect_0.copy_(hidden_states_intersect_0)
                    layer.input_layernorm(
                        hidden_states[:dim_0], output=hidden_states[:dim_0]
                    )
                    layer.self_attn.qkv_proj(
                        hidden_states[:dim_0], output=qkv_buffer[:dim_0]
                    )

                with torch.cuda.stream(s1):
                    if i == 0:
                        residual_intersect_1.copy_(hidden_states_intersect_1)
                        layer.input_layernorm(
                            hidden_states[dim_0:], output=hidden_states[dim_0:]
                        )
                    else:
                        if self.allgather_dtype is not None:
                            if self.allgather_dtype == "static_fp8e4m3":
                                qkv_input_scale_1 = torch.full(
                                    [1],
                                    layer.self_attn.qkv_proj.input_scales,
                                    device="cuda",
                                    dtype=torch.float32,
                                )
                                torch.ops.OptimusFp8.rms_norm_quantize_infer(
                                    residual_intersect_1,
                                    layer.input_layernorm.weight,
                                    qkv_input_scale_1,
                                    out=hidden_states_fp8_intersect_1,
                                )
                            elif self.allgather_dtype == "dynamic_fp8e4m3":
                                layer.input_layernorm(
                                    residual_intersect_1,
                                    output=hidden_states_intersect_1,
                                )
                                qkv_input_scale_1 = (
                                    dynamic_fp8_pertensor_quantize(
                                        hidden_states_intersect_1
                                    )
                                )
                                torch.distributed.all_reduce(
                                    qkv_input_scale_1,
                                    torch.distributed.ReduceOp.MIN,
                                    group=get_tensor_model_parallel_group().device_group,
                                )
                                hidden_states_fp8_intersect_1 = (
                                    torch.ops.OptimusFp8.as_type(
                                        hidden_states_fp8_intersect_1,
                                        torch.float8_e4m3fn,
                                    )
                                )
                                torch.ops.OptimusFp8.quantize(
                                    hidden_states_intersect_1,
                                    qkv_input_scale_1,
                                    out=hidden_states_fp8_intersect_1,
                                )
                                hidden_states_fp8_intersect_1 = (
                                    torch.ops.OptimusFp8.as_type(
                                        hidden_states_fp8_intersect_1,
                                        torch.uint8,
                                    )
                                )
                            else:
                                raise ValueError(
                                    f"Unsupported allgather_dtype: {self.allgather_dtype}"
                                )

                            torch.distributed.all_gather_into_tensor(
                                hidden_states_fp8_1,
                                hidden_states_fp8_intersect_1,
                                group=get_tensor_model_parallel_group().device_group,
                            )

                            hidden_states_fp8_1 = torch.ops.OptimusFp8.as_type(
                                hidden_states_fp8_1, torch.float8_e4m3fn
                            )
                            torch.ops.OptimusFp8.dequantize(
                                hidden_states_fp8_1,
                                qkv_input_scale_1.reciprocal(),
                                torch.bfloat16,
                                out=hidden_states[dim_0:],
                            )
                            hidden_states_fp8_1 = torch.ops.OptimusFp8.as_type(
                                hidden_states_fp8_1, torch.uint8
                            )
                        else:
                            layer.input_layernorm(
                                residual_intersect_1,
                                output=hidden_states_intersect_1,
                            )
                            torch.distributed.all_gather_into_tensor(
                                hidden_states[dim_0:],
                                hidden_states_intersect_1,
                                group=get_tensor_model_parallel_group().device_group,
                            )

                    layer.self_attn.qkv_proj(
                        hidden_states[dim_0:], output=qkv_buffer[dim_0:]
                    )

                torch.cuda.current_stream().wait_stream(s1)
                q, k, v = qkv_buffer.split(
                    [
                        layer.self_attn.q_size,
                        layer.self_attn.kv_size,
                        layer.self_attn.kv_size,
                    ],
                    dim=-1,
                )

                if pad_len > 0:
                    attn_output = layer.self_attn.attn(
                        q[: S - pad_len], k[: S - pad_len], v[: S - pad_len]
                    )
                    attn_output = torch.cat(
                        [
                            attn_output,
                            torch.zeros(
                                pad_len,
                                attn_output.shape[1],
                                dtype=attn_output.dtype,
                                device=attn_output.device,
                            ),
                        ],
                        dim=0,
                    )
                else:
                    attn_output = layer.self_attn.attn(q, k, v)

                layer.self_attn.o_proj(
                    attn_output[:dim_0],
                    output=hidden_states[:dim_0],
                    disable_allreduce=True,
                )
                hidden_states_intersect_0.add_(residual_intersect_0)

                torch.distributed.reduce_scatter_tensor(
                    residual_intersect_0,
                    hidden_states[:dim_0],
                    group=get_tensor_model_parallel_group().device_group,
                )

                with torch.cuda.stream(s1):
                    layer.self_attn.o_proj(
                        attn_output[dim_0:],
                        output=hidden_states[dim_0:],
                        disable_allreduce=True,
                    )
                    hidden_states_intersect_1.add_(residual_intersect_1)

                del attn_output

                if self.allgather_dtype is not None:
                    if self.allgather_dtype == "static_fp8e4m3":
                        gate_up_input_scale_0 = torch.full(
                            [1],
                            layer.mlp.gate_up_proj.input_scales,
                            device="cuda",
                            dtype=torch.float32,
                        )
                        torch.ops.OptimusFp8.rms_norm_quantize_infer(
                            residual_intersect_0,
                            layer.post_attention_layernorm.weight,
                            gate_up_input_scale_0,
                            out=hidden_states_fp8_intersect_0,
                        )
                    elif self.allgather_dtype == "dynamic_fp8e4m3":
                        layer.post_attention_layernorm(
                            residual_intersect_0,
                            output=hidden_states_intersect_0,
                        )
                        gate_up_input_scale_0 = dynamic_fp8_pertensor_quantize(
                            hidden_states_intersect_0
                        )
                        torch.distributed.all_reduce(
                            gate_up_input_scale_0,
                            torch.distributed.ReduceOp.MIN,
                            group=get_tensor_model_parallel_group().device_group,
                        )
                        hidden_states_fp8_intersect_0 = (
                            torch.ops.OptimusFp8.as_type(
                                hidden_states_fp8_intersect_0,
                                torch.float8_e4m3fn,
                            )
                        )
                        torch.ops.OptimusFp8.quantize(
                            hidden_states_intersect_0,
                            gate_up_input_scale_0,
                            out=hidden_states_fp8_intersect_0,
                        )
                        hidden_states_fp8_intersect_0 = (
                            torch.ops.OptimusFp8.as_type(
                                hidden_states_fp8_intersect_0, torch.uint8
                            )
                        )
                    else:
                        raise ValueError(
                            f"Unsupported allgather_dtype: {self.allgather_dtype}"
                        )

                    torch.distributed.all_gather_into_tensor(
                        hidden_states_fp8_0,
                        hidden_states_fp8_intersect_0,
                        group=get_tensor_model_parallel_group().device_group,
                    )
                else:
                    layer.post_attention_layernorm(
                        residual_intersect_0, output=hidden_states_intersect_0
                    )
                    torch.distributed.all_gather_into_tensor(
                        hidden_states[:dim_0],
                        hidden_states_intersect_0,
                        group=get_tensor_model_parallel_group().device_group,
                    )

                with torch.cuda.stream(s1):
                    torch.distributed.reduce_scatter_tensor(
                        residual_intersect_1,
                        hidden_states[dim_0:],
                        group=get_tensor_model_parallel_group().device_group,
                    )

                if self.allgather_dtype is not None:
                    hidden_states_fp8_0 = torch.ops.OptimusFp8.as_type(
                        hidden_states_fp8_0, torch.float8_e4m3fn
                    )
                    torch.ops.OptimusFp8.dequantize(
                        hidden_states_fp8_0,
                        gate_up_input_scale_0.reciprocal(),
                        torch.bfloat16,
                        out=hidden_states[:dim_0],
                    )
                    hidden_states_fp8_0 = torch.ops.OptimusFp8.as_type(
                        hidden_states_fp8_0, torch.uint8
                    )

                num_batch_size = (
                    dim_0 + self.mlp_batch_size - 1
                ) // self.mlp_batch_size
                for idx in range(num_batch_size):
                    start = idx * self.mlp_batch_size
                    end = min((idx + 1) * self.mlp_batch_size, dim_0)
                    w0_out_0, _ = layer.mlp.gate_up_proj(
                        hidden_states_0[start:end]
                    )
                    layer.mlp.act_fn(w0_out_0, output=mlp_buffer_0[start:end])
                del w0_out_0

                with torch.cuda.stream(s1):
                    if self.allgather_dtype is not None:
                        if self.allgather_dtype == "static_fp8e4m3":
                            gate_up_input_scale_1 = torch.full(
                                [1],
                                layer.mlp.gate_up_proj.input_scales,
                                device="cuda",
                                dtype=torch.float32,
                            )
                            torch.ops.OptimusFp8.rms_norm_quantize_infer(
                                residual_intersect_1,
                                layer.post_attention_layernorm.weight,
                                gate_up_input_scale_1,
                                out=hidden_states_fp8_intersect_1,
                            )
                        elif self.allgather_dtype == "dynamic_fp8e4m3":
                            layer.post_attention_layernorm(
                                residual_intersect_1,
                                output=hidden_states_intersect_1,
                            )
                            gate_up_input_scale_1 = (
                                dynamic_fp8_pertensor_quantize(
                                    hidden_states_intersect_1
                                )
                            )
                            torch.distributed.all_reduce(
                                gate_up_input_scale_1,
                                torch.distributed.ReduceOp.MIN,
                                group=get_tensor_model_parallel_group().device_group,
                            )
                            hidden_states_fp8_intersect_1 = (
                                torch.ops.OptimusFp8.as_type(
                                    hidden_states_fp8_intersect_1,
                                    torch.float8_e4m3fn,
                                )
                            )
                            torch.ops.OptimusFp8.quantize(
                                hidden_states_intersect_1,
                                gate_up_input_scale_1,
                                out=hidden_states_fp8_intersect_1,
                            )
                            hidden_states_fp8_intersect_1 = (
                                torch.ops.OptimusFp8.as_type(
                                    hidden_states_fp8_intersect_1, torch.uint8
                                )
                            )
                        else:
                            raise ValueError(
                                f"Unsupported allgather_dtype: {self.allgather_dtype}"
                            )

                        torch.distributed.all_gather_into_tensor(
                            hidden_states_fp8_1,
                            hidden_states_fp8_intersect_1,
                            group=get_tensor_model_parallel_group().device_group,
                        )
                    else:
                        layer.post_attention_layernorm(
                            residual_intersect_1,
                            output=hidden_states_intersect_1,
                        )
                        torch.distributed.all_gather_into_tensor(
                            hidden_states[dim_0:],
                            hidden_states_intersect_1,
                            group=get_tensor_model_parallel_group().device_group,
                        )

                layer.mlp.down_proj(
                    mlp_buffer_0,
                    output=hidden_states[:dim_0],
                    disable_allreduce=True,
                )

                hidden_states_intersect_0.add_(residual_intersect_0)

                if i < len(self.layers) - 1:
                    torch.distributed.reduce_scatter_tensor(
                        residual_intersect_0,
                        hidden_states[:dim_0],
                        group=get_tensor_model_parallel_group().device_group,
                    )
                else:
                    torch.distributed.all_reduce(
                        hidden_states[:dim_0],
                        group=get_tensor_model_parallel_group().device_group,
                    )

                with torch.cuda.stream(s1):
                    if self.allgather_dtype is not None:
                        hidden_states_fp8_1 = torch.ops.OptimusFp8.as_type(
                            hidden_states_fp8_1, torch.float8_e4m3fn
                        )
                        torch.ops.OptimusFp8.dequantize(
                            hidden_states_fp8_1,
                            gate_up_input_scale_1.reciprocal(),
                            torch.bfloat16,
                            out=hidden_states[dim_0:],
                        )
                        hidden_states_fp8_1 = torch.ops.OptimusFp8.as_type(
                            hidden_states_fp8_1, torch.uint8
                        )

                    num_batch_size = (
                        dim_1 + self.mlp_batch_size - 1
                    ) // self.mlp_batch_size
                    for idx in range(num_batch_size):
                        start = idx * self.mlp_batch_size
                        end = min((idx + 1) * self.mlp_batch_size, dim_1)
                        w0_out_1, _ = layer.mlp.gate_up_proj(
                            hidden_states_1[start:end]
                        )
                        layer.mlp.act_fn(
                            w0_out_1, output=mlp_buffer_1[start:end]
                        )
                    del w0_out_1

                if i < len(self.layers) - 1:
                    next_layer = self.layers[i + 1]
                    if self.allgather_dtype is not None:
                        if self.allgather_dtype == "static_fp8e4m3":
                            qkv_input_scale_0 = torch.full(
                                [1],
                                next_layer.self_attn.qkv_proj.input_scales,
                                device="cuda",
                                dtype=torch.float32,
                            )
                            torch.ops.OptimusFp8.rms_norm_quantize_infer(
                                residual_intersect_0,
                                next_layer.input_layernorm.weight,
                                qkv_input_scale_0,
                                out=hidden_states_fp8_intersect_0,
                            )
                        elif self.allgather_dtype == "dynamic_fp8e4m3":
                            next_layer.input_layernorm(
                                residual_intersect_0,
                                output=hidden_states_intersect_0,
                            )
                            qkv_input_scale_0 = dynamic_fp8_pertensor_quantize(
                                hidden_states_intersect_0
                            )
                            torch.distributed.all_reduce(
                                qkv_input_scale_0,
                                torch.distributed.ReduceOp.MIN,
                                group=get_tensor_model_parallel_group().device_group,
                            )
                            hidden_states_fp8_intersect_0 = (
                                torch.ops.OptimusFp8.as_type(
                                    hidden_states_fp8_intersect_0,
                                    torch.float8_e4m3fn,
                                )
                            )
                            torch.ops.OptimusFp8.quantize(
                                hidden_states_intersect_0,
                                qkv_input_scale_0,
                                out=hidden_states_fp8_intersect_0,
                            )
                            hidden_states_fp8_intersect_0 = (
                                torch.ops.OptimusFp8.as_type(
                                    hidden_states_fp8_intersect_0, torch.uint8
                                )
                            )
                        else:
                            raise ValueError(
                                f"Unsupported allgather_dtype: {self.allgather_dtype}"
                            )

                        torch.distributed.all_gather_into_tensor(
                            hidden_states_fp8_0,
                            hidden_states_fp8_intersect_0,
                            group=get_tensor_model_parallel_group().device_group,
                        )
                    else:
                        next_layer.input_layernorm(
                            residual_intersect_0,
                            output=hidden_states_intersect_0,
                        )
                        torch.distributed.all_gather_into_tensor(
                            hidden_states[:dim_0],
                            hidden_states_intersect_0,
                            group=get_tensor_model_parallel_group().device_group,
                        )

                with torch.cuda.stream(s1):
                    layer.mlp.down_proj(
                        mlp_buffer_1,
                        output=hidden_states[dim_0:],
                        disable_allreduce=True,
                    )

                    hidden_states_intersect_1.add_(residual_intersect_1)
                    if i < len(self.layers) - 1:
                        torch.distributed.reduce_scatter_tensor(
                            residual_intersect_1,
                            hidden_states[dim_0:],
                            group=get_tensor_model_parallel_group().device_group,
                        )
                    else:
                        torch.distributed.all_reduce(
                            hidden_states[dim_0:],
                            group=get_tensor_model_parallel_group().device_group,
                        )

                if i < len(self.layers) - 1:
                    next_layer = self.layers[i + 1]
                    if self.allgather_dtype is not None:
                        hidden_states_fp8_0 = torch.ops.OptimusFp8.as_type(
                            hidden_states_fp8_0, torch.float8_e4m3fn
                        )
                        torch.ops.OptimusFp8.dequantize(
                            hidden_states_fp8_0,
                            qkv_input_scale_0.reciprocal(),
                            torch.bfloat16,
                            out=hidden_states[:dim_0],
                        )
                        hidden_states_fp8_0 = torch.ops.OptimusFp8.as_type(
                            hidden_states_fp8_0, torch.uint8
                        )
                    next_layer.self_attn.qkv_proj(
                        hidden_states[:dim_0], output=qkv_buffer[:dim_0]
                    )

            torch.cuda.current_stream().wait_stream(s1)
            del buffer, residual
            self.norm(hidden_states, output=hidden_states)
            return hidden_states[: S - pad_len]
        else:
            for i in range(len(self.layers)):
                layer = self.layers[i]
                hidden_states = layer(
                    positions,
                    hidden_states,
                )
            hidden_states = self.norm(hidden_states)
            return hidden_states


class Step1PretrainedModel(nn.Module, SupportsPP):

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
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
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        params_need_to_load = []
        for name in params_dict:
            if not (
                "vision_model" in name
                or "latent_query_tokens" in name
                or "sam_model" in name
            ):
                params_need_to_load.append(name)
        params_need_to_load = set(params_need_to_load)

        if params_need_to_load != loaded_params:
            param_name_example = list(params_need_to_load - loaded_params)[0]
            raise RuntimeError(
                f"Some parameters like {param_name_example} are not in the checkpoint and will falsely use random initialization"
            )

    def load_fp8_input_scales(self, input_scales_path):
        for name, loaded_weight in fp8_input_scales_loader(input_scales_path):
            if name.startswith("reference_model."):
                name = name.replace("reference_model.", "")
            idx = int(name.split(".")[2])
            layer = self.model.layers[idx]
            if "qkv_proj" in name:
                layer.self_attn.qkv_proj.input_scales = loaded_weight[:].item()
            elif "gate_up_proj" in name:
                layer.mlp.gate_up_proj.input_scales = loaded_weight[:].item()


class Step1ForCausalLM(Step1PretrainedModel):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.model = Step1Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
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
                    # We need bigger padding if using lora for kernel
                    # compatibility
                    if not lora_config
                    else lora_config.lora_vocab_padding_size
                ),
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(
                self.unpadded_vocab_size,
                config.vocab_size,
                logit_scale,
                need_fp32_logits=False,
            )
            #self.sampler = get_sampler()
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
    ) -> Union[torch.Tensor, IntermediateTensors]:
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
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens


class Step1ForSequenceClassification(Step1PretrainedModel):
    """\
    Step1 Transformer with a sequence classification head.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):

        super().__init__()
        self.model = Step1Model(vllm_config, prefix)
        config = vllm_config.model_config.hf_config
        assert len(config.id2label.keys()) == config.num_labels
        if get_pp_group().is_last_rank:
            self.score = ReplicatedLinear(
                config.hidden_size, config.num_labels, bias=False
            )
            pooler_config = vllm_config.model_config.pooler_config
            self._pooler = Pooler.from_config_with_defaults(
                pooler_config,
                pooling_type=PoolingType.LAST,
                normalize=False,
                softmax=False,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
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
