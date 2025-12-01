from typing import Callable, List, Optional, Tuple

import torch

from vllm.config import get_current_vllm_config
from vllm.distributed import (get_dp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)

from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)

from vllm.model_executor.layers.fused_moe.layer import (FusedMoE,
                                                        UnquantizedFusedMoEMethod,
                                                        determine_expert_map)
from unittest.mock import patch
from vllm.model_executor.utils import set_weight_attrs

def get_expert_map(
    ep_size: int, ep_rank: int, global_num_experts: int,
    layer_prior_expert_map: Optional[torch.Tensor]
) -> Tuple[int, Optional[torch.Tensor]]:
    """Get the expert map for the current rank.
    Args:
        ep_rank: The rank of the current process.
        layer_prior_expert_map: The expert map from the prior knowledge.
    Returns:
        A tuple containing the local number of experts and the expert map.
    """
    if layer_prior_expert_map is not None:
        # If a prior expert map is provided, use it.
        ranks, expert_nums = layer_prior_expert_map.shape
        assert ep_size == ranks, "The expert ranks doesn't match ep_size."
        assert global_num_experts == expert_nums, \
            "The expert nums doesn't match global experts."

        expert_map = layer_prior_expert_map[ep_rank].to(torch.gcu.current_device())
        local_num_experts = torch.sum(torch.ne(expert_map, -1)).item()
        return (local_num_experts, expert_map)
    else:
        return determine_expert_map(ep_size, ep_rank, global_num_experts)


class PatchedFusedMoE(FusedMoE):
    """FusedMoE layer for MoE models.

    This layer contains both MergedColumnParallel weights (gate_up_proj /
    w13) and RowParallelLinear weights (down_proj/ w2).

    Note: Mixtral uses w1, w2, and w3 for gate, up, and down_proj. We
    copy that naming convention here and handle any remapping in the
    load_weights function in each model implementation.

    Args:
        num_experts: Number of experts in the model
        top_k: Number of experts selected for each token
        hidden_size: Input hidden state size of the transformer
        intermediate_size: Intermediate size of the experts
        params_dtype: Data type for the parameters.
        reduce_results: Whether to all all_reduce on the output of the layer
        renomalize: Whether to renormalize the logits in the fused_moe kernel
        quant_config: Quantization configure.
    """

    def __init__(
        self,
        num_experts: int,  # Global number of experts
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        ep_size: Optional[int] = None,
        dp_size: Optional[int] = None,
        prefix: str = "",
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        layer_prior_expert_map=None,
        with_b1: Optional[bool] = False,
        with_b2: Optional[bool] = False,
    ):
        torch.nn.Module.__init__(self)

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        # Use expert parallelism instead of tensor parallelism?
        vllm_config = get_current_vllm_config()

        # when this is in mtp, it may have different parallel state compared with target model.
        self.tp_size = (tp_size if tp_size is not None else
                        vllm_config.parallel_config.tensor_parallel_size)
        tp_rank = 0 if self.tp_size == 1 else get_tensor_model_parallel_rank()
        # same as tp_size
        self.dp_size = (dp_size
                        if dp_size is not None else vllm_config.parallel_config.data_parallel_size)
        self.dp_rank = (0
                        if self.dp_size == 1 else get_dp_group().rank_in_group)
        self.global_num_experts = num_experts

        # NOTE: vllm_gcu patch
        use_ep = (vllm_config.parallel_config.enable_expert_parallel
                  and self.tp_size * self.dp_size > 1)

        # For smuggling this layer into the fused moe custom op
        self.use_direct_call = self.dp_size == 1

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError("Duplicate layer name: {}".format(prefix))
        compilation_config.static_forward_context[prefix] = self
        self.layer_name = prefix

        if with_b1:
            intermediate_size_per_partition = intermediate_size // self.tp_size
            self.w13_bias = torch.nn.Parameter(
                torch.empty(num_experts, intermediate_size_per_partition * 2, dtype=params_dtype),
                requires_grad=False)
            # use shard_id to judge tp instead of output_dim
            set_weight_attrs(self.w13_bias, {
                "weight_loader": self._load_bias,
            })
        if with_b2:
            self.w2_bias = torch.nn.Parameter(
                torch.empty(num_experts, hidden_size, dtype=params_dtype),
                requires_grad=False)
            # use shard_id to judge tp instead of output_dim
            set_weight_attrs(self.w2_bias, {
                "weight_loader": self._load_bias,
            })

        if use_ep:
            # Set TP size to 1 to adjust for EP and adjust EP size and rank
            # for DP attention.
            self.ep_rank = tp_rank + self.tp_size * self.dp_rank
            self.tp_rank = 0
            self.ep_size = self.tp_size * self.dp_size
            self.tp_size = 1

            self.local_num_experts, self.expert_map = get_expert_map(
                                                        self.ep_size,
                                                        self.ep_rank,
                                                        self.global_num_experts,
                                                        layer_prior_expert_map
                                                      )
        else:
            # Adjust TP size for DP attention
            self.tp_rank = tp_rank + self.tp_size * self.dp_rank
            self.ep_rank = 0
            self.tp_size = self.tp_size * self.dp_size
            self.ep_size = 1
            self.local_num_experts = self.global_num_experts
            self.expert_map = None
        self.top_k = top_k
        self.global_num_experts = num_experts

        assert intermediate_size % self.tp_size == 0
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.e_score_correction_bias = e_score_correction_bias
        self.activation = f"{activation}_{prefix}"

        if self.scoring_func != "softmax" and not self.use_grouped_topk:
            raise ValueError("Only softmax scoring function is supported for "
                             "non-grouped topk.")

        # Note: get_quant_method will look at the layer's local_num_experts
        # for heuristic purposes, so it must be initialized first.
        if quant_config is None:
            self.quant_method: Optional[QuantizeMethodBase] = (
                UnquantizedFusedMoEMethod())
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix)
        assert self.quant_method is not None

        moe_quant_params = {
            "num_experts": self.local_num_experts,
            "hidden_size": hidden_size,
            "intermediate_size_per_partition":
            self.intermediate_size_per_partition,
            "params_dtype": params_dtype,
            "weight_loader": self.weight_loader,
        }
        # need full intermediate size pre-sharding for WNA16 act order
        if (self.quant_method.__class__.__name__
                in ("GPTQMarlinMoEMethod", "CompressedTensorsWNA16MoEMethod")):
            moe_quant_params["intermediate_size_full"] = intermediate_size

        self.quant_method.create_weights(layer=self, **moe_quant_params)
        self.use_direct_call = True

        # NOTE: vllm_gcu patch
        if self.dp_size > 1 and self.ep_size > 1:
            # ep impl is not for dpa
            self.dp_size = 1

    def _load_bias(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor, weight_name: str,
                      shard_id: str, expert_id: int) -> None:
        tp_rank = get_tensor_model_parallel_rank()
        if shard_id ==  "w1":
            shard_size = param.shape[1] // 2
            expert_data = param[expert_id].narrow(0, 0, shard_size)
            loaded_weight = loaded_weight.narrow(0, shard_size * tp_rank,
                                                 shard_size)
        elif shard_id == "w2":
            expert_data = param[expert_id]
        elif shard_id ==  "w3":
            shard_size = param.shape[1] // 2
            expert_data = param[expert_id].narrow(0, shard_size, shard_size)
            loaded_weight = loaded_weight.narrow(0, shard_size * tp_rank,
                                                 shard_size)
        else:
            raise ValueError(f"shard_id must be ['w1','w2','w3'] but "
                             f"got {shard_id}.")
        expert_data.copy_(loaded_weight)

patcher1 = patch("vllm.model_executor.layers.fused_moe.layer.FusedMoE", PatchedFusedMoE)
patcher1.start()
patcher2 = patch("vllm.model_executor.layers.fused_moe.FusedMoE", PatchedFusedMoE)
patcher2.start()
