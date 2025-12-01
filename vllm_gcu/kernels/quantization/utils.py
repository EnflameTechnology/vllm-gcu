from typing import Optional
import torch
from vllm.model_executor.layers.quantization import (
    QUANTIZATION_METHODS,
    register_quantization_config,
)
from vllm.distributed import get_ep_group
from vllm_gcu.kernels._custom_ops import eplb_map_to_physical_and_record

def register_gcu_quantization_config(quantization: str):
    gcu_quantization = f"{quantization}_gcu"
    if gcu_quantization not in QUANTIZATION_METHODS:
        return register_quantization_config(gcu_quantization)


def register_weight_loader_v2_supported(cls):
    from vllm.model_executor.layers.linear import WEIGHT_LOADER_V2_SUPPORTED

    WEIGHT_LOADER_V2_SUPPORTED += [cls.__name__]
    return cls


def eplb_update(
    topk_ids: torch.Tensor,
    expert_load_view: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
    indices_type: Optional[torch.dtype] = None
):
    assert expert_load_view is not None
    assert logical_to_physical_map is not None
    assert logical_replica_count is not None

    ep_size = get_ep_group().world_size
    ep_rank = get_ep_group().rank_in_group
    num_local_physical_experts = expert_load_view.shape[0]
    num_physical_experts = num_local_physical_experts * ep_size
    global_expert_load = torch.zeros(num_physical_experts,
                                        device=expert_load_view.device,
                                        dtype=expert_load_view.dtype)

    topk_ids = eplb_map_to_physical_and_record(
        topk_ids=topk_ids,
        expert_load_view=global_expert_load,
        logical_to_physical_map=logical_to_physical_map,
        logical_replica_count=logical_replica_count,
        indices_type=indices_type,
    )

    expert_load_view += global_expert_load[
        ep_rank * num_local_physical_experts:
        (ep_rank + 1) * num_local_physical_experts]

    return topk_ids
