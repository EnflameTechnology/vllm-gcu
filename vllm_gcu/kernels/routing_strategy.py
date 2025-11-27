#!/usr/bin/env python
# coding=utf-8
from typing import Optional
import torch

from vllm.distributed import get_ep_group
from vllm.model_executor.layers.fused_moe.routing_simulator import RoutingStrategy, RoutingSimulator


class DistributionBalanceRouting(RoutingStrategy):

    def __init__(self):
        self.ep_group = get_ep_group()
        self.ep_rank = self.ep_group.rank
        self.ep_size = self.ep_group.world_size
        self.num_tokens = torch.empty(self.ep_size + 1,
                                      device="cuda",
                                      dtype=torch.int32)

    def route_tokens(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        indices_type: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = router_logits.device
        num_tokens = hidden_states.shape[0]
        num_experts = router_logits.shape[-1]

        if indices_type is None:
            indices_type = torch.int32

        self.num_tokens.fill_(0)
        self.num_tokens[self.ep_rank + 1] = num_tokens
        num_tokens_across_ep = self.ep_group.all_reduce(self.num_tokens)
        cu_num_tokens_across_ep = torch.cumsum(num_tokens_across_ep, dim=0)

        num_groups = num_experts // top_k
        start_idx = cu_num_tokens_across_ep[self.ep_rank]
        token_indices = torch.arange(num_tokens, device=device)
        group_ids = ((start_idx + token_indices) % num_groups).unsqueeze(1)
        topk_ids = group_ids * top_k + torch.arange(top_k, device=device)

        topk_weights = torch.ones(
            (num_tokens, top_k),
            dtype=torch.float32,
            device=device,
        )

        return topk_weights, topk_ids.to(indices_type)


RoutingSimulator.register_strategy("balance", DistributionBalanceRouting)
