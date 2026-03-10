#!/usr/bin/env python
# coding=utf-8
import pytest
import torch
from unittest.mock import Mock, patch

import vllm
from vllm_gcu.kernels.routing_strategy import DistributionBalanceRouting


class TestDistributionBalanceRoutingExpertBalance:

    @pytest.fixture
    def mock_ep_group(self):
        mock_group = Mock()
        mock_group.rank = 0
        mock_group.world_size = 2
        return mock_group

    @pytest.fixture
    def routing_strategy(self, mock_ep_group):
        with patch('vllm_gcu.kernels.routing_strategy.get_ep_group',
                   return_value=mock_ep_group):
            strategy = DistributionBalanceRouting()
            return strategy

    def test_expert_distribution_single_rank(self, routing_strategy):
        routing_strategy.ep_rank = 0
        routing_strategy.ep_size = 1
        routing_strategy.num_tokens = torch.empty(2,
                                                  device="cuda",
                                                  dtype=torch.int32)

        routing_strategy.ep_group.all_reduce = Mock(
            return_value=torch.tensor([0, 100], device="cuda"))

        num_tokens = 64
        num_experts = 256
        top_k = 8

        hidden_states = torch.randn(num_tokens, 64, device="cuda")
        router_logits = torch.randn(num_tokens, num_experts, device="cuda")

        _, topk_ids = routing_strategy.route_tokens(hidden_states,
                                                    router_logits, top_k)

        expert_counts = torch.zeros(num_experts, device="cuda")
        for i in range(num_tokens):
            for j in range(top_k):
                expert_id = topk_ids[i, j]
                expert_counts[expert_id] += 1

        total_selections = num_tokens * top_k
        expected_per_expert = total_selections / num_experts

        for expert_id in range(num_experts):
            count = expert_counts[expert_id]
            assert abs(count - expected_per_expert) <= expected_per_expert * 0.1, \
                f"expert {expert_id} was chosen {count} times，expected {expected_per_expert}"

    def test_expert_distribution_multiple_ranks(self, mock_ep_group):
        for rank in [0, 1]:
            mock_ep_group.rank = rank
            mock_ep_group.world_size = 2

            with patch('vllm_gcu.kernels.routing_strategy.get_ep_group',
                       return_value=mock_ep_group):
                strategy = DistributionBalanceRouting()

                num_tokens_per_rank = [64, 128]
                num_tokens = num_tokens_per_rank[rank]

                strategy.ep_group.all_reduce = Mock(
                    return_value=torch.tensor([0, 64, 128], device="cuda"))

                num_experts = 256
                top_k = 8

                hidden_states = torch.randn(num_tokens, 64, device="cuda")
                router_logits = torch.randn(num_tokens,
                                            num_experts,
                                            device="cuda")

                _, topk_ids = strategy.route_tokens(hidden_states,
                                                    router_logits, top_k)

                if rank == 0:
                    topk_ids_rank0 = topk_ids
                else:
                    topk_ids_rank1 = topk_ids

        all_topk_ids = torch.cat([topk_ids_rank0, topk_ids_rank1], dim=0)
        total_tokens = 64 + 128

        expert_counts = torch.zeros(num_experts, device="cuda")
        for i in range(total_tokens):
            for j in range(top_k):
                expert_id = all_topk_ids[i, j]
                expert_counts[expert_id] += 1

        total_selections = total_tokens * top_k
        expected_per_expert = total_selections / num_experts

        for expert_id in range(num_experts):
            count = expert_counts[expert_id]
            assert abs(count - expected_per_expert) <= expected_per_expert * 0.1, \
                f"expert {expert_id} was chosen {count} times，expected {expected_per_expert}"
