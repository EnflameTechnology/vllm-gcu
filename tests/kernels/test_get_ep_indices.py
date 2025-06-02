import torch
import pytest
from vllm_gcu.kernels import _custom_ops as ops

def get_ep_indices_ref(topk_ids, expert_per_rank, ep_size):
    idx = torch.arange(0, topk_ids.shape[0], device=topk_ids.device)
    idx_list = []
    count_list = []
    for i in range(ep_size):
        mask = torch.logical_and(
            topk_ids >= (expert_per_rank * i),
            topk_ids < (expert_per_rank * (i + 1)),
        )
        mask = mask.sum(dim=1) > 0
        count_list.append(mask.sum((0,), keepdim=True))
        idx_list.append(idx[mask])
    ep_count = torch.cat(count_list)
    ep_token_indices = torch.cat(idx_list)
    return ep_token_indices, ep_count

@pytest.mark.parametrize("batch_size,top_k", [
    (8192, 8),
    (4096, 4),
    (2048, 2),
])
def test_get_ep_indices(batch_size, top_k):
    ep_size = 16
    num_experts = 256
    expert_per_rank = num_experts // ep_size
    
    # Generate random topk_ids
    topk_ids = torch.randint(0, num_experts-1, (batch_size, top_k), dtype=torch.int32).gcu()
    
    # Reference implementation
    ref_ep_token_indices, ref_ep_count = get_ep_indices_ref(topk_ids, expert_per_rank, ep_size)
    
    # Custom op implementation
    ep_split_size = torch.empty([ep_size], dtype=torch.int32, device=topk_ids.device)
    ep_token_indices = torch.zeros(
        [topk_ids.shape[0]*topk_ids.shape[1]], dtype=torch.int32, device=topk_ids.device)
    send_token_total = torch.empty([1], dtype=torch.int32, device=topk_ids.device)
    
    ops.get_ep_indices(ep_split_size, ep_token_indices,
                      send_token_total, topk_ids, expert_per_rank, ep_size)
    
    # Compare results
    # Note: You might need to adjust the comparison based on how the custom op formats its output
    assert torch.allclose(ep_split_size, ref_ep_count.to(torch.int32))
    assert send_token_total.item() == ref_ep_token_indices.shape[0]
    
    # Check that all indices are accounted for
    actual_indices = ep_token_indices[:send_token_total.item()]
    assert torch.all(torch.sort(actual_indices)[0] == torch.sort(ref_ep_token_indices)[0])

@pytest.mark.parametrize("num_experts,ep_size", [
    (256, 16),
    (128, 8),
    (64, 4),
])
def test_different_configs(num_experts, ep_size):
    batch_size = 4096
    top_k = 4
    expert_per_rank = num_experts // ep_size
    
    topk_ids = torch.randint(0, num_experts-1, (batch_size, top_k), dtype=torch.int32).gcu()
    
    # Reference and custom op implementations
    ref_ep_token_indices, ref_ep_count = get_ep_indices_ref(topk_ids, expert_per_rank, ep_size)
    
    ep_split_size = torch.empty([ep_size], dtype=torch.int32, device=topk_ids.device)
    ep_token_indices = torch.zeros(
        [topk_ids.shape[0]*topk_ids.shape[1]], dtype=torch.int32, device=topk_ids.device)
    send_token_total = torch.empty([1], dtype=torch.int32, device=topk_ids.device)
    
    ops.get_ep_indices(ep_split_size, ep_token_indices,
                      send_token_total, topk_ids, expert_per_rank, ep_size)
    
    # Validate results
    assert torch.allclose(ep_split_size, ref_ep_count.to(torch.int32))
    assert send_token_total.item() == ref_ep_token_indices.shape[0]
    actual_indices = ep_token_indices[:send_token_total.item()]
    assert torch.all(torch.sort(actual_indices)[0] == torch.sort(ref_ep_token_indices)[0])
