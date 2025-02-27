import torch
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

ep_size = 16
num_experts = 256
expert_per_rank = num_experts // ep_size
topk_ids = torch.randint(0, num_experts-1, (8192, 8), dtype=torch.int32).gcu()

ep_split_size = torch.empty(
    [ep_size], dtype=torch.int32, device=topk_ids.device)
ep_token_indices = torch.zeros(
    [topk_ids.shape[0]*topk_ids.shape[1]], dtype=torch.int32, device=topk_ids.device)
send_token_total = torch.empty([1], dtype=torch.int32, device=topk_ids.device)
ops.get_ep_indices(ep_split_size, ep_token_indices,
                   send_token_total, topk_ids, expert_per_rank, ep_size)
print(ep_split_size)
print(ep_token_indices)
print(send_token_total)
