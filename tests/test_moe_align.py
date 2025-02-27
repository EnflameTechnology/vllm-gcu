import torch
import torch_gcu
import vllm
from vllm_gcu.kernels import _custom_ops as ops


block_size = 64
num_experts = 256
topk_ids = torch.randint(0, num_experts-1, (8192,8),dtype=torch.int32).gcu()
topk_ids_size = torch.tensor([4],dtype=torch.int32,device='gcu')

max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
sorted_ids = torch.empty(
    (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
)
sorted_ids.fill_(topk_ids.numel())
max_num_m_blocks = max_num_tokens_padded // block_size
expert_ids = torch.zeros(
    (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
)
num_tokens_post_pad = torch.zeros((1), dtype=torch.int32, device=topk_ids.device)

ops.moe_align_block_size_pad(
    topk_ids,
    topk_ids_size,
    num_experts,
    block_size,
    sorted_ids,
    expert_ids,
    num_tokens_post_pad,
)
print('topk_ids',topk_ids.shape, 'valid',topk_ids[0:4])
print('topk_ids_size',topk_ids_size)
print('num_experts',num_experts)
print('block_size',block_size)
print('---pad---')
print('num_tokens_post_pad', num_tokens_post_pad)
print('sorted_ids first block',sorted_ids[:block_size])
print('expert_ids',expert_ids)

ops.moe_align_block_size(
    topk_ids[0:topk_ids_size],
    num_experts,
    block_size,
    sorted_ids,
    expert_ids,
    num_tokens_post_pad,
)
print('---origin---')
print('num_tokens_post_pad', num_tokens_post_pad)
print('sorted_ids first block',sorted_ids[:block_size])
print('expert_ids',expert_ids)