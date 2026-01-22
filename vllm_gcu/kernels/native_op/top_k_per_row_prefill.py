import torch
from vllm_gcu.kernels.native_op.utils import register_native

@register_native("_C", "top_k_per_row_prefill")
def _ref_top_k_per_row_prefill(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, topk):
    topk_indices = logits.topk(min(topk, logits.shape[-1]),
                                dim=-1)[1]
    topk_indices -= row_starts[:, None]
    mask_lo = topk_indices >= 0
    mask_hi = topk_indices - (row_ends -
                                row_starts)[:, None] < 0
    mask = torch.full_like(topk_indices,
                            False,
                            dtype=torch.bool,
                            device=topk_indices.device)
    mask = mask_lo & mask_hi
    topk_indices = topk_indices.masked_fill(~mask, -1)
    indices[:, :topk_indices.shape[1]] = topk_indices.to(dtype=torch.int32)

