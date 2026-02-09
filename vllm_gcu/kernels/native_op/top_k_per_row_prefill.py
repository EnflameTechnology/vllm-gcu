import torch
from vllm_gcu.kernels.native_op.utils import register_native

@register_native("_C", "top_k_per_row_prefill")
def _ref_top_k_per_row_prefill(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    topk: int,
    threshold:float
):
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

