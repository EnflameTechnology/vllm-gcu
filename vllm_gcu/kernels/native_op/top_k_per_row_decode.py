import torch
from vllm_gcu.kernels.native_op.utils import register_native

@register_native("_C", "top_k_per_row_decode")
def _ref_top_k_per_row_decode(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    topk: int,
    threshold:float
):
    # padded query len
    current_device = logits.device
    batch_size = seq_lens.shape[0]
    padded_num_tokens = batch_size * next_n
    max_model_len = logits.shape[-1]

    positions = torch.arange(max_model_len,
                            device=current_device).unsqueeze(0).expand(
                            batch_size * next_n, -1)
    row_indices = torch.arange(padded_num_tokens,
                                device=current_device) // next_n
    next_n_offset = torch.arange(
        padded_num_tokens,
        device=current_device) % next_n
    index_end_pos = (seq_lens[row_indices] - next_n +
                        next_n_offset).unsqueeze(1)
    # index_end_pos: [B * N, 1]
    mask = positions <= index_end_pos
    # mask: [B * N, L]
    logits = logits.masked_fill(~mask, float('-inf'))
    topk_indices = logits.topk(topk,
                                dim=-1)[1].to(torch.int32)  # [B * N, K]
    # ensure we don't set indices for the top k
    # that is out of range(masked already)
    # this will happen if context length is shorter than K
    topk_indices[topk_indices > index_end_pos] = -1
    indices[:] = topk_indices
