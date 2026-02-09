import torch 
from vllm_gcu.kernels.native_op.utils import register_native

@register_native("_C_cache_ops", "convert_req_index_to_global_index")
def _ref_convert_req_index_to_global_index(
        out: torch.Tensor,
        req_id: torch.Tensor,
        block_table: torch.Tensor,
        token_indices: torch.Tensor,
        prefill_workspace_request_ids: torch.Tensor,
        prefill_workspace_starts: torch.Tensor,
        block_size: int,
        num_topk_tokens: int,
        block_n: int,
        has_prefill_workspace: bool,
        seq_lens: torch.Tensor,
        threshold:int
    ):
    """Reference implementation for triton_convert_req_index_to_global_index."""
    # out[i][j] = block_table[req_id[i], block_id[i][j]] * block_size + inblock_off[i][j]
    
    # block_id = token_indices // block_size
    block_id = torch.div(token_indices, block_size, rounding_mode="floor")
    # inblock_off = token_indices % block_size
    inblock_off  = torch.remainder(token_indices, block_size)

    max_num_blocks_per_req = block_table.shape[1]
    valid_block = block_id < max_num_blocks_per_req
    invalid = (token_indices < 0) | (~valid_block)

    block_id_safe = block_id.clamp(0, max_num_blocks_per_req - 1)

    base = block_table[req_id[:, None], block_id_safe]
    tmp_out = base * block_size + inblock_off 

    tmp_out = torch.where(invalid, torch.full_like(tmp_out, -1), tmp_out)
    out.copy_(tmp_out)