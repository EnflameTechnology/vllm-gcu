import torch 
from vllm_gcu.kernels.native_op.utils import register_native

@register_native("_C_cache_ops", "convert_req_index_to_global_index")
def _ref_convert_req_index_to_global_index(
        out: torch.Tensor,
        req_id,
        block_table,
        token_indices,
        prefill_workspace_request_ids,
        prefill_workspace_starts,
        block_size,
        num_topk_tokens,
        block_n,
        has_prefill_workspace,
        None,
    ):
    """Reference implementation for triton_convert_req_index_to_global_index."""
    num_tokens = req_id.shape[0]
    max_blocks_per_req = block_table.shape[1]
    result = torch.empty_like(out, dtype=torch.int32, device=req_id.device)

    for token_id in range(num_tokens):
        req_id = req_id[token_id].item()

        # Determine if this token uses workspace or paged cache
        use_prefill_workspace = False
        workspace_start = 0
        if has_prefill_workspace and prefill_workspace_request_ids is not None:
            assert prefill_workspace_starts is not None
            prefill_req_id = prefill_workspace_request_ids[token_id].item()
            if prefill_req_id >= 0:
                use_prefill_workspace = True
                workspace_start = prefill_workspace_starts[prefill_req_id].item()

        for idx_id in range(num_topk_tokens):
            token_idx = token_indices[token_id, idx_id].item()

            if token_idx == -1:
                result[token_id, idx_id] = -1
            elif use_prefill_workspace:
                # Prefill + using prefill workspace: map to workspace offset
                result[token_id, idx_id] = workspace_start + token_idx
            else:
                # Decode: map to paged cache
                block_id = token_idx // block_size
                if block_id >= max_blocks_per_req:
                    result[token_id, idx_id] = -1
                else:
                    block_num = block_table[req_id, block_id].item()
                    offset = token_idx % block_size
                    result[token_id, idx_id] = block_num * block_size + offset 
    out.copy_(result)