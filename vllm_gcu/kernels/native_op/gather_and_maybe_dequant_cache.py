import torch
from vllm import _custom_ops as ops
from vllm_gcu.kernels.native_op.utils import register_native

@register_native("_C_cache_ops", "gather_and_maybe_dequant_cache")
def _ref_gather_and_maybe_dequant_cache(
    src_cache,
    dst,
    block_table,
    cu_seq_lens,
    token_to_seq,
    total_tokens,
    kv_cache_dtype,
    scale,
    seq_starts):

    batch_size = block_table.size(0)
    block_size = src_cache.size(1)
    seq_len_tensor = (cu_seq_lens[1:] - cu_seq_lens[:-1]).to(torch.int32)
    
    out_dtype = dst.dtype
    if seq_starts is None:
        offsets = torch.zeros((batch_size,), device=cu_seq_lens.device, dtype=torch.int64)
    else:
        offsets = seq_starts.to(torch.int64)
    tot_blocks_tensor = torch.zeros((batch_size,), device=cu_seq_lens.device, dtype=torch.int64)
    nonzero = seq_len_tensor > 0
    if nonzero.any():
        offs_nz = offsets[nonzero]
        lens_nz = seq_len_tensor[nonzero]
        block_start = offs_nz // block_size
        block_end = (offs_nz + lens_nz - 1) // block_size
        tot_blocks_tensor[nonzero] = (block_end - block_start + 1)

    expected_batches = []
    for b in range(batch_size):
        s = seq_len_tensor[b]
        if s == 0:
            continue
        tot = tot_blocks_tensor[b]
        blocks = block_table[b, :tot].tolist()

        gathered_rows = []
        for i in range(tot - 1):
            block_data = src_cache[blocks[i]]
            if kv_cache_dtype == "fp8":
                dequantized_block = torch.empty_like(block_data, dtype=dtype)
                ops.convert_fp8(dequantized_block, block_data, scale.item())
                gathered_rows.append(dequantized_block)
            else:
                gathered_rows.append(block_data)
        remaining = s - (tot - 1) * block_size
        last_block_data = src_cache[blocks[-1], :remaining, :]
        if kv_cache_dtype == "fp8":
            dequantized_last_block = torch.empty_like(last_block_data, dtype=dtype)
            ops.convert_fp8(dequantized_last_block, last_block_data, scale.item())
            gathered_rows.append(dequantized_last_block)
        else:
            gathered_rows.append(last_block_data)

        batch_expected = torch.cat(gathered_rows, dim=0)
        expected_batches.append(batch_expected)
    expected = torch.cat(expected_batches, dim=0)

    dst.copy_(expected, True)