# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch_gcu
import vllm

from vllm_gcu.kernels import _custom_ops as ops
from vllm.platforms import current_platform


def _ref_convert_req_index_to_global_index(
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    max_num_blocks_per_req = block_table.shape[1]
    block_id = torch.div(token_indices, block_size, rounding_mode="floor")
    inblock_off = torch.remainder(token_indices, block_size)

    valid_block = block_id < max_num_blocks_per_req
    invalid = (token_indices < 0) | (~valid_block)

    block_id_safe = block_id.clamp(0, max_num_blocks_per_req - 1)
    base = block_table[req_id[:, None], block_id_safe]
    out = base * block_size + inblock_off
    out = torch.where(invalid, torch.full_like(out, -1), out)
    return out


@pytest.mark.parametrize("case", [
    # num_tokens, num_requests, max_num_blocks_per_req, num_topk_tokens,
    # block_size, block_n
    (128, 16, 8, 64, 64, 64),
    (512, 32, 16, 128, 64, 128),
    (1024, 64, 32, 256, 64, 128),
])
@pytest.mark.parametrize("add_invalid", [False, True])
@torch.inference_mode()
def test_convert_req_index_to_global_index(case, add_invalid):
    current_platform.seed_everything(0)
    torch.set_default_device("gcu")

    num_tokens, num_requests, max_num_blocks_per_req, num_topk_tokens, \
        block_size, block_n = case

    if num_topk_tokens % block_n != 0:
        pytest.skip("num_topk_tokens must be divisible by block_n")

    num_blocks = max_num_blocks_per_req * 2
    device = "gcu"

    req_id = torch.randint(0,
                           num_requests, (num_tokens, ),
                           dtype=torch.int32,
                           device=device)
    block_table = torch.randint(0,
                                num_blocks,
                                (num_requests, max_num_blocks_per_req),
                                dtype=torch.int32,
                                device=device)
    token_indices = torch.randint(0,
                                  max_num_blocks_per_req * block_size,
                                  (num_tokens, num_topk_tokens),
                                  dtype=torch.int32,
                                  device=device)

    if add_invalid:
        invalid_mask = torch.rand(token_indices.shape, device=device) < 0.05
        oob_mask = torch.rand(token_indices.shape, device=device) < 0.05

        token_indices = token_indices.clone()
        token_indices[invalid_mask] = -1

        if oob_mask.any():
            oob_vals = max_num_blocks_per_req * block_size + torch.randint(
                0,
                block_size,
                (int(oob_mask.sum().item()), ),
                dtype=torch.int32,
                device=device,
            )
            token_indices[oob_mask] = oob_vals

    prefill_workspace_request_ids = torch.empty((0, ),
                                                dtype=torch.int32,
                                                device=device)
    prefill_workspace_starts = torch.empty((0, ),
                                           dtype=torch.int32,
                                           device=device)
    has_prefill_workspace = False

    out = torch.empty_like(token_indices)
    torch.ops._C_cache_ops.convert_req_index_to_global_index(
        out,
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
        -1,
    )

    ref = _ref_convert_req_index_to_global_index(req_id, block_table,
                                                 token_indices, block_size)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)