import torch
from unittest.mock import patch
import triton
import triton.language as tl
from vllm.utils import direct_register_custom_op
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding
)
from vllm.distributed import tensor_model_parallel_all_reduce
import vllm_gcu.envs as gcu_envs
try:
    import triton_gcu.triton
    USE_TRITON_GCU = True & gcu_envs.VLLM_GCU_TRITON_EAGLE
except:
    USE_TRITON_GCU = False

@triton.jit
def masked_input_kernel(
    input_ptr,
    org_vocab_start_index,
    org_vocab_end_index,
    num_org_vocab_padding,
    added_vocab_start_index,
    added_vocab_end_index,
    output_ptr,
    mask_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0)
    
    org_vocab_mask = (input_vals >= org_vocab_start_index) & (input_vals < org_vocab_end_index)
    added_vocab_mask = (input_vals >= added_vocab_start_index) & (input_vals < added_vocab_end_index)

    added_offset = added_vocab_start_index - (org_vocab_end_index - org_vocab_start_index) - num_org_vocab_padding
    valid_offset = (org_vocab_start_index * org_vocab_mask) + (added_offset * added_vocab_mask)

    vocab_mask = org_vocab_mask | added_vocab_mask
    masked_input = vocab_mask * (input_vals - valid_offset)
    inverse_mask = ~vocab_mask
    
    tl.store(output_ptr + offsets, masked_input, mask=mask)
    tl.store(mask_ptr + offsets, inverse_mask, mask=mask)


def get_masked_input_and_mask_triton(
    input_: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_elements = input_.numel()

    output = torch.empty_like(input_)
    mask = torch.empty_like(input_, dtype=torch.bool)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    masked_input_kernel[grid](
        input_,
        org_vocab_start_index,
        org_vocab_end_index,
        num_org_vocab_padding,
        added_vocab_start_index,
        added_vocab_end_index,
        output,
        mask,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output, mask


def get_masked_input_and_mask_triton_fake(
    input_: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(input_)
    mask = torch.empty_like(input_, dtype=torch.bool)
    return output, mask


direct_register_custom_op(
        op_name="get_masked_input_and_mask_triton",
        op_func=get_masked_input_and_mask_triton,
        mutates_args=[],
        fake_impl=get_masked_input_and_mask_triton_fake,
    )


def forward_oot(self, input_):
    if self.tp_size > 1:
        # Build the mask.
        masked_input, input_mask = torch.ops.vllm.get_masked_input_and_mask_triton(
            input_, self.shard_indices.org_vocab_start_index,
            self.shard_indices.org_vocab_end_index,
            self.shard_indices.num_org_vocab_padding,
            self.shard_indices.added_vocab_start_index,
            self.shard_indices.added_vocab_end_index)
    else:
        masked_input = input_
    # Get the embeddings.
    output_parallel = self.quant_method.embedding(self,
                                                    masked_input.long())
    # Mask the output embedding.
    if self.tp_size > 1:
        output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
    # Reduce across all the model parallel GPUs.
    output = tensor_model_parallel_all_reduce(output_parallel)
    return output

if USE_TRITON_GCU:
    VocabParallelEmbedding.forward_oot = forward_oot

