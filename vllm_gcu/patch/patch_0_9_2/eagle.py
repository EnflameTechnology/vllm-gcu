from unittest.mock import patch
import torch

def prepare_eagle_input_sequential(out_tensor: torch.Tensor,
                                   cu_query_lens: torch.Tensor,
                                   cu_num_tokens: torch.Tensor,
                                   block_size: int):
    num_programs = len(cu_num_tokens) - 1
    for pid in range(num_programs):
        start_pos = cu_num_tokens[pid].item()
        end_pos = cu_num_tokens[pid + 1].item()
        num_tokens = end_pos - start_pos
        index_start = cu_query_lens[pid].item()
        num_blocks = int(
            torch.ceil(torch.tensor(num_tokens / block_size)).item())

        for i in range(num_blocks):
            offset_tensor = torch.arange(0,
                                         block_size,
                                         dtype=torch.int32,
                                         device=out_tensor.device)
            global_start_offset = i * block_size
            target_indices = torch.tensor(
                start_pos + global_start_offset,
                dtype=torch.int32,
                device=out_tensor.device) + offset_tensor
            values_to_store = torch.tensor(
                index_start, dtype=torch.int32,
                device=out_tensor.device) + offset_tensor
            mask = (target_indices >= start_pos) & \
                   (target_indices < end_pos) & \
                   (offset_tensor < num_tokens)
            out_tensor[target_indices[mask]] = values_to_store[mask]

@staticmethod
def prepare_inputs(
    # [batch_size + 1]
    cu_target_query_lens: torch.Tensor,
    # [batch_size]
    num_rejected_tokens: torch.Tensor,
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # cu_target_query_lens: [0, a, a + b, a + b + c]
    # num_rejected_tokens: [n1, n2, n3]
    # num_tokens_per_req: [a - n1, b - n2, c - n3]
    # cu_num_tokens: [0, a - n1, a + b - n1 - n2, a + b + c - n1 - n2 - n3]
    # token_indices: [0, 1, ..., a - n1 - 1,
    #                 a, a + 1, ..., a + b - n2 - 1,
    #                 a + b, a + b + 1, ..., a + b + c - n3 - 1]

    # [0, a, a + b, a + b + c] -> [a, b, c]
    query_len_per_req = (cu_target_query_lens[1:] -
                            cu_target_query_lens[:-1])
    # [a, b, c] -> [a - n1, b - n2, c - n3]
    num_tokens_per_req = query_len_per_req - num_rejected_tokens

    # [a - n1, b - n2, c - n3] ->
    # [0, a - n1, a + b - n1 - n2, a + b + c - n1 - n2 - n3]
    cu_num_tokens = torch.zeros_like(cu_target_query_lens)
    torch.cumsum(num_tokens_per_req, dim=0, out=cu_num_tokens[1:])
    token_indices = torch.empty(
        num_tokens,
        dtype=torch.int32,
        device=cu_target_query_lens.device,
    )
    BLOCK_SIZE = 1024
    prepare_eagle_input_sequential(
        token_indices,
        cu_target_query_lens,
        cu_num_tokens,
        block_size=BLOCK_SIZE,
    )
    return cu_num_tokens, token_indices

patch("vllm.v1.spec_decode.eagle.EagleProposer.prepare_inputs", prepare_inputs).start()