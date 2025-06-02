import pytest
import torch
import torch_gcu
import vllm
import vllm_gcu.kernels._custom_ops


def ref_advance_step_flashattn(num_seqs: int, num_queries: int, block_size: int,
                               input_tokens: torch.Tensor,
                               sampled_token_ids: torch.Tensor,
                               input_positions: torch.Tensor,
                               seq_lens: torch.Tensor,  # int
                               slot_mapping: torch.Tensor,
                               block_tables: torch.Tensor  # int
                               ) -> None:
    """Advance a step on GPU for existing inputs for a multi-step runner"""
    if hasattr(input_tokens, "is_gcu") and input_tokens.is_gcu:
        input_tokens[0:num_queries].copy_(sampled_token_ids[:, 0])
        seq_lens = seq_lens[0:num_queries]
        block_index = seq_lens // block_size
        block_offset = seq_lens % block_size
        tmp = torch.gather(block_tables[0:num_queries], dim=1,
                           index=block_index.long().view(-1, 1)).squeeze(1)
        slot_mapping[0:num_queries].copy_(tmp * block_size + block_offset)
        input_positions[0:num_queries].copy_(seq_lens)
        seq_lens.add_(1)
        return


@pytest.mark.parametrize("num_seqs", [16, 32, 64])
@pytest.mark.parametrize("pad", [0, 3])  # pad<num_seqs
@pytest.mark.parametrize("max_seq_len", [4096, 128*1024])
@pytest.mark.parametrize("block_size", [16, 64])
def test_advance_step(
    num_seqs,
    pad,
    max_seq_len,
    block_size,
):
    num_queries = num_seqs - pad
    block_num = max_seq_len // block_size
    # does not matter
    total_block_num = 5555
    vocab = 65536
    input_tokens = torch.randint(
        0, vocab, (num_seqs,), dtype=torch.int64).gcu()
    sampled_token_ids = torch.randint(
        0, vocab, (num_queries, 1), dtype=torch.int64).gcu()
    input_positions = torch.randint(
        0, max_seq_len, (num_seqs,), dtype=torch.int64).gcu()
    seq_lens = torch.randint(
        1, max_seq_len, (num_seqs,), dtype=torch.int32).gcu()
    slot_mapping = torch.randint(
        1, max_seq_len, (num_seqs,), dtype=torch.int64).gcu()
    block_tables = torch.randint(
        0, total_block_num, (num_seqs, block_num), dtype=torch.int32).gcu()
    input_tokens_ref = input_tokens.clone()
    sampled_token_ids_ref = sampled_token_ids.clone()
    input_positions_ref = input_positions.clone()
    seq_lens_ref = seq_lens.clone()
    slot_mapping_ref = slot_mapping.clone()
    block_tables_ref = block_tables.clone()
    ref_advance_step_flashattn(num_seqs, num_queries, block_size, input_tokens_ref,
                               sampled_token_ids_ref, input_positions_ref, seq_lens_ref, slot_mapping_ref, block_tables_ref)
    torch.ops._C.advance_step_flashattn(num_seqs, num_queries, block_size, input_tokens,
                           sampled_token_ids, input_positions, seq_lens, slot_mapping, block_tables)
    assert torch.allclose(input_tokens_ref, input_tokens, rtol=1e-2, atol=1e-2)
    assert torch.allclose(sampled_token_ids_ref,
                          sampled_token_ids, rtol=1e-2, atol=1e-2)
    assert torch.allclose(input_positions_ref,
                          input_positions, rtol=1e-2, atol=1e-2)
    assert torch.allclose(seq_lens_ref, seq_lens, rtol=1e-2, atol=1e-2)
    assert torch.allclose(slot_mapping_ref, slot_mapping, rtol=1e-2, atol=1e-2)
    assert torch.allclose(block_tables_ref, block_tables, rtol=1e-2, atol=1e-2)
