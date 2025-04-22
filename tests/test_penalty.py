#!/usr/bin/env python
# coding=utf-8
import pytest
import torch
import torch_gcu  # noqa


def get_token_bin_counts_and_mask(
    tokens: torch.Tensor,
    vocab_size: int,
    num_seqs: int,
):
    bin_counts = torch.zeros(
        (num_seqs, vocab_size + 1), dtype=torch.long, device=tokens.device
    )
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
    bin_counts = bin_counts[:, :vocab_size]
    mask = bin_counts > 0

    return bin_counts, mask


def ref_penalty(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
):
    logits[logits > 0] /= torch.where(
        prompt_mask | output_mask, repetition_penalties, 1.0
    )[logits > 0]
    logits[logits <= 0] *= torch.where(
        prompt_mask | output_mask, repetition_penalties, 1.0
    )[logits <= 0]

    return logits


@pytest.mark.parametrize("num_seqs", [4])
@pytest.mark.parametrize("vocab_size", [129280])
@pytest.mark.parametrize("prompt_length", [7])
@pytest.mark.parametrize("output_length", [1])
def test_penalty(num_seqs, vocab_size, prompt_length, output_length):
    logits = torch.rand((num_seqs, vocab_size)).to("gcu")
    repetition_penalties = torch.tensor(1.1).to("gcu")

    prompt_tokens_tensor = torch.randint(0, vocab_size, [num_seqs, prompt_length]).to(
        "gcu"
    )
    _, prompt_mask = get_token_bin_counts_and_mask(
        prompt_tokens_tensor, vocab_size, num_seqs
    )

    output_tokens_tensor = torch.randint(0, vocab_size, [num_seqs, 1]).to("gcu")
    output_bin_counts, output_mask = get_token_bin_counts_and_mask(
        output_tokens_tensor, vocab_size, num_seqs
    )

    ref_logits = logits.clone()
    ref = ref_penalty(ref_logits, prompt_mask, output_mask, repetition_penalties)

    test_logits = logits.clone()
    mask0 = torch.where(
        (prompt_mask | output_mask) & (logits > 0), 1.0 / repetition_penalties, 1.0
    )
    mask1 = torch.where(
        (prompt_mask | output_mask) & (logits <= 0), repetition_penalties, 1.0
    )
    mask = mask0 * mask1
    test = ref_logits * mask

    assert (ref - test).sum() == 0
