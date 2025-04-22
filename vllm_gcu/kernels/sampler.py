from unittest.mock import patch

import torch

from vllm.model_executor.layers.utils import get_token_bin_counts_and_mask


def apply_penalties(
    logits: torch.Tensor,
    prompt_tokens_tensor: torch.Tensor,
    output_tokens_tensor: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> torch.Tensor:
    num_seqs, vocab_size = logits.shape
    _, prompt_mask = get_token_bin_counts_and_mask(
        prompt_tokens_tensor, vocab_size, num_seqs
    )
    output_bin_counts, output_mask = get_token_bin_counts_and_mask(
        output_tokens_tensor, vocab_size, num_seqs
    )
    repetition_penalties = repetition_penalties.unsqueeze(dim=1).repeat(1, vocab_size)
    mask0 = torch.where(
        (prompt_mask | output_mask) & (logits > 0), 1.0 / repetition_penalties, 1.0
    )
    mask1 = torch.where(
        (prompt_mask | output_mask) & (logits <= 0), repetition_penalties, 1.0
    )
    mask = mask0 * mask1
    logits *= mask

    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    logits -= frequency_penalties.unsqueeze(dim=1) * output_bin_counts
    logits -= presence_penalties.unsqueeze(dim=1) * output_mask
    return logits


patcher = patch("vllm.model_executor.layers.sampler.apply_penalties", apply_penalties)
patcher.start()
