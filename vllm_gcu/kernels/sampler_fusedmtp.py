import torch
from typing import Optional, Union
from vllm.v1.sample.ops.penalties import _convert_to_tensors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.config import LogprobsMode
from vllm_gcu.kernels.sampler import GCUSampler


class FusedMTPSampler(GCUSampler):

    def __init__(self, logprobs_mode: LogprobsMode = "raw_logprobs", spec_k: int = 0):
        super().__init__(logprobs_mode)
        self.spec_k = spec_k

    def apply_penalties(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        if not sampling_metadata.no_penalties:
            assert sampling_metadata.prompt_token_ids is not None
            logits = self.apply_all_penalties_fusedmtp(
                logits,
                sampling_metadata.prompt_token_ids,
                sampling_metadata.presence_penalties,
                sampling_metadata.frequency_penalties,
                sampling_metadata.repetition_penalties,
                sampling_metadata.output_token_ids,
            )
        return logits

    def apply_penalties_fused_mtp_torch(
        self,
        logits: torch.Tensor,
        repetition_penalties: torch.Tensor,
        frequency_penalties: torch.Tensor,
        presence_penalties: torch.Tensor,
        output_bin_counts: torch.Tensor,
        prompt_mask: torch.Tensor,
        output_mask: torch.Tensor,
        spec_k: int,
    ) -> torch.Tensor:
        num_seqs, num_reqs = logits.shape[0], repetition_penalties.shape[0]
        num_decodes = (num_seqs - num_reqs) // (spec_k)
        penalties_final = torch.ones(logits.shape, dtype=logits.dtype, device=logits.device)
        frequency_penalties_minus_final = torch.zeros(logits.shape, dtype=logits.dtype, device=logits.device)
        presence_penalties_minus_final = torch.zeros(logits.shape, dtype=logits.dtype, device=logits.device)

        repetition_penalties = repetition_penalties.unsqueeze(dim=1).repeat(1, logits.size(1))
        penalties = torch.where(prompt_mask | output_mask, repetition_penalties, 1.0)
        decodes_part = penalties[:num_decodes]
        penalties_final[:num_decodes*(1+spec_k)] = decodes_part.repeat_interleave(1+spec_k, dim=0)
        scaling = torch.where(logits > 0, 1.0 / penalties_final, penalties_final)
        logits *= scaling

        frequency_penalties_minus = frequency_penalties.unsqueeze(dim=1) * output_bin_counts
        frequency_penalties_minus_final[:num_decodes*(1+spec_k)] = frequency_penalties_minus[:num_decodes].repeat_interleave(1+spec_k, dim=0)
        presence_penalties_minus = presence_penalties.unsqueeze(dim=1) * output_mask
        presence_penalties_minus_final[:num_decodes*(1+spec_k)] = presence_penalties_minus[:num_decodes].repeat_interleave(1+spec_k, dim=0)
        logits -= frequency_penalties_minus_final
        logits -= presence_penalties_minus_final 
        return logits

    def apply_all_penalties_fusedmtp(
        self,
        logits: torch.Tensor,
        prompt_token_ids: torch.Tensor,
        presence_penalties: torch.Tensor,
        frequency_penalties: torch.Tensor,
        repetition_penalties: torch.Tensor,
        output_token_ids: Union[list[list[int]], torch.Tensor],
    ) -> torch.Tensor:
        """
        Applies presence, frequency and repetition penalties to the logits.
        """
        _, vocab_size = logits.shape
        if isinstance(output_token_ids, torch.Tensor):
            output_tokens_t = output_token_ids
        else:
            output_tokens_t = _convert_to_tensors(output_token_ids, vocab_size,
                                                logits.device)
        prompt_tokens_tensor = prompt_token_ids
        output_tokens_tensor = output_tokens_t
        from vllm_gcu.kernels._custom_ops import get_token_bin_counts_and_mask
        num_reqs = prompt_tokens_tensor.shape[0]
        
        _, prompt_mask = get_token_bin_counts_and_mask(prompt_tokens_tensor,
                                                    vocab_size, num_reqs)
        output_bin_counts, output_mask = get_token_bin_counts_and_mask(
            output_tokens_tensor, vocab_size, num_reqs)

        logits = self.apply_penalties_fused_mtp_torch(
            logits, repetition_penalties, frequency_penalties, presence_penalties,
            output_bin_counts, prompt_mask, output_mask, self.spec_k)
        return logits
