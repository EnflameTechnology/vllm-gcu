import torch

from vllm.model_executor.layers.utils import get_token_bin_counts_and_mask
from vllm.distributed import get_tp_group
from vllm.v1.sample.sampler import Sampler
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.outputs import SamplerOutput
from vllm_gcu.utils import scatter
from vllm.platforms import current_platform
from typing import Optional
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler

class DPParallelSampler(Sampler):

    def forward(self, logits, sampling_metadata):
        num_logprobs = sampling_metadata.max_num_logprobs
        if num_logprobs is not None:
            # disable parallel when num_logprobs is not None
            return super().forward(logits, sampling_metadata)

        tp_group = get_tp_group()
        world_size = tp_group.world_size
        local_rank = tp_group.rank_in_group

        scatter_counts = scatter(logits.shape[0], world_size)
        start = sum(scatter_counts[:local_rank])
        end = sum(scatter_counts[:local_rank + 1])

        # slice logits (B, V) -> (B/tp_size, V)
        dp_logits = logits[start:end]
        dp_generators = dict(
            sorted(sampling_metadata.generators.items(),
                   key=lambda x: x[0])[start:end])

        # sampling_metadata
        dp_sampling_metadata = SamplingMetadata(
            temperature=None if sampling_metadata.temperature is None else
            sampling_metadata.temperature[start:end],
            all_greedy=sampling_metadata.all_greedy,
            all_random=sampling_metadata.all_random,
            top_p=None if sampling_metadata.top_p is None else
            sampling_metadata.top_p[start:end],
            top_k=None if sampling_metadata.top_k is None else
            sampling_metadata.top_k[start:end],
            generators=dp_generators,
            max_num_logprobs=sampling_metadata.max_num_logprobs,
            no_penalties=sampling_metadata.no_penalties,
            prompt_token_ids=None if sampling_metadata.prompt_token_ids is None
            else sampling_metadata.prompt_token_ids[start:end],
            frequency_penalties=sampling_metadata.
            frequency_penalties[start:end],
            presence_penalties=sampling_metadata.presence_penalties[start:end],
            repetition_penalties=sampling_metadata.
            repetition_penalties[start:end],
            output_token_ids=sampling_metadata.output_token_ids[start:end],
            allowed_token_ids_mask=None
            if sampling_metadata.allowed_token_ids_mask is None else
            sampling_metadata.allowed_token_ids_mask[start:end],
            bad_words_token_ids=sampling_metadata.bad_words_token_ids,
            logitsprocs=sampling_metadata.logitsprocs,
        )

        dp_sampler_output = super().forward(dp_logits, dp_sampling_metadata)

        gathered_sampled_token_ids = torch.ops.vllm.all_gather_v(
            dp_sampler_output.sampled_token_ids,
            scatter_counts,
            group_name=tp_group.unique_name)

        return SamplerOutput(sampled_token_ids=gathered_sampled_token_ids,
                             logprobs_tensors=None)


def apply_penalties(
    logits: torch.Tensor,
    prompt_tokens_tensor: torch.Tensor,
    output_tokens_tensor: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> torch.Tensor:
    num_seqs, vocab_size = logits.shape
    _, prompt_mask = get_token_bin_counts_and_mask(prompt_tokens_tensor,
                                                   vocab_size, num_seqs)
    output_bin_counts, output_mask = get_token_bin_counts_and_mask(
        output_tokens_tensor, vocab_size, num_seqs)
    repetition_penalties = repetition_penalties.unsqueeze(dim=1).repeat(
        1, vocab_size)
    mask0 = torch.where((prompt_mask | output_mask) & (logits > 0),
                        1.0 / repetition_penalties, 1.0)
    mask1 = torch.where((prompt_mask | output_mask) & (logits <= 0),
                        repetition_penalties, 1.0)
    mask = mask0 * mask1
    logits *= mask

    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    logits -= frequency_penalties.unsqueeze(dim=1) * output_bin_counts
    logits -= presence_penalties.unsqueeze(dim=1) * output_mask
    return logits

class GCUTopKTopPSampler(TopKTopPSampler):
    def forward_native(
            self,
            logits: torch.Tensor,
            generators: dict[int, torch.Generator],
            k: Optional[torch.Tensor],
            p: Optional[torch.Tensor],) -> torch.Tensor:
        if current_platform.has_device_capability(140):
            output_token_ids = torch.empty(
                size=(logits.shape[0],), dtype=torch.int32, device=logits.device)
            q = torch.empty_like(logits,dtype=torch.float32,device=logits.device)
            # NOTE(woosuk): To batch-process the requests without their own seeds,
            # which is the common case, we first assume that every request does
            # not have its own seed. Then, we overwrite the values for the requests
            # that have their own seeds.
            if len(generators) != logits.shape[0]:
                q.exponential_()
            if generators:
                # TODO(woosuk): This can be slow because we handle each request
                # one by one. Optimize this.
                for i, generator in generators.items():
                    q[i].exponential_(generator=generator)

            torch.ops._C.topk_topp_random_sampler_from_logits(
                output_token_ids, logits, k, p, q, dim=-1)
            return output_token_ids
        else:
            return super().forward_native(logits, generators, k, p)
