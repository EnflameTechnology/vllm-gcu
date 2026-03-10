from typing import Optional, Tuple
import torch
from vllm.config import get_current_vllm_config
from vllm.platforms import current_platform
from vllm.distributed.parallel_state import get_tp_group
from vllm.v1.sample.rejection_sampler import (
    RejectionSampler,
    MAX_SPEC_LEN,
    PLACEHOLDER_TOKEN_ID,
    generate_uniform_probs,
)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm_gcu.kernels.sampler import apply_top_k_top_p

GREEDY_TEMPERATURE = float('inf')

def sample_recovered_tokens(
    max_spec_len: int,
    num_draft_tokens: list[int],
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: Optional[torch.Tensor],
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    device: torch.device,
) -> torch.Tensor:
    # NOTE(woosuk): Create only one distribution for each request.
    batch_size = len(num_draft_tokens)
    vocab_size = target_probs.shape[-1]
    q = torch.empty(
        (batch_size, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    q.exponential_()
    for i, generator in sampling_metadata.generators.items():
        # Do not generate random numbers for requests with no draft tokens.
        # This can be important for reproducibility.
        if num_draft_tokens[i] > 0:
            q[i].exponential_(generator=generator)

    recovered_token_ids = torch.empty_like(draft_token_ids)

    torch.ops._C.sample_recovered_tokens(
        recovered_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        target_probs,
        q,
        draft_probs,
    )

    return recovered_token_ids


def compute_probs(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    sampling_metadata: SamplingMetadata,
    world_size: int = 1,
) -> torch.Tensor:
    assert logits.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    if sampling_metadata.all_greedy:
        return logits

    num_tokens = logits.shape[0]
    temperature = torch.empty(size=[num_tokens], device=logits.device)
    torch.ops._C.expand_batch_to_tokens(
        temperature,
        sampling_metadata.temperature,
        cu_num_draft_tokens,
        num_tokens,
        replace_from=GREEDY_TEMPERATURE,
        replace_to=1,
    )

    logits.mul_(temperature.unsqueeze(-1))

    top_k = None
    if sampling_metadata.top_k is not None:
        top_k = torch.empty(size=[num_tokens],
                            device=logits.device,
                            dtype=torch.int32)
        torch.ops._C.expand_batch_to_tokens(
            top_k,
            sampling_metadata.top_k.to(dtype=torch.int32),
            cu_num_draft_tokens,
            num_tokens,
            replace_from=0,
            replace_to=0,
        )

    top_p = None
    if sampling_metadata.top_p is not None:
        top_p = torch.empty(size=[num_tokens], device=logits.device)
        torch.ops._C.expand_batch_to_tokens(
            top_p,
            sampling_metadata.top_p,
            cu_num_draft_tokens,
            num_tokens,
            replace_from=0,
            replace_to=0,
        )

    logits = apply_top_k_top_p(logits, top_k, top_p, world_size)
    output_prob = logits.softmax(dim=-1, dtype=torch.float32)
    return output_prob


def rejection_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: Optional[torch.Tensor],
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    is_greedy: Optional[torch.Tensor] = None
) -> torch.Tensor:
    assert draft_token_ids.ndim == 1
    assert draft_probs is None or draft_probs.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    assert target_probs.ndim == 2

    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    vocab_size = target_probs.shape[-1]
    device = target_probs.device
    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert target_probs.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_probs.shape == (num_tokens, vocab_size)

    # Create output buffer.
    output_token_ids = torch.empty(
        (batch_size, max_spec_len + 1),
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )
    output_token_ids.fill_(PLACEHOLDER_TOKEN_ID)

    if sampling_metadata.all_greedy:
        is_greedy = None
    elif is_greedy is None:
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE
    if not sampling_metadata.all_random:
        # Rejection sampling for greedy sampling requests.
        target_argmax = target_probs.argmax(dim=-1)
        torch.ops._C.rejection_greedy_sample(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
        )

        ###
        if sampling_metadata.all_greedy:
            return output_token_ids

    # Generate uniform probabilities for rejection sampling.
    # [num_tokens]
    uniform_probs = generate_uniform_probs(
        num_tokens,
        num_draft_tokens,
        sampling_metadata.generators,
        device,
    )

    # Sample recovered tokens for each position.
    # [num_tokens]
    recovered_token_ids = sample_recovered_tokens(
        max_spec_len,
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        sampling_metadata,
        device,
    )

    # # Rejection sampling for random sampling requests.
    torch.ops._C.rejection_random_sample(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids.view(-1),
        recovered_token_ids,
        uniform_probs,
        is_greedy,
    )

    return output_token_ids

def compute_probs_fused_mtp(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    num_decodes: int,
    max_spec_len: int,
    sampling_metadata: SamplingMetadata,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute probability distribution from logits based on sampling metadata.

    This function applies temperature scaling to the logits and converts
    them to probabilities using softmax. For greedy decoding, it returns
    the original logits.

    Args:
        logits: Input logits tensor to be converted to probabilities.
        cu_num_draft_tokens: Cumulative number of draft tokens.
        sampling_metadata: Metadata containing sampling parameters such as
            temperature and whether greedy sampling is used.

    Returns:
        torch.Tensor: Probability distribution (softmax of scaled logits)
            if non-greedy sampling is used, otherwise returns the
            original logits.
    """
    assert logits.ndim == 2
    #assert cu_num_draft_tokens.ndim == 1
    if sampling_metadata.all_greedy:
        return logits, None
    
    temperature = sampling_metadata.temperature[:(num_decodes * (max_spec_len + 1))]
    temperature = temperature.reshape(num_decodes, max_spec_len + 1)[:, :-1].contiguous()
    is_greedy = temperature[:, 0] == GREEDY_TEMPERATURE
    temperature[is_greedy, :] = 1
    temperature = temperature.reshape(num_decodes * max_spec_len)
    top_k = sampling_metadata.top_k[:(num_decodes * (max_spec_len + 1))]
    top_k = top_k.reshape(num_decodes, max_spec_len + 1)[:, :-1].contiguous()
    top_k = top_k.reshape(num_decodes * max_spec_len)
    top_p = sampling_metadata.top_p[:(num_decodes * (max_spec_len + 1))]
    top_p = top_p.reshape(num_decodes, max_spec_len + 1)[:, :-1].contiguous()
    top_p = top_p.reshape(num_decodes * max_spec_len)

    num_tokens = logits.shape[0]
    assert temperature.shape[0] == num_tokens and top_k.shape[0] == num_tokens and \
        top_p.shape[0] == num_tokens
    # NOTE(woosuk): Update `logits` in place to avoid allocating a new tensor.
    logits.mul_(temperature.unsqueeze(-1))

    
    # NOTE(woosuk): `apply_top_k_top_p` uses sorting to calculate the mask,
    # which is slow for large vocab sizes. This may cause performance issues.
    torch.ops._C.top_k_top_p(logits, top_k, top_p)
    output_prob = logits.softmax(dim=-1, dtype=torch.float32)
    return output_prob, is_greedy

class GCURejectionSampler(RejectionSampler):
    def __init__(self):
        super().__init__()
        vllm_config = get_current_vllm_config()
        self.enable_dp_parallel = not vllm_config.additional_config.get("disable_dp_sampler", False)

    def forward(
        self,
        metadata: SpecDecodeMetadata,
        draft_probs: Optional[torch.Tensor],
        target_logits: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        assert current_platform.has_device_capability(140)
        assert metadata.max_spec_len <= MAX_SPEC_LEN

        tp_group = get_tp_group()
        world_size = tp_group.world_size if self.enable_dp_parallel else 1

        target_probs = compute_probs(
            target_logits,
            metadata.cu_num_draft_tokens,
            sampling_metadata,
            world_size,
        )
        output_token_ids = rejection_sample(
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            draft_probs,
            target_probs,
            bonus_token_ids,
            sampling_metadata,
        )
        return output_token_ids
    
    

    def rejection_sampler_forward_with_fused_mtp(
        self,
        num_decodes: int,
        max_spec_len: int,
        # [num_decodes, spec_k]
        draft_token_ids: torch.Tensor,
        # [num_tokens, vocab_size]
        draft_probs: Optional[torch.Tensor],
        # [num_decodes * (spec_k + 1), vocab_size]
        selected_logits: torch.Tensor,
        # [num_decodes, (spec_k + 1)]
        selected_token_ids: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        '''
        Args:
            metadata:
                Metadata for spec decoding.
            draft_probs (Optional[torch.Tensor]):
                Probability distribution for the draft tokens. Shape is
                [num_tokens, vocab_size]. Can be None if probabilities are
                not provided, which is the case for ngram spec decode.
            target_logits (torch.Tensor):
                Target model's logits probability distribution.
                Shape is [num_tokens, vocab_size]. Here, probabilities from
                different requests are flattened into a single tensor because
                this is the shape of the output logits.
                NOTE: `target_logits` can be updated in place to save memory.
            bonus_token_ids_tensor (torch.Tensor):
                A tensor containing bonus tokens. Shape is [batch_size, 1].
                Bonus tokens are added to the end of the sequence if all
                proposed tokens are accepted. We generate the bonus tokens
                outside of the rejection sampler with the default sampling
                strategy. It allows for more flexibility in the sampling
                process such as top_p, top_k sampling.
            sampling_metadata (vllm.v1.sample.metadata.SamplingMetadata):
                Additional metadata needed for sampling, such as temperature,
                top-k/top-p parameters, or other relevant information.
        Returns:
            output_token_ids (torch.Tensor):
                A tensor containing the final output token IDs.
        '''
        target_logits = selected_logits.reshape(num_decodes, max_spec_len + 1, selected_logits.shape[-1])[:, :-1, :].contiguous()
        target_logits = target_logits.reshape(num_decodes * max_spec_len, selected_logits.shape[-1])
        bonus_token_ids = selected_token_ids[:, -1:].contiguous()
        # [num_tokens, vocab_size]
        # NOTE(woosuk): `target_logits` can be updated in place inside the
        # `compute_probs` function.
        num_draft_tokens_lst = [max_spec_len] * num_decodes
        num_draft_tokens = torch.full((num_decodes, ), max_spec_len, device=target_logits.device)
        cu_num_draft_tokens = num_draft_tokens.cumsum(dim=0).to(torch.int32)
        target_probs, is_greedy = compute_probs_fused_mtp(
            target_logits,
            num_decodes,
            max_spec_len,
            sampling_metadata,
        )
        draft_token_ids_1d = draft_token_ids.reshape(num_decodes * max_spec_len)
        output_token_ids = rejection_sample(
            draft_token_ids_1d,
            num_draft_tokens_lst,
            max_spec_len,
            cu_num_draft_tokens,
            draft_probs,
            target_probs,
            bonus_token_ids,
            sampling_metadata,
            is_greedy
        )
        mask = output_token_ids.eq(-1).to(torch.bool)
        accepted_lens = mask.logical_not().sum(dim=1)
        return output_token_ids, accepted_lens
