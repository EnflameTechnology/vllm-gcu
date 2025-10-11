from unittest.mock import patch
from typing import Optional
import torch
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import generate_uniform_probs
PLACEHOLDER_TOKEN_ID = -1
GREEDY_TEMPERATURE = -1
import vllm_gcu.envs as gcu_envs
from vllm_gcu.kernels.sampler import apply_top_k_top_p

def rejection_greedy_sample_kernel_torch(
    output_token_ids,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,  # [num_tokens]
    target_argmax,  # [num_tokens]
    bonus_token_ids,  # [batch_size]
    is_greedy,  # [batch_size] or None
    max_spec_len,):
    
    start_pos = 0
    for seq_index,end_pos in enumerate(cu_num_draft_tokens):
        rejected = False
        if isinstance(is_greedy,torch.Tensor):
            if not is_greedy[seq_index]:
                continue

        if start_pos == end_pos:
            output_token_ids[seq_index, 0] = bonus_token_ids[seq_index]
            continue

        for token_index in range(start_pos, end_pos):
            draft_token_id = draft_token_ids[token_index]
            target_argmax_id = target_argmax[token_index]
            output_token_ids[seq_index,token_index-start_pos] = target_argmax_id
            if draft_token_id != target_argmax_id:
                rejected = True
                break

        if not rejected:
            output_token_ids[seq_index, max_spec_len] = bonus_token_ids[seq_index]

        start_pos = end_pos

def rejection_random_sample_kernel_torch(
    output_token_ids,
    cu_num_draft_tokens,
    draft_token_ids,
    draft_probs,
    target_probs,
    bonus_token_ids,
    recovered_token_ids,
    uniform_probs,
    is_greedy,
    max_spec_len):

    if draft_probs is None:
        draft_probs = torch.ones(target_probs.shape, dtype=target_probs.dtype, device=target_probs.device)
    
    start_pos = 0
    for seq_index,end_pos in enumerate(cu_num_draft_tokens):
        if start_pos == end_pos:
            output_token_ids[seq_index,0] = bonus_token_ids[seq_index]
            continue

        reject = False
        for token_index in range(start_pos,end_pos):
            token = draft_token_ids[token_index]
            if draft_probs[token_index,token] > 0 and \
                target_probs[token_index,token] / draft_probs[token_index,token] >= uniform_probs[token_index]:
                output_token_ids[seq_index,token_index-start_pos] = draft_token_ids[token_index]
            else:
                reject = True
                output_token_ids[seq_index,token_index-start_pos] = recovered_token_ids[token_index]
                break
        
        if not reject:
            output_token_ids[seq_index,max_spec_len] = bonus_token_ids[seq_index]

        start_pos = end_pos

def expand_kernel_torch(
    expanded_x,
    x,
    cu_num_tokens,
    replace_from,
    replace_to):

    x = torch.where(x==replace_from, replace_to, x)
    prev_index = 0
    for end_index, value in zip(cu_num_tokens, x):
        expanded_x[prev_index: end_index] = value
        prev_index = end_index

def sample_recovered_tokens_kernel_torch(
    recovered_token_ids,
    cu_num_draft_tokens,
    draft_token_ids,
    draft_probs,
    target_probs,
    q):

    save_probs = torch.empty(draft_token_ids.shape, \
                                 dtype=target_probs.dtype, device=target_probs.device)
    start_pos = 0
    for seq_index, end_pos in enumerate(cu_num_draft_tokens):
        if start_pos == end_pos:
            continue

        for pos in range(start_pos,end_pos):
            token = draft_token_ids[pos]
            save_probs[pos] = target_probs[pos,token]
            target_probs[pos,token] = 0

        recovered_token_ids[start_pos:end_pos] = torch.argmax(target_probs[start_pos:end_pos]/q[seq_index], dim=-1)

        for pos in range(start_pos,end_pos):
            token = draft_token_ids[pos]
            target_probs[pos,token] = save_probs[pos]
        
        start_pos = end_pos

def expand_batch_to_tokens(
    x: torch.Tensor,  # [batch_size]
    cu_num_tokens: torch.Tensor,  # [batch_size]
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int = 0,
) -> torch.Tensor:
    """Expand [batch_size] tensor to [num_tokens] tensor based on the number of
    tokens per batch in cu_num_tokens.

    For example, if x = [a, b, c] and cu_num_tokens = [2, 5, 6], then
    num_tokens = 6, and expanded_x = [a, a, b, b, b, c].

    Args:
        x: [batch_size] tensor to expand.
        cu_num_tokens: [batch_size] tensor containing the cumulative number of
            tokens per batch. Each element represents the total number of
            tokens up to and including that batch.
        num_tokens: Total number of tokens.
        replace_from: int = 0
            Value to be replaced if it is found in x.
        replace_to: int = 0
            Value to replace with when replace_from is found.
    Returns:
        expanded_x: [num_tokens] tensor.
    """
    batch_size = x.shape[0]
    assert cu_num_tokens.shape[0] == batch_size
    expanded_x = x.new_empty(num_tokens)
    # expand_kernel[(batch_size, )](
    #     expanded_x,
    #     x,
    #     cu_num_tokens,
    #     replace_from,
    #     replace_to,
    #     MAX_NUM_TOKENS=MAX_SPEC_LEN,  # To avoid recompilation.
    #     num_warps=1,
    # )

    # expanded_x_v1 = x.new_empty(num_tokens)

    expand_kernel_torch(
        expanded_x,
        x,
        cu_num_tokens,
        replace_from,
        replace_to)

    # assert torch.all(expanded_x == expanded_x_v1)

    return expanded_x

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
        draft_probs,)

    ###
    if gcu_envs.VLLM_GCU_REJECT_SAMPLER_CHECK:
        recovered_token_ids_v1 = \
            torch.empty_like(draft_token_ids, dtype=draft_token_ids.dtype,device=draft_token_ids.device)

        sample_recovered_tokens_kernel_torch(
            recovered_token_ids_v1,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            q)

        assert torch.all(recovered_token_ids_v1 == recovered_token_ids)

    return recovered_token_ids

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
    else:
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
            is_greedy
        )

        ###
        if gcu_envs.VLLM_GCU_REJECT_SAMPLER_CHECK:
            output_token_ids_v1 = torch.empty(
                (batch_size, max_spec_len + 1),
                dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
                device=device,
            )
            output_token_ids_v1.fill_(PLACEHOLDER_TOKEN_ID)
            rejection_greedy_sample_kernel_torch(
                output_token_ids_v1,
                cu_num_draft_tokens,
                draft_token_ids,
                target_argmax,
                bonus_token_ids,
                is_greedy,
                max_spec_len,
            )

            assert torch.all(output_token_ids == output_token_ids_v1)

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
        is_greedy)

    # ###
    if gcu_envs.VLLM_GCU_REJECT_SAMPLER_CHECK:
        output_token_ids_v1 = torch.empty(
            (batch_size, max_spec_len + 1),
            dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
            device=device,
        )
        output_token_ids_v1.fill_(PLACEHOLDER_TOKEN_ID)

        rejection_random_sample_kernel_torch(
            output_token_ids_v1,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len)

        assert torch.all(output_token_ids == output_token_ids_v1)

    return output_token_ids

def compute_probs(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
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
    
    if gcu_envs.VLLM_GCU_REJECT_SAMPLER_CHECK:
        temperature_new = torch.empty(size=[num_tokens], device=logits.device)
        temperature_new = expand_batch_to_tokens(
            sampling_metadata.temperature,
            cu_num_draft_tokens,
            num_tokens,
            replace_from=GREEDY_TEMPERATURE,
            replace_to=1,
        )

        assert torch.all(temperature == temperature_new)
    # NOTE(woosuk): Update `logits` in place to avoid allocating a new tensor.
    logits.div_(temperature.unsqueeze(-1))

    # Get expanded top_k and top_p tensors.
    top_k = None
    if sampling_metadata.top_k is not None:
        top_k = torch.empty(size=[num_tokens], device=logits.device,dtype=torch.int32)
        torch.ops._C.expand_batch_to_tokens(
            top_k,
            sampling_metadata.top_k.to(dtype=torch.int32),
            cu_num_draft_tokens,
            num_tokens,
            replace_from=0,
            replace_to=0,
        )

        if gcu_envs.VLLM_GCU_REJECT_SAMPLER_CHECK:
            top_k_new = expand_batch_to_tokens(
                sampling_metadata.top_k,
                cu_num_draft_tokens,
                num_tokens,
            )
            
            assert torch.all(top_k == top_k_new)
    top_p = None
    if sampling_metadata.top_p is not None:
        top_p = torch.empty(size=[num_tokens], device=logits.device)
        torch.ops._C.expand_batch_to_tokens(
            top_p,
            sampling_metadata.top_p,
            cu_num_draft_tokens,
            num_tokens,
            replace_from = 0,
            replace_to = 0,
        )

        if gcu_envs.VLLM_GCU_REJECT_SAMPLER_CHECK:
            top_p_new = expand_batch_to_tokens(
                sampling_metadata.top_p,
                cu_num_draft_tokens,
                num_tokens,
            )
            assert torch.all(top_p == top_p_new)

    # NOTE(woosuk): `apply_top_k_top_p` uses sorting to calculate the mask,
    # which is slow for large vocab sizes. This may cause performance issues.
    logits = apply_top_k_top_p(logits, top_k, top_p)
    output_prob = logits.softmax(dim=-1, dtype=torch.float32)
    return output_prob

patch("vllm.v1.sample.rejection_sampler.compute_probs", compute_probs).start()
patch("vllm.v1.sample.rejection_sampler.rejection_sample", rejection_sample).start()
