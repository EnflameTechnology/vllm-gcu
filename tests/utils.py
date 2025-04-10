import time
from typing import Optional

from vllm import SamplingParams
from vllm.inputs import token_inputs
from vllm.lora.request import LoRARequest
from vllm.sequence import Logprob, Sequence, SequenceGroup


def create_dummy_prompt(
    request_id: str,
    prompt_length: int = -1,
    block_size: Optional[int] = None,
    lora_request: Optional[LoRARequest] = None,
    prompt_tokens: Optional[list[int]] = None,
    min_tokens: int = 0,
    max_tokens: int = 16,
    priority: Optional[int] = None,
) -> tuple[Sequence, SequenceGroup]:
    if not block_size:
        block_size = prompt_length

    if prompt_tokens is None:
        # Create dummy prompt sequence with tokens 0...block_size-1
        # and prompt "0 ... block_size".
        prompt_tokens = list(range(prompt_length))

    prompt_str = " ".join([str(t) for t in prompt_tokens])
    prompt = Sequence(
        int(request_id),
        inputs=token_inputs(prompt_tokens, prompt=prompt_str),
        block_size=block_size,
    )
    seq_group = SequenceGroup(
        request_id=request_id,
        seqs=[prompt],
        arrival_time=time.time(),
        sampling_params=SamplingParams(max_tokens=max_tokens, min_tokens=min_tokens),
        lora_request=lora_request,
        priority=priority,
    )

    return prompt, seq_group


def schedule_and_update_computed_tokens(scheduler):
    metas, out, _ = scheduler.schedule()
    for s in out.scheduled_seq_groups:
        s.seq_group.update_num_computed_tokens(s.token_chunk_size)
    return metas, out


def get_sequence_groups(scheduler_output):
    return [s.seq_group for s in scheduler_output.scheduled_seq_groups]


def append_new_token(out, token_id: int):
    seq_groups = get_sequence_groups(out)
    for seq_group in seq_groups:
        for seq in seq_group.get_seqs():
            seq.append_token_id(token_id, {token_id: Logprob(token_id)})
