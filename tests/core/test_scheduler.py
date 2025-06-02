import vllm
from vllm.config import CacheConfig, SchedulerConfig
from vllm_gcu.scheduler import PriorityScheduler

from utils import (
    append_new_token,
    create_dummy_prompt,
    get_sequence_groups,
    schedule_and_update_computed_tokens,
)


def test_priority_scheduler():
    block_size = 4
    max_model_len = 30
    max_batched_num_tokens = 30

    scheduler_config = SchedulerConfig(
        "generate",
        max_num_batched_tokens=max_batched_num_tokens,
        max_num_seqs=2,
        max_model_len=max_model_len,
        policy="priority",
    )

    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 16
    cache_config.num_gpu_blocks = 16
    scheduler = PriorityScheduler(scheduler_config, cache_config, None)

    _, seq_group_a = create_dummy_prompt("1", 1, block_size=block_size, priority=1)
    scheduler.add_seq_group(seq_group_a)

    _, out = schedule_and_update_computed_tokens(scheduler)
    assert get_sequence_groups(out) == [seq_group_a]
    append_new_token(out, 1)

    _, seq_group_b = create_dummy_prompt("2", 30, block_size=block_size, priority=100)
    scheduler.add_seq_group(seq_group_b)

    _, out = schedule_and_update_computed_tokens(scheduler)
    assert get_sequence_groups(out) == [seq_group_a]
