from contextlib import contextmanager

from vllm.distributed.parallel_state import get_dp_group

_DP_STATE_PATCHED = False

@contextmanager
def patch_data_parallel_group():
    global _DP_STATE_PATCHED
    assert not _DP_STATE_PATCHED, "Should not call when it's already patched"

    _DP_STATE_PATCHED = True
    old_cpu_group = get_dp_group().cpu_group
    get_dp_group().cpu_group = None
    try:
        yield
    finally:
        # restore the original state
        _DP_STATE_PATCHED = False
        get_dp_group().cpu_group = old_cpu_group
