from unittest.mock import patch

import vllm_gcu.envs as gcu_envs
from vllm_gcu.distributed.eplb.eplb_state import step
from vllm_gcu.distributed.eplb.rebalance_execute import (
    rearrange_expert_weights_inplace,
    shuffle_layer,
)

if gcu_envs.VLLM_GCU_EPLB_ASYNC_ENABLED:
    from vllm.distributed.eplb.eplb_state import EplbState
    from vllm_gcu.distributed.eplb.async_eplb_step import (
        async_step,
        start_async_loop,
    )

    patch("vllm.distributed.eplb.eplb_state.EplbState.step", async_step).start()

    # Wrap EplbState.build so that the async worker is started during
    # initialization (inside load_model), matching v0.14.1 where
    # start_async_loop is called right after add_model.
    _original_build = EplbState.build.__func__

    @classmethod
    def _async_build(cls, model, device, parallel_config,
                     global_expert_load=None,
                     old_global_expert_indices=None,
                     rank_mapping=None):
        state = _original_build(
            cls, model, device, parallel_config,
            global_expert_load, old_global_expert_indices, rank_mapping,
        )
        start_async_loop(state, model, device, rank_mapping=rank_mapping)
        return state

    EplbState.build = _async_build
else:
    patch("vllm.distributed.eplb.eplb_state.EplbState.step", step).start()

patch("vllm.distributed.eplb.rebalance_execute.rearrange_expert_weights_inplace", rearrange_expert_weights_inplace).start()
patch("vllm.distributed.eplb.eplb_state.rearrange_expert_weights_inplace", rearrange_expert_weights_inplace).start()
patch("vllm.distributed.eplb.rebalance_execute.shuffle_layer", shuffle_layer).start()
