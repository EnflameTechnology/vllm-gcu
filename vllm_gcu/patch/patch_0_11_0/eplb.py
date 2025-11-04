from unittest.mock import patch
from vllm_gcu.distributed.eplb.eplb_state import step
from vllm_gcu.distributed.eplb.rebalance_execute import rearrange_expert_weights_inplace, shuffle_layer

patch("vllm.distributed.eplb.eplb_state.EplbState.step", step).start()
patch("vllm.distributed.eplb.rebalance_execute.rearrange_expert_weights_inplace", rearrange_expert_weights_inplace).start()
patch("vllm.distributed.eplb.eplb_state.rearrange_expert_weights_inplace", rearrange_expert_weights_inplace).start()
patch("vllm.distributed.eplb.rebalance_execute.shuffle_layer", shuffle_layer).start()
