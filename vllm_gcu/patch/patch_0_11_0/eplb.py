from unittest.mock import patch
from vllm_gcu.distributed.eplb.eplb_state import step

patch("vllm.distributed.eplb.eplb_state.EplbState.step", step).start()
