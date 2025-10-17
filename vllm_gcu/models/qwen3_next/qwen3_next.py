import triton
import triton_gcu.triton
import torch

from torch_gcu import transfer_to_gcu
from unittest.mock import patch



from .fused_recurrent import fused_recurrent_gated_delta_rule
from .solve_tril import merge_16x16_to_64x64_inverse_kernel
from .chunk_delta_h import chunk_gated_delta_rule_fwd_h

from vllm.model_executor.models.qwen3_next import Qwen3NextForCausalLM


patch("vllm.model_executor.layers.fla.ops.chunk.chunk_gated_delta_rule_fwd_h", chunk_gated_delta_rule_fwd_h).start()
patch("vllm.model_executor.layers.fla.ops.solve_tril.merge_16x16_to_64x64_inverse_kernel", merge_16x16_to_64x64_inverse_kernel).start()
patch("vllm.model_executor.models.qwen3_next.fused_recurrent_gated_delta_rule", fused_recurrent_gated_delta_rule).start()
