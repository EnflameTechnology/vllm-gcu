import triton
import triton_gcu.triton
import torch

from torch_gcu import transfer_to_gcu
from unittest.mock import patch


from .fused_recurrent import fused_recurrent_gated_delta_rule
from .solve_tril import merge_16x16_to_64x64_inverse_kernel
from .chunk_delta_h import chunk_gated_delta_rule_fwd_h

from vllm.model_executor.models.qwen3_next import Qwen3NextForCausalLM
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention, Qwen3NextGatedDeltaNet


class Qwen3NextPatchedAttention(Qwen3NextAttention):
    def forward(
        self,
        positions: torch.Tensor,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        qkv, _ = self.qkv_proj(hidden_states)

        if self.attn_output_gate:
            q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
            orig_shape = q_gate.shape[:-1]
            last_shape = q_gate.shape[-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, last_shape // self.num_heads)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, last_shape // 2)
            gate = gate.reshape(*orig_shape, last_shape // 2)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(-1, self.num_heads * self.head_dim)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(-1, self.num_kv_heads * self.head_dim)

        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        output[:], _ = self.o_proj(attn_output)


class Qwen3NextPatchedGatedDeltaNet(Qwen3NextGatedDeltaNet):
    def fix_query_key_value_ordering(
        self,
        mixed_qkvz,
        mixed_ba,
    ):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvzba`.
        """
        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            (self.head_k_dim + self.head_k_dim + (self.head_v_dim + self.head_v_dim) * self.num_v_heads // self.num_k_heads),
        )
        new_tensor_shape_ba = mixed_qkvz.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            2 * self.num_v_heads // self.num_k_heads,
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [self.num_v_heads // self.num_k_heads, self.num_v_heads // self.num_k_heads]

        # [b, sq, ng, (hn + hn + np/ng * hn + np/ng + np/ng)]
        # --> [b, sq, ng, hn], [b, sq, ng, hn], [b, sq, ng, np/ng * hn],
        #  [b, sq, ng, np/ng * hn], [b, sq, ng, np/ng], [b, sq, ng, np/ng]
        (query, key, value, z) = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=2)
        (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=2)

        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), value.size()[1:].numel() // self.head_v_dim, self.head_v_dim)
        z = z.reshape(z.size(0), z.size()[1:].numel() // self.head_v_dim, self.head_v_dim)
        b = b.reshape(b.size(0), self.num_v_heads // self.tp_size)
        a = a.reshape(a.size(0), self.num_v_heads // self.tp_size)

        return query, key, value, z, b, a


patch("vllm.model_executor.layers.fla.ops.chunk.chunk_gated_delta_rule_fwd_h", chunk_gated_delta_rule_fwd_h).start()
patch("vllm.model_executor.layers.fla.ops.solve_tril.merge_16x16_to_64x64_inverse_kernel", merge_16x16_to_64x64_inverse_kernel).start()
patch("vllm.model_executor.models.qwen3_next.fused_recurrent_gated_delta_rule", fused_recurrent_gated_delta_rule).start()
patch("vllm.model_executor.models.qwen3_next.Qwen3NextAttention", Qwen3NextPatchedAttention).start()
patch("vllm.model_executor.models.qwen3_next.Qwen3NextGatedDeltaNet", Qwen3NextPatchedGatedDeltaNet).start()
