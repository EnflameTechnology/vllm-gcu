import torch
import pytest
import uuid
from typing import Optional

from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.layernorm import LayerNorm
from vllm.model_executor.layers.mla import MLAModules
from vllm_gcu.kernels.mla import GCUMultiHeadLatentAttention
from vllm.attention.layer import Attention

LinearBase.name = "linear"


@pytest.fixture(autouse=True)
def init_fake_env(monkeypatch):

    class FakeTPGroup:
        def __init__(self):
            self.world_size = 1
            self.rank = 0
            self.rank_in_group = 0

        def size(self):
            return 1

    fake_group = FakeTPGroup()

    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group",
        lambda: fake_group,
    )
    monkeypatch.setattr(
        "vllm.distributed.get_tp_group",
        lambda: fake_group,
    )
    monkeypatch.setattr(
        "vllm_gcu.kernels.mla.get_tp_group",
        lambda: fake_group,
    )

    def fake_forward_native(self, x):
        out_features = self.output_size
        in_features = x.shape[-1]

        torch.manual_seed(0)
        weight = torch.randn(out_features, in_features, device=x.device) * 0.01
        bias = torch.zeros(out_features, device=x.device)

        out = torch.nn.functional.linear(x, weight, bias)
        return out, None

    monkeypatch.setattr(
        LinearBase,
        "forward_native",
        fake_forward_native,
    )
    from vllm.attention.layer import Attention as GCUAttention

    def fake_attention_forward(self, q, k, v, *args, **kwargs):
        return q

    monkeypatch.setattr(
        GCUAttention,
        "forward",
        fake_attention_forward,
    )


def make_linear(inp, out, prefix):
    layer = LinearBase(
        input_size=inp,
        output_size=out,
        skip_bias_add=False,
        params_dtype=torch.float32,
        quant_config=None,
        prefix=prefix,
        return_bias=True,
        disable_tp=True,
    )

    return layer


def build_mla_modules(
    hidden_size=32,
    q_lora_rank: Optional[int] = 8,
    kv_lora_rank=8,
    qk_rope_head_dim=8,
):

    total_lora_out = (q_lora_rank or 0) + kv_lora_rank + qk_rope_head_dim

    fused_qkv_a_proj = make_linear(hidden_size, total_lora_out, "fused_")
    kv_a_proj_with_mqa = make_linear(
        hidden_size,
        kv_lora_rank + qk_rope_head_dim,
        "kv_",
    )

    q_proj = make_linear(hidden_size, hidden_size, "q_")

    q_b_proj = make_linear(q_lora_rank, hidden_size, "qb_") if q_lora_rank else None

    kv_a_layernorm = LayerNorm(kv_lora_rank)
    q_a_layernorm = LayerNorm(q_lora_rank) if q_lora_rank else None

    o_proj = make_linear(hidden_size, hidden_size, "o_")

    modules = MLAModules(
        kv_a_layernorm=kv_a_layernorm.to("gcu"),
        kv_b_proj=None,
        rotary_emb=None,
        o_proj=o_proj.to("gcu"),
        fused_qkv_a_proj=fused_qkv_a_proj.to("gcu"),
        kv_a_proj_with_mqa=kv_a_proj_with_mqa.to("gcu"),
        q_a_layernorm=q_a_layernorm.to("gcu") if q_a_layernorm else None,
        q_b_proj=q_b_proj.to("gcu") if q_b_proj else None,
        q_proj=q_proj.to("gcu"),
        indexer_rotary_emb=None,
        indexer=None,
        is_sparse=False,
        topk_indices_buffer=None,
    )

    return modules


@pytest.fixture
def mla_dense():
    modules = build_mla_modules(
        hidden_size=32,
        q_lora_rank=8,
    )

    layer = GCUMultiHeadLatentAttention(
        hidden_size=32,
        num_heads=1,
        scale=1.0,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        q_lora_rank=8,
        kv_lora_rank=8,
        mla_modules=modules,
        prefix=str(uuid.uuid4()),
    )

    return layer.to("gcu")


@pytest.fixture
def mla_no_q_lora():
    modules = build_mla_modules(
        hidden_size=32,
        q_lora_rank=None,
    )

    layer = GCUMultiHeadLatentAttention(
        hidden_size=32,
        num_heads=1,
        scale=1.0,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        q_lora_rank=None,
        kv_lora_rank=8,
        mla_modules=modules,
        prefix=str(uuid.uuid4()),
    )

    return layer.to("gcu")


def run_forward_check(layer, batch):
    torch.set_default_device("gcu")

    hidden = torch.randn(batch, 32, requires_grad=True)
    pos = torch.arange(batch)

    out = layer.forward_oot(pos, hidden)

    assert out.dtype == torch.float32
    assert not torch.isnan(out).any()

    out.sum().backward()
    assert hidden.grad is not None


def test_forward_dense(mla_dense):
    run_forward_check(mla_dense, batch=4)


def test_forward_no_q_lora(mla_no_q_lora):
    run_forward_check(mla_no_q_lora, batch=3)
