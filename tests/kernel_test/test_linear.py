import pytest
import torch

import vllm.distributed.parallel_state as ps
from vllm_gcu.kernels.linear import (
    MergedReplicatedLinear,
    CustomMergedColumnParallelLinear,
)


@pytest.fixture(autouse=True)
def fake_tensor_parallel(monkeypatch):
    class FakeTPGroup:
        rank_in_group = 0
        world_size = 1

        def all_gather(self, x, dim=-1):
            return x

    fake_group = FakeTPGroup()

    monkeypatch.setattr(ps, "_TP", fake_group, raising=False)
    monkeypatch.setattr(ps, "get_tp_group", lambda: fake_group)
    monkeypatch.setattr(ps, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(ps, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.destroy_model_parallel",
        lambda: None,
    )


def test_merged_replicated_weight_loader():
    layer = MergedReplicatedLinear(
        input_size=4,
        output_sizes=[3, 5],
        bias=True,
    )

    weight_cpu = torch.randn(8, 4)
    weight_gcu = weight_cpu.gcu()

    layer.weight_loader(layer.weight, weight_gcu)
    assert layer.weight.shape == (8, 4)
    torch.testing.assert_close(
        layer.weight.cpu(),
        weight_cpu,
        rtol=0,
        atol=0,
    )


def test_column_parallel_no_scale_no_gather_no_bias():
    layer_cpu = CustomMergedColumnParallelLinear(
        input_size=4,
        output_sizes=[8],
        bias=True,
        gather_output=False,
    )
    layer_cpu.return_bias = False

    layer_gcu = CustomMergedColumnParallelLinear(
        input_size=4,
        output_sizes=[8],
        bias=True,
        gather_output=False,
    )
    layer_gcu.return_bias = False
    layer_gcu.to("gcu")

    with torch.no_grad():
        layer_gcu.weight.copy_(layer_cpu.weight)
        layer_gcu.bias.copy_(layer_cpu.bias)

    x_cpu = torch.randn(2, 4)
    x_gcu = x_cpu.gcu()

    out_cpu = layer_cpu(x_cpu)
    out_gcu = layer_gcu(x_gcu)

    torch.testing.assert_close(
        out_gcu.cpu(),
        out_cpu,
        rtol=1e-5,
        atol=1e-5,
    )


def test_column_parallel_no_scale_with_gather_and_bias():
    layer_cpu = CustomMergedColumnParallelLinear(
        input_size=4,
        output_sizes=[8],
        bias=True,
        gather_output=True,
    )
    layer_cpu.return_bias = True

    layer_gcu = CustomMergedColumnParallelLinear(
        input_size=4,
        output_sizes=[8],
        bias=True,
        gather_output=True,
    )
    layer_gcu.return_bias = True
    layer_gcu.to("gcu")

    with torch.no_grad():
        layer_gcu.weight.copy_(layer_cpu.weight)
        layer_gcu.bias.copy_(layer_cpu.bias)

    x_cpu = torch.randn(2, 4)
    x_gcu = x_cpu.gcu()

    out_cpu, bias_cpu = layer_cpu(x_cpu)
    out_gcu, bias_gcu = layer_gcu(x_gcu)

    torch.testing.assert_close(
        out_gcu.cpu(),
        out_cpu,
        rtol=1e-5,
        atol=1e-5,
    )

    assert bias_cpu is None
    assert bias_gcu is None
