import pytest
import torch
import torch_gcu


@pytest.mark.parametrize("b", [8192, 1024, 250, 1, 2, 3, 4])
@pytest.mark.parametrize("h", [7168])
@pytest.mark.parametrize("scale", [2.5, 1.0])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("topk", [8])
def test_index_add(
    b, h, scale, dtype, topk
) -> None:
    shared_output = torch.rand((b, h), dtype=dtype).gcu()
    sp_hidden_states = torch.rand((b*topk, h), dtype=dtype).gcu()
    ep_indices = torch.randint(0, b, (b*topk, ), dtype=torch.int32).gcu()

    def impl1():
        output = torch.zeros_like(shared_output)
        output.index_add_(0, ep_indices, sp_hidden_states)
        return output * scale + shared_output

    def impl2():
        output = torch.zeros_like(shared_output)
        output.copy_(shared_output)
        output.index_add_(0, ep_indices, sp_hidden_states, alpha=scale)
        return output

    assert torch.cosine_similarity(
        impl1().float().flatten(),
        impl2().float().flatten(),
        dim=0
    ).item() > 0.9995
