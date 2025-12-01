import torch

from vllm.forward_context import get_forward_context
import vllm.envs as envs
from vllm.distributed.parallel_state import get_tp_group
from vllm.platforms import current_platform


def scatter(seqlen, size):
    indices = list(range(size))
    return [(seqlen + indices[i]) // size - indices[i] // size
            for i in range(size)]


def align_up(seqlen, size):
    return (seqlen + size - 1) // size * size


def tp_to_sp(input_, seqlen):
    tp_group = get_tp_group()
    if current_platform.has_device_capability(140):
        scatter_counts = scatter(seqlen, tp_group.world_size)
        return torch.ops.vllm.reduce_scatter_v(
            input_,
            scatter_counts,
            tp_group.unique_name,
        )
    else:
        pad_size = align_up(seqlen, tp_group.world_size) - seqlen
        hidden_states = torch.nn.functional.pad(
            input_,
            (0, 0, 0, pad_size),
            mode="constant",
            value=0,
        )
        sp_hidden_states = torch.empty(hidden_states.shape[0] //
                                       tp_group.world_size,
                                       *hidden_states.shape[1:],
                                       device=hidden_states.device,
                                       dtype=hidden_states.dtype)
        torch.distributed.reduce_scatter_tensor(
            sp_hidden_states,
            hidden_states,
            group=tp_group.device_group,
        )
        return sp_hidden_states


def sp_to_tp(input_, seqlen):
    tp_group = get_tp_group()
    if current_platform.has_device_capability(140):
        scatter_counts = scatter(seqlen, tp_group.world_size)
        return torch.ops.vllm.all_gather_v(input_, scatter_counts,
                                           tp_group.unique_name)
    else:
        input_ = get_tp_group().all_gather(input_, dim=0)
        return input_[:seqlen]


def slice_tensor_sp(tensor, seqlen):
    tp_group = get_tp_group()
    if current_platform.has_device_capability(140):
        scatter_counts = scatter(seqlen, tp_group.world_size)
        tensor = tensor[sum(scatter_counts[:tp_group.rank_in_group]
                            ):sum(scatter_counts[:tp_group.rank_in_group +
                                                 1])].clone()
    else:
        pad_size = align_up(seqlen, tp_group.world_size) - seqlen
        tensor = torch.nn.functional.pad(
            tensor,
            (0, 0, 0, pad_size),
            mode="constant",
            value=0,
        )
        tensor_list = list(tensor.chunk(tp_group.world_size, dim=0))
        tensor = tensor_list[tp_group.rank_in_group].clone()
    return tensor
