from typing import Optional

import torch
import torch_gcu
from torch.distributed import ProcessGroup
from vllm.distributed.parallel_state import GroupCoordinator, _groups
from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op

from vllm_gcu.distributed.pyeccl import PyEcclCommunicator


def all_to_all_v2(
    output: torch.Tensor,
    input: torch.Tensor,
    output_split_sizes: Optional[torch.Tensor] = None,
    input_split_sizes: Optional[torch.Tensor] = None,
    group: Optional["ProcessGroup"] = None,
    flag: Optional[int] = None,
    async_op=False,
) -> None:
    return torch_gcu.distributed.all_to_all_vd(output,
                                               input,
                                               output_split_sizes,
                                               input_split_sizes,
                                               group=group,
                                               flag=flag,
                                               async_op=async_op)


def all_to_all_cpu(output,
                   input,
                   output_split_size=None,
                   input_split_size=None,
                   group=None) -> None:
    output_ori = output
    output = output.cpu()
    input = input.cpu()
    world_size = torch.distributed.get_world_size(group)
    rank = torch.distributed.get_rank(group)
    if output_split_size is None:
        output_split_size = [output.shape[0] // world_size] * world_size
    if input_split_size is None:
        input_split_size = [input.shape[0] // world_size] * world_size
    s1 = 0
    s2 = 0
    input_offsets = []
    output_offsets = []
    for i in range(world_size):
        input_offsets.append(s1)
        s1 += input_split_size[i]
        output_offsets.append(s2)
        s2 += output_split_size[i]

    for send_rank in range(world_size):
        for recv_rank in range(world_size):
            send_buffer = input[
                input_offsets[recv_rank]:input_offsets[recv_rank] +
                input_split_size[recv_rank]]
            recv_buffer = output[
                output_offsets[send_rank]:output_offsets[send_rank] +
                output_split_size[send_rank]]
            if send_rank == recv_rank:
                if rank == send_rank:
                    recv_buffer.copy_(send_buffer)
            else:
                if rank == send_rank:
                    torch.distributed.send(send_buffer, recv_rank, group=group)
                if rank == recv_rank:
                    torch.distributed.recv(recv_buffer, send_rank, group=group)
    output_ori.copy_(output)


def all_to_all_v2_ref(
    output,
    input,
    output_split_sizes,
    input_split_sizes,
    group=None,
    flag=0,
    async_op=False,
) -> None:
    assert output.is_contiguous(), 'output is not contiguous'
    assert input.is_contiguous(), 'input is not contiguous'
    assert output_split_sizes is not None
    assert input_split_sizes is not None

    if flag == 1:
        torch.distributed.all_to_all_single(output_split_sizes,
                                            input_split_sizes,
                                            group=group)

    assert output.shape[0] >= output_split_sizes.sum().item(
    ), 'output shape error'
    assert input.shape[0] >= input_split_sizes.sum().item(
    ), 'input shape error'

    return torch.distributed.all_to_all_single(
        output[:output_split_sizes.sum().item()],
        input[:input_split_sizes.sum().item()],
        output_split_sizes.cpu().tolist(),
        input_split_sizes.cpu().tolist(),
        group=group,
        async_op=async_op,
    )


def reduce_scatter_v(inp: torch.Tensor, scatter_counts: list[int],
                     group_name: str) -> torch.Tensor:
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    output_size = (scatter_counts[group.rank_in_group], ) + inp.size()[1:]
    output = torch.empty(output_size, dtype=inp.dtype, device=inp.device)
    torch_gcu.distributed.reduce_scatter_tensor_v(output,
                                                  inp,
                                                  scatter_counts,
                                                  group=group.device_group)
    return output


def reduce_scatter_v_fake(inp: torch.Tensor, scatter_counts: list[int],
                          group_name: str) -> torch.Tensor:
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    output_size = (scatter_counts[group.rank_in_group], ) + inp.size()[1:]
    output = torch.empty(output_size, dtype=inp.dtype, device=inp.device)
    return output


direct_register_custom_op(
    op_name="reduce_scatter_v",
    op_func=reduce_scatter_v,
    mutates_args=[],
    fake_impl=reduce_scatter_v_fake,
    dispatch_key=current_platform.dispatch_key,
)


def all_gather_v(inp: torch.Tensor, recv_counts: list[int],
                 group_name: str) -> torch.Tensor:
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")

    output_size = (sum(recv_counts), ) + inp.size()[1:]
    output = torch.empty(output_size, dtype=inp.dtype, device=inp.device)

    torch_gcu.distributed.all_gather_into_tensor_v(output,
                                                   inp,
                                                   recv_counts,
                                                   group=group.device_group)
    return output


def all_gather_v_fake(inp: torch.Tensor, recv_counts: list[int],
                      group_name: str) -> torch.Tensor:
    output_size = (sum(recv_counts), ) + inp.size()[1:]
    output = torch.empty(output_size, dtype=inp.dtype, device=inp.device)
    return output


direct_register_custom_op(
    op_name="all_gather_v",
    op_func=all_gather_v,
    mutates_args=[],
    fake_impl=all_gather_v_fake,
    dispatch_key=current_platform.dispatch_key,
)
