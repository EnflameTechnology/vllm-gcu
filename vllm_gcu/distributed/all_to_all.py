import torch
import torch_gcu
import pickle
N = 0


def all_to_all_v2_bypass(output, input, output_split_sizes=None, input_split_sizes=None, group=None, flag=None):
    if flag == 1:
        output_split_sizes.copy_(input_split_sizes)
    if output.numel() > input.numel():
        output.view(-1)[:input.numel()].copy_(input.view(-1))
    else:
        output.view(-1).copy_(input.view(-1)[:output.numel()])


def all_to_all_v2_ref(output, input, output_split_sizes=None, input_split_sizes=None, group=None, flag=None):
    assert output.is_contiguous()
    assert input.is_contiguous()
    if flag == 1:
        torch.distributed.all_to_all_single(
            output_split_sizes, input_split_sizes, group=group)
    assert output.shape[0] >= output_split_sizes.sum().item()
    assert input.shape[0] >= input_split_sizes.sum().item()
    torch.distributed.all_to_all_single(
        output[:output_split_sizes.sum().item()],
        input[:input_split_sizes.sum().item()],
        output_split_sizes.cpu().tolist(),
        input_split_sizes.cpu().tolist(),
        group=group,
    )


def all_to_all_v2(output, input, output_split_sizes=None, input_split_sizes=None, group=None, flag=None):
    torch_gcu.distributed.all_to_all_vd(
        output, input, output_split_sizes, input_split_sizes, group=group, flag=flag)


def all_to_all_v2_dump(output, input, output_split_sizes=None, input_split_sizes=None, group=None, flag=None):
    assert output.is_contiguous()
    assert input.is_contiguous()
    global N
    print('flag', flag)
    print('input', input.shape, input)
    print('input_split_sizes', input_split_sizes)
    rank = torch.distributed.get_rank()
    with open(f'all_to_all_v2_N{N}_rank{rank}_in.pkl', 'wb') as f:
        pickle.dump((output.cpu(), input.cpu(),
                    output_split_sizes.cpu(), input_split_sizes.cpu(), flag), f)
    torch_gcu.distributed.all_to_all_vd(
        output, input, output_split_sizes, input_split_sizes, group=group, flag=flag)
    with open(f'all_to_all_v2_N{N}_rank{rank}_out.pkl', 'wb') as f:
        pickle.dump((output.cpu(), input.cpu(),
                    output_split_sizes.cpu(), input_split_sizes.cpu(), flag), f)
    print('output_split_sizes', output_split_sizes)
    print('output', output.shape, output)
    print('-' * 20)
    N += 1


def all_to_all_cpu(output, input, output_split_size=None, input_split_size=None, group=None):
    output_ori = output
    input_ori = input
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
            send_buffer = input[input_offsets[recv_rank]
                :input_offsets[recv_rank]+input_split_size[recv_rank]]
            recv_buffer = output[output_offsets[send_rank]
                :output_offsets[send_rank]+output_split_size[send_rank]]
            if send_rank == recv_rank:
                if rank == send_rank:
                    recv_buffer.copy_(send_buffer)
            else:
                if rank == send_rank:
                    torch.distributed.send(send_buffer, recv_rank, group=group)
                if rank == recv_rank:
                    torch.distributed.recv(recv_buffer, send_rank, group=group)
    output_ori.copy_(output)
