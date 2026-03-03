from collections.abc import Iterable, MutableSequence, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Optional

import numpy as np
import torch
from torch.distributed import (P2POp, ProcessGroup, all_gather,
                               batch_isend_irecv, get_global_rank)
from vllm.distributed.eplb.rebalance_execute import (
    idx_local_to_global,
    get_ep_ranks_with_expert,
    _map_old_expert_indices_with_rank_mapping,
    _map_new_expert_indices_with_rank_mapping,
)


def shuffle_layer(
    num_local_experts: int,
    ep_rank: int,
    old_indices: Sequence[int],
    new_indices: Sequence[int],
    expert_weights: Iterable[torch.Tensor],
    expert_weights_buffer: Sequence[torch.Tensor],
    whole_buffer: torch.Tensor,
    ep_group: ProcessGroup,
) -> None:
    """
    Perform expert weights rearrangement of one layer.
    """
    local2global = partial(
        idx_local_to_global,
        local_cnt=num_local_experts,
        ep_rank=ep_rank,
    )

    # 0. Do nothing for experts that did not change.
    is_unchanged = [
        old_indices[local2global(i)] == new_indices[local2global(i)]
        for i in range(num_local_experts)
    ]

    # 1. Perform weight copy inside the local rank.
    is_received_locally = is_unchanged[:]
    for src in range(num_local_experts):
        src_global = local2global(src)
        for dst in range(num_local_experts):
            dst_global = local2global(dst)
            if is_received_locally[dst]:
                continue
            if old_indices[src_global] == -1 or new_indices[dst_global] == -1:
                continue
            if old_indices[src_global] == new_indices[dst_global]:
                is_received_locally[dst] = True
                for weight, buffer in zip(expert_weights,
                                          expert_weights_buffer):
                    buffer[dst].copy_(weight[src])

    p2p_ops: list[P2POp] = []

    # 2. Initiate sending of weights.
    experts_send_loc: dict[int, int] = {}
    for src in range(num_local_experts):
        expert = old_indices[local2global(src)]
        if expert == -1:
            continue
        if expert in experts_send_loc:
            continue
        experts_send_loc[expert] = src

    # We need to sort here to match send/recv
    for expert, src in sorted(experts_send_loc.items()):
        ranks_to_send, ranks_to_recv = get_ep_ranks_with_expert(
            expert,
            num_local_experts,
            old_indices,
            new_indices,
        )

        # Calculate the ranks to send by this rank
        num_dst_per_sender = len(ranks_to_recv) // len(ranks_to_send)
        sender_pos = ranks_to_send.index(ep_rank)
        recv_begin = sender_pos * num_dst_per_sender
        recv_end = recv_begin + num_dst_per_sender
        recv_ranks = ranks_to_recv[recv_begin:recv_end]

        # Tackle remainders
        remainder_start = len(ranks_to_send) * num_dst_per_sender
        recver_pos = remainder_start + sender_pos
        if recver_pos < len(ranks_to_recv):
            recv_ranks.append(ranks_to_recv[recver_pos])

        for dst in recv_ranks:
            dst_global = get_global_rank(ep_group, dst)
            p2p_ops += [
                P2POp(
                    torch.distributed.isend,
                    torch.cat([w[src].view(torch.uint8) for w in expert_weights], dim=0),
                    dst_global,
                )
            ]

    # 3. Initiate receiving of weights.
    experts_recv_loc: dict[int, int] = {}
    for dst in range(num_local_experts):
        if is_received_locally[dst]:
            continue
        expert = new_indices[local2global(dst)]
        if expert == -1:
            continue
        if expert in experts_recv_loc:
            continue
        experts_recv_loc[expert] = dst

    # We need to sort here to match send/recv
    for expert, dst in sorted(experts_recv_loc.items()):
        ranks_to_send, ranks_to_recv = get_ep_ranks_with_expert(
            expert,
            num_local_experts,
            old_indices,
            new_indices,
        )

        # Calculate the rank to recv by this rank
        num_dst_per_sender = len(ranks_to_recv) // len(ranks_to_send)
        recver_pos = ranks_to_recv.index(ep_rank)
        remainder_start = len(ranks_to_send) * num_dst_per_sender
        if recver_pos < remainder_start:
            src = ranks_to_send[recver_pos // num_dst_per_sender]
        else:
            src = ranks_to_send[recver_pos - remainder_start]

        src_global = get_global_rank(ep_group, src)
        p2p_ops += [
            P2POp(
                torch.distributed.irecv,
                whole_buffer[dst],
                src_global,
            )
        ]

    # 4. Execute the P2P operations. The real communication happens here.
    if p2p_ops:
        reqs = batch_isend_irecv(p2p_ops)
        for req in reqs:
            req.wait()

    # 5. Copy the weights from the buffer back to the original weights.
    for dst in range(num_local_experts):
        if is_unchanged[dst]:
            continue
        if is_received_locally[dst]:
            for weight, buffer in zip(expert_weights, expert_weights_buffer):
                weight[dst].copy_(buffer[dst])
        else:
            expert = new_indices[local2global(dst)]
            if expert == -1:
                continue
            src = experts_recv_loc[expert]
            for weight, buffer in zip(expert_weights, expert_weights_buffer):
                weight[dst].copy_(buffer[src])


def rearrange_expert_weights_inplace(
    old_global_expert_indices: torch.Tensor,
    new_global_expert_indices: torch.Tensor,
    expert_weights: Sequence[Iterable[torch.Tensor]],
    ep_group: ProcessGroup,
    is_profile: bool = False,
    rank_mapping: Optional[dict[int, int]] = None,
) -> None:
    """
    Rearranges the expert weights in place according to the new expert indices.

    The value of the indices arguments are logical indices of the experts,
    while keys are physical.

    Args:
        old_global_expert_indices: Shape (num_moe_layers, num_physical_experts).
        new_global_expert_indices: Shape (num_moe_layers, num_physical_experts).
        expert_weights: A sequence of shape (num_moe_layers)(weight_count)
            of tensors of shape (num_local_physical_experts, hidden_size_i).
            For example, a linear layer may have up and down projection,
            so weight_count = 2. Each weight's hidden size can be different.
        ep_group: The device process group for expert parallelism.
        is_profile (bool): If `True`, do not perform any actual weight copy.
            This is used during profile run, where we only perform dummy
            communications to reserve enough memory for the buffers.
        rank_mapping: A dictionary mapping old rank to new rank.
    """
    if rank_mapping is not None:
        if len(rank_mapping) == ep_group.size():
            # scale down
            new_global_expert_indices = \
                _map_new_expert_indices_with_rank_mapping(
                new_global_expert_indices,
                rank_mapping,
            )
        else:
            # scale up
            old_global_expert_indices = \
                _map_old_expert_indices_with_rank_mapping(
                old_global_expert_indices,
                rank_mapping,
                ep_group.size(),
            )

    assert old_global_expert_indices.shape[
        1] == new_global_expert_indices.shape[1]

    num_moe_layers, num_physical_experts = old_global_expert_indices.shape
    assert len(expert_weights) == num_moe_layers

    num_local_physical_experts = next(iter(expert_weights[0])).shape[0]
    assert new_global_expert_indices.shape == (num_moe_layers,
                                               num_physical_experts)

    ep_rank = ep_group.rank()
    ep_size = ep_group.size()
    assert num_physical_experts == ep_size * num_local_physical_experts

    # A buffer to hold the expert weights in one layer during the exchange.
    # NOTE: Currently we assume the same weights across different layers
    # have the same shape.
    # expert_weights_buffer = [torch.empty_like(w) for w in expert_weights[0]]
    total_hidden_dim_in_bytes = 0
    for w in expert_weights[0]:
        bpe = w.element_size()
        total_hidden_dim_in_bytes += (w.shape[1] * bpe)

    whole_buffer = torch.empty(num_local_physical_experts,
                               total_hidden_dim_in_bytes,
                               dtype=torch.uint8,
                               device=expert_weights[0][0].device)

    expert_weights_buffer = []

    start_idx = 0
    for w in expert_weights[0]:
        stride = w.shape[1] * w.element_size()
        expert_weights_buffer.append(whole_buffer[:, start_idx: start_idx + stride].view(w.dtype))
        start_idx += stride

    if is_profile:
        # Maximum send size is to send all local experts to all ranks,
        # So we use a dummy `all_gather` to reserve enough communication buffer
        for weight, buffer in zip(expert_weights[0], expert_weights_buffer):
            # A `/dev/null`-like buffer to avoid real memory allocation
            dummy_recv_buffer = [buffer for _ in range(ep_size)]
            # NOTE(bowen): Needed this barrier to avoid OOM during actual
            # execution. I'm not very sure why this is needed
            torch.distributed.barrier()
            all_gather(
                dummy_recv_buffer,
                weight,
                group=ep_group,
            )
        return

    for layer in range(num_moe_layers):
        # NOTE(bowen): We need this synchronize to run, but I don't know why.
        # If you figure out the reason, please let me know -- thank you!
        torch.cuda.synchronize()
        shuffle_layer(
            num_local_physical_experts,
            ep_rank,
            old_global_expert_indices[layer].tolist(),
            new_global_expert_indices[layer].tolist(),
            expert_weights[layer],
            expert_weights_buffer,
            whole_buffer,
            ep_group,
        )


# ---------------------------------------------------------------------------
# Async EPLB: move_to_buffer / move_from_buffer / transfer_layer
#
# Ported from vLLM v0.14.1 with minimal modifications.
# ---------------------------------------------------------------------------


@dataclass
class RecvMetadata:
    """Metadata describing remote receives during EPLB rebalancing."""

    recv_primary_mask: np.ndarray
    recv_count: int
    recv_expert_ids: np.ndarray
    recv_dst_rows: np.ndarray


MoveToBufferResult = tuple[np.ndarray, np.ndarray, RecvMetadata]


def get_ep_ranks_with_experts_batch(
    expert_ids: np.ndarray,
    num_local_experts: int,
    old_indices: np.ndarray,
    new_indices: np.ndarray,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """
    Vectorised batch version of get_ep_ranks_with_expert (from v0.14.1).

    Returns:
        A tuple of two dicts mapping expert_id to:
        - ranks_to_send: ranks that have this expert and need to send.
        - ranks_to_recv: ranks that need to receive this expert.
    """
    ranks_to_send_map: dict[int, list[int]] = {}
    ranks_to_recv_map: dict[int, list[int]] = {}

    if expert_ids.size == 0:
        return ranks_to_send_map, ranks_to_recv_map

    unique_experts = np.unique(expert_ids)
    num_positions = len(old_indices)
    position_indices = np.arange(num_positions, dtype=np.int32)

    old_relevant_mask = np.isin(old_indices, unique_experts)
    new_relevant_mask = np.isin(new_indices, unique_experts)

    if np.any(old_relevant_mask):
        old_relevant_positions = position_indices[old_relevant_mask]
        old_relevant_experts = old_indices[old_relevant_mask]
        old_relevant_ranks = old_relevant_positions // num_local_experts

        sort_order = np.lexsort(
            (old_relevant_positions, old_relevant_experts)
        )
        sorted_experts = old_relevant_experts[sort_order]
        sorted_ranks = old_relevant_ranks[sort_order]

        expert_boundaries = np.concatenate(
            [[0],
             np.where(np.diff(sorted_experts) != 0)[0] + 1,
             [len(sorted_experts)]]
        )

        for i in range(len(expert_boundaries) - 1):
            start, end = expert_boundaries[i], expert_boundaries[i + 1]
            expert = int(sorted_experts[start])
            expert_ranks = sorted_ranks[start:end]
            _, unique_idx = np.unique(expert_ranks, return_index=True)
            unique_ranks = expert_ranks[np.sort(unique_idx)]
            ranks_to_send_map[expert] = unique_ranks.tolist()

    if np.any(new_relevant_mask):
        new_relevant_positions = position_indices[new_relevant_mask]
        new_relevant_experts = new_indices[new_relevant_mask]
        new_relevant_ranks = new_relevant_positions // num_local_experts

        sort_order = np.lexsort(
            (new_relevant_positions, new_relevant_experts)
        )
        sorted_experts = new_relevant_experts[sort_order]
        sorted_ranks = new_relevant_ranks[sort_order]

        expert_boundaries = np.concatenate(
            [[0],
             np.where(np.diff(sorted_experts) != 0)[0] + 1,
             [len(sorted_experts)]]
        )

        for i in range(len(expert_boundaries) - 1):
            start, end = expert_boundaries[i], expert_boundaries[i + 1]
            expert = int(sorted_experts[start])
            expert_ranks = sorted_ranks[start:end]
            _, unique_idx = np.unique(expert_ranks, return_index=True)
            unique_ranks = expert_ranks[np.sort(unique_idx)]
            send_ranks_set = set(ranks_to_send_map.get(expert, []))
            recv_ranks_actual = [
                int(r) for r in unique_ranks if r not in send_ranks_set
            ]
            ranks_to_recv_map[expert] = recv_ranks_actual

    for expert in unique_experts:
        expert = int(expert)
        if expert not in ranks_to_send_map:
            ranks_to_send_map[expert] = []
        if expert not in ranks_to_recv_map:
            ranks_to_recv_map[expert] = []

    return ranks_to_send_map, ranks_to_recv_map


def move_to_buffer_gcu(
    num_local_experts: int,
    old_indices: np.ndarray,
    new_indices: np.ndarray,
    expert_weights: Iterable[torch.Tensor],
    expert_weights_buffers: Sequence[torch.Tensor],
    cuda_stream: Optional[torch.cuda.Stream],
    ep_group: ProcessGroup,
) -> MoveToBufferResult:
    """
    Rearranges expert weights during EPLB rebalancing (phase 1).

    Ported from vLLM v0.14.1 ``move_to_buffer`` with minimal changes.
    """
    assert old_indices.shape == new_indices.shape
    ep_rank = ep_group.rank()

    recv_primary_mask = np.zeros((num_local_experts,), dtype=np.bool_)
    send_expert_ids = np.full((num_local_experts,), -1, dtype=np.int64)
    send_src_rows = np.full((num_local_experts,), -1, dtype=np.int32)
    recv_expert_ids = np.full((num_local_experts,), -1, dtype=np.int64)
    recv_dst_rows = np.full((num_local_experts,), -1, dtype=np.int32)

    base = ep_rank * num_local_experts
    local_rows = np.arange(num_local_experts, dtype=np.int32)
    local_global = base + local_rows

    old_local_expert_ids = old_indices[local_global]
    new_local_expert_ids = new_indices[local_global]

    is_unchanged = old_local_expert_ids == new_local_expert_ids

    new_valid = new_local_expert_ids != -1
    can_recv_local = np.isin(
        new_local_expert_ids, old_local_expert_ids, assume_unique=False
    )
    is_received_locally = np.logical_or(
        is_unchanged, np.logical_and(new_valid, can_recv_local)
    )

    send_count = 0
    valid_old = old_local_expert_ids != -1
    if np.any(valid_old):
        uniq_experts, first_idx = np.unique(
            old_local_expert_ids[valid_old], return_index=True
        )
        filtered_rows = local_rows[valid_old]
        src_rows = filtered_rows[first_idx]
        send_count = int(uniq_experts.shape[0])
        send_expert_ids[:send_count] = uniq_experts
        send_src_rows[:send_count] = src_rows

    recv_count = 0
    need_recv_mask = np.logical_and(~is_received_locally, new_valid)
    if np.any(need_recv_mask):
        desired_experts = new_local_expert_ids[need_recv_mask]
        desired_dsts = local_rows[need_recv_mask]
        uniq_recv_experts, uniq_indices = np.unique(
            desired_experts, return_index=True
        )
        dst_rows = desired_dsts[uniq_indices]
        recv_count = int(uniq_recv_experts.shape[0])
        recv_expert_ids[:recv_count] = uniq_recv_experts
        recv_dst_rows[:recv_count] = dst_rows
        recv_primary_mask[dst_rows] = True

    eligible_local_buffer_mask = np.logical_and(
        ~is_unchanged, is_received_locally
    )

    # 1. Local moves into tmp buffers
    if bool(eligible_local_buffer_mask.any()) and send_count > 0:
        dest_indices = np.nonzero(eligible_local_buffer_mask)[0].tolist()
        expert_to_src_map = dict(
            zip(send_expert_ids[:send_count], send_src_rows[:send_count])
        )
        for dst in dest_indices:
            expert = new_local_expert_ids[dst]
            src_local = expert_to_src_map.get(expert, -1)
            if src_local != -1:
                for w, b in zip(expert_weights, expert_weights_buffers):
                    b[dst].copy_(w[src_local], non_blocking=True)

    p2p_ops: list[P2POp] = []

    ep_size = ep_group.size()
    rank_to_global = {
        rank: get_global_rank(ep_group, rank) for rank in range(ep_size)
    }

    # 2. Post sends — one P2POp per weight tensor (v0.14.1 style)
    if send_count > 0:
        experts = send_expert_ids[:send_count]
        srcs = send_src_rows[:send_count]
        order = np.argsort(experts, kind="stable")
        experts = experts[order]
        srcs = srcs[order]

        send_map, recv_map = get_ep_ranks_with_experts_batch(
            experts, num_local_experts, old_indices, new_indices,
        )

        for expert, src in zip(experts.tolist(), srcs.tolist()):
            ranks_to_send = send_map[expert]
            ranks_to_recv = recv_map[expert]
            if not ranks_to_send or not ranks_to_recv:
                continue
            num_dst_per_sender = len(ranks_to_recv) // len(ranks_to_send)
            sender_pos = ranks_to_send.index(ep_rank)
            recv_begin = sender_pos * num_dst_per_sender
            recv_end = recv_begin + num_dst_per_sender
            recv_ranks = ranks_to_recv[recv_begin:recv_end]
            remainder_start = len(ranks_to_send) * num_dst_per_sender
            recver_pos = remainder_start + sender_pos
            if recver_pos < len(ranks_to_recv):
                recv_ranks.append(ranks_to_recv[recver_pos])
            for dst in recv_ranks:
                dst_global = rank_to_global[dst]
                p2p_ops += [
                    P2POp(
                        torch.distributed.isend,
                        w[src],
                        dst_global,
                    )
                    for w in expert_weights
                ]

    # 3. Post recvs — one P2POp per buffer tensor (v0.14.1 style)
    if recv_count > 0:
        experts = recv_expert_ids[:recv_count]
        dsts = recv_dst_rows[:recv_count]
        order = np.argsort(experts, kind="stable")
        experts = experts[order]
        dsts = dsts[order]

        send_map, recv_map = get_ep_ranks_with_experts_batch(
            experts, num_local_experts, old_indices, new_indices,
        )

        for expert, dst in zip(experts.tolist(), dsts.tolist()):
            ranks_to_send = send_map[expert]
            ranks_to_recv = recv_map[expert]
            if not ranks_to_send or not ranks_to_recv:
                continue
            num_dst_per_sender = len(ranks_to_recv) // len(ranks_to_send)
            recver_pos = ranks_to_recv.index(ep_rank)
            remainder_start = len(ranks_to_send) * num_dst_per_sender
            if recver_pos < remainder_start:
                src = ranks_to_send[recver_pos // num_dst_per_sender]
            else:
                src = ranks_to_send[recver_pos - remainder_start]
            src_global = rank_to_global[src]
            p2p_ops += [
                P2POp(
                    torch.distributed.irecv,
                    b[dst],
                    src_global,
                )
                for b in expert_weights_buffers
            ]

    # 4. Execute the P2P operations
    if p2p_ops and cuda_stream is not None:
        with torch.cuda.stream(cuda_stream):
            reqs = batch_isend_irecv(p2p_ops)
            for req in reqs:
                req.wait()
    elif p2p_ops:
        reqs = batch_isend_irecv(p2p_ops)
        for req in reqs:
            req.wait()

    return (
        is_unchanged,
        is_received_locally,
        RecvMetadata(
            recv_primary_mask=recv_primary_mask,
            recv_count=recv_count,
            recv_expert_ids=recv_expert_ids,
            recv_dst_rows=recv_dst_rows,
        ),
    )


def move_from_buffer_gcu(
    expert_weights: Iterable[torch.Tensor],
    expert_weights_buffers: Sequence[torch.Tensor],
    is_unchanged: np.ndarray,
    is_received_locally: np.ndarray,
    recv_metadata: RecvMetadata,
    new_indices: np.ndarray,
    ep_rank: int,
) -> None:
    """
    Copies expert weights from communication buffers back to the target
    weight tensors after EPLB rebalancing (phase 2).

    Ported from vLLM v0.14.1 ``move_from_buffer``.
    """
    recv_primary_mask = recv_metadata.recv_primary_mask
    recv_count = recv_metadata.recv_count
    recv_expert_ids = recv_metadata.recv_expert_ids
    recv_dst_rows = recv_metadata.recv_dst_rows
    num_local_experts = is_unchanged.shape[0]

    copy_mask = np.logical_or(is_received_locally, recv_primary_mask)
    dest_mask_np = np.logical_and(~is_unchanged, copy_mask)
    if bool(dest_mask_np.any()):
        dest_indices = np.nonzero(dest_mask_np)[0].tolist()
        for dst in dest_indices:
            for w, b in zip(expert_weights, expert_weights_buffers):
                w[dst].copy_(b[dst], non_blocking=True)

    if recv_count == 0:
        return

    # Duplicate remote received rows to non-primary duplicate dsts
    base = ep_rank * num_local_experts
    local_experts = new_indices[
        base + np.arange(num_local_experts, dtype=np.int32)
    ]
    duplicate_mask = np.logical_and(
        np.logical_and(~is_unchanged, ~is_received_locally),
        np.logical_and(~recv_primary_mask, local_experts != -1),
    )
    if not bool(duplicate_mask.any()):
        return

    dup_dst_rows = np.nonzero(duplicate_mask)[0]
    dup_experts = local_experts[dup_dst_rows]

    prim_experts = recv_expert_ids[:recv_count]
    prim_dsts = recv_dst_rows[:recv_count]
    order = np.argsort(prim_experts, kind="stable")
    prim_experts_sorted = prim_experts[order]
    prim_dsts_sorted = prim_dsts[order]
    pos = np.searchsorted(prim_experts_sorted, dup_experts)
    valid = np.logical_and(
        pos < prim_experts_sorted.shape[0],
        prim_experts_sorted[
            np.minimum(pos, prim_experts_sorted.shape[0] - 1)
        ] == dup_experts,
    )
    if not bool(valid.any()):
        return

    matched_dst_rows = dup_dst_rows[valid]
    matched_src_rows = prim_dsts_sorted[pos[valid]]

    for dst, src in zip(matched_dst_rows.tolist(), matched_src_rows.tolist()):
        for w in expert_weights:
            w[dst].copy_(w[src], non_blocking=True)


async def transfer_layer_gcu(
    old_global_expert_indices: torch.Tensor,
    new_global_expert_indices: torch.Tensor,
    expert_weights: Sequence[Iterable[torch.Tensor]],
    expert_weights_buffer: Sequence[torch.Tensor],
    ep_group: ProcessGroup,
    is_profile: bool = False,
    layer: int = 0,
    cuda_stream: Optional[torch.cuda.Stream] = None,
    rank_mapping: Optional[dict[int, int]] = None,
) -> MoveToBufferResult:
    """
    Transfer a single MoE layer's expert weights into the buffer (async).

    Ported from vLLM v0.14.1 ``transfer_layer``.
    """
    if rank_mapping is not None:
        if len(rank_mapping) == ep_group.size():
            new_global_expert_indices = _map_new_expert_indices_with_rank_mapping(
                new_global_expert_indices, rank_mapping,
            )
        else:
            old_global_expert_indices = _map_old_expert_indices_with_rank_mapping(
                old_global_expert_indices, rank_mapping, ep_group.size(),
            )

    assert old_global_expert_indices.shape[1] == new_global_expert_indices.shape[1]
    num_moe_layers, num_physical_experts = old_global_expert_indices.shape
    assert len(expert_weights) == num_moe_layers
    num_local_physical_experts = next(iter(expert_weights[0])).shape[0]
    assert num_physical_experts == ep_group.size() * num_local_physical_experts

    old_np = old_global_expert_indices.cpu().numpy()
    new_np = new_global_expert_indices.cpu().numpy()

    if is_profile:
        return (
            np.ones(num_local_physical_experts, dtype=np.bool_),
            np.ones(num_local_physical_experts, dtype=np.bool_),
            RecvMetadata(
                recv_primary_mask=np.zeros(
                    num_local_physical_experts, dtype=np.bool_,
                ),
                recv_count=0,
                recv_expert_ids=np.full(
                    num_local_physical_experts, -1, dtype=np.int64,
                ),
                recv_dst_rows=np.full(
                    num_local_physical_experts, -1, dtype=np.int32,
                ),
            ),
        )

    return move_to_buffer_gcu(
        num_local_experts=num_local_physical_experts,
        old_indices=old_np[layer],
        new_indices=new_np[layer],
        expert_weights=expert_weights[layer],
        expert_weights_buffers=expert_weights_buffer,
        cuda_stream=cuda_stream,
        ep_group=ep_group,
    )
