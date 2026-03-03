"""
Async EPLB step function and helpers.

This module provides an async-capable replacement for EplbState.step().
When patched in (via VLLM_GCU_EPLB_ASYNC_ENABLED=1), expert weight transfers
happen in a background thread one layer at a time, overlapping with
inference.

This file is completely independent from eplb_state.py (the sync step).
The choice of which step function to patch is made in
patch/patch_0_11_0/eplb.py at import time.

The helper functions below correspond to v0.14.1's EplbState methods but
are implemented as standalone functions that operate on v0.11.0's flat
EplbState dataclass (``eplb_state``) plus an AsyncEplbExtension
(``async_ext``) that holds the async-specific state.

    v0.14.1 EplbState method             GCU standalone function
    ──────────────────────────            ──────────────────────────
    _all_ranks_buffer_ready       →      _all_ranks_buffer_ready
    _update_layer_mapping_from_new →     _update_layer_mapping_from_new
    move_to_workspace             →      _move_to_workspace
    post_eplb                     →      _post_eplb
    rearrange (async branch)      →      _rearrange_async
    step      (async sections)    →      async_step
"""

import torch
from torch.distributed import all_reduce

from vllm.distributed.eplb.eplb_state import logger
from vllm.distributed.eplb.rebalance_algo import rebalance_experts
from vllm.distributed.parallel_state import get_ep_group, get_node_count
from vllm.model_executor.models.interfaces import MixtureOfExperts

import vllm.envs as envs
from vllm_gcu.utils import get_tx_mark_func

try:
    import orjson
except ImportError:
    orjson = None

if envs.VLLM_NVTX_SCOPES_FOR_PROFILING:
    step_idx = 0


# ── initialisation (called once from patched EplbState.build) ────────
# Corresponds to v0.14.1 EplbState.start_async_loop

def start_async_loop(
    eplb_state,
    model: MixtureOfExperts,
    device,
    rank_mapping=None,
    is_profile: bool = False,
):
    """
    Initialize the async EPLB extension and start the background worker.

    Called once during ``EplbState.build`` (patched in eplb.py) so that
    the worker is already running before the first ``step`` call.
    This mirrors v0.14.1 where ``start_async_loop`` is invoked inside
    ``load_model`` right after ``add_model``.
    """
    from .async_state import AsyncEplbExtension
    from .async_worker import start_async_worker

    ext = AsyncEplbExtension(model, device)
    eplb_state._async_ext = ext

    ext.async_worker = start_async_worker(
        eplb_state=eplb_state,
        async_ext=ext,
        rank_mapping=rank_mapping,
        is_profile=is_profile,
    )
    logger.info("Async EPLB extension initialised and worker started")


# ── cross-rank readiness ─────────────────────────────────────────────
# Corresponds to v0.14.1 EplbState._all_ranks_buffer_ready

def _all_ranks_buffer_ready(eplb_state, async_ext) -> bool:
    """Check whether ALL EP ranks have their buffer ready."""
    parallel_state = get_ep_group()
    cpu_group = getattr(parallel_state, "cpu_group", None)
    if cpu_group is not None and cpu_group.size() > 1:
        flag = torch.tensor(
            (int(async_ext.ep_buffer_ready),),
            dtype=torch.int32,
            device="cpu",
        )
        all_reduce(flag, group=cpu_group)
        return int(flag.item()) == cpu_group.size()

    device_group = parallel_state.device_group
    if device_group.size() <= 1:
        return bool(async_ext.ep_buffer_ready)

    device = getattr(
        parallel_state, "device",
        eplb_state.physical_to_logical_map.device,
    )
    flag = torch.tensor(
        (int(async_ext.ep_buffer_ready),),
        dtype=torch.int32,
        device=device,
    )
    all_reduce(flag, group=device_group)
    return int(flag.item()) == device_group.size()


# ── per-layer mapping update ─────────────────────────────────────────
# Corresponds to v0.14.1 EplbState._update_layer_mapping_from_new

def _update_layer_mapping_from_new(eplb_state, async_ext, layer: int):
    """Progressively update the EPLB mapping tensors for a single layer."""
    if (
        async_ext.new_physical_to_logical_map is None
        or async_ext.new_logical_to_physical_map is None
        or async_ext.new_logical_replica_count is None
    ):
        return

    target_device = eplb_state.physical_to_logical_map.device
    new_physical = async_ext.new_physical_to_logical_map
    if eplb_state.physical_to_logical_map.shape[1] != new_physical.shape[1]:
        eplb_state.physical_to_logical_map = new_physical.to(target_device)
    else:
        eplb_state.physical_to_logical_map[layer].copy_(
            new_physical[layer].to(target_device)
        )

    logical_device = eplb_state.logical_to_physical_map.device
    new_logical = async_ext.new_logical_to_physical_map[layer].to(
        logical_device
    )
    max_slots = eplb_state.logical_to_physical_map.shape[-1]
    slot_delta = max_slots - new_logical.shape[-1]
    if slot_delta > 0:
        new_logical = torch.nn.functional.pad(
            new_logical, (0, slot_delta), value=-1,
        )
    eplb_state.logical_to_physical_map[layer].copy_(new_logical)

    replica_device = eplb_state.logical_replica_count.device
    eplb_state.logical_replica_count[layer].copy_(
        async_ext.new_logical_replica_count[layer].to(replica_device)
    )


# ── buffer → workspace copy ──────────────────────────────────────────
# Corresponds to v0.14.1 EplbState.move_to_workspace

def _move_to_workspace(eplb_state, async_ext, model, ep_group,
                       is_profile=False):
    """Copy one completed layer from the staging buffer into model weights."""
    from .rebalance_execute import move_from_buffer_gcu

    max_retries = 6
    retries = 0
    while not async_ext.buffer_lock.acquire(blocking=True, timeout=10.0):
        retries += 1
        if retries >= max_retries:
            raise RuntimeError(
                f"Rank {ep_group.rank()}: buffer_lock timeout after "
                f"{max_retries * 10}s"
            )
        logger.warning(
            "Rank %d: EPLB buffer_lock acquire failed, retrying (%d/%d)",
            ep_group.rank(),
            retries,
            max_retries,
        )

    try:
        assert async_ext.new_physical_to_logical_map is not None
        device_index = async_ext.cuda_device_index
        if async_ext.buffer_ready_event is not None and device_index is not None:
            stream = torch.cuda.current_stream(device=device_index)
            stream.wait_event(async_ext.buffer_ready_event)
            async_ext.buffer_ready_event = None

        expert_weights = model.expert_weights[async_ext.layer_to_transfer]
        expert_weights_buffers = async_ext.expert_buffer
        new_indices = (
            async_ext.new_physical_to_logical_map[async_ext.layer_to_transfer]
            .numpy()
        )

        move_from_buffer_gcu(
            expert_weights=expert_weights,
            expert_weights_buffers=expert_weights_buffers,
            is_unchanged=async_ext.is_unchanged,
            is_received_locally=async_ext.is_received_locally,
            recv_metadata=async_ext.recv_metadata,
            new_indices=new_indices,
            ep_rank=ep_group.rank(),
        )

        consumed_event = torch.cuda.Event()
        consumed_event.record()
        async_ext.buffer_consumed_event = consumed_event

        transferred_layer = async_ext.layer_to_transfer
        _update_layer_mapping_from_new(eplb_state, async_ext, transferred_layer)
        async_ext.layer_to_transfer += 1
        async_ext.ep_buffer_ready = 0
        logger.debug(
            "async EPLB: successfully move_to_workspace layer %d",
            transferred_layer,
        )
    finally:
        try:
            async_ext.buffer_lock.release()
        except Exception as e:
            logger.error(
                "Rank %d: buffer_lock release failed in move_to_workspace: %s",
                ep_group.rank(),
                str(e),
            )


# ── post-EPLB finalisation ───────────────────────────────────────────
# Corresponds to v0.14.1 EplbState.post_eplb

def _post_eplb(eplb_state, async_ext, is_profile=False):
    """Finalise async EPLB after all layers have been transferred."""
    assert async_ext.new_physical_to_logical_map is not None
    assert async_ext.new_logical_to_physical_map is not None
    assert async_ext.new_logical_replica_count is not None

    if not is_profile:
        for layer_idx in range(eplb_state.physical_to_logical_map.shape[0]):
            _update_layer_mapping_from_new(eplb_state, async_ext, layer_idx)

    async_ext.new_physical_to_logical_map = None
    async_ext.new_logical_to_physical_map = None
    async_ext.new_logical_replica_count = None


# ── async rearrange (compute mapping, signal worker) ─────────────────
# Corresponds to v0.14.1 EplbState.rearrange (async branch)

def _rearrange_async(eplb_state, async_ext, model, is_profile=False):
    """
    Compute new expert placement and signal the async worker thread.

    Unlike the sync rearrange, this does NOT move any weights — that is
    handled layer-by-layer by the background worker.
    """
    ep_group = get_ep_group().device_group
    ep_rank = ep_group.rank()
    is_main_rank = ep_rank == 0

    start_event = None
    end_event = None
    if is_main_rank:
        if is_profile:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        logger.info(
            "Rearranging experts %s %s...",
            "(async mode)",
            "(profile)" if is_profile else "",
        )

    # Map physical expert load to global logical experts
    logical_expert_load_window = torch.zeros(
        eplb_state.expert_load_window_size,
        model.num_moe_layers,
        model.num_logical_experts,
        dtype=eplb_state.expert_load_window.dtype,
        device=eplb_state.expert_load_window.device,
    )
    logical_expert_load_window.scatter_add_(
        dim=-1,
        index=eplb_state.physical_to_logical_map.unsqueeze(0)
            .expand_as(eplb_state.expert_load_window).long(),
        src=eplb_state.expert_load_window,
    )
    global_expert_load_window = logical_expert_load_window.sum(dim=0)
    all_reduce(global_expert_load_window, group=ep_group)

    num_replicas = model.num_physical_experts
    num_groups = model.num_expert_groups
    num_nodes = get_node_count()
    num_gpus = ep_group.size()

    if num_gpus % num_nodes != 0:
        num_nodes = 1
        logger.warning_once(
            f"num_gpus % num_nodes != 0, "
            "not using hierarchical rearrangement algorithm.\n"
            f"{num_gpus=}, {num_nodes=}"
        )

    (
        new_physical_to_logical_map,
        new_logical_to_physical_map,
        new_logical_replica_count,
    ) = rebalance_experts(
        global_expert_load_window,
        num_replicas,
        num_groups,
        num_nodes,
        num_gpus,
    )

    if is_profile:
        from .rebalance_execute import rearrange_expert_weights_inplace
        rearrange_expert_weights_inplace(
            eplb_state.physical_to_logical_map,
            new_physical_to_logical_map,
            model.expert_weights,
            ep_group,
            True,
        )
        if is_main_rank:
            assert start_event is not None
            assert end_event is not None
            end_event.record()
            end_event.synchronize()
            gpu_elapsed = start_event.elapsed_time(end_event) / 1000.0
            logger.info(
                "Rearranged experts (profile) in %.2f s.",
                gpu_elapsed,
            )
        return

    # Async path: store new maps and signal worker
    max_slots = eplb_state.logical_to_physical_map.shape[-1]
    padded_logical = torch.nn.functional.pad(
        new_logical_to_physical_map,
        (0, max(0, max_slots - new_logical_to_physical_map.shape[-1])),
        value=-1,
    ).to(eplb_state.logical_to_physical_map.device)
    new_replica = new_logical_replica_count.to(
        eplb_state.logical_replica_count.device
    )

    async_ext.new_physical_to_logical_map = new_physical_to_logical_map.cpu()
    async_ext.new_logical_to_physical_map = padded_logical
    async_ext.new_logical_replica_count = new_replica

    async_ext.rebalanced = True
    async_ext.layer_to_transfer = 0
    async_ext.pending_global_ready_check = True

    async_ext.rearrange_event.set()

    if is_main_rank:
        logger.info("Async EPLB: new mapping computed, worker signalled")


# ── async step function (replaces EplbState.step when enabled) ───────
# Corresponds to v0.14.1 EplbState.step (async sections)

def async_step(
    self,
    model: MixtureOfExperts,
    is_dummy: bool = False,
    is_profile: bool = False,
    log_stats: bool = False,
) -> None:
    """
    Async-capable replacement for EplbState.step().

    The common bookkeeping (NVTX, stats, sliding window, step counter)
    is identical to the original sync step.  The only difference is in
    the rearrangement trigger at the end, which delegates to the async
    worker instead of doing a blocking weight shuffle.
    """
    async_ext = self._async_ext  # initialized in EplbState.build (patched)
    ep_group = get_ep_group().device_group

    # ── profile path ─────────────────────────────────────────────────
    if is_profile:
        _rearrange_async(self, async_ext, model, is_profile=True)
        return

    if is_dummy:
        self.expert_load_pass.zero_()

    # ── NVTX profiling (GCU-specific) ────────────────────────────────
    if envs.VLLM_NVTX_SCOPES_FOR_PROFILING:
        global step_idx
        if orjson is not None:
            payload = {
                "step_idx": step_idx,
                "ep_size": get_ep_group().world_size,
                "expert_load_pass": {
                    "shape": list(self.expert_load_pass.shape),
                    "dtype": str(self.expert_load_pass.dtype),
                    "value": self.expert_load_pass.flatten().tolist(),
                },
                "physical_to_logical_map": {
                    "shape": list(self.physical_to_logical_map.shape),
                    "dtype": str(self.physical_to_logical_map.dtype),
                    "value": self.physical_to_logical_map.flatten().tolist(),
                },
            }
            payload_str = orjson.dumps(payload)
            step_idx += 1
            tx_mark_func = get_tx_mark_func()
            tx_mark_func(
                "expert_load_pass", "green", "VLLM", "Eplb", payload_str,
            )

    # ── log stats (GCU-specific extended metrics) ────────────────────
    if log_stats:
        total_expert_load_pass = self.expert_load_pass.clone()
        all_reduce(total_expert_load_pass, group=ep_group)

        num_tokens_per_rank = (
            total_expert_load_pass.reshape(
                total_expert_load_pass.shape[0], ep_group.size(), -1
            )
            .sum(dim=-1)
            .float()
        )

        avg_tokens_per_layer_tensor = num_tokens_per_rank.mean(dim=1)
        max_tokens_per_layer_tensor = num_tokens_per_rank.max(dim=1).values
        min_tokens_per_layer_tensor = num_tokens_per_rank.min(dim=1).values

        avg_tokens_tensor = avg_tokens_per_layer_tensor.sum(dim=0)
        max_tokens_tensor = max_tokens_per_layer_tensor.sum(dim=0)
        min_tokens_tensor = min_tokens_per_layer_tensor.sum(dim=0)

        tokens_tensors: list[float] = torch.stack(
            [avg_tokens_tensor, max_tokens_tensor, min_tokens_tensor]
        ).tolist()
        avg_tokens, max_tokens, min_tokens = tokens_tensors
        balancedness = avg_tokens / max_tokens if max_tokens > 0 else 0.0
        imbalanceness = (
            (max_tokens_per_layer_tensor - avg_tokens_per_layer_tensor)
            / avg_tokens_per_layer_tensor
        ).mean() if torch.all(avg_tokens_per_layer_tensor > 0) else torch.inf
        lower_bound = min_tokens / avg_tokens if avg_tokens > 0 else 0.0
        upper_bound = max_tokens / avg_tokens if avg_tokens > 0 else 0.0

        if ep_group.rank() == 0:
            logger.info(
                "EPLB step: avg_tokens=%.2f, max_tokens=%d, "
                "balancedness=%.4f, imbalanceness=%.4f, "
                "lower_bound=%.4f, upper_bound=%.4f",
                avg_tokens, max_tokens, balancedness,
                imbalanceness, lower_bound, upper_bound,
            )

    # ── update sliding window ────────────────────────────────────────
    if not is_dummy:
        self.expert_load_window[self.expert_load_window_step] = (
            self.expert_load_pass.clone()
        )
        self.expert_load_pass.zero_()
        self.expert_load_window_step += 1
        if self.expert_load_window_step >= self.expert_load_window_size:
            self.expert_load_window_step = 0

    # ── rearrangement step counter ───────────────────────────────────
    self.expert_rearrangement_step += 1

    # ── async: consume ready buffer ──────────────────────────────────
    all_ranks_buffer_ready = False
    if async_ext.pending_global_ready_check:
        all_ranks_buffer_ready = _all_ranks_buffer_ready(self, async_ext)

    if async_ext.ep_buffer_ready and all_ranks_buffer_ready:
        _move_to_workspace(
            self, async_ext, model, ep_group, is_profile=is_profile,
        )
        if async_ext.layer_to_transfer >= model.num_moe_layers:
            _post_eplb(self, async_ext, is_profile)
            async_ext.rebalanced = False
            async_ext.layer_to_transfer = 0
            async_ext.pending_global_ready_check = False
            logger.info(
                "finish async transfer rank %d layer %d",
                ep_group.rank(),
                model.num_moe_layers,
            )

    # ── async: trigger rearrange when interval reached ───────────────
    if self.expert_rearrangement_step >= self.expert_rearrangement_step_interval:
        if async_ext.rebalanced:
            return
        self.expert_rearrangement_step = 0
        _rearrange_async(self, async_ext, model)
