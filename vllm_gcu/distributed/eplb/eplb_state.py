import torch
from torch.distributed import ProcessGroup, all_reduce
from vllm.model_executor.models.interfaces import MixtureOfExperts
from vllm.distributed.parallel_state import get_ep_group
from vllm.distributed.eplb.eplb_state import logger
import vllm_gcu.envs as gcu_envs
import vllm.envs as envs
from vllm_gcu.utils import get_tx_ctx, get_tx_mark_func
import orjson


if envs.VLLM_NVTX_SCOPES_FOR_PROFILING:
    step_idx = 0

def step(self,
        model: MixtureOfExperts,
        is_dummy: bool = False,
        is_profile: bool = False,
        log_stats: bool = False) -> None:
    """
    Step the EPLB state.

    Args:
        model (MixtureOfExperts): The MoE model.
        is_dummy (bool): If `True`, this is a dummy step and the load
            metrics recorded in this forward pass will not count.
            Defaults to `False`.
        is_profile (bool): If `True`, perform a dummy rearrangement
            with maximum communication cost. This is used in
            `profile_run` to reserve enough memory
            for the communication buffer.
        log_stats (bool): If `True`, log the expert load metrics.

    # Stats
        The metrics are all summed up across layers.
        - `avg_tokens`: The average load across ranks.
        - `max_tokens`: The maximum load across ranks.
        - `balancedness`: The ratio of average load to maximum load.
    """

    if is_profile:
        self.rearrange(model, is_profile=True)
        return

    if is_dummy:
        # Do not record load metrics for dummy steps
        self.expert_load_pass.zero_()

    if envs.VLLM_NVTX_SCOPES_FOR_PROFILING:
        global step_idx

        message = "expert_load_pass"
        color = "green"
        domain = "VLLM"
        category = "Eplb"
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
            }
        }

        payload_str = orjson.dumps(payload)

        step_idx += 1

        tx_mark_func = get_tx_mark_func()
        tx_mark_func(message, color, domain, category, payload_str)

    if log_stats:
        # total_expert_load_pass: (num_moe_layers, num_physical_experts)
        total_expert_load_pass = self.expert_load_pass.clone()

        # Collect load metrics from all ranks
        ep_group = get_ep_group().device_group
        all_reduce(total_expert_load_pass, group=ep_group)

        # num_tokens_per_rank: (num_moe_layers, num_ranks)
        num_tokens_per_rank = total_expert_load_pass.reshape(
            total_expert_load_pass.shape[0], ep_group.size(),
            -1).sum(dim=-1).float()

        # Compute balancedness ratio:
        # for each layer:
        #   (mean load across ranks) / (max load across ranks)
        avg_tokens_per_layer_tensor = num_tokens_per_rank.mean(dim=1)
        max_tokens_per_layer_tensor = num_tokens_per_rank.max(dim=1).values
        min_tokens_per_layer_tensor = num_tokens_per_rank.min(dim=1).values

        avg_tokens_tensor = avg_tokens_per_layer_tensor.sum(dim=0)
        max_tokens_tensor = max_tokens_per_layer_tensor.sum(dim=0)
        min_tokens_tensor = min_tokens_per_layer_tensor.sum(dim=0)

        # Just to make type checker happy
        tokens_tensors: list[float] = torch.stack(
            [avg_tokens_tensor, max_tokens_tensor, min_tokens_tensor]).tolist()
        avg_tokens, max_tokens, min_tokens = tokens_tensors
        balancedness = avg_tokens / max_tokens if max_tokens > 0 else 0.0

        imbalanceness = (
            (max_tokens_per_layer_tensor - avg_tokens_per_layer_tensor) /
             avg_tokens_per_layer_tensor).mean() if \
            torch.all(avg_tokens_per_layer_tensor > 0) else torch.inf

        lower_bound = min_tokens / avg_tokens if avg_tokens > 0 else 0.0
        upper_bound = max_tokens / avg_tokens if avg_tokens > 0 else 0.0

        if ep_group.rank() == 0:
            logger.info(
                "EPLB step: avg_tokens=%.2f, max_tokens=%d, "
                "balancedness=%.4f, imbalanceness=%.4f, lower_bound=%.4f, upper_bound=%.4f",
                avg_tokens, max_tokens, balancedness, imbalanceness, lower_bound, upper_bound)

    # Update the expert load sliding window
    if not is_dummy:
        self.expert_load_window[self.expert_load_window_step] = (
            self.expert_load_pass.clone())
        self.expert_load_window_step += 1
        if self.expert_load_window_step >= self.expert_load_window_size:
            self.expert_load_window_step = 0
        self.expert_load_pass.zero_()

    # Step the expert rearrangement step
    # Note that even if this is a dummy step, we still increment the
    # rearrangement step and perform rearrangement to ensure all ranks are
    # performing collective communication.
    self.expert_rearrangement_step += 1
    if (self.expert_rearrangement_step
            >= self.expert_rearrangement_step_interval):
        self.expert_rearrangement_step = 0
        self.rearrange(model)
