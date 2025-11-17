import torch
from typing import Optional
from vllm.utils import is_pin_memory_available
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler, random_sample
from vllm.v1.sample.sampler import Sampler, _SAMPLING_EPS
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.config import LogprobsMode, get_current_vllm_config
from vllm.platforms import current_platform
from vllm.distributed.parallel_state import get_tp_group
from vllm_gcu.utils import scatter

_SAMPLING_EPS_INV = 1.0 / _SAMPLING_EPS


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
    world_size: int = 1,
) -> torch.Tensor:
    if world_size > 1:
        tp_group = get_tp_group()
        local_rank = tp_group.rank_in_group

        scatter_counts = scatter(logits.shape[0], world_size)
        start = sum(scatter_counts[:local_rank])
        end = sum(scatter_counts[:local_rank + 1])

        dp_logits = logits[start:end]
        dp_k = None if k is None else k[start:end]
        dp_p = None if p is None else p[start:end]
        torch.ops._C.top_k_top_p(dp_logits, dp_k, dp_p)

        logits = tp_group.all_gatherv(dp_logits, sizes=scatter_counts)
    else:
        torch.ops._C.top_k_top_p(logits, k, p)

    return logits


class ParallelTopKTopPSampler(TopKTopPSampler):

    def __init__(self, logprobs_mode: LogprobsMode = "raw_logprobs") -> None:
        super().__init__(logprobs_mode)

        vllm_config = get_current_vllm_config()
        self.enable_dp_parallel = not vllm_config.additional_config.get("disable_dp_sampler", False)
        self.forward = self.forward_oot

    def forward_oot(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: Optional[torch.Tensor],
        p: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not current_platform.has_device_capability(140):
            return super().forward_native(logits, generators, k, p)

        tp_group = get_tp_group()
        world_size = tp_group.world_size if self.enable_dp_parallel else 1

        logits = apply_top_k_top_p(logits, k, p, world_size)

        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

        sampled_tokens = torch.empty(size=(logits.shape[0], ),
                                     dtype=torch.int32,
                                     device=logits.device)

        q = torch.empty_like(logits, dtype=torch.float32, device=logits.device)
        if len(generators) != logits.shape[0]:
            q.exponential_()
        if generators:
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)

        torch.ops._C.topk_topp_random_sampler_from_logits(sampled_tokens,
                                                          logits,
                                                          None,
                                                          None,
                                                          q,
                                                          dim=-1)
        return sampled_tokens, logits_to_return


class GCUSampler(Sampler):

    def __init__(self, logprobs_mode: LogprobsMode = "raw_logprobs"):
        super().__init__(logprobs_mode)
        self.topk_topp_sampler = ParallelTopKTopPSampler(logprobs_mode)

    def apply_temperature(
        self,
        logits: torch.Tensor,
        temp_inv: torch.Tensor,
        all_random: bool,
    ) -> torch.Tensor:
        # Use in-place division to avoid creating a new tensor.
        # Avoid division by zero if there are greedy requests.
        if not all_random:
            temp_inv = torch.where(temp_inv > _SAMPLING_EPS_INV, 1.0, temp_inv)
        return logits.mul_(temp_inv.unsqueeze(dim=1))

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sample logits based on sampling metadata.

        The various logits processing functions called in this method
        may update the logits tensor in-place.
        """

        assert not (sampling_metadata.all_greedy
                    and sampling_metadata.all_random)
        if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.greedy_sample(logits)
            if sampling_metadata.all_greedy:
                processed_logprobs = None
                if sampling_metadata.max_num_logprobs is not None:
                    if self.logprobs_mode == "processed_logits":
                        processed_logprobs = logits
                    elif self.logprobs_mode == "processed_logprobs":
                        processed_logprobs = self.compute_logprobs(logits)
                return greedy_sampled, processed_logprobs

        assert sampling_metadata.temperature is not None

        # Apply temperature.
        logits = self.apply_temperature(logits, sampling_metadata.temperature,
                                        sampling_metadata.all_random)

        # Apply logits processors that only apply to random sampling
        # (argmax invariant)
        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)

        # Apply top_k and/or top_p.
        random_sampled, processed_logprobs = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )

        if greedy_sampled is None:
            return random_sampled, processed_logprobs

        # 以下做了修改，社区版 where 的判断条件是 sampling_metadata.temperature < _SAMPLING_EPS
        # 但因为GCU版对temperature做了inversion处理，所以这里改成了 > _SAMPLING_EPS_INV
        sampled = torch.where(
            sampling_metadata.temperature > _SAMPLING_EPS_INV,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,  # Reuse tensor
        )
        return sampled, processed_logprobs
