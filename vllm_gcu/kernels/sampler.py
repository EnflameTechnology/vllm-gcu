import torch
from typing import Optional
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler, random_sample
from vllm.config import LogprobsMode, get_current_vllm_config
from vllm.platforms import current_platform
from vllm.distributed.parallel_state import get_tp_group
from vllm_gcu.utils import scatter


class ParallelTopKTopPSampler(TopKTopPSampler):

    def __init__(self, logprobs_mode: LogprobsMode = "raw_logprobs") -> None:
        super().__init__(logprobs_mode)

        vllm_config = get_current_vllm_config()
        self.enable_dp_parallel = (not vllm_config.additional_config.get(
            "disable_dp_sampler", False) and
                                   current_platform.has_device_capability(140))
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

        if self.enable_dp_parallel:
            tp_group = get_tp_group()
            world_size = tp_group.world_size
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
