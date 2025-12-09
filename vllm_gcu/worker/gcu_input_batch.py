import torch
from typing import Optional
from vllm.sampling_params import SamplingType
from vllm.v1.worker.gpu_input_batch import InputBatch, CachedRequestState
from vllm_gcu.kernels.rejection_sampler import GCURejectionSampler
import vllm_gcu.envs as gcu_envs

class GCUInputBatch(InputBatch):

    def __init__(self, *args, **kwargs):
        # for custom logits processor, we dont support now
        self.logitsprocs_need_output_token_ids = False
        super().__init__(*args, **kwargs)
        self.sampled_token_ids_cpu: Optional[torch.Tensor] = None
        self.async_copy_ready_event: Optional[torch.cuda.Event] = None

    def add_request(self,
        request: "CachedRequestState",
    ) -> int:
        req_index = super().add_request(request)
        if sampling_params := request.sampling_params:
            if sampling_params.sampling_type == SamplingType.GREEDY:
                self.temperature_cpu[req_index] = float('inf')
            else:
                self.temperature_cpu[req_index] = 1.0 / sampling_params.temperature
        return req_index

    def set_async_sampled_token_ids(
        self,
        sampled_token_ids_cpu: torch.Tensor,
        async_copy_ready_event: torch.cuda.Event,
    ) -> None:
        """
        In async scheduling case, store ref to sampled_token_ids_cpu
        tensor and corresponding copy-ready event. Used to repair
        output_token_ids prior to sampling, if needed by logits processors.
        """
        self.sampled_token_ids_cpu = sampled_token_ids_cpu
        self.async_copy_ready_event = async_copy_ready_event

    def update_async_output_token_ids(self) -> None:
        """
        In async scheduling case, update output_token_ids in sampling metadata
        from prior steps sampled token ids once they've finished copying to CPU.
        This is called right before they are needed by the logits processors.
        """
        output_token_ids = self.sampling_metadata.output_token_ids
        if self.sampled_token_ids_cpu is None or not output_token_ids:
            # Output token ids not needed or not async scheduling.
            return

        assert self.prev_req_id_to_index is not None
        sampled_token_ids = None
        for index, req_id in enumerate(self.req_ids):
            prev_index = self.prev_req_id_to_index.get(req_id)
            if prev_index is None:
                continue
            req_output_token_ids = output_token_ids[index]
            if not req_output_token_ids or req_output_token_ids[-1] != -1:
                # Final output id is not a placeholder, some tokens must have
                # been discarded after a kv-load failure.
                continue
            if sampled_token_ids is None:
                assert self.async_copy_ready_event is not None
                self.async_copy_ready_event.synchronize()
                max_gen_len = self.sampled_token_ids_cpu.shape[-1]
                if max_gen_len != 1:
                    sampled_token_ids = GCURejectionSampler.parse_output(
                        self.sampled_token_ids_cpu,
                        self.vocab_size,
                    )
                else:
                    sampled_token_ids = self.sampled_token_ids_cpu.tolist()
            # Replace placeholder token id with actual sampled id.
            del req_output_token_ids[-1]
            req_output_token_ids.extend(sampled_token_ids[prev_index])

    def _make_prompt_token_ids_tensor(self) -> torch.Tensor:
        num_reqs = self.num_reqs
        max_prompt_len = self.num_prompt_tokens[:num_reqs].max()
        prompt_token_ids_cpu_tensor = torch.empty(
            (self.num_reqs, max_prompt_len),
            device="cpu",
            dtype=torch.int64,
            pin_memory=self.pin_memory,
        )
        prompt_token_ids = prompt_token_ids_cpu_tensor.numpy()
        prompt_token_ids[:] = self.token_ids_cpu[:num_reqs, :max_prompt_len]
        # Use the value of vocab_size as a pad since we don't have a
        # token_id of this value.
        for i in range(num_reqs):
            prompt_token_ids[i, self.num_prompt_tokens[i]:] = self.vocab_size
        
        if gcu_envs.VLLM_GCU_ENABLE_DEEPSEEK_MTP_FUSION:
            # for fused_mtp, no need to transfer to device in advance
            return prompt_token_ids_cpu_tensor
        return prompt_token_ids_cpu_tensor.to(device=self.device,
                                                non_blocking=True)
