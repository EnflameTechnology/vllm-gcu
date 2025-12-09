import torch
from vllm.v1.attention.backends.mla.indexer import (DeepseekV32IndexerMetadataBuilder,
                                                    get_max_prefill_buffer_size,
                                                    DeepseekV32IndexerBackend)
from vllm.v1.attention.backends.utils import AttentionMetadataBuilder

class GCUDeepseekV32IndexerMetadataBuilder(DeepseekV32IndexerMetadataBuilder):
    def __init__(self, *args, **kwargs):
        AttentionMetadataBuilder.__init__(self, *args, **kwargs)
        scheduler_config = self.vllm_config.scheduler_config
        #NOTE(Chen):an estimated max size of flattened_kv. Need to double check.
        self.max_prefill_buffer_size = get_max_prefill_buffer_size(
            self.vllm_config)
        self.num_speculative_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config else 0)
        # Now deepgemm fp8_paged_mqa_logits does not support next_n > 2
        self.reorder_batch_threshold += min(self.num_speculative_tokens, 1)

        props = torch.cuda.get_device_properties(self.device)
        # patch start
        # get_paged_mqa_logits_metadata op need sip count
        props_str = str(props)
        sip_count_loc = props_str.find("sip_count=")
        self.num_sms = int(props_str[sip_count_loc+len("sip_count="):-1])
        # patch end

        self.decode_lens_buffer = torch.empty(
            (scheduler_config.max_num_seqs, ),
            dtype=torch.int32,
            device=self.device)

        # See: DeepGMM/csrc/apis/attention.hpp
        self.scheduler_metadata_buffer = torch.empty((self.num_sms + 1, 2),
                                                     dtype=torch.int32,
                                                     device=self.device)

@staticmethod
def get_builder_cls() -> type["GCUDeepseekV32IndexerMetadataBuilder"]:
    return GCUDeepseekV32IndexerMetadataBuilder