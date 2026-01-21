from unittest.mock import patch
from functools import partial
import torch
from typing import ClassVar
from vllm.v1.attention.backends.mla.indexer import (DeepseekV32IndexerMetadataBuilder,
                                                    get_max_prefill_buffer_size,
                                                    DeepseekV32IndexerMetadata)
from vllm.v1.attention.backends.utils import (AttentionCGSupport,
                                              AttentionMetadataBuilder,
                                              CommonAttentionMetadata,
                                              split_decodes_and_prefills)
from vllm.v1.attention.backends.utils import AttentionMetadataBuilder
from vllm_gcu.attention.backends.mla_v1 import customized_split_decodes_and_prefills

class GCUDeepseekV32IndexerMetadataBuilder(DeepseekV32IndexerMetadataBuilder):
    cudagraph_support: ClassVar[AttentionCGSupport] = \
        AttentionCGSupport.UNIFORM_BATCH
    
    def __init__(self, *args, **kwargs):
        AttentionMetadataBuilder.__init__(self, *args, **kwargs)
        scheduler_config = self.vllm_config.scheduler_config
        #NOTE(Chen):an estimated max size of flattened_kv. Need to double check.
        self.max_prefill_buffer_size = get_max_prefill_buffer_size(
            self.vllm_config)
        self.num_speculative_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config else 0)

        # topsdeepgemmFp8PagedMqaLogits op support next_n=1/2/3/4
        self.reorder_batch_threshold = 4

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

    def build(self,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> DeepseekV32IndexerMetadata:
        with patch(
                'vllm.v1.attention.backends.mla.indexer.split_decodes_and_prefills',
                partial(customized_split_decodes_and_prefills, builder = self, \
                         require_uniform=True)):
            return super().build(common_prefix_len, common_attn_metadata,
                                 fast_build)

@staticmethod
def get_builder_cls() -> type["GCUDeepseekV32IndexerMetadataBuilder"]:
    return GCUDeepseekV32IndexerMetadataBuilder