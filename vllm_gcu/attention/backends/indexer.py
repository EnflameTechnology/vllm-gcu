from unittest.mock import patch
from functools import partial
import torch
from typing import ClassVar
from vllm.distributed import get_tp_group
from vllm.v1.attention.backends.mla.indexer import (DeepseekV32IndexerMetadataBuilder,
                                                    get_max_prefill_buffer_size,
                                                    DeepseekV32IndexerMetadata,
                                                    DeepseekV32IndexerPrefillChunkMetadata)
from vllm.v1.attention.backends.utils import (AttentionCGSupport,
                                              AttentionMetadataBuilder,
                                              CommonAttentionMetadata,
                                              split_decodes_and_prefills)
from vllm.v1.attention.backends.utils import AttentionMetadataBuilder
from vllm_gcu.attention.backends.mla_v1 import customized_split_decodes_and_prefills
from vllm_gcu.distributed.sp import scatter

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
        indexer_parallel = self.vllm_config.additional_config.get("indexer_parallel", '')
        self.indexer_use_sp_q = indexer_parallel == 'sp_q'

    def chunk_sequence_parallel(
        self,
        chunk_in: DeepseekV32IndexerPrefillChunkMetadata,
        cu_seq_q,
    ):
        sum_seq_q = chunk_in.token_end - chunk_in.token_start

        tp_group = get_tp_group()
        sp_seq_q = scatter(sum_seq_q, tp_group.world_size)
        # q[chunk_in.token_start:chunk_in.token_end][sp_chunk_start:sp_chunk_end]
        sp_chunk_start = sum(sp_seq_q[: tp_group.rank_in_group])
        sp_chunk_end = sum(sp_seq_q[: tp_group.rank_in_group + 1])

        sp_req_offset = torch.sum(cu_seq_q <= chunk_in.token_start).item() - 1
        sp_req_start = torch.sum(cu_seq_q <= sp_chunk_start + chunk_in.token_start).item() - 1
        sp_req_end = torch.sum(cu_seq_q < sp_chunk_end + chunk_in.token_start).item()
        sp_req_start = sp_req_start - sp_req_offset
        sp_req_end = sp_req_end - sp_req_offset

        chunk_out = DeepseekV32IndexerPrefillChunkMetadata(
            block_table=chunk_in.block_table[sp_req_start:sp_req_end],
            cu_seqlen_ks=chunk_in.cu_seqlen_ks[sp_chunk_start:sp_chunk_end],
            cu_seqlen_ke=chunk_in.cu_seqlen_ke[sp_chunk_start:sp_chunk_end],
            cu_seq_lens=chunk_in.cu_seq_lens[sp_req_start:sp_req_end+1],
            total_seq_lens=chunk_in.total_seq_lens,
            token_start=sp_chunk_start + chunk_in.token_start,
            token_end=sp_chunk_end + chunk_in.token_start,
            num_reqs=sp_req_end - sp_req_start,
        )
        chunk_out.gathered_slice = slice(chunk_in.token_start, chunk_in.token_end)
        return chunk_out

    def build(self,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> DeepseekV32IndexerMetadata:
        with patch(
                'vllm.v1.attention.backends.mla.indexer.split_decodes_and_prefills',
                partial(customized_split_decodes_and_prefills, builder = self, \
                         require_uniform=True)):
            r = super().build(common_prefix_len, common_attn_metadata,
                              fast_build)
            if self.indexer_use_sp_q and r.prefill:
                r.prefill.chunks = [
                    self.chunk_sequence_parallel(
                        chunk, common_attn_metadata.query_start_loc_cpu
                    )
                    for chunk in r.prefill.chunks
                ]

            return r

@staticmethod
def get_builder_cls() -> type["GCUDeepseekV32IndexerMetadataBuilder"]:
    return GCUDeepseekV32IndexerMetadataBuilder