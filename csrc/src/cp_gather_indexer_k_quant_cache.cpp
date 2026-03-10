/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "cp_gather_indexer_k_quant_cache.h"

#include <topsaten/topsaten_vllm.h>

#include <vector>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void cp_gather_indexer_k_quant_cache(
    const at::Tensor& kv_cache,// [num_blocks, block_size, cache_stride]
    at::Tensor& dst_k,         // [num_tokens, head_dim]
    at::Tensor& dst_scale,     // [num_tokens, head_dim / quant_block_size * 4]
    const at::Tensor& block_table,   // [batch_size, num_blocks]
    const at::Tensor& cu_seq_lens) { // [batch_size + 1]
    const torch_gcu::OptionalGCUGuard device_guard(device_of(kv_cache));
    const topsStream_t stream = torch_gcu::getCurrentGCUStream();

    std::vector<at::Tensor> out_vector = {dst_k, dst_scale};
    ATEN_ATENOP_CHECK(
      ATEN_ATENOP_CALL(topsvllm::topsvllmCpGatherIndexerKQuantAndCache)(
          out_vector, kv_cache, block_table, cu_seq_lens, stream));

    return;
}

}  // namespace vllm_gcu::llm_ops
