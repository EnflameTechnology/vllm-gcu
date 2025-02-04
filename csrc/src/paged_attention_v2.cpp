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

#include "paged_attention_v2.h"

#include <topsaten/topsaten_vllm.h>
#include <torch/all.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void paged_attention_v2(
    at::Tensor &out, const at::Tensor &exp_sums, const at::Tensor &max_logits,
    const at::Tensor &tmp_output, const at::Tensor &query,
    const at::Tensor &key_cache, const at::Tensor &value_cache,
    int64_t num_kv_heads, double scale, const at::Tensor &block_tables,
    const at::Tensor &context_lens, int64_t block_size, int64_t max_context_len,
    const c10::optional<at::Tensor> &alibi_slopes,
    c10::string_view kv_cache_dtype, double k_scale, double v_scale,
    int64_t tp_rank, int64_t blocksparse_local_blocks,
    int64_t blocksparse_vert_stride, int64_t blocksparse_block_size,
    int64_t blocksparse_head_sliding_step, double k_zero, double v_zero,
    const c10::optional<at::Tensor> &out_scales) {
  TORCH_CHECK(blocksparse_vert_stride <= 0,
              "block sparse attention is not supported for gcu");
  const torch_gcu::OptionalGCUGuard device_guard(device_of(query));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  int64_t num_heads = query.size(1);
  int64_t num_queries_per_kv = num_heads / num_kv_heads;
  torch::Tensor head_mapping_tensor;

  const char *kv_dtype = kv_cache_dtype.data();
  at::Scalar scale_scalar(scale);
  at::Scalar block_size_scalar(block_size);
  at::Scalar max_context_len_scalar(max_context_len);

  at::Scalar k_scale_scalar(k_scale);
  at::Scalar k_zp_scalar(k_zero);
  at::Scalar v_scale_scalar(v_scale);
  at::Scalar v_zp_scalar(v_zero);

  at::Tensor out_scales_tensor;

  at::Tensor alibi_slopes_tensor;
  if (alibi_slopes.has_value()) {
    alibi_slopes_tensor = alibi_slopes.value();
  }

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmPagedAttentionV1)(
      out, query, key_cache, value_cache, head_mapping_tensor, scale_scalar,
      block_tables, context_lens, block_size_scalar, max_context_len_scalar,
      alibi_slopes_tensor, kv_dtype, k_scale_scalar, k_zp_scalar,
      v_scale_scalar, v_zp_scalar, out_scales_tensor, stream));
}

}  // namespace vllm_gcu::llm_ops
