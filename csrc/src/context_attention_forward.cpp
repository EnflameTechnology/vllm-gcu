/*
 * Copyright 2024 Enflame. All Rights Reserved.

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

#include "context_attention_forward.h"

#include <topsaten/topsaten_vllm.h>
#include <torch/all.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void context_attention_forward(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    at::Tensor &o, const at::Tensor &key_cache, const at::Tensor &value_cache,
    const at::Tensor &block_tables, const at::Tensor &subquery_start_loc,
    const at::Tensor &seq_lens_tensor, const at::Tensor &context_lens,
    int64_t max_query_len, const c10::optional<at::Tensor> &alibi_slopes,
    c10::optional<int64_t> sliding_window) {
  at::Scalar scaler_max_query_len(max_query_len);

  at::Tensor alibi_slopes_tensor;
  if (alibi_slopes.has_value()) {
    alibi_slopes_tensor = alibi_slopes.value();
  }

  int64_t sliding_window_value = 0;
  if (sliding_window.has_value()) {
    sliding_window_value = sliding_window.value();
  }
  at::Scalar scaler_sliding_window(sliding_window_value);

  const torch_gcu::OptionalGCUGuard device_guard(device_of(q));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmContextAttentionForward)(
      o, q, k, v, key_cache, value_cache, block_tables, subquery_start_loc,
      seq_lens_tensor, context_lens, scaler_max_query_len, alibi_slopes_tensor,
      scaler_sliding_window, stream));
}

}  // namespace vllm_gcu::llm_ops
