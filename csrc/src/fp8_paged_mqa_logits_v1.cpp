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

#include "fp8_paged_mqa_logits_v1.h"

#include <topsaten/topsaten_deepgemm.h>
#include <torch/all.h>

#include <tuple>
#include <vector>

#include "get_paged_mqa_logits_metadata_v1.h"
#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
at::Tensor get_paged_mqa_logits_metadata_v1(const at::Tensor& context_lens,
                                            int64_t block_kv, int64_t num_sms,
                                            int64_t threshold) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(context_lens));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();
  auto out = torch::empty({num_sms + 1, 2},
                          context_lens.options().dtype(torch::kInt32));

  at::Scalar block_kv_scalar(block_kv);
  at::Scalar num_sms_scalar(num_sms);
  at::Scalar threshold_scalar(threshold);

  ATEN_ATENOP_CHECK(
      ATEN_ATENOP_CALL(topsdeepgemm::topsdeepgemmGetPagedMqaLogitsMetaData_V1)(
          out, context_lens, block_kv_scalar, num_sms_scalar, threshold_scalar,
          stream));

  return out;
}
at::Tensor fp8_paged_mqa_logits_v1(
    const at::Tensor& q, const at::Tensor& fused_kv_cache,
    const at::Tensor& weights, const at::Tensor& context_lens,
    const at::Tensor& block_table, const at::Tensor& schedule_meta,
    int64_t max_context_len, bool clean_logits, int64_t threshold) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(q));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();
  auto batch_size = q.size(0);
  auto next_n = q.size(1);
  auto out = torch::empty({batch_size * next_n, max_context_len},
                          q.options().dtype(torch::kFloat));

  at::Scalar max_context_len_scalar(max_context_len);
  at::Scalar threshold_scalar(threshold);

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(
      topsdeepgemm::topsdeepgemmFp8PagedMqaLogits_V1)(
      out, q, fused_kv_cache, weights, context_lens, block_table, schedule_meta,
      max_context_len_scalar, threshold_scalar, clean_logits, stream));

  return out;
}
}  // namespace vllm_gcu::llm_ops
