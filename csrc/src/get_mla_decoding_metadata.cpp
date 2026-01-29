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

#include "get_mla_decoding_metadata.h"

#include <topsaten/topsaten_vllm.h>
#include <torch/all.h>

#include <tuple>
#include <vector>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void get_mla_decoding_metadata(at::Tensor& out, const at::Tensor& seqlens_k,
                               int64_t num_q_tokens_per_head_k, int64_t h_k,
                               std::optional<int64_t> h_q, bool is_fp8_kvcache,
                               std::optional<int64_t> topk,
                               std::optional<int64_t> threshold,
                               const std::optional<at::Tensor>& cu_seq_q) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(seqlens_k));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();
  bool is_sparse_attn = topk.has_value();
  if (is_sparse_attn) {
    TORCH_CHECK(h_q.has_value(),
                "num_heads_q must be provided when topk is provided");
    TORCH_CHECK(threshold.has_value(),
                "threshold must be provided when topk is provided");
    TORCH_CHECK(cu_seq_q.has_value(),
                "cu_seq_q must be provided when topk is provided");
    at::Scalar topk_scalar(topk.value());
    at::Scalar h_q_scalar(h_q.value());
    at::Scalar threshold_scalar(threshold.value());

    ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmFwdKvcacheMlaMetaData)(
        out, seqlens_k, topk_scalar, threshold_scalar, cu_seq_q.value(),
        h_q_scalar, stream));
  } else {
    ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmFwdKvcacheMlaMetaData)(
        out, seqlens_k, stream));
  }
}

}  // namespace vllm_gcu::llm_ops
