/*
 * Copyright 2025 Enflame. All Rights Reserved.

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

#include "get_ep_indices.h"

#include <topsaten/topsaten_vllm.h>
#include <torch/all.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
// x, qweight, zeros, scale, g_idx, bit, bias, group_size
void get_ep_indices(at::Tensor &ep_count, at::Tensor &ep_token_indices,
                    at::Tensor &ep_valid_token_indices,
                    const at::Tensor &topk_ids, int64_t expert_per_rank,
                    int64_t ep_size) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(topk_ids));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmGetEPIndices)(
      ep_count, ep_token_indices, ep_valid_token_indices, topk_ids,
      static_cast<int>(expert_per_rank), static_cast<int>(ep_size), stream));
}
}  // namespace vllm_gcu::llm_ops
