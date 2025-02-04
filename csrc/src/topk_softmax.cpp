/*
 * Copyright 2021-2023 Enflame. All Rights Reserved.

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

#include "topk_softmax.h"

#include <topsaten/topsaten_vllm.h>
#include <torch/all.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void topk_softmax(at::Tensor &topk_weights, at::Tensor &topk_indices,
                  at::Tensor &token_expert_indices, at::Tensor &gating_output) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(topk_indices));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmTopkSoftmax)(
      topk_weights, topk_indices, token_expert_indices, gating_output, stream));
}

}  // namespace vllm_gcu::llm_ops
