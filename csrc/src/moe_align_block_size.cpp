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

#include "moe_align_block_size.h"

#include <topsaten/topsaten_vllm.h>
#include <torch/all.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void moe_align_block_size(at::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, at::Tensor sorted_token_ids,
                          at::Tensor experts_ids,
                          at::Tensor num_tokens_post_pad) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(topk_ids));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmMoeAlignBlockSize)(
      sorted_token_ids, experts_ids, num_tokens_post_pad, topk_ids, num_experts,
      block_size, stream));
}

}  // namespace vllm_gcu::llm_ops
