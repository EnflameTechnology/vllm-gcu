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

#include "fused_moe_kernel.h"

#include <c10/util/env.h>
#include <topsaten/topsaten_vllm.h>
#include <torch/all.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void fused_moe_kernel(at::Tensor &C, const at::Tensor &A, const at::Tensor &B,
                      const at::Tensor &topk_weights,
                      const at::Tensor &topk_ids,
                      const at::Tensor &sorted_topk_ids,
                      const at::Tensor &expert_ids,
                      const at::Tensor &num_tokens_post_pad,
                      bool mul_routed_weight, int64_t topk, int64_t block_size,
                      const c10::optional<at::Tensor> &bias) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(A));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  auto use_legacy = c10::utils::check_env("VLLM_FUSED_MOE_IMPL_LEGACY");
  if (use_legacy) {
    if (bias.has_value()) {
      at::Tensor bias_tensor = bias.value();
      ATEN_ATENOP_CHECK(
          ATEN_ATENOP_CALL(topsvllm::topsvllmInvokeFusedMoeKernel)(
              C, A, B, bias_tensor, topk_weights, topk_ids, sorted_topk_ids,
              expert_ids, num_tokens_post_pad, mul_routed_weight, topk,
              block_size, stream));
    } else {
      ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(
          topsvllm::topsvllmInvokeFusedMoeKernel)(
          C, A, B, topk_weights, topk_ids, sorted_topk_ids, expert_ids,
          num_tokens_post_pad, mul_routed_weight, topk, block_size, stream));
    }
  } else {
    if (bias.has_value()) {
      at::Tensor bias_tensor = bias.value();
      ATEN_ATENOP_CHECK(
          ATEN_ATENOP_CALL(topsvllm::topsvllmInvokeFusedMoeNonGatherKernel)(
              C, A, B, bias_tensor, topk_weights, topk_ids, sorted_topk_ids,
              expert_ids, num_tokens_post_pad, mul_routed_weight, topk,
              block_size, stream));
    } else {
      ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(
          topsvllm::topsvllmInvokeFusedMoeNonGatherKernel)(
          C, A, B, topk_weights, topk_ids, sorted_topk_ids, expert_ids,
          num_tokens_post_pad, mul_routed_weight, topk, block_size, stream));
    }
  }
}

}  // namespace vllm_gcu::llm_ops
