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

#include "fused_moe_quant_kernel.h"

#include <c10/util/env.h>
#include <topsaten/topsaten_vllm.h>
#include <torch/all.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void fused_moe_quant_kernel(
    at::Tensor& C, const at::Tensor& A, const at::Tensor& B,
    const c10::optional<at::Tensor>& A_scale, const at::Tensor& B_scale,
    int64_t gs, const c10::optional<at::Tensor>& B_zp,
    const at::Tensor& topk_weights, const at::Tensor& topk_ids,
    const at::Tensor& sorted_token_ids, const at::Tensor& experts_ids,
    const at::Tensor& num_tokens_post_pad, bool mul_routed_weight, int64_t topk,
    int64_t block_size, const c10::optional<at::Tensor>& bias) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(C));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  TORCH_CHECK(!bias.has_value(), "bias should be None.");

  if (A_scale.has_value()) {
    at::Tensor A_scale_tensor = A_scale.value();
    ATEN_ATENOP_CHECK(
        ATEN_ATENOP_CALL(topsvllm::topsvllmInvokeFusedMoeNonGatherQuantKernel)(
            C, A, B, A_scale_tensor, B_scale, topk_weights, topk_ids,
            sorted_token_ids, experts_ids, num_tokens_post_pad,
            mul_routed_weight, topk, block_size, stream));
  } else {
    at::Tensor B_zp_tensor;
    if (B_zp.has_value()) {
      B_zp_tensor = B_zp.value();
    }
    ATEN_ATENOP_CHECK(
        ATEN_ATENOP_CALL(topsvllm::topsvllmInvokeFusedMoeNonGatherQuantKernel)(
            C, A, B, B_scale, gs, B_zp_tensor, topk_weights, topk_ids,
            sorted_token_ids, experts_ids, num_tokens_post_pad,
            mul_routed_weight, topk, block_size, stream));
  }
}

}  // namespace vllm_gcu::llm_ops
