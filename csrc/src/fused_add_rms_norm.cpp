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

#include "fused_add_rms_norm.h"

#include <topsaten/topsaten_vllm.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

std::tuple<at::Tensor, at::Tensor>
fused_add_rms_norm_native(const at::Tensor &input, const at::Tensor &residual,
                          const at::Tensor &weight, double epsilon) {
  at::ScalarType orig_dtype = input.scalar_type();

  at::Tensor input_float = input.to(at::kFloat);
  at::Tensor residual_float = residual.to(at::kFloat);
  at::Tensor result = input_float + residual_float;
  at::Tensor residual_out = result.to(orig_dtype);

  at::Tensor variance = result.pow(2).mean(-1, /*keepdim=*/true);

  at::Tensor normalized = result * at::rsqrt(variance + epsilon);
  normalized = normalized.to(orig_dtype);
  at::Tensor final_result = normalized * weight;

  return std::make_tuple(final_result, residual_out);
}

void fused_add_rms_norm(at::Tensor &input, at::Tensor &residual,
                        const at::Tensor &weight, double epsilon) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(input));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  if (input.numel() == 0) return;

  auto use_native = c10::utils::check_env("VLLM_GCU_NATIVE");
  auto fallback_cpu = c10::utils::check_env("VLLM_GCU_FALLBACK_CPU");

  if (use_native) {
    std::tuple<at::Tensor, at::Tensor> ret;
    if (fallback_cpu) {
      ret = TORCH_FALLBACK_CALL(fused_add_rms_norm_native)(input, residual,
                                                           weight, epsilon);
    } else {
      ret = fused_add_rms_norm_native(input, residual, weight, epsilon);
    }
    input.copy_(std::get<0>(ret));
    residual.copy_(std::get<1>(ret));
    return;
  }

  at::Tensor device_weight;
  if (!weight.device().is_privateuseone()) {
    device_weight = weight.to(at::kPrivateUse1);
  } else {
    device_weight = weight;
  }

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmFusedAddRmsNorm)(
      input, residual, device_weight, epsilon, stream));
}

} // namespace vllm_gcu::llm_ops
