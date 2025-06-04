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

#include <tuple>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void fused_add_rms_norm_gcu(at::Tensor &input, at::Tensor &residual,
                            const at::Tensor &weight, double epsilon) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(input));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  at::Tensor device_weight;
  if (!weight.device().is_privateuseone()) {
    device_weight = weight.to(at::kPrivateUse1);
  } else {
    device_weight = weight;
  }

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmFusedAddRmsNorm)(
      input, residual, device_weight, epsilon, stream));
}

void fused_add_rms_norm(at::Tensor &input, at::Tensor &residual,
                        const at::Tensor &weight, double epsilon) {
  if (input.numel() == 0) return;

#ifndef NDEBUG
  auto fallback_ops = c10::utils::get_env("VLLM_GCU_FALLBACK_CPU");
  bool is_fallback = false;
  at::Tensor input_cpu, residual_cpu, weight_cpu;

  if (fallback_ops.has_value()) {
    if (fallback_ops->find("fused_add_rms_norm") != std::string::npos ||
        (*fallback_ops) == "all") {
      is_fallback = true;

      // Convert tensors to CPU for native implementation
      input_cpu = input.to(at::kCPU);
      residual_cpu = residual.to(at::kCPU);
      weight_cpu = weight.to(at::kCPU);

      // Call native implementation on CPU tensors
      vllmFusedAddRmsNorm(input_cpu, residual_cpu, weight_cpu,
                          static_cast<float>(epsilon));
    }
  }
#endif

  fused_add_rms_norm_gcu(input, residual, weight, epsilon);

#ifndef NDEBUG
  if (is_fallback) {
    const topsStream_t stream = torch_gcu::getCurrentGCUStream();
    topsStreamSynchronize(stream);

    auto cpu_output = std::make_tuple(input_cpu);
    auto gcu_output = std::make_tuple(input.to(at::kCPU));
    EXPECT_TRUE(vllmFusedAddRmsNormCheck(cpu_output, gcu_output),
                "fused_add_rms_norm");
    input.copy_(input_cpu);
    residual.copy_(residual_cpu);
  }
#endif
}

}  // namespace vllm_gcu::llm_ops
