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

#include "rms_norm.h"

#include <topsaten/topsaten_te.h>
#include <topsaten/topsaten_vllm.h>

#include <tuple>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void rms_norm_gcu(at::Tensor &out, const at::Tensor &input,
                  const at::Tensor &device_weight, double epsilon) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(input));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  at::Tensor view_out = out.view({-1, out.size(-1)});
  at::Tensor view_input = input.view({-1, input.size(-1)});
  at::Scalar xeps(epsilon);

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmRmsNorm)(
      view_out, view_input, device_weight, xeps, stream));
}

void rms_norm(at::Tensor &out, const at::Tensor &input,
              const at::Tensor &weight, double epsilon) {
  if (input.numel() == 0) return;

  at::Tensor device_weight;
  if (!weight.device().is_privateuseone()) {
    device_weight = weight.to(at::kPrivateUse1);
  } else {
    device_weight = weight;
  }

#ifndef NDEBUG
  auto fallback_ops = c10::utils::get_env("VLLM_GCU_FALLBACK_CPU");
  bool is_fallback = false;
  at::Tensor view_out_cpu, view_input_cpu, device_weight_cpu;

  if (fallback_ops.has_value()) {
    if (fallback_ops->find("rms_norm") != std::string::npos ||
        (*fallback_ops) == "all") {
      is_fallback = true;

      // Convert tensors to CPU for native implementation
      at::Tensor view_out = out.view({-1, out.size(-1)});
      at::Tensor view_input = input.view({-1, input.size(-1)});
      view_out_cpu = view_out.to(at::kCPU);
      view_input_cpu = view_input.to(at::kCPU);
      device_weight_cpu = device_weight.to(at::kCPU);

      // Call native implementation on CPU tensors
      vllmRmsNorm(view_out_cpu, view_input_cpu, device_weight_cpu, epsilon);
    }
  }
#endif

  rms_norm_gcu(out, input, device_weight, epsilon);

#ifndef NDEBUG
  if (is_fallback) {
    const topsStream_t stream = torch_gcu::getCurrentGCUStream();
    topsStreamSynchronize(stream);

    auto cpu_output = std::make_tuple(view_out_cpu);
    auto reshaped_out = out.view({-1, out.size(-1)});
    auto device_outputs = std::make_tuple(reshaped_out.to(at::kCPU));
    EXPECT_TRUE(vllmRmsNormCheck(cpu_output, device_outputs), "rms_norm");
    out.copy_(view_out_cpu.view_as(out));
  }
#endif
}
}  // namespace vllm_gcu::llm_ops
