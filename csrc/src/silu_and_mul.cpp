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

#include "silu_and_mul.h"

#include <ATen/ATen.h>
#include <topsaten/topsaten_vllm.h>

#include <tuple>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

at::Tensor silu_and_mul_native(const at::Tensor &input) {
  int64_t d = input.size(-1) / 2;
  auto left = input.slice(-1, 0, d);
  auto right = input.slice(-1, d);

  auto res = left * at::sigmoid(left) * right;
  return res;
}

void silu_and_mul_gcu(at::Tensor &out, const at::Tensor &input) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(input));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  at::Tensor view_out = out.view({-1, out.size(-1)});
  at::Tensor view_input = input.view({-1, input.size(-1)});
  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmSiluAndMul)(
      view_out, view_input, stream));
}

void silu_and_mul(at::Tensor &out, const at::Tensor &input) {
  auto use_native = c10::utils::check_env("VLLM_GCU_NATIVE");
  auto fallback_cpu = c10::utils::check_env("VLLM_GCU_FALLBACK_CPU");

  if (input.numel() == 0) return;

  if (use_native) {
    at::Tensor res;
    if (fallback_cpu) {
      res = TORCH_FALLBACK_CALL(silu_and_mul_native)(input);
    } else {
      res = silu_and_mul_native(input);
    }
    out.copy_(res);
    return;
  }

#ifndef NDEBUG
  auto fallback_ops = c10::utils::get_env("VLLM_GCU_FALLBACK_CPU");
  bool is_fallback = false;
  at::Tensor out_cpu, input_cpu;

  if (fallback_ops.has_value()) {
    if (fallback_ops->find("silu_and_mul") != std::string::npos ||
        (*fallback_ops) == "all") {
      is_fallback = true;

      // Log fallback CPU usage
      VLLM_FALLBACK_CPU_LOG("silu_and_mul",
                            "Using CPU fallback implementation");

      // Convert tensors to CPU for native implementation
      out_cpu = out.to(at::kCPU);
      input_cpu = input.to(at::kCPU);

      // Call native implementation on CPU tensors
      vllmSiluAndMul(out_cpu, input_cpu);

      VLLM_FALLBACK_CPU_LOG("silu_and_mul",
                            "CPU fallback computation completed");
    }
  }
#endif

  silu_and_mul_gcu(out, input);

#ifndef NDEBUG
  if (is_fallback) {
    const topsStream_t stream = torch_gcu::getCurrentGCUStream();
    topsStreamSynchronize(stream);

    VLLM_FALLBACK_CPU_LOG("silu_and_mul", "Starting result verification");
    auto cpu_output = std::make_tuple(out_cpu);
    auto device_outputs = std::make_tuple(out.to(at::kCPU));
    EXPECT_TRUE(vllmSiluAndMulCheck(cpu_output, device_outputs),
                "silu_and_mul");
    out.copy_(out_cpu);
    VLLM_FALLBACK_CPU_LOG("silu_and_mul",
                          "Fallback CPU results copied back to device");
  }
#endif
}

}  // namespace vllm_gcu::llm_ops
