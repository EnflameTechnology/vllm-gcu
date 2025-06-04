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

#include "weight_only_quant.h"

#include <topsaten/topsaten_ops.h>

#include <tuple>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void weight_only_quant_gcu(at::Tensor &output, const at::Tensor &input,
                           const at::Tensor &qweight,
                           const at::Tensor &bias_tensor,
                           const at::Tensor &scale, int64_t group_size) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(input));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsaten::topsatenLinearQuant)(
      output, input, qweight, bias_tensor, scale, static_cast<int>(group_size),
      stream));
}

// output, x, qweight, None, qscales
void weight_only_quant(at::Tensor &output, const at::Tensor &input,
                       const at::Tensor &qweight,
                       const c10::optional<at::Tensor> &bias,
                       const at::Tensor &scale, int64_t group_size = -1) {
  at::Tensor bias_tensor;
  if (bias.has_value()) {
    bias_tensor = bias.value();
  }

#ifndef NDEBUG
  auto fallback_ops = c10::utils::get_env("VLLM_GCU_FALLBACK_CPU");
  bool is_fallback = false;
  at::Tensor output_cpu, input_cpu, qweight_cpu, scale_cpu, bias_tensor_cpu;

  if (fallback_ops.has_value()) {
    if (fallback_ops->find("weight_only_quant") != std::string::npos ||
        (*fallback_ops) == "all") {
      is_fallback = true;

      // Convert tensors to CPU for native implementation
      output_cpu = output.to(at::kCPU);
      input_cpu = input.to(at::kCPU);
      qweight_cpu = qweight.to(at::kCPU);
      scale_cpu = scale.to(at::kCPU);
      if (bias.has_value()) {
        bias_tensor_cpu = bias_tensor.to(at::kCPU);
      }

      // Call native implementation on CPU tensors
      // Note: Assuming there's a corresponding native function
      atenLinearQuant(output_cpu, input_cpu, qweight_cpu, bias_tensor_cpu,
                      scale_cpu, scale_cpu);
    }
  }
#endif

  weight_only_quant_gcu(output, input, qweight, bias_tensor, scale, group_size);

#ifndef NDEBUG
  if (is_fallback) {
    const topsStream_t stream = torch_gcu::getCurrentGCUStream();
    topsStreamSynchronize(stream);

    auto cpu_output = std::make_tuple(output_cpu);
    auto device_outputs = std::make_tuple(output.to(at::kCPU));
    EXPECT_TRUE(atenLinearQuantCheck(cpu_output, device_outputs),
                "weight_only_quant");
    output.copy_(output_cpu);
  }
#endif
}

}  // namespace vllm_gcu::llm_ops
