/**
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "linear_quant.h"

#include <topsaten/topsaten_vllm.h>

#include <iostream>
#include <tuple>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void linear_quant_gcu(at::Tensor &out, const at::Tensor &lhs,
                      const at::Tensor &rhs, const at::Tensor &bias_tensor,
                      const at::Tensor &lhs_scale,
                      const at::Tensor &rhs_scale) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(out));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsaten::topsatenLinearQuant)(
      out, lhs, rhs, bias_tensor, lhs_scale, rhs_scale, stream));
}

void linear_quant(at::Tensor &out, const at::Tensor &lhs, const at::Tensor &rhs,
                  const c10::optional<at::Tensor> &bias,
                  const at::Tensor &lhs_scale, const at::Tensor &rhs_scale) {
  if (lhs.numel() == 0) return;

  at::Tensor bias_tensor;
  if (bias.has_value()) {
    bias_tensor = bias.value();
  }

#ifndef NDEBUG
  auto fallback_ops = c10::utils::get_env("VLLM_GCU_FALLBACK_CPU");
  bool is_fallback = false;
  at::Tensor out_cpu, lhs_cpu, rhs_cpu, bias_tensor_cpu, lhs_scale_cpu,
      rhs_scale_cpu;

  if (fallback_ops.has_value()) {
    if (fallback_ops->find("linear_quant") != std::string::npos ||
        (*fallback_ops) == "all") {
      is_fallback = true;

      // Log fallback CPU usage
      VLLM_FALLBACK_CPU_LOG("linear_quant",
                            "Using CPU fallback implementation");

      // Convert tensors to CPU for native implementation
      out_cpu = out.to(at::kCPU);
      lhs_cpu = lhs.to(at::kCPU);
      rhs_cpu = rhs.to(at::kCPU);
      //bias_tensor_cpu = bias_tensor.to(at::kCPU);
      if (bias.has_value()) {
        bias_tensor_cpu = bias.value().to(at::kCPU);
      } else {
        bias_tensor_cpu = at::Tensor();
      }
      lhs_scale_cpu = lhs_scale.to(at::kCPU);
      rhs_scale_cpu = rhs_scale.to(at::kCPU);

      // Call native implementation on CPU tensors
      atenLinearQuant(out_cpu, lhs_cpu, rhs_cpu, bias_tensor_cpu, lhs_scale_cpu,
                      rhs_scale_cpu);

      VLLM_FALLBACK_CPU_LOG("linear_quant",
                            "CPU fallback computation completed");
    }
  }
#endif

  linear_quant_gcu(out, lhs, rhs, bias_tensor, lhs_scale, rhs_scale);

#ifndef NDEBUG
  if (is_fallback) {
    const topsStream_t stream = torch_gcu::getCurrentGCUStream();
    topsStreamSynchronize(stream);

    VLLM_FALLBACK_CPU_LOG("linear_quant", "Starting result verification");

    auto cpu_output = std::make_tuple(out_cpu);
    auto device_outputs = std::make_tuple(out.to(at::kCPU));
    EXPECT_TRUE(atenLinearQuantCheck(cpu_output, device_outputs),
                "linear_quant");

    out.copy_(out_cpu);

    VLLM_FALLBACK_CPU_LOG("linear_quant",
                          "Fallback CPU results copied back to device");
  }
#endif
}

}  // namespace vllm_gcu::llm_ops
