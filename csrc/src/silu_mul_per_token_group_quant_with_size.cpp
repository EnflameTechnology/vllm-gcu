/**
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "silu_mul_per_token_group_quant_with_size.h"

#include <topsaten/topsaten_extensions.h>

#include <tuple>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void silu_mul_per_token_group_quant_with_size_gcu(at::Tensor &out,
                                                  at::Tensor &scale,
                                                  const at::Tensor &input,
                                                  const at::Tensor &size,
                                                  int64_t group_size) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(out));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  ATEN_ATENOP_CHECK(
      ATEN_ATENOP_CALL(topsexts::topsextsSiluMulPerTokenGroupQuant)(
          out, scale, input, size, group_size, stream));
}

void silu_mul_per_token_group_quant_with_size(at::Tensor &out,
                                              at::Tensor &scale,
                                              const at::Tensor &input,
                                              const at::Tensor &size,
                                              int64_t group_size) {
  if (input.numel() == 0) return;

#ifndef NDEBUG
  auto fallback_ops = c10::utils::get_env("VLLM_GCU_FALLBACK_CPU");
  bool is_fallback = false;
  at::Tensor out_cpu, scale_cpu, input_cpu, size_cpu;

  if (fallback_ops.has_value()) {
    if (fallback_ops->find("silu_mul_per_token_group_quant_with_size") !=
            std::string::npos ||
        (*fallback_ops) == "all") {
      is_fallback = true;

      // Log fallback CPU usage
      VLLM_FALLBACK_CPU_LOG("silu_mul_per_token_group_quant_with_size",
                            "Using CPU fallback implementation");

      // Convert tensors to CPU for native implementation
      out_cpu = out.to(at::kCPU);
      scale_cpu = scale.to(at::kCPU);
      input_cpu = input.to(at::kCPU);
      size_cpu = size.to(at::kCPU);

      // Call native implementation on CPU tensors
      extsSiluMulPerTokenGroupQuant(out_cpu, scale_cpu, input_cpu, size_cpu,
                                    static_cast<int32_t>(group_size));

      VLLM_FALLBACK_CPU_LOG("silu_mul_per_token_group_quant_with_size",
                            "CPU fallback computation completed");
    }
  }
#endif

  silu_mul_per_token_group_quant_with_size_gcu(out, scale, input, size,
                                               group_size);

#ifndef NDEBUG
  if (is_fallback) {
    const topsStream_t stream = torch_gcu::getCurrentGCUStream();
    topsStreamSynchronize(stream);

    VLLM_FALLBACK_CPU_LOG("silu_mul_per_token_group_quant_with_size",
                          "Starting result verification");
    auto cpu_output = std::make_tuple(out_cpu, scale_cpu);
    auto device_outputs = std::make_tuple(out.to(at::kCPU),
                                          scale.to(at::kCPU));
    EXPECT_TRUE(extsSiluMulPerTokenGroupQuantCheck(cpu_output, device_outputs),
                "silu_mul_per_token_group_quant_with_size");
    out.copy_(out_cpu);
    scale.copy_(scale_cpu);
    VLLM_FALLBACK_CPU_LOG("silu_mul_per_token_group_quant_with_size",
                          "Fallback CPU results copied back to device");
  }
#endif
}

}  // namespace vllm_gcu::llm_ops
