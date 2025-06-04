/**
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "dynamic_per_token_group_fp8_quant.h"

#include <topsaten/topsaten_vllm.h>

#include <tuple>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void dynamic_per_token_group_fp8_quant_gcu(at::Tensor &out, at::Tensor &scale,
                                           const at::Tensor &input,
                                           int64_t group_size) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(out));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  ATEN_ATENOP_CHECK(
      ATEN_ATENOP_CALL(topsvllm::topsvllmDynamicPerTokenGroupFP8Quant)(
          out, scale, input, group_size, stream));
}

void dynamic_per_token_group_fp8_quant(at::Tensor &out, at::Tensor &scale,
                                       const at::Tensor &input,
                                       int64_t group_size) {
  if (input.numel() == 0) return;

#ifndef NDEBUG
  auto fallback_ops = c10::utils::get_env("VLLM_GCU_FALLBACK_CPU");
  bool is_fallback = false;
  at::Tensor out_cpu, scale_cpu, input_cpu;

  if (fallback_ops.has_value()) {
    if (fallback_ops->find("dynamic_per_token_group_fp8_quant") !=
            std::string::npos ||
        (*fallback_ops) == "all") {
      is_fallback = true;

      // Convert tensors to CPU for native implementation
      out_cpu = out.to(at::kCPU);
      scale_cpu = scale.to(at::kCPU);
      input_cpu = input.to(at::kCPU);

      // Call native implementation on CPU tensors
      vllmDynamicPerTokenGroupFP8Quant(out_cpu, scale_cpu, input_cpu,
                                       group_size);
    }
  }
#endif

  dynamic_per_token_group_fp8_quant_gcu(out, scale, input, group_size);

#ifndef NDEBUG
  if (is_fallback) {
    const topsStream_t stream = torch_gcu::getCurrentGCUStream();
    topsStreamSynchronize(stream);

    auto cpu_output = std::make_tuple(out_cpu, scale_cpu);
    auto device_outputs = std::make_tuple(out.to(at::kCPU),
                                          scale.to(at::kCPU));
    EXPECT_TRUE(
        vllmDynamicPerTokenGroupFP8QuantCheck(cpu_output, device_outputs),
        "dynamic_per_token_group_fp8_quant");
    out.copy_(out_cpu);
    scale.copy_(scale_cpu);
  }
#endif
}

}  // namespace vllm_gcu::llm_ops
