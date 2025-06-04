/**
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "fused_add_rms_norm_per_token_group_quant_fp8.h"

#include <topsaten/topsaten_vllm.h>

#include <tuple>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void fused_add_rms_norm_per_token_group_quant_fp8_gcu(
    at::Tensor &out, at::Tensor &residual, at::Tensor &scale,
    const at::Tensor &input, const at::Tensor &weight, double epsilon,
    int64_t group_size) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(out));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  ATEN_ATENOP_CHECK(
      ATEN_ATENOP_CALL(topsvllm::topsvllmFusedAddRmsNormPerTokenGroupQuantFp8)(
          out, residual, scale, input, const_cast<at::Tensor &>(residual),
          weight, epsilon, group_size, stream));
}

void fused_add_rms_norm_per_token_group_quant_fp8(
    at::Tensor &out, at::Tensor &residual, at::Tensor &scale,
    const at::Tensor &input, const at::Tensor &weight, double epsilon,
    int64_t group_size) {
  if (input.numel() == 0) return;

#ifndef NDEBUG
  auto fallback_ops = c10::utils::get_env("VLLM_GCU_FALLBACK_CPU");
  bool is_fallback = false;
  at::Tensor out_cpu, residual_cpu, scale_cpu, input_cpu, weight_cpu;

  if (fallback_ops.has_value()) {
    if (fallback_ops->find("fused_add_rms_norm_per_token_group_quant_fp8") !=
            std::string::npos ||
        (*fallback_ops) == "all") {
      is_fallback = true;

      // Convert tensors to CPU for native implementation
      out_cpu = out.to(at::kCPU);
      residual_cpu = residual.to(at::kCPU);
      scale_cpu = scale.to(at::kCPU);
      input_cpu = input.to(at::kCPU);
      weight_cpu = weight.to(at::kCPU);

      // Call native implementation on CPU tensors
      vllmFusedAddRmsNormPerTokenGroupQuantFp8(
          out_cpu, residual_cpu, scale_cpu, input_cpu, residual_cpu, weight_cpu,
          static_cast<float>(epsilon), group_size);
    }
  }
#endif

  fused_add_rms_norm_per_token_group_quant_fp8_gcu(out, residual, scale, input,
                                                   weight, epsilon, group_size);

#ifndef NDEBUG
  if (is_fallback) {
    const topsStream_t stream = torch_gcu::getCurrentGCUStream();
    topsStreamSynchronize(stream);

    auto cpu_output = std::make_tuple(out_cpu, residual_cpu, scale_cpu);
    auto gcu_output = std::make_tuple(out.to(at::kCPU),
                                      residual.to(at::kCPU),
                                      scale.to(at::kCPU));
    EXPECT_TRUE(
        vllmFusedAddRmsNormPerTokenGroupQuantFp8Check(cpu_output, gcu_output),
        "fused_add_rms_norm_per_token_group_quant_fp8");
    out.copy_(out_cpu);
    residual.copy_(residual_cpu);
    scale.copy_(scale_cpu);
  }
#endif
}

}  // namespace vllm_gcu::llm_ops
