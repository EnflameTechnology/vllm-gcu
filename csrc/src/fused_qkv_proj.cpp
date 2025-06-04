/**
 * Copyright 2025 Enflame. All Rights Reserved.
 */
#include "fused_qkv_proj.h"

#include <topsaten/topsaten_extensions.h>

#include <tuple>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void fused_qkv_proj_gcu(at::Tensor &q, at::Tensor &kv, const at::Tensor &x,
                        const at::Tensor &weight, const at::Tensor &x_scale,
                        const at::Tensor &weight_scale, int64_t group_size) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(q));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsexts::topsextsFusedQKVProj)(
      q, kv, x, weight, x_scale, weight_scale, group_size, stream));
}

void fused_qkv_proj(at::Tensor &q, at::Tensor &kv, const at::Tensor &x,
                    const at::Tensor &weight, const at::Tensor &x_scale,
                    const at::Tensor &weight_scale, int64_t group_size) {
#ifndef NDEBUG
  auto fallback_ops = c10::utils::get_env("VLLM_GCU_FALLBACK_CPU");
  bool is_fallback = false;
  at::Tensor q_cpu, kv_cpu, x_cpu, weight_cpu, x_scale_cpu, weight_scale_cpu;

  if (fallback_ops.has_value()) {
    if (fallback_ops->find("fused_qkv_proj") != std::string::npos) {
      is_fallback = true;

      // Convert tensors to CPU for native implementation
      q_cpu = q.to(at::kCPU);
      kv_cpu = kv.to(at::kCPU);
      x_cpu = x.to(at::kCPU);
      weight_cpu = weight.to(at::kCPU);
      x_scale_cpu = x_scale.to(at::kCPU);
      weight_scale_cpu = weight_scale.to(at::kCPU);

      // Call native implementation on CPU tensors
      extsFusedQKVProj(q_cpu, kv_cpu, x_cpu, weight_cpu, x_scale_cpu,
                       weight_scale_cpu, group_size);
    }
  }
#endif

  fused_qkv_proj_gcu(q, kv, x, weight, x_scale, weight_scale, group_size);

#ifndef NDEBUG
  if (is_fallback) {
    auto cpu_output = std::make_tuple(q_cpu, kv_cpu);
    auto gcu_output = std::make_tuple(q.to(at::kCPU), kv.to(at::kCPU));
    EXPECT_TRUE(extsFusedQKVProjCheck(cpu_output, gcu_output),
                "fused_qkv_proj");
    q.copy_(q_cpu);
    kv.copy_(kv_cpu);
  }
#endif
}

}  // namespace vllm_gcu::llm_ops
