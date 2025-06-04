/**
 * Copyright 2025 Enflame. All Rights Reserved.
 */
#include "fused_dispatch_decode.h"

#include <topsaten/topsaten_extensions.h>

#include <tuple>
#include <vector>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void fused_dispatch_decode_gcu(at::TensorList outputs,
                               const at::Tensor &recv_packed,
                               const at::Tensor &sp_split_size,
                               at::IntArrayRef split_sizes) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(outputs[0]));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  topsatenSize_t topsaten_split_sizes(split_sizes.data(),
                                      static_cast<int64_t>(split_sizes.size()));

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsexts::topsextsFusedDispatchDecode)(
      outputs, recv_packed, sp_split_size, topsaten_split_sizes, stream));
}

void fused_dispatch_decode(at::TensorList outputs,
                           const at::Tensor &recv_packed,
                           const at::Tensor &sp_split_size,
                           at::IntArrayRef split_sizes) {
#ifndef NDEBUG
  auto fallback_ops = c10::utils::get_env("VLLM_GCU_FALLBACK_CPU");
  bool is_fallback = false;
  std::vector<at::Tensor> outputs_cpu;
  at::Tensor recv_packed_cpu, sp_split_size_cpu;

  if (fallback_ops.has_value()) {
    if (fallback_ops->find("fused_dispatch_decode") != std::string::npos ||
        (*fallback_ops) == "all") {
      is_fallback = true;

      // Convert tensors to CPU for native implementation
      outputs_cpu.reserve(outputs.size());
      for (const auto &tensor : outputs) {
        outputs_cpu.push_back(tensor.to(at::kCPU));
      }
      recv_packed_cpu = recv_packed.to(at::kCPU);
      sp_split_size_cpu = sp_split_size.to(at::kCPU);
      std::vector<int64_t> split_sizes_vec(split_sizes.begin(),
                                           split_sizes.end());

      // Call native implementation on CPU tensors
      extsFusedDispatchDecode(outputs_cpu, recv_packed_cpu, sp_split_size_cpu,
                              split_sizes_vec);
    }
  }
#endif

  fused_dispatch_decode_gcu(outputs, recv_packed, sp_split_size, split_sizes);

#ifndef NDEBUG
  if (is_fallback) {
    const topsStream_t stream = torch_gcu::getCurrentGCUStream();
    topsStreamSynchronize(stream);
    auto concat_outputs_cpu = torch::cat(outputs_cpu, 0);
    auto concat_outputs = torch::cat(outputs, 0);
    auto cpu_output = std::make_tuple(concat_outputs_cpu);
    auto device_outputs = std::make_tuple(concat_outputs.to(at::kCPU));
    EXPECT_TRUE(extsFusedDispatchDecodeCheck(cpu_output, device_outputs),
                "fused_dispatch_decode");

    // Copy results back to original tensors
    for (size_t i = 0; i < outputs.size(); ++i) {
      outputs[i].copy_(outputs_cpu[i]);
    }
  }
#endif
}

}  // namespace vllm_gcu::llm_ops
