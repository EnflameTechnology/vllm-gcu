/**
 * Copyright 2025 Enflame. All Rights Reserved.
 */
#include "exts_moe_align_block_size.h"

#include <topsaten/topsaten_extensions.h>

#include <tuple>
#include <vector>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void exts_moe_align_block_size_gcu(at::Tensor& sorted_token_ids,
                                   at::Tensor& experts_ids,
                                   at::Tensor& num_tokens_post_pad,
                                   const at::Tensor& topk_ids,
                                   const at::Tensor& real_token_num,
                                   const at::Tensor& expert_map,
                                   int64_t num_experts, int64_t block_size) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(sorted_token_ids));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsexts::topsextsMoeAlignBlockSize)(
      sorted_token_ids, experts_ids, num_tokens_post_pad, topk_ids,
      real_token_num, expert_map, static_cast<int>(num_experts),
      static_cast<int>(block_size), stream));
}

void exts_moe_align_block_size(at::Tensor& sorted_token_ids,
                               at::Tensor& experts_ids,
                               at::Tensor& num_tokens_post_pad,
                               const at::Tensor& topk_ids,
                               const at::Tensor& real_token_num,
                               const at::Tensor& expert_map,
                               int64_t num_experts, int64_t block_size) {
#ifndef NDEBUG
  auto fallback_ops = c10::utils::get_env("VLLM_GCU_FALLBACK_CPU");
  bool is_fallback = false;
  at::Tensor sorted_token_ids_cpu, experts_ids_cpu, num_tokens_post_pad_cpu;
  at::Tensor topk_ids_cpu, real_token_num_cpu, expert_map_cpu;

  if (fallback_ops.has_value()) {
    if (fallback_ops->find("ets_moe_align_block_size") != std::string::npos ||
        (*fallback_ops) == "all") {
      is_fallback = true;

      // Convert tensors to CPU for native implementation
      sorted_token_ids_cpu = sorted_token_ids.to(at::kCPU);
      experts_ids_cpu = experts_ids.to(at::kCPU);
      num_tokens_post_pad_cpu = num_tokens_post_pad.to(at::kCPU);
      topk_ids_cpu = topk_ids.to(at::kCPU);
      real_token_num_cpu = real_token_num.to(at::kCPU);
      expert_map_cpu = expert_map.to(at::kCPU);

      // Call native implementation on CPU tensors
      extsMoeAlignBlockSize(sorted_token_ids_cpu, experts_ids_cpu,
                            num_tokens_post_pad_cpu, topk_ids_cpu,
                            real_token_num_cpu, expert_map_cpu, num_experts,
                            block_size);
    }
  }
#endif

  exts_moe_align_block_size_gcu(sorted_token_ids, experts_ids,
                                num_tokens_post_pad, topk_ids, real_token_num,
                                expert_map, num_experts, block_size);

#ifndef NDEBUG
  if (is_fallback) {
    const topsStream_t stream = torch_gcu::getCurrentGCUStream();
    topsStreamSynchronize(stream);

    auto cpu_output = std::make_tuple(sorted_token_ids_cpu, experts_ids_cpu,
                                      num_tokens_post_pad_cpu);
    auto device_outputs =
        std::make_tuple(sorted_token_ids.to(at::kCPU),
                        experts_ids.to(at::kCPU),
                        num_tokens_post_pad.to(at::kCPU));
    EXPECT_TRUE(extsMoeAlignBlockSizeCheck(cpu_output, device_outputs),
                "ets_moe_align_block_size");
    sorted_token_ids.copy_(sorted_token_ids_cpu);
    experts_ids.copy_(experts_ids_cpu);
    num_tokens_post_pad.copy_(num_tokens_post_pad_cpu);
  }
#endif
}

}  // namespace vllm_gcu::llm_ops
