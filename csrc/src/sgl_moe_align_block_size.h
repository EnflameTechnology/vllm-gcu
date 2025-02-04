#pragma once

#include <ATen/ATen.h>

namespace vllm_gcu::llm_ops {

void sgl_moe_align_block_size(at::Tensor topk_ids, int64_t num_experts,
                              int64_t block_size, at::Tensor sorted_token_ids,
                              at::Tensor experts_ids,
                              at::Tensor num_tokens_post_pad);

}  // namespace vllm_gcu::llm_ops
