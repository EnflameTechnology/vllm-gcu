/*
 * Copyright 2025 Enflame. All Rights Reserved.
 */
#pragma once

#include <ATen/ATen.h>
#ifndef NDEBUG
#include "native_vllm.h"
#endif

namespace vllm_gcu::llm_ops {

void moe_align_block_size_pad(at::Tensor topk_ids, at::Tensor topk_ids_size,
                              int64_t num_experts, int64_t block_size,
                              at::Tensor sorted_token_ids,
                              at::Tensor experts_ids,
                              at::Tensor num_tokens_post_pad);

}  // namespace vllm_gcu::llm_ops
