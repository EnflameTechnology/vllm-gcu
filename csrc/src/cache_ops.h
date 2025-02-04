
#pragma once

#include <ATen/ATen.h>
#include <torch/all.h>

namespace vllm_gcu::llm_ops {
// TODO: use tensor instead of map
void swap_blocks(torch::Tensor &src, torch::Tensor &dst,
                 const torch::Tensor &block_mapping);

void copy_blocks(std::vector<torch::Tensor> const &key_caches,
                 std::vector<torch::Tensor> const &value_caches,
                 const torch::Tensor &block_mapping);

void reshape_and_cache(const at::Tensor &key, const at::Tensor &value,
                       at::Tensor &key_cache, at::Tensor &value_cache,
                       const at::Tensor &slot_mapping,
                       const std::string &kv_cache_dtype, double k_scale,
                       double v_scale, double k_zero, double v_zero);
}  // namespace vllm_gcu::llm_ops
