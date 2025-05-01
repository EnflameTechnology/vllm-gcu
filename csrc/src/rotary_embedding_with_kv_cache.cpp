/**
 * Copyright 2025 Enflame. All Rights Reserved.
 */
#include "rotary_embedding_with_kv_cache.h"

#include <topsaten/topsaten_extensions.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void rotary_embedding_with_kv_cache(
    at::Tensor &q_out, at::Tensor &kv_cache, const at::Tensor &q,
    const at::Tensor &kv, const at::Tensor &positions,
    const at::Tensor &cos_sin_cache, const at::Tensor &weight,
    const at::Tensor &slot_mapping, const at::Tensor &scale, double eps,
    at::IntArrayRef split_size, c10::string_view kv_cache_dtype) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(q_out));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  topsatenSize_t topsaten_split_sizes(split_size.data(),
                                      static_cast<int64_t>(split_size.size()));

  const char *kv_dtype = kv_cache_dtype.data();
  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(
      topsexts::topsextsRotaryEmbeddingWithKVCache)(
      q_out, kv_cache, q, kv, positions, cos_sin_cache, weight, slot_mapping,
      scale, static_cast<double>(eps), topsaten_split_sizes, kv_dtype, stream));
}

}  // namespace vllm_gcu::llm_ops
