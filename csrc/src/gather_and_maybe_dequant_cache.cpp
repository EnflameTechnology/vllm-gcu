/**
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "gather_and_maybe_dequant_cache.h"

#include <topsaten/topsaten_vllm.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void gather_and_maybe_dequant_cache(
    const at::Tensor& src_cache, const at::Tensor& dst,
    const at::Tensor& block_table, const at::Tensor& cu_seq_lens,
    int64_t batch_size, c10::string_view kv_cache_dtype,
    const at::Tensor& scale, const c10::optional<at::Tensor> &seq_starts) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(dst));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();
  at::Tensor seq_starts_tensor;
  if (seq_starts.has_value()) {
    seq_starts_tensor = seq_starts.value();
  }
  at::Tensor dst_kvdtype;
  if (kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3") {
    dst_kvdtype = dst.to(src_cache.dtype());
  } else if (kv_cache_dtype == "auto") {
    dst_kvdtype = dst;
  } else {
    assert(false);
  }

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmGatherCache)(
      src_cache, dst_kvdtype, block_table, cu_seq_lens, batch_size,
      seq_starts_tensor, stream));
  if (kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3") {
    dst.copy_(dst_kvdtype);
  }
}

}  // namespace vllm_gcu::llm_ops
