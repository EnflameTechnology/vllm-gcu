/**
 * Copyright 2025 Enflame. All Rights Reserved.
 */
#include "reshape_and_cache_flash_int8kv.h"

#include <topsaten/topsaten_vllm.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu {
namespace llm_ops {

void reshape_and_cache_flash_int8kv(
    const at::Tensor& key, const at::Tensor& value,
    at::Tensor& key_cache, at::Tensor& value_cache,
    const at::Tensor& slot_mapping,
    c10::string_view kv_cache_dtype,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale,
    const at::Tensor& k_zp,
    const at::Tensor& v_zp
) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(key_cache));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  const char *kv_dtype = kv_cache_dtype.data();

  ATEN_ATENOP_CHECK(
    ATEN_ATENOP_CALL(topsvllm::topsvllmReshapeAndCacheFlashInt8KV)(
      key_cache, value_cache,
      key, value,
      slot_mapping,
      kv_dtype,
      k_scale, v_scale,
      k_zp, v_zp,
      stream));
}

}  // namespace llm_ops
}  // namespace vllm_gcu
